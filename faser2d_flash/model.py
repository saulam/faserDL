from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .attention import PackedSelfAttention
from .dataset import NUM_TASKS, ProjectionBatch
from .tokenization import (
    LONGITUDINAL_PATCH,
    MAX_PATCH_PIXELS,
    TRANSVERSE_PATCH,
    XY_PATCH,
)


class DropPath(nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class FourierPositionEmbedding(nn.Module):
    def __init__(self, dim: int, num_bands: int = 16):
        super().__init__()
        self.num_bands = int(num_bands)
        frequencies = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        input_dim = 3 + 3 * 2 * num_bands
        self.projection = nn.Linear(input_dim, dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * positions.unsqueeze(-1) * self.frequencies
        encoded = torch.cat(
            (
                positions,
                torch.sin(angles).flatten(1),
                torch.cos(angles).flatten(1),
            ),
            dim=-1,
        )
        return self.projection(encoded)


class LocalPatchEncoder(nn.Module):
    def __init__(self, height: int, width: int, dim: int, channels: int):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        pixels = self.height * self.width
        self.convolution = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.projection = nn.Linear(channels * pixels, dim)

    def forward(self, charge: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        packed = torch.cat((charge, occupancy), dim=-1)
        image = packed.reshape(-1, 2, self.height, self.width)
        return self.projection(self.convolution(image).flatten(1))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        dropout: float,
        attention_dropout: float,
        drop_path: float,
        backend: str,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = PackedSelfAttention(
            dim,
            num_heads,
            qkv_bias=True,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
            backend=backend,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x = x + self.drop_path1(
            self.attention(self.norm1(x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        )
        return x + self.drop_path2(self.mlp(self.norm2(x)))


class CylindricalHeadNormalized(nn.Module):
    def __init__(self, stats: dict[str, Any], hidden: int):
        super().__init__()
        self.k_t = float(stats["k_T"])
        self.mu_ut = float(stats["mu_uT"])
        self.sigma_ut = float(max(stats["sigma_uT"], 1e-8))
        self.k_z = float(stats["k_Z"])
        self.mu_uz = float(stats["mu_uZ"])
        self.sigma_uz = float(max(stats["sigma_uZ"], 1e-8))
        self.mlp = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> dict[str, torch.Tensor]:
        z_t, a, b, z_z = self.mlp(x).unbind(-1)
        norm = torch.sqrt(a * a + b * b + eps)
        cos_phi = a / norm
        sin_phi = b / norm
        u_t = self.mu_ut + self.sigma_ut * z_t
        p_t = self.k_t * torch.expm1(u_t)
        u_z = self.mu_uz + self.sigma_uz * z_z
        p_z = self.k_z * torch.expm1(u_z)
        p_cart = torch.stack((p_t * cos_phi, p_t * sin_phi, p_z), dim=-1)
        return {
            "p_cart": p_cart,
            "pT": p_t,
            "cos_phi": cos_phi,
            "sin_phi": sin_phi,
            "pz": p_z,
            "latents": torch.stack((z_t, z_z), dim=-1),
        }


class ProjectionTransformer(nn.Module):
    def __init__(self, model_config: dict[str, Any], metadata: dict[str, Any]):
        super().__init__()
        dim = int(model_config["embed_dim"])
        depth = int(model_config["depth"])
        backend = model_config["attention_backend"]
        self.attention_backend = (
            "flash_attn_varlen" if backend == "flash" else "torch_sdpa_explicit"
        )
        patch_channels = int(model_config.get("patch_encoder_channels", 16))
        self.longitudinal_patch_embedding = LocalPatchEncoder(
            TRANSVERSE_PATCH,
            LONGITUDINAL_PATCH,
            dim,
            patch_channels,
        )
        self.xy_patch_embedding = LocalPatchEncoder(
            XY_PATCH,
            XY_PATCH,
            dim,
            patch_channels,
        )
        self.position_embedding = FourierPositionEmbedding(
            dim, num_bands=int(model_config.get("position_bands", 16))
        )
        self.view_embedding = nn.Embedding(4, dim)
        self.kind_embedding = nn.Embedding(3, dim)
        self.task_tokens = nn.Parameter(torch.empty(1, NUM_TASKS, dim))
        self.input_dropout = nn.Dropout(float(model_config["dropout"]))

        drop_paths = torch.linspace(0, float(model_config["drop_path_rate"]), depth).tolist()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                dim,
                int(model_config["num_heads"]),
                float(model_config["mlp_ratio"]),
                dropout=float(model_config["dropout"]),
                attention_dropout=float(model_config["attention_dropout"]),
                drop_path=drop_paths[index],
                backend=backend,
            )
            for index in range(depth)
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        head_dropout_cls = float(model_config.get("head_dropout_cls", 0.0))
        head_dropout_reg = float(model_config.get("head_dropout_reg", 0.0))
        self.flavour_head = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6), nn.Dropout(head_dropout_cls), nn.Linear(dim, 6)
        )
        self.charm_head = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6), nn.Dropout(head_dropout_cls), nn.Linear(dim, 4)
        )
        self.vis_norm = nn.Sequential(nn.LayerNorm(dim, eps=1e-6), nn.Dropout(head_dropout_reg))
        self.jet_norm = nn.Sequential(nn.LayerNorm(dim, eps=1e-6), nn.Dropout(head_dropout_reg))
        self.vis_head = CylindricalHeadNormalized(metadata["target_stats"]["vis"], dim)
        self.jet_head = CylindricalHeadNormalized(metadata["target_stats"]["jet"], dim)
        self.vertex_head = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6), nn.Dropout(head_dropout_cls), nn.Linear(dim, 3)
        )
        self._initialize(float(model_config.get("head_init", 2e-5)))

    def _initialize(self, head_init: float) -> None:
        nn.init.normal_(self.task_tokens, std=0.02)
        nn.init.normal_(self.view_embedding.weight, std=0.02)
        nn.init.normal_(self.kind_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for head in (self.flavour_head[-1], self.charm_head[-1], self.vertex_head[-1]):
            nn.init.trunc_normal_(head.weight, std=head_init)
        nn.init.trunc_normal_(self.vis_head.mlp.weight, std=head_init)
        nn.init.trunc_normal_(self.jet_head.mlp.weight, std=head_init)

    def no_weight_decay(self) -> set[str]:
        return {"task_tokens", "view_embedding.weight", "kind_embedding.weight"}

    def forward(self, batch: ProjectionBatch) -> dict[str, Any]:
        x = (
            self.position_embedding(batch.positions)
            + self.view_embedding(batch.view_ids)
            + self.kind_embedding(batch.token_kinds)
        )
        encoder_dependency = x.new_zeros(())
        for encoder in (
            self.longitudinal_patch_embedding,
            self.xy_patch_embedding,
        ):
            for parameter in encoder.parameters():
                encoder_dependency = encoder_dependency + 0.0 * parameter.reshape(-1)[0]
        x = x + encoder_dependency
        patch_mask = batch.token_kinds == 0
        longitudinal_indices = torch.nonzero(
            patch_mask & (batch.view_ids < 2), as_tuple=False
        ).flatten()
        if longitudinal_indices.numel():
            features = batch.features.index_select(0, longitudinal_indices)
            pixels = TRANSVERSE_PATCH * LONGITUDINAL_PATCH
            encoded = self.longitudinal_patch_embedding(
                features[:, :pixels],
                features[:, MAX_PATCH_PIXELS : MAX_PATCH_PIXELS + pixels],
            )
            x = x.index_add(0, longitudinal_indices, encoded.to(dtype=x.dtype))
        xy_indices = torch.nonzero(
            patch_mask & (batch.view_ids == 2), as_tuple=False
        ).flatten()
        if xy_indices.numel():
            features = batch.features.index_select(0, xy_indices)
            pixels = XY_PATCH * XY_PATCH
            encoded = self.xy_patch_embedding(
                features[:, :pixels],
                features[:, MAX_PATCH_PIXELS : MAX_PATCH_PIXELS + pixels],
            )
            x = x.index_add(0, xy_indices, encoded.to(dtype=x.dtype))
        task_source = self.task_tokens.expand(batch.batch_size, -1, -1).reshape(
            batch.batch_size * NUM_TASKS, -1
        )
        x = x.index_copy(0, batch.task_indices.reshape(-1), task_source)
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x, batch.cu_seqlens, batch.max_seqlen)
        x = self.norm(x)
        task_outputs = x[batch.task_indices]
        return {
            "out_flavour": self.flavour_head(task_outputs[:, 0]),
            "out_charm": self.charm_head(task_outputs[:, 1]),
            "out_vis": self.vis_head(self.vis_norm(task_outputs[:, 2])),
            "out_jet": self.jet_head(self.jet_norm(task_outputs[:, 3])),
            "out_vertex": self.vertex_head(task_outputs[:, 4]),
        }


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
