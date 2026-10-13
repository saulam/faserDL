"""JEPA-style latent prediction using the existing sparse FASERCal encoder."""

from copy import deepcopy

import torch
import torch.nn as nn
from spconv.pytorch import SparseConv3d
from torch.nn import functional as F

from .sparsemaevit import mae_vit_base, mae_vit_tiny


class LatentPredictor(nn.Module):
    """Small discardable MLP applied independently to each Perceiver latent slot."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)


class SparseJEPA(nn.Module):
    """Predict full-view EMA-encoder latents from the existing masked encoder view.

    The backbone, sparse patch embeddings, module attention, auxiliary detector
    tokens, and Perceiver latent slots are unchanged from ``SparseMAEViT``.  Only
    its reconstruction decoder is omitted.  The context encoder uses the
    existing occupancy-aware patch masking; the target encoder sees the full
    event and is updated only through EMA.
    """

    _VALID_NORMALIZATIONS = {"none", "layernorm", "l2"}
    _VALID_LOSSES = {"smooth_l1", "mse", "cosine"}
    _VALID_TARGET_SELECTIONS = {"masked_modules", "all"}

    def __init__(
        self,
        context_encoder: nn.Module,
        predictor_hidden_dim: int = 384,
        predictor_dropout: float = 0.0,
        target_normalization: str = "layernorm",
        loss_type: str = "smooth_l1",
        target_selection: str = "masked_modules",
    ):
        super().__init__()
        if getattr(context_encoder, "build_decoder", True):
            raise ValueError("JEPA requires an encoder-only SparseMAEViT (build_decoder=False).")
        if target_normalization not in self._VALID_NORMALIZATIONS:
            raise ValueError(f"Unknown JEPA normalization: {target_normalization}")
        if loss_type not in self._VALID_LOSSES:
            raise ValueError(f"Unknown JEPA loss: {loss_type}")
        if target_selection not in self._VALID_TARGET_SELECTIONS:
            raise ValueError(f"Unknown JEPA target selection: {target_selection}")

        self.context_encoder = context_encoder
        self.target_encoder = deepcopy(context_encoder)
        embed_dim = int(context_encoder.module_cls_token.shape[-1])
        self.predictor = LatentPredictor(
            embed_dim=embed_dim,
            hidden_dim=int(predictor_hidden_dim),
            dropout=float(predictor_dropout),
        )
        self.target_normalization = target_normalization
        self.loss_type = loss_type
        self.target_selection = target_selection

        self._initialize_target_encoder()

    @torch.no_grad()
    def _initialize_target_encoder(self) -> None:
        """Copy the context encoder exactly and permanently disable target gradients."""
        self.target_encoder.load_state_dict(self.context_encoder.state_dict(), strict=True)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The EMA branch is always a deterministic teacher, including while the
        # context encoder and predictor are in training mode.
        self.target_encoder.eval()
        return self

    def _configure_target_sparse_convs(self, device: torch.device) -> None:
        """Keep the deterministic target compatible with generic spconv on CPU.

        Generic (non-CUDA) spconv cannot use its eval-time fused bias path.  The
        sparse convolution has no stochastic state, so selecting its training
        kernel on CPU does not alter the JEPA teacher semantics; dropout and
        stochastic-depth modules remain in evaluation mode.
        """
        use_training_kernel = device.type == "cpu"
        for module in self.target_encoder.modules():
            if isinstance(module, SparseConv3d):
                module.train(use_training_kernel)

    def no_weight_decay(self):
        return {
            f"context_encoder.{name}"
            for name in self.context_encoder.no_weight_decay()
        }

    @torch.no_grad()
    def update_target_encoder(self, decay: float) -> None:
        """Apply one EMA update after the context optimiser step."""
        decay = float(decay)
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")

        context_params = dict(self.context_encoder.named_parameters())
        target_params = dict(self.target_encoder.named_parameters())
        if context_params.keys() != target_params.keys():
            raise RuntimeError("Context and target encoder parameters no longer match.")
        for name, target in target_params.items():
            target.mul_(decay).add_(context_params[name].detach(), alpha=1.0 - decay)

        # The encoder currently has no running-statistics buffers.  Copying all
        # buffers keeps integer geometry templates and any future buffers exact.
        context_buffers = dict(self.context_encoder.named_buffers())
        target_buffers = dict(self.target_encoder.named_buffers())
        if context_buffers.keys() != target_buffers.keys():
            raise RuntimeError("Context and target encoder buffers no longer match.")
        for name, target in target_buffers.items():
            target.copy_(context_buffers[name])

    def _latent_prediction_mask(
        self,
        rand_mask: torch.Tensor,
        ah_rand_mask: torch.Tensor,
        target_latent_keep: torch.Tensor,
    ) -> torch.Tensor:
        """Map masked patches to the fixed module/AHCAL Perceiver latent slots."""
        if self.target_selection == "all":
            return target_latent_keep

        # A FASERCal latent slot is a target when its detector module contains
        # at least one masked occupied patch.  AHCAL slots are selected in the
        # same way for the single AHCAL token group.
        fas_modules = rand_mask.any(dim=-1)
        fas_targets = fas_modules.repeat_interleave(
            self.context_encoder.num_module_cls, dim=1
        )
        ah_targets = ah_rand_mask.any(dim=-1, keepdim=True).expand(
            -1, self.context_encoder.num_ahcal_cls
        )
        prediction_mask = torch.cat([fas_targets, ah_targets], dim=1)
        prediction_mask = prediction_mask & target_latent_keep

        # Degenerate very-low-occupancy events can have no patch selected.  Use
        # their valid slots so every event still contributes a finite loss.
        empty = ~prediction_mask.any(dim=1)
        if empty.any():
            prediction_mask = prediction_mask.clone()
            prediction_mask[empty] = target_latent_keep[empty]
        return prediction_mask

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_normalization == "none":
            return x
        if self.target_normalization == "layernorm":
            return F.layer_norm(x, (x.shape[-1],))
        return F.normalize(x, dim=-1)

    def _prediction_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self._normalize(prediction)
        target = self._normalize(target)

        if self.loss_type == "smooth_l1":
            per_slot = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
        elif self.loss_type == "mse":
            per_slot = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
        else:
            per_slot = 1.0 - F.cosine_similarity(prediction, target, dim=-1)

        weights = prediction_mask.to(dtype=per_slot.dtype)
        return (per_slot * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(self, x, x_glob, mask_ratio: float = 0.75):
        (
            context_latents,
            context_latent_keep,
            rand_mask,
            attn_mask_mod,
            ah_mask,
            ah_rand_mask,
            ecal_drop,
            muon_drop,
        ) = self.context_encoder.forward_encoder(x, x_glob, mask_ratio=mask_ratio)

        with torch.no_grad():
            input_device = x.features.device if hasattr(x, "features") else x.device
            self._configure_target_sparse_convs(input_device)
            target_outputs = self.target_encoder.forward_encoder(x, x_glob, mask_ratio=0.0)
            target_latents, target_latent_keep = target_outputs[:2]

        prediction = self.predictor(context_latents)
        prediction_mask = self._latent_prediction_mask(
            rand_mask=rand_mask,
            ah_rand_mask=ah_rand_mask,
            target_latent_keep=target_latent_keep,
        )
        loss = self._prediction_loss(prediction, target_latents, prediction_mask)

        return {
            "loss": loss,
            "prediction": prediction,
            "target": target_latents.detach(),
            "prediction_mask": prediction_mask,
            "context_latents": context_latents,
            "context_latent_keep": context_latent_keep,
            "target_latent_keep": target_latent_keep,
            "fcal_patch_mask": rand_mask,
            "fcal_patch_keep": attn_mask_mod,
            "ahcal_patch_mask": ah_rand_mask,
            "ahcal_patch_keep": ah_mask,
            "ecal_drop": ecal_drop,
            "muon_drop": muon_drop,
        }


def _make_jepa(encoder_factory, **kwargs):
    predictor_hidden_dim = kwargs.pop("predictor_hidden_dim", 384)
    predictor_dropout = kwargs.pop("predictor_dropout", 0.0)
    target_normalization = kwargs.pop("target_normalization", "layernorm")
    loss_type = kwargs.pop("loss_type", "smooth_l1")
    target_selection = kwargs.pop("target_selection", "masked_modules")
    context_encoder = encoder_factory(build_decoder=False, **kwargs)
    return SparseJEPA(
        context_encoder=context_encoder,
        predictor_hidden_dim=predictor_hidden_dim,
        predictor_dropout=predictor_dropout,
        target_normalization=target_normalization,
        loss_type=loss_type,
        target_selection=target_selection,
    )


def jepa_vit_tiny(**kwargs):
    return _make_jepa(mae_vit_tiny, **kwargs)


def jepa_vit_base(**kwargs):
    return _make_jepa(mae_vit_base, **kwargs)
