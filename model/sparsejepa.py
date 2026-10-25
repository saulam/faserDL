"""JEPA-style latent prediction using the existing sparse FASERCal encoder."""

from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from spconv.pytorch import SparseConv3d
from torch.nn import functional as F

from .sparsemaevit import mae_vit_base, mae_vit_tiny
from .utils import CrossAttnBlock, get_3d_sincos_pos_embed


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


class PatchLatentPredictor(nn.Module):
    """Positional predictor of masked-patch encoder tokens from context latents.

    It mirrors the MAE Perceiver-IO decoder: one query per masked occupied
    FASERCal patch (fixed sin-cos intra-module position + learned module
    embedding + learned query token), one per masked AHCAL patch, and one per
    dropped ECAL/muon token.  Queries cross-attend only to the context latent
    slots, so all predictive information must pass through the representation
    used downstream.  A norm and projection to the encoder width replace the
    MAE voxel heads.
    """

    def __init__(self, encoder, embed_dim=256, depth=2, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        encoder_dim = int(encoder.module_cls_token.shape[-1])
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.enc_to_pred = nn.Linear(encoder_dim, embed_dim)
        self.fcal_pos_embed = nn.Embedding(encoder.num_intra_positions, embed_dim)    # frozen sin-cos
        self.module_embed = nn.Embedding(encoder.num_modules, embed_dim)
        self.fcal_query = nn.Parameter(torch.zeros(1, embed_dim))
        self.ahcal_pos_embed = nn.Embedding(encoder.num_ahcal_positions, embed_dim)   # frozen sin-cos
        self.ahcal_query = nn.Parameter(torch.zeros(1, embed_dim))
        self.global_query = nn.Parameter(torch.zeros(2, embed_dim))                    # 0=ECAL, 1=MUON
        self.blocks = nn.ModuleList([
            CrossAttnBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=dropout, norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.proj = nn.Linear(embed_dim, encoder_dim)

        self.apply(encoder._init_weights)
        with torch.no_grad():
            for table, grid in (
                (self.fcal_pos_embed, encoder.intra_grid_size),
                (self.ahcal_pos_embed, encoder.ahcal_grid_size),
            ):
                table.weight.copy_(torch.from_numpy(get_3d_sincos_pos_embed(embed_dim, grid)).float())
                table.weight.requires_grad_(False)
            for parameter in (self.module_embed.weight, self.fcal_query, self.ahcal_query, self.global_query):
                nn.init.normal_(parameter, std=0.02)

    def no_weight_decay(self):
        return {"module_embed.weight", "fcal_query", "ahcal_query", "global_query"}

    def forward(self, latents, latent_keep, fcal_ids, ahcal_ids, ecal_ids, muon_ids):
        """Return one prediction per query, ordered FASERCal, AHCAL, ECAL, muon.

        fcal_ids:  (batch, module, intra position) of masked occupied patches
        ahcal_ids: (batch, position) of masked occupied AHCAL patches
        ecal_ids, muon_ids: batch indices of events whose token was dropped
        """
        fb, fm, fl = fcal_ids
        ab, al = ahcal_ids
        queries = torch.cat([
            self.fcal_pos_embed(fl) + self.module_embed(fm) + self.fcal_query,
            self.ahcal_pos_embed(al) + self.ahcal_query,
            self.global_query[0].expand(ecal_ids.numel(), -1),
            self.global_query[1].expand(muon_ids.numel(), -1),
        ])
        batch_ids = torch.cat([fb, ab, ecal_ids, muon_ids])
        if batch_ids.numel() == 0:
            return queries.new_zeros(0, self.proj.out_features)

        # Pack queries per event.  Queries never attend to one another, so the
        # packing (and its zero padding) does not affect any prediction.
        batch_size = latents.size(0)
        sorted_ids, order = torch.sort(batch_ids, stable=True)
        counts = torch.bincount(sorted_ids, minlength=batch_size)
        starts = torch.cumsum(counts, 0) - counts
        within = torch.empty_like(batch_ids)
        within[order] = torch.arange(batch_ids.numel(), device=batch_ids.device) - starts[sorted_ids]
        packed = queries.new_zeros(batch_size, int(counts.max().item()), queries.size(-1))
        packed[batch_ids, within] = queries

        context = self.enc_to_pred(latents)
        for block in self.blocks:
            packed = block(packed, context, attn_mask=latent_keep)
        return self.proj(self.norm(packed[batch_ids, within]))


class SparseJEPA(nn.Module):
    """Predict full-view EMA-encoder representations from the masked encoder view.

    The backbone, sparse patch embeddings, module attention, auxiliary detector
    tokens, and Perceiver latent slots are unchanged from ``SparseMAEViT``.  Only
    its reconstruction decoder is omitted.  The context encoder uses the
    existing occupancy-aware patch masking; the target encoder sees the full
    event and is updated only through EMA.

    ``target_level="patch"`` predicts the target encoder's per-patch tokens at
    the masked occupied patches (and its ECAL/muon tokens when dropped) with a
    positional predictor reading the context latents.  ``target_level="slot"``
    is the original objective of the first JEPA run: a slot-wise MLP maps each
    context latent slot to the corresponding full-view latent slot.
    """

    _VALID_NORMALIZATIONS = {"none", "layernorm", "l2"}
    _VALID_LOSSES = {"smooth_l1", "mse", "cosine"}
    _VALID_TARGET_SELECTIONS = {"masked_modules", "all"}
    _VALID_TARGET_LEVELS = {"slot", "patch"}

    def __init__(
        self,
        context_encoder: nn.Module,
        predictor_hidden_dim: int = 384,
        predictor_dropout: float = 0.0,
        target_normalization: str = "layernorm",
        loss_type: str = "smooth_l1",
        target_selection: str = "masked_modules",
        target_level: str = "slot",
        predictor_embed_dim: int = 256,
        predictor_depth: int = 2,
        predictor_num_heads: int = 8,
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
        if target_level not in self._VALID_TARGET_LEVELS:
            raise ValueError(f"Unknown JEPA target level: {target_level}")
        if target_level == "patch" and target_selection != "masked_modules":
            raise ValueError(
                "Patch-level JEPA always predicts the masked patches; "
                f"target_selection={target_selection!r} applies only to slot targets."
            )

        self.context_encoder = context_encoder
        self.target_encoder = deepcopy(context_encoder)
        embed_dim = int(context_encoder.module_cls_token.shape[-1])
        if target_level == "patch":
            self.predictor = PatchLatentPredictor(
                context_encoder,
                embed_dim=int(predictor_embed_dim),
                depth=int(predictor_depth),
                num_heads=int(predictor_num_heads),
                dropout=float(predictor_dropout),
            )
        else:
            self.predictor = LatentPredictor(
                embed_dim=embed_dim,
                hidden_dim=int(predictor_hidden_dim),
                dropout=float(predictor_dropout),
            )
        self.target_normalization = target_normalization
        self.loss_type = loss_type
        self.target_selection = target_selection
        self.target_level = target_level

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
        names = {
            f"context_encoder.{name}"
            for name in self.context_encoder.no_weight_decay()
        }
        if self.target_level == "patch":
            names.update(f"predictor.{name}" for name in self.predictor.no_weight_decay())
        return names

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

    def _per_vector_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = self._normalize(prediction)
        target = self._normalize(target)

        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
        if self.loss_type == "mse":
            return F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
        return 1.0 - F.cosine_similarity(prediction, target, dim=-1)

    def _prediction_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        prediction_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_slot = self._per_vector_loss(prediction, target)
        weights = prediction_mask.to(dtype=per_slot.dtype)
        return (per_slot * weights).sum() / weights.sum().clamp_min(1.0)

    def _target_view(self, x, x_glob, return_tokens=False):
        with torch.no_grad():
            input_device = x.features.device if hasattr(x, "features") else x.device
            self._configure_target_sparse_convs(input_device)
            extra = {"return_tokens": True} if return_tokens else {}
            return self.target_encoder.forward_encoder(x, x_glob, mask_ratio=0.0, **extra)

    def forward(self, x, x_glob, mask_ratio: float = 0.75):
        context = self.context_encoder.forward_encoder(x, x_glob, mask_ratio=mask_ratio)
        if self.target_level == "patch":
            return self._forward_patch(x, x_glob, context)
        return self._forward_slot(x, x_glob, context)

    @staticmethod
    def _view_outputs(context, target_latents, target_latent_keep):
        (
            context_latents,
            context_latent_keep,
            rand_mask,
            attn_mask_mod,
            ah_mask,
            ah_rand_mask,
            ecal_drop,
            muon_drop,
        ) = context
        return {
            "context_latents": context_latents,
            "context_latent_keep": context_latent_keep,
            "target_latents": target_latents.detach(),
            "target_latent_keep": target_latent_keep,
            "fcal_patch_mask": rand_mask,
            "fcal_patch_keep": attn_mask_mod,
            "ahcal_patch_mask": ah_rand_mask,
            "ahcal_patch_keep": ah_mask,
            "ecal_drop": ecal_drop,
            "muon_drop": muon_drop,
        }

    def _forward_slot(self, x, x_glob, context):
        context_latents, _, rand_mask, _, _, ah_rand_mask = context[:6]
        target_latents, target_latent_keep = self._target_view(x, x_glob)[:2]

        prediction = self.predictor(context_latents)
        prediction_mask = self._latent_prediction_mask(
            rand_mask=rand_mask,
            ah_rand_mask=ah_rand_mask,
            target_latent_keep=target_latent_keep,
        )
        loss = self._prediction_loss(prediction, target_latents, prediction_mask)

        output = self._view_outputs(context, target_latents, target_latent_keep)
        output.update({
            "loss": loss,
            "prediction": prediction,
            "target": target_latents.detach(),
            "prediction_mask": prediction_mask,
            "selected_prediction": prediction[prediction_mask],
            "selected_target": target_latents.detach()[prediction_mask],
            "targets_per_event": prediction_mask.float().sum(dim=1),
            "target_candidates_per_event": target_latent_keep.float().sum(dim=1),
        })
        return output

    def _forward_patch(self, x, x_glob, context):
        (
            context_latents,
            context_latent_keep,
            rand_mask,
            attn_mask_mod,
            ah_mask,
            ah_rand_mask,
            ecal_drop,
            muon_drop,
        ) = context
        target_outputs = self._target_view(x, x_glob, return_tokens=True)
        target_latents, target_latent_keep = target_outputs[:2]
        tokens = target_outputs[8]

        # Full-view target tokens indexed by (module, intra position) and AHCAL
        # position.  The unmasked target gathers every position, so scattering
        # by the returned ids restores the canonical order exactly.
        fcal_targets = torch.zeros_like(tokens["fcal"]).scatter(
            2, tokens["fcal_ids"].unsqueeze(-1).expand_as(tokens["fcal"]), tokens["fcal"]
        )
        ahcal_targets = torch.zeros_like(tokens["ahcal"]).scatter(
            1, tokens["ahcal_ids"].unsqueeze(-1).expand_as(tokens["ahcal"]), tokens["ahcal"]
        )

        # rand_mask/ah_rand_mask mark masked *occupied* patches of the context view.
        fcal_ids = torch.nonzero(rand_mask, as_tuple=True)
        ahcal_ids = torch.nonzero(ah_rand_mask, as_tuple=True)
        ecal_ids = torch.nonzero(ecal_drop, as_tuple=True)[0]
        muon_ids = torch.nonzero(muon_drop, as_tuple=True)[0]
        target = torch.cat([
            fcal_targets[fcal_ids],
            ahcal_targets[ahcal_ids],
            tokens["ecal"][ecal_ids],
            tokens["muon"][muon_ids],
        ]).detach()
        groups = torch.cat([
            ids.new_full((ids.numel(),), index)
            for index, ids in enumerate((fcal_ids[0], ahcal_ids[0], ecal_ids, muon_ids))
        ])
        batch_ids = torch.cat([fcal_ids[0], ahcal_ids[0], ecal_ids, muon_ids])

        prediction = self.predictor(
            context_latents, context_latent_keep, fcal_ids, ahcal_ids, ecal_ids, muon_ids
        )
        if target.numel():
            per_target = self._per_vector_loss(prediction.float(), target.float())
            loss = per_target.mean()
        else:
            # No masked patch or dropped token anywhere in the batch.
            per_target = prediction.new_zeros(0)
            loss = context_latents.sum() * 0.0

        batch_size = context_latents.size(0)
        output = self._view_outputs(context, target_latents, target_latent_keep)
        output.update({
            "loss": loss,
            "selected_prediction": prediction,
            "selected_target": target,
            "target_groups": groups,
            "group_losses": {
                name: per_target.detach()[groups == index].mean()
                for index, name in enumerate(("fcal", "ahcal", "ecal", "muon"))
                if bool((groups == index).any())
            },
            "targets_per_event": torch.bincount(batch_ids, minlength=batch_size).float(),
            "target_candidates_per_event": (
                attn_mask_mod.flatten(1).sum(dim=1) + ah_mask.sum(dim=1) + 2
            ).float(),
        })
        return output


def _make_jepa(encoder_factory, **kwargs):
    jepa_kwargs = {
        name: kwargs.pop(name)
        for name in (
            "predictor_hidden_dim", "predictor_dropout", "target_normalization",
            "loss_type", "target_selection", "target_level",
            "predictor_embed_dim", "predictor_depth", "predictor_num_heads",
        )
        if name in kwargs
    }
    context_encoder = encoder_factory(build_decoder=False, **kwargs)
    return SparseJEPA(context_encoder=context_encoder, **jepa_kwargs)


def jepa_vit_tiny(**kwargs):
    return _make_jepa(mae_vit_tiny, **kwargs)


def jepa_vit_base(**kwargs):
    return _make_jepa(mae_vit_base, **kwargs)
