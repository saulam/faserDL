"""
Description: I-JEPA-style pretraining of the spconv-free flash-attention encoder.

- Context encoder: FlashEncoder over the visible (context) tokens only.
- Target encoder: EMA copy of the context encoder over all tokens; its final
  token representations, layer-normalised, are the regression targets.
  Optionally (target_layers = K > 1) the targets are data2vec's instead: the
  average of the layer-normalised MLP outputs (before the residual addition)
  of its last K blocks, layer-normalised again, which keeps more of the
  lower-level content of the deposits.  Optionally (target_norm = "batch")
  each target channel is then standardised over the batch's tokens of the
  same detector type, so a component shared by every token is no target.
- Predictor: a narrow transformer over the context tokens plus one mask token
  per target, each carrying the target's position and detector type; it
  predicts the target encoder's representation of every target token.
- Masking: multi-block masking adapted to sparse 3D events.  Each target block
  is the set of occupied patches nearest to a random occupied patch (a
  randomly stretched 3D neighbourhood), so the targets are contiguous regions
  of the actual tracks and showers rather than isolated patches or whole
  modules.  Blocks are drawn separately in FASERCal and AHCAL; the ECAL and
  muon-spectrometer tokens are targets with a fixed probability.
"""

import copy
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flash_vit import (
    AHCAL, ECAL, FCAL, FLASH_ENCODER_PRESETS, MUON, NUM_TOKEN_KINDS, TOKEN_KIND_NAMES,
    Block, FlashEncoder, SeqInfo,
)


class MultiBlockMasker(nn.Module):
    """Choose target tokens: nearest-neighbour 3D blocks of occupied patches."""

    def __init__(self, num_blocks=4, block_scale=(0.15, 0.2), block_aspect=(0.5, 2.0),
                 global_mask_prob=0.25):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        if not 0.0 < block_scale[0] <= block_scale[1] < 1.0:
            raise ValueError(f"block_scale must satisfy 0 < min <= max < 1, got {block_scale}")
        if not 0.0 < block_aspect[0] <= block_aspect[1]:
            raise ValueError(f"block_aspect must satisfy 0 < min <= max, got {block_aspect}")
        if not 0.0 <= global_mask_prob <= 1.0:
            raise ValueError(f"global_mask_prob must be in [0, 1], got {global_mask_prob}")
        self.num_blocks = int(num_blocks)
        self.block_scale = tuple(float(v) for v in block_scale)
        self.block_aspect = tuple(float(v) for v in block_aspect)
        self.global_mask_prob = float(global_mask_prob)

    def _blocks(self, coords, event, batch_size):
        """Union of ``num_blocks`` nearest-neighbour blocks per event; [N] bool.

        coords [N, 3] patch-grid coordinates, event [N] non-decreasing.  A block
        holds ceil(scale * n) of the event's n patches (scale ~ U(block_scale)),
        never all of them, and at least one patch per event stays visible.
        """
        B, M = batch_size, self.num_blocks
        device = coords.device
        counts = torch.bincount(event, minlength=B)
        starts = counts.cumsum(0) - counts
        rank = torch.arange(event.numel(), device=device) - starts[event]
        L = int(counts.max())

        grid = coords.new_zeros(B, L, 3, dtype=torch.float32)
        grid[event, rank] = coords.float()
        valid = torch.zeros(B, L, dtype=torch.bool, device=device)
        valid[event, rank] = True

        last = (counts - 1).clamp_min(0)
        center = (torch.rand(B, M, device=device) * counts.unsqueeze(1)).long()
        center = torch.minimum(center, last.unsqueeze(1))
        centers = grid[torch.arange(B, device=device).unsqueeze(1), center]           # [B, M, 3]
        log_aspect = torch.empty(B, M, 3, device=device).uniform_(
            math.log(self.block_aspect[0]), math.log(self.block_aspect[1]))
        distance = (((grid.unsqueeze(1) - centers.unsqueeze(2)) / log_aspect.exp().unsqueeze(2))
                    .square().sum(-1))                                                # [B, M, L]
        # Distinct grid distances differ by >= 1 / max_aspect^2; the noise only breaks ties.
        distance = distance + 1e-3 * torch.rand_like(distance)
        distance = distance.masked_fill(~valid.unsqueeze(1), float("inf"))
        order = distance.argsort(dim=-1)
        block_rank = torch.empty_like(order)
        block_rank.scatter_(-1, order, torch.arange(L, device=device).expand_as(order))

        scale = torch.empty(B, M, device=device).uniform_(*self.block_scale)
        size = torch.ceil(scale * counts.unsqueeze(1)).long()
        size = torch.minimum(size, last.unsqueeze(1))
        target = (block_rank < size.unsqueeze(-1)).any(dim=1) & valid                 # [B, L]

        # If the union of blocks covers an event, reveal its least-covered patch.
        covered = (target.sum(dim=1) >= counts) & (counts > 0)
        if bool(covered.any()):
            least = block_rank.min(dim=1).values.masked_fill(~valid, -1).argmax(dim=-1)
            target[covered, least[covered]] = False
        return target[event, rank]

    @torch.no_grad()
    def forward(self, tokens):
        target = torch.zeros_like(tokens.event, dtype=torch.bool)
        for kind in (FCAL, AHCAL):
            index = (tokens.kind == kind).nonzero().squeeze(1)
            if index.numel():
                target[index] = self._blocks(
                    tokens.coords[index], tokens.event[index], tokens.batch_size)
        for kind in (ECAL, MUON):
            index = (tokens.kind == kind).nonzero().squeeze(1)
            target[index] = torch.rand(index.numel(), device=index.device) < self.global_mask_prob
        return target


class FlashPredictor(nn.Module):
    """Narrow transformer predicting target-encoder tokens from context tokens and mask tokens."""

    def __init__(self, encoder, embed_dim=256, depth=6, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.predictor_embed = nn.Linear(encoder.embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.token_type_embed = nn.Embedding(NUM_TOKEN_KINDS, embed_dim)
        self.register_buffer(
            "fcal_pos_embed", FlashEncoder._sincos(embed_dim, encoder.fcal_patch_embed.patch_grid),
            persistent=False)
        self.register_buffer(
            "ahcal_pos_embed", FlashEncoder._sincos(embed_dim, encoder.ahcal_patch_embed.patch_grid),
            persistent=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.predictor_proj = nn.Linear(embed_dim, encoder.embed_dim)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.token_type_embed.weight, std=0.02)
        self.blocks.apply(FlashEncoder._init_weights)
        for module in (self.predictor_embed, self.norm, self.predictor_proj):
            module.apply(FlashEncoder._init_weights)

    def position(self, kind, pos):
        """Detector type plus sin-cos patch position (FASERCal / AHCAL grids)."""
        fcal = self.fcal_pos_embed[pos.clamp_max(self.fcal_pos_embed.size(0) - 1)]
        ahcal = self.ahcal_pos_embed[pos.clamp_max(self.ahcal_pos_embed.size(0) - 1)]
        zero = fcal.new_zeros(())
        grid = torch.where((kind == FCAL).unsqueeze(-1), fcal,
                           torch.where((kind == AHCAL).unsqueeze(-1), ahcal, zero))
        return grid + self.token_type_embed(kind)

    def forward(self, context_x, context, targets):
        """context_x [Tc, D]: context-encoder outputs of the ``context`` TokenSet.

        Returns predictions [Tt, D] for the ``targets`` TokenSet (same order).
        """
        x = torch.cat([
            self.predictor_embed(context_x).float() + self.position(context.kind, context.pos),
            self.mask_token.float() + self.position(targets.kind, targets.pos),
        ])
        event = torch.cat([context.event, targets.event])
        order = torch.argsort(event, stable=True)
        seq = SeqInfo.from_event(event[order], context.batch_size)
        x = x[order]
        for block in self.blocks:
            x = block(x, seq)
        x = self.predictor_proj(self.norm(x))
        position = torch.empty_like(order)
        position[order] = torch.arange(order.numel(), device=order.device)
        return x[position[context.event.numel():]]


def standardise_per_kind(x, kind, eps=1e-5):
    """Zero mean and unit variance of every channel over the tokens of each
    detector type (FASERCal, AHCAL, ECAL, muon) in the batch: [T, C] -> [T, C].

    A component that every token of a type shares becomes zero, so targets
    that collapse onto one vector carry no easy-to-predict part; differences
    between tokens and between events are kept at unit scale."""
    counts = torch.bincount(kind, minlength=NUM_TOKEN_KINDS).clamp_min(1).unsqueeze(-1).to(x.dtype)
    mean = x.new_zeros(NUM_TOKEN_KINDS, x.size(-1)).index_add_(0, kind, x) / counts
    centred = x - mean[kind]
    var = x.new_zeros(NUM_TOKEN_KINDS, x.size(-1)).index_add_(0, kind, centred.square()) / counts
    return centred * torch.rsqrt(var[kind] + eps)


class FlashJEPA(nn.Module):
    def __init__(
        self,
        predictor_embed_dim=256,
        predictor_depth=6,
        predictor_num_heads=4,
        mask_num_blocks=4,
        mask_block_scale=(0.15, 0.2),
        mask_block_aspect=(0.5, 2.0),
        mask_global_prob=0.25,
        target_layers=1,
        target_norm="token",
        **encoder_kwargs,
    ):
        super().__init__()
        self.context_encoder = FlashEncoder(**encoder_kwargs)
        depth = len(self.context_encoder.blocks)
        if not 1 <= int(target_layers) <= depth:
            raise ValueError(f"target_layers must be in [1, {depth}], got {target_layers}")
        if target_norm not in ("token", "batch"):
            raise ValueError(f"target_norm must be 'token' or 'batch', got {target_norm!r}")
        self.target_layers = int(target_layers)
        self.target_norm = target_norm
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = FlashPredictor(
            self.context_encoder, embed_dim=predictor_embed_dim, depth=predictor_depth,
            num_heads=predictor_num_heads, mlp_ratio=self.context_encoder.encoder_config["mlp_ratio"],
        )
        self.masker = MultiBlockMasker(
            num_blocks=mask_num_blocks, block_scale=mask_block_scale,
            block_aspect=mask_block_aspect, global_mask_prob=mask_global_prob,
        )

    @property
    def encoder_config(self):
        return dict(self.context_encoder.encoder_config)

    def no_weight_decay(self):
        names = {f"context_encoder.{name}" for name in self.context_encoder.no_weight_decay()}
        return names | {"predictor.mask_token", "predictor.token_type_embed.weight"}

    @torch.no_grad()
    def update_target_encoder(self, decay):
        """Apply one EMA update after the context optimiser step."""
        decay = float(decay)
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")
        context_params = dict(self.context_encoder.named_parameters())
        target_params = dict(self.target_encoder.named_parameters())
        if context_params.keys() != target_params.keys():
            raise RuntimeError("Context and target encoder parameters no longer match.")
        targets = list(target_params.values())
        torch._foreach_mul_(targets, decay)
        torch._foreach_add_(targets, [context_params[name].detach() for name in target_params],
                            alpha=1.0 - decay)

    @torch.no_grad()
    def _targets(self, batch):
        """Layer-normalised target-encoder representations of all tokens (no dropout).

        target_layers = 1: the final output.  K > 1: the average of the
        layer-normalised MLP outputs of the last K blocks (data2vec).
        target_norm = "batch": then standardised per channel and detector type
        over the batch (standardise_per_kind).
        """
        was_training = self.target_encoder.training
        self.target_encoder.eval()
        try:
            tokens = self.target_encoder.embed(batch)
            seq = tokens.seq()
            if self.target_layers == 1:
                out = self.target_encoder.forward_tokens(tokens.x, seq)
            else:
                layers = self.target_encoder.forward_mlp_outputs(tokens.x, seq, self.target_layers)
                out = sum(F.layer_norm(layer.float(), (layer.size(-1),)) for layer in layers) / len(layers)
        finally:
            self.target_encoder.train(was_training)
        out = F.layer_norm(out.float(), (out.size(-1),))
        if self.target_norm == "batch":
            out = standardise_per_kind(out, tokens.kind)
        return tokens, seq, out

    def forward(self, batch):
        tokens = self.context_encoder.embed(batch)
        target_mask = self.masker(tokens)
        target_tokens, token_seq, target_out = self._targets(batch)
        if not (torch.equal(target_tokens.kind, tokens.kind) and torch.equal(target_tokens.pos, tokens.pos)):
            raise RuntimeError("Context and target encoders tokenised the batch differently.")

        context = tokens.select(~target_mask)
        targets = tokens.select(target_mask)
        context_seq = context.seq()
        context_out = self.context_encoder.forward_tokens(context.x, context_seq)
        prediction = self.predictor(context_out, context, targets).float()
        target = target_out[target_mask]

        token_loss = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
        loss = token_loss.mean() if token_loss.numel() else prediction.sum() * 0.0

        group_losses = {}
        with torch.no_grad():
            for kind, name in enumerate(TOKEN_KIND_NAMES):
                selected = targets.kind == kind
                if bool(selected.any()):
                    group_losses[name] = token_loss[selected].mean()
        return {
            "loss": loss,
            "prediction": prediction,
            "target": target,
            "target_kind": targets.kind,
            "group_losses": group_losses,
            "token_kind": tokens.kind,
            "target_mask": target_mask,
            "context_tokens": context_out.detach(),
            "context_seq": context_seq,
            "target_tokens": target_out,
            "token_seq": token_seq,
        }


def _flash_jepa(preset, **kwargs):
    config = dict(FLASH_ENCODER_PRESETS[preset])
    config.update(kwargs)
    return FlashJEPA(**config)


def flash_jepa_tiny(**kwargs):
    return _flash_jepa("tiny", **kwargs)


def flash_jepa_base(**kwargs):
    return _flash_jepa("base", **kwargs)
