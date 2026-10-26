"""
Description: spconv-free sparse-token ViT with variable-length flash attention.

Every occupied FASERCal and AHCAL patch is one token, plus one ECAL token and
one muon-spectrometer token per event.  The events of a batch are packed into
one token sequence without padding, and attention runs within each event with
torch.nn.attention.varlen (FlashAttention-2 kernels).  There is no
Perceiver bottleneck: all tokens attend to all tokens of their event.

The patch embedding reproduces the spconv models' sparse convolution with
kernel = stride (including their two-step embedding of large patches) with
scatter operations and dense matrix products, so it needs no spconv and has no
voxel-count limit.
"""

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.funcs import collate_targets
from .common import CylindricalHeadNormalized, choose_k1_k2, get_3d_sincos_pos_embed

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:  # torch < 2.9: exact padded attention only (tests, spconv env)
    varlen_attn = None


FCAL, AHCAL, ECAL, MUON = 0, 1, 2, 3
NUM_TOKEN_KINDS = 4
TOKEN_KIND_NAMES = ("fcal", "ahcal", "ecal", "muon")

# Encoder architectures.  "base" has ~30M parameters, matching the 32M spconv
# vit_base; "tiny" is for quick experiments.
FLASH_ENCODER_PRESETS = {
    "tiny": dict(embed_dim=384, depth=6, num_heads=6),
    "base": dict(embed_dim=384, depth=16, num_heads=6),
}


def _flat_index(coords, shape):
    """Row-major flat index of integer coordinates [..., 3] in a grid of ``shape``."""
    return (coords[..., 0] * shape[1] + coords[..., 1]) * shape[2] + coords[..., 2]


def _unflatten_index(index, shape):
    z = index % shape[2]
    y = (index // shape[2]) % shape[1]
    x = index // (shape[1] * shape[2])
    return torch.stack([x, y, z], dim=-1)


def _autocast_dtype(tensor):
    device_type = tensor.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return tensor.dtype


@dataclass
class SeqInfo:
    """Packing of a token sequence whose tokens are grouped contiguously by event."""
    event: torch.Tensor        # [T] long, non-decreasing
    cu_seqlens: torch.Tensor   # [B+1] int32
    max_seqlen: int
    batch_size: int

    @classmethod
    def from_event(cls, event, batch_size):
        counts = torch.bincount(event, minlength=batch_size)
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=event.device)
        cu_seqlens[1:] = counts.cumsum(0)
        max_seqlen = int(counts.max()) if counts.numel() else 0
        return cls(event, cu_seqlens, max_seqlen, batch_size)

    def rank(self):
        """Position of every token within its event."""
        positions = torch.arange(self.event.numel(), device=self.event.device)
        return positions - self.cu_seqlens[:-1].long()[self.event]

    def mean(self, x):
        """Per-event mean of token features: [T, C] -> [B, C]."""
        out = x.new_zeros(self.batch_size, x.size(-1)).index_add_(0, self.event, x)
        counts = torch.bincount(self.event, minlength=self.batch_size).clamp_min(1)
        return out / counts.unsqueeze(-1).to(x.dtype)


@dataclass
class TokenSet:
    """Packed tokens of a batch, grouped by event and ordered by (kind, position)."""
    x: torch.Tensor        # [T, C]
    event: torch.Tensor    # [T] long, non-decreasing
    kind: torch.Tensor     # [T] long: FCAL, AHCAL, ECAL, MUON
    pos: torch.Tensor      # [T] long: patch index within the kind's grid (0 for ECAL/muon)
    coords: torch.Tensor   # [T, 3] long: patch-grid coordinates (0 for ECAL/muon)
    batch_size: int

    def select(self, keep):
        return TokenSet(
            self.x[keep], self.event[keep], self.kind[keep], self.pos[keep],
            self.coords[keep], self.batch_size,
        )

    def seq(self):
        return SeqInfo.from_event(self.event, self.batch_size)


class EventBatch(dict):
    """A collated batch (dict) that exposes ``batch_size`` like a SparseConvTensor."""

    @property
    def batch_size(self):
        return int(self["batch_size"])


def padded_attention(q, k, v, seq):
    """Exact per-event attention through padding; [T, H, D] in and out."""
    T, H, D = q.shape
    B, L = seq.batch_size, max(1, seq.max_seqlen)
    rank = seq.rank()

    def pad(t):
        out = t.new_zeros(B, L, H, D)
        out[seq.event, rank] = t
        return out.transpose(1, 2)                                           # [B, H, L, D]

    counts = (seq.cu_seqlens[1:] - seq.cu_seqlens[:-1]).long()
    key_valid = torch.arange(L, device=q.device).unsqueeze(0) < counts.unsqueeze(1)
    # Events without tokens attend to their padding so that no row is fully
    # masked (fully masked rows give NaN, which would leak into the gradients).
    key_valid = key_valid | (counts == 0).unsqueeze(1)
    out = F.scaled_dot_product_attention(
        pad(q), pad(k), pad(v), attn_mask=key_valid[:, None, None, :],
    )
    return out.transpose(1, 2)[seq.event, rank]


def varlen_attention(q, k, v, seq):
    """Self-attention within each event of a packed sequence; [T, H, D] in and out."""
    if varlen_attn is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        # A window covering the whole sequence is full attention (FlashAttention
        # treats windows >= max_seqlen as unbounded) and selects the
        # FlashAttention-2 kernels for forward and backward on every GPU.
        # Without it, sm90/sm100 GPUs use cuDNN, which rebuilds its graph for
        # every new batch shape (~100 ms per call on an H100 when shapes vary).
        window = (seq.max_seqlen, seq.max_seqlen)
        return varlen_attn(
            q, k, v, seq.cu_seqlens, seq.cu_seqlens, seq.max_seqlen, seq.max_seqlen,
            window_size=window,
        )
    return padded_attention(q, k, v, seq)


class DropPath(nn.Module):
    """Stochastic depth with one keep/drop decision per event of a packed sequence."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x, seq):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        keep = x.new_empty(seq.batch_size).bernoulli_(keep_prob).div_(keep_prob)
        return x * keep[seq.event].unsqueeze(-1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seq):
        T, C = x.shape
        q, k, v = self.qkv(x).view(T, 3, self.num_heads, self.head_dim).unbind(1)
        x = varlen_attention(q, k, v, seq).reshape(T, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm transformer block over a packed variable-length sequence."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, seq):
        x = x + self.drop_path(self.attn(self.norm1(x), seq), seq)
        return x + self.drop_path(self.mlp(self.norm2(x)), seq)


class SparsePatchEmbed(nn.Module):
    """Sparse convolution with kernel = stride (non-overlapping patches) without spconv.

    Equivalent to the spconv models' patch embedding: a patch of more than 512
    voxels is embedded in two steps (kernel k1 without bias, LayerNorm, GELU,
    then kernel k2 = patch / k1), each step creating outputs only where it has
    inputs.  Coincident voxels are summed and voxels outside the grid are
    dropped, as in spconv.  Linear weights are laid out like spconv's
    [out, kx, ky, kz, in] kernels flattened over (kx, ky, kz, in).
    """

    def __init__(self, grid_size, patch_size, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.grid_size = tuple(int(g) for g in grid_size)
        self.patch_size = tuple(int(p) for p in patch_size)
        assert all(g % p == 0 for g, p in zip(self.grid_size, self.patch_size)), \
            "grid_size must be divisible by patch_size"
        self.patch_grid = tuple(g // p for g, p in zip(self.grid_size, self.patch_size))
        self.num_patches = math.prod(self.patch_grid)
        self.in_chans = int(in_chans)
        self.two_step = math.prod(self.patch_size) > 512
        if self.two_step:
            self.k1, self.k2 = choose_k1_k2(self.patch_size)
            mid = embed_dim // 4
            self.proj1 = nn.Linear(math.prod(self.k1) * in_chans, mid, bias=False)
            self.norm1 = norm_layer(mid)
            self.act = nn.GELU()
            self.proj2 = nn.Linear(math.prod(self.k2) * mid, embed_dim)
        else:
            self.proj = nn.Linear(math.prod(self.patch_size) * in_chans, embed_dim)

    def _scatter_cells(self, cell_key, offset, feats, cell_volume):
        """Sum voxel features into dense per-cell vectors [cells, cell_volume * C]."""
        cells, inverse = torch.unique(cell_key, sorted=True, return_inverse=True)
        C = self.in_chans
        index = ((inverse * cell_volume + offset).unsqueeze(-1) * C
                 + torch.arange(C, device=feats.device)).reshape(-1)
        dense = feats.new_zeros(cells.numel() * cell_volume * C)
        dense.index_add_(0, index, feats.reshape(-1))
        return cells, dense.view(cells.numel(), cell_volume * C)

    def forward(self, coords, feats, event):
        """Embed voxels into patch tokens.

        coords [N, 3] voxel indices, feats [N, C], event [N].  Returns tokens
        [P, embed_dim], their event [P] and patch index [P], sorted by
        (event, patch index).
        """
        coords = coords.long()
        event = event.long()
        feats = feats.reshape(coords.size(0), self.in_chans).float()
        grid = torch.tensor(self.grid_size, device=coords.device)
        inside = ((coords >= 0) & (coords < grid)).all(dim=1)
        if not bool(inside.all()):
            coords, feats, event = coords[inside], feats[inside], event[inside]

        if not self.two_step:
            patch = torch.tensor(self.patch_size, device=coords.device)
            key = event * self.num_patches + _flat_index(coords // patch, self.patch_grid)
            patches, dense = self._scatter_cells(
                key, _flat_index(coords % patch, self.patch_size), feats,
                math.prod(self.patch_size),
            )
            tokens = self.proj(dense)
        else:
            k1 = torch.tensor(self.k1, device=coords.device)
            k2 = torch.tensor(self.k2, device=coords.device)
            cell_grid = tuple(g // k for g, k in zip(self.grid_size, self.k1))
            cells_per_event = math.prod(cell_grid)
            cell_key = event * cells_per_event + _flat_index(coords // k1, cell_grid)
            cells, dense = self._scatter_cells(
                cell_key, _flat_index(coords % k1, self.k1), feats, math.prod(self.k1),
            )
            hidden = self.act(self.norm1(self.proj1(dense)))
            hidden = hidden.to(_autocast_dtype(hidden))

            cell_event = cells // cells_per_event
            cell_coords = _unflatten_index(cells % cells_per_event, cell_grid)
            key = cell_event * self.num_patches + _flat_index(cell_coords // k2, self.patch_grid)
            patches, patch_inverse = torch.unique(key, sorted=True, return_inverse=True)
            stacked = hidden.new_zeros(patches.numel(), math.prod(self.k2), hidden.size(-1))
            stacked[patch_inverse, _flat_index(cell_coords % k2, self.k2)] = hidden
            tokens = self.proj2(stacked.flatten(1))

        return tokens, patches // self.num_patches, patches % self.num_patches


class FlashEncoder(nn.Module):
    """Sparse-token ViT encoder over all FASERCal, AHCAL, ECAL and muon tokens of an event."""

    def __init__(
        self,
        embed_dim=384,
        depth=16,
        num_heads=6,
        mlp_ratio=4.0,
        in_chans=1,
        fcal_size=(48, 48, 200),
        fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40),
        ahcal_patch_size=(6, 6, 5),
        ecal_dim=25,
        muon_dim=5,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        if attn_drop_rate > 0:
            raise ValueError("The flash-attention encoder does not support attention dropout.")
        # Architecture (not regularisation), stored in checkpoints so that
        # fine-tuning and evaluation rebuild exactly the pretrained encoder.
        self.encoder_config = dict(
            embed_dim=int(embed_dim), depth=int(depth), num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio), in_chans=int(in_chans),
            fcal_size=[int(v) for v in fcal_size], fcal_patch_size=[int(v) for v in fcal_patch_size],
            ahcal_size=[int(v) for v in ahcal_size], ahcal_patch_size=[int(v) for v in ahcal_patch_size],
            ecal_dim=int(ecal_dim), muon_dim=int(muon_dim),
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim

        self.fcal_patch_embed = SparsePatchEmbed(fcal_size, fcal_patch_size, in_chans, embed_dim, norm_layer)
        self.ahcal_patch_embed = SparsePatchEmbed(ahcal_size, ahcal_patch_size, in_chans, embed_dim, norm_layer)
        self.ecal_embed = nn.Linear(ecal_dim, embed_dim)
        self.muon_spec_embed = nn.Sequential(
            nn.Linear(muon_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim),
        )
        self.muon_spec_count_encoder = nn.Linear(1, embed_dim)
        self.muon_state_embed = nn.Embedding(2, embed_dim)  # 0 = no tracks, 1 = tracks
        self.token_type_embed = nn.Embedding(NUM_TOKEN_KINDS, embed_dim)
        self.register_buffer("fcal_pos_embed", self._sincos(embed_dim, self.fcal_patch_embed.patch_grid), persistent=False)
        self.register_buffer("ahcal_pos_embed", self._sincos(embed_dim, self.ahcal_patch_embed.patch_grid), persistent=False)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [float(v) for v in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop=drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.initialize_weights()

    @staticmethod
    def _sincos(dim, grid):
        return torch.from_numpy(get_3d_sincos_pos_embed(dim, grid)).float()

    def initialize_weights(self):
        nn.init.normal_(self.muon_state_embed.weight, std=0.02)
        nn.init.normal_(self.token_type_embed.weight, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Same initialisation as the spconv MAE/ViT models.  I-JEPA's
        # (truncated normal 0.02, output projections / sqrt(2 l)) gave
        # markedly less diverse JEPA targets in a short pretraining test.
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {"muon_state_embed.weight", "token_type_embed.weight"}

    def _muon_tokens(self, batch, batch_size):
        tracks = self.muon_spec_embed(batch["muspec_feats"].float())
        track_event = batch["muspec_event"].long()
        summed = tracks.new_zeros(batch_size, tracks.size(-1)).index_add_(0, track_event, tracks)
        counts = torch.bincount(track_event, minlength=batch_size)
        mean = summed / counts.clamp_min(1).unsqueeze(-1).to(summed.dtype)
        has_tracks = (counts > 0).long()
        count = batch["muspec_counts"].reshape(batch_size, 1).float()
        return mean.float() + self.muon_spec_count_encoder(count).float() + self.muon_state_embed(has_tracks)

    def embed(self, batch):
        """Embed all detector inputs of a collated batch into a packed TokenSet."""
        B = int(batch["batch_size"])
        device = self.token_type_embed.weight.device
        f_tok, f_event, f_pos = self.fcal_patch_embed(
            batch["fcal_coords"], batch["fcal_feats"], batch["fcal_event"])
        a_tok, a_event, a_pos = self.ahcal_patch_embed(
            batch["ahcal_coords"], batch["ahcal_feats"], batch["ahcal_event"])
        ecal = self.ecal_embed(batch["ecal_hits"].reshape(B, -1).float())
        muon = self._muon_tokens(batch, B)

        events = torch.arange(B, device=device)
        zeros = torch.zeros(B, dtype=torch.long, device=device)
        x = torch.cat([
            f_tok.float() + self.fcal_pos_embed[f_pos],
            a_tok.float() + self.ahcal_pos_embed[a_pos],
            ecal.float(),
            muon.float(),
        ])
        kind = torch.cat([
            torch.full_like(f_pos, FCAL), torch.full_like(a_pos, AHCAL), zeros + ECAL, zeros + MUON,
        ])
        event = torch.cat([f_event, a_event, events, events])
        pos = torch.cat([f_pos, a_pos, zeros, zeros])
        coords = torch.cat([
            _unflatten_index(f_pos, self.fcal_patch_embed.patch_grid),
            _unflatten_index(a_pos, self.ahcal_patch_embed.patch_grid),
            zeros.new_zeros(2 * B, 3),
        ])
        x = x + self.token_type_embed(kind)

        # Group by event; within an event keep FASERCal, AHCAL, ECAL, muon order
        # (patches are already sorted by index within each kind).
        order = torch.argsort(event * NUM_TOKEN_KINDS + kind, stable=True)
        return TokenSet(x[order], event[order], kind[order], pos[order], coords[order], B)

    def forward_tokens(self, x, seq):
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x, seq)
        return self.norm(x)

    def forward_features(self, batch):
        tokens = self.embed(batch)
        seq = tokens.seq()
        return self.forward_tokens(tokens.x, seq), seq


class FlashViT(FlashEncoder):
    """Fine-tuning model: FlashEncoder, per-event mean pooling and the SparseViT task heads."""

    def __init__(self, metadata=None, head_init=2e-5, head_dropout_cls=0.0,
                 head_dropout_reg=0.0, **encoder_kwargs):
        super().__init__(**encoder_kwargs)
        self.metadata = metadata
        self.head_init = head_init
        embed_dim = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Same heads as SparseViT with global pooling.
        self.head_channels = {"flavour": 6, "charm": 4, "vis": 3, "jet": 3, "vertex": 3}
        cls_tasks = {"flavour", "charm", "vertex"}
        self.heads = nn.ModuleDict()
        for name, channels in self.head_channels.items():
            dropout = head_dropout_cls if name in cls_tasks else head_dropout_reg
            if metadata is not None and name in metadata:
                stats = metadata[name]
                output = CylindricalHeadNormalized(
                    k_T=stats["k_T"], mu_uT=stats["mu_uT"], sigma_uT=stats["sigma_uT"],
                    k_Z=stats["k_Z"], mu_uZ=stats["mu_uZ"], sigma_uZ=stats["sigma_uZ"],
                    hidden=embed_dim,
                )
            else:
                output = nn.Linear(embed_dim, channels)
            self.heads[name] = nn.Sequential(norm_layer(embed_dim), nn.Dropout(dropout), output)
        self.heads.apply(self._init_weights)
        for name, head in self.heads.items():
            linear = head[2].mlp if isinstance(head[2], CylindricalHeadNormalized) else head[2]
            nn.init.trunc_normal_(linear.weight, std=self.head_init)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, x, x_glob=None):
        tokens, seq = self.forward_features(x)
        pooled = seq.mean(tokens)
        return {f"out_{name}": head(pooled) for name, head in self.heads.items()}


def _flash_vit(preset, **kwargs):
    config = dict(FLASH_ENCODER_PRESETS[preset])
    config.update(kwargs)
    return FlashViT(**config)


def flash_vit_tiny(**kwargs):
    return _flash_vit("tiny", **kwargs)


def flash_vit_base(**kwargs):
    return _flash_vit("base", **kwargs)


def collate_tokens(batch, test=True):
    """Collate events into packed voxel/track tensors for the flash-attention models.

    Same event filtering and targets as utils.funcs.collate, but no spconv
    tensors: voxels and muon tracks are concatenated with their event index.
    """
    mode = "test" if test else "train"
    batch = [d for d in batch if len(d["coords"]) > 0]

    def packed(coords_key, feats_key):
        counts = torch.tensor([len(d[coords_key]) for d in batch], dtype=torch.long)
        coords = torch.cat([d[coords_key].reshape(-1, 3) for d in batch]).to(torch.int32)
        feats = torch.cat([d[feats_key].reshape(-1, d[feats_key].shape[-1]) for d in batch]).float()
        event = torch.repeat_interleave(torch.arange(len(batch), dtype=torch.int32), counts)
        return coords, feats, event

    ret = {"batch_size": len(batch)}
    ret["fcal_coords"], ret["fcal_feats"], ret["fcal_event"] = packed("coords", "feats")
    ret["ahcal_coords"], ret["ahcal_feats"], ret["ahcal_event"] = packed("ahcal_hits_coords", "ahcal_hits_feats")
    ret["ecal_hits"] = torch.stack([d["ecal_hits"] for d in batch])
    muon_counts = torch.tensor([len(d["muspec_info"]) for d in batch], dtype=torch.long)
    ret["muspec_feats"] = torch.cat([d["muspec_info"].reshape(-1, 5) for d in batch]).float()
    ret["muspec_event"] = torch.repeat_interleave(torch.arange(len(batch), dtype=torch.int32), muon_counts)
    ret["muspec_counts"] = torch.stack([d["nb_muspec_tracks"] for d in batch]).float()
    return collate_targets(batch, ret, mode)
