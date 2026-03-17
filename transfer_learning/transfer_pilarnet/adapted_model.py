"""PILArNet sparse encoder adapted from the FASERCal fine-tuning model."""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from spconv.pytorch import SparseConv3d, SparseSequential

from model.utils import (
    get_3d_sincos_pos_embed,
    BlockWithMask,
    CrossAttnBlock,
    CrossAttention,
)


def _choose_k1_k2(patch_size, max_k1_vol=32):
    """Pick small per-axis factors so prod(k1) <= max_k1_vol and k1 | patch."""
    k1 = []
    vol = 1
    for p in patch_size:
        f = 1
        for cand in (4, 3, 2):
            if p % cand == 0 and vol * cand <= max_k1_vol:
                f = cand
                break
        k1.append(f)
        vol *= f
    k2 = [p // f for p, f in zip(patch_size, k1)]
    return tuple(k1), tuple(k2)


class PILArNetEncoder(nn.Module):
    """Sparse encoder adapted for PILArNet particle-level crops."""

    def __init__(
        self,
        spatial_shape=(168, 168, 180),
        patch_size=(12, 12, 10),
        window_size=(2, 2, 3),
        in_chans=1,
        embed_dim=384,
        depth=4,
        io_depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        num_cls=2,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=None,
        meta_dim=0,
        meta_dropout=0.0,
        num_pid_classes=5,
        head_dropout_cls=0.0,
        head_init=2e-5,
        global_pool=True,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.spatial_shape = spatial_shape
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.head_init = head_init
        self.global_pool = global_pool
        self.meta_dim = int(meta_dim)
        self._pretrained_loaded_keys = set()
        self._patch_embed_loaded = False

        # Patch grid dimensions
        S = spatial_shape
        P = patch_size
        assert all(s % p == 0 for s, p in zip(S, P)), \
            "spatial_shape must be divisible by patch_size"
        self.grid_size = tuple(s // p for s, p in zip(S, P))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        W = window_size
        assert all(g % w == 0 for g, w in zip(self.grid_size, W)), \
            "grid_size must be divisible by window_size"
        self.window_grid_size = tuple(g // w for g, w in zip(self.grid_size, W))
        self.num_windows = (
            self.window_grid_size[0]
            * self.window_grid_size[1]
            * self.window_grid_size[2]
        )
        self.num_window_positions = W[0] * W[1] * W[2]

        # Sparse patch embedding
        vol = P[0] * P[1] * P[2]
        if vol > 512:
            mid = embed_dim // 4
            k1, k2 = _choose_k1_k2(P)
            self.patch_embed = SparseSequential(
                SparseConv3d(in_chans, mid, kernel_size=k1, stride=k1, padding=0, bias=False),
                norm_layer(mid),
                nn.GELU(),
                SparseConv3d(mid, embed_dim, kernel_size=k2, stride=k2, padding=0, bias=True),
            )
        else:
            self.patch_embed = SparseConv3d(
                in_chans, embed_dim, kernel_size=P, stride=P, padding=0, bias=True
            )

        # Frozen local/window positional embeddings mirror the pretraining layout.
        self.local_pos_embed = nn.Embedding(self.num_window_positions, embed_dim)
        local_pos = get_3d_sincos_pos_embed(embed_dim, self.window_size, cls_token=False)
        with torch.no_grad():
            self.local_pos_embed.weight.copy_(torch.from_numpy(local_pos).float())
            self.local_pos_embed.weight.requires_grad_(False)

        self.window_pos_embed = nn.Embedding(self.num_windows, embed_dim)
        window_pos = get_3d_sincos_pos_embed(
            embed_dim,
            self.window_grid_size,
            cls_token=False,
        )
        with torch.no_grad():
            self.window_pos_embed.weight.copy_(torch.from_numpy(window_pos).float())
            self.window_pos_embed.weight.requires_grad_(False)

        # Precompute the local-window token order once so the runtime path only gathers.
        Gx, Gy, Gz = self.grid_size
        Wx, Wy, Wz = self.window_size
        Wgx, Wgy, Wgz = self.window_grid_size
        gx = torch.arange(Gx)
        gy = torch.arange(Gy)
        gz = torch.arange(Gz)
        XX, YY, ZZ = torch.meshgrid(gx, gy, gz, indexing="ij")
        flat_ids = (XX * (Gy * Gz) + YY * Gz + ZZ).reshape(-1)
        window_ids = (
            (XX // Wx) * (Wgy * Wgz)
            + (YY // Wy) * Wgz
            + (ZZ // Wz)
        ).reshape(-1)
        local_ids = (
            (XX % Wx) * (Wy * Wz)
            + (YY % Wy) * Wz
            + (ZZ % Wz)
        ).reshape(-1)
        grouped = []
        for window_idx in range(self.num_windows):
            mask = window_ids == window_idx
            order = torch.argsort(local_ids[mask])
            grouped.append(flat_ids[mask][order])
        self.register_buffer("window_token_indices", torch.stack(grouped, dim=0))
        self.register_buffer(
            "window_ids",
            torch.arange(self.num_windows, dtype=torch.long),
        )
        self.t_patches_per_cls = max(
            1,
            self.num_window_positions // max(1, int(num_cls)),
        )

        # CLS token(s)
        self.num_cls = num_cls
        self.cls_token = nn.Parameter(torch.zeros(1, num_cls, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Drop-path schedule for the self-attention blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Self-attention blocks (same architecture/role as FASERCal local blocks)
        self.blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.tokens_norm = norm_layer(embed_dim)

        # Perceiver-IO bottleneck
        dp_lat_x = [0.0] * io_depth
        dp_lat_s = [0.0] * io_depth
        if drop_path_rate > 0:
            merge_dp = torch.linspace(
                dpr[-1] if depth > 0 else 0.0,
                drop_path_rate,
                2 * io_depth,
            )
            dp_lat_x = (merge_dp[0::2] * 0.5).tolist()
            dp_lat_s = merge_dp[1::2].tolist()

        self.lat_xattn_blocks = nn.ModuleList([
            CrossAttnBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dp_lat_x[i], norm_layer=norm_layer,
            )
            for i in range(io_depth)
        ])
        self.latent_self_blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dp_lat_s[i], norm_layer=norm_layer,
            )
            for i in range(io_depth)
        ])

        # Task head
        num_tasks = 1
        if not global_pool:
            self.latents_norm = norm_layer(embed_dim)
            self.task_tokens = nn.Parameter(torch.zeros(1, num_tasks, embed_dim))
            nn.init.normal_(self.task_tokens, std=0.02)
            self.task_cross_attn = CrossAttention(
                dim=embed_dim, num_heads=num_heads, qkv_bias=True,
                attn_drop=attn_drop_rate,
            )
            self.gamma = nn.Parameter(torch.ones(1) * 1e-4)

        self.head_pid = nn.Sequential(
            norm_layer(embed_dim),
            nn.Dropout(head_dropout_cls),
            nn.Linear(embed_dim, num_pid_classes),
        )
        self.meta_proj = None
        self.meta_drop = None
        if self.meta_dim > 0:
            self.meta_proj = nn.Sequential(
                nn.LayerNorm(self.meta_dim),
                nn.Linear(self.meta_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.meta_drop = nn.Dropout(meta_dropout)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        self.apply(self.__init_module)
        # Small head init
        lin = self.head_pid[-1]
        nn.init.trunc_normal_(lin.weight, std=self.head_init)
        if lin.bias is not None:
            nn.init.zeros_(lin.bias)

    @staticmethod
    def __init_module(m):
        if isinstance(m, SparseConv3d):
            w = m.weight
            nn.init.xavier_uniform_(w.view(w.size(0), -1))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def no_weight_decay(self):
        return {"cls_token", "task_tokens"}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def densify_patches(self, x_sp):
        """Convert SparseConvTensor to dense tokens + occupancy mask."""
        B = x_sp.batch_size
        C = x_sp.features.size(1)
        Gx, Gy, Gz = x_sp.spatial_shape

        x_dense = x_sp.dense()  # [B, C, Gx, Gy, Gz]
        tokens = x_dense.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C)

        # Occupancy mask from sparse indices
        idx = x_sp.indices.long()
        b, h, w, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        occ = torch.zeros((B, Gx, Gy, Gz), dtype=torch.bool, device=x_sp.features.device)
        occ[b, h, w, d] = True
        attn_mask = occ.view(B, -1)
        return tokens, attn_mask

    def _group_tokens_by_window(self, x, attn_mask):
        """Group dense patch tokens into fixed local windows."""
        B, _, C = x.shape
        W = self.num_windows
        L = self.num_window_positions
        idx = self.window_token_indices.unsqueeze(0).expand(B, -1, -1)
        flat_idx = idx.reshape(B, W * L)
        x_win = torch.gather(
            x,
            1,
            flat_idx.unsqueeze(-1).expand(-1, -1, C),
        ).contiguous()
        mask_win = torch.gather(attn_mask, 1, flat_idx).contiguous()
        return x_win.view(B, W, L, C), mask_win.view(B, W, L)

    @staticmethod
    def _pack_flat_by_mask(seq, mask):
        """Pack variable-length rows to the rowwise max number of real tokens."""
        N_rows, _, C = seq.shape
        device = seq.device
        row_ids, tok_ids = torch.nonzero(mask, as_tuple=True)
        N_real = row_ids.numel()
        if N_real == 0:
            packed = seq.new_zeros(N_rows, 0, C)
            keep = torch.zeros(N_rows, 0, dtype=torch.bool, device=device)
            pos_ids = torch.zeros(N_rows, 0, dtype=torch.long, device=device)
            return packed, keep, pos_ids

        _, counts = torch.unique_consecutive(row_ids, return_counts=True)
        n_max = int(counts.max().item())
        starts = torch.cumsum(counts, 0) - counts
        within = torch.arange(N_real, device=device) - torch.repeat_interleave(starts, counts)

        packed = seq.new_zeros(N_rows, n_max, C)
        keep = torch.zeros(N_rows, n_max, dtype=torch.bool, device=device)
        pos_ids = torch.zeros(N_rows, n_max, dtype=torch.long, device=device)
        packed[row_ids, within] = seq[row_ids, tok_ids]
        keep[row_ids, within] = True
        pos_ids[row_ids, within] = tok_ids
        return packed, keep, pos_ids

    def _pack_by_mask(self, kv, mask):
        """Pack tokens from [B, W, L, C] to [B, N_max, C] using only real patches."""
        B, W, L, C = kv.shape
        device = kv.device
        b_ids, w_ids, l_ids = torch.nonzero(mask, as_tuple=True)
        N = b_ids.numel()
        if N == 0:
            packed = kv.new_zeros(B, 0, C)
            keep = torch.zeros(B, 0, dtype=torch.bool, device=device)
            return packed, keep, b_ids, w_ids, l_ids

        _, counts = torch.unique_consecutive(b_ids, return_counts=True)
        n_max = int(counts.max().item())
        starts = torch.cumsum(counts, 0) - counts
        within = torch.arange(N, device=device) - torch.repeat_interleave(starts, counts)

        packed = kv.new_zeros(B, n_max, C)
        keep = torch.zeros(B, n_max, dtype=torch.bool, device=device)
        packed[b_ids, within] = kv[b_ids, w_ids, l_ids]
        keep[b_ids, within] = True
        return packed, keep, b_ids, w_ids, l_ids

    def _pack_active_windows(self, x_sp):
        """Pack only occupied sparse patch tokens into active local windows."""
        idx = x_sp.indices.long()
        tok = x_sp.features
        device = tok.device
        C = tok.size(1)

        if idx.numel() == 0:
            empty = tok.new_zeros((0, 0, C))
            empty_keep = torch.zeros((0, 0), dtype=torch.bool, device=device)
            empty_ids = torch.zeros((0, 0), dtype=torch.long, device=device)
            empty_rows = torch.zeros((0,), dtype=torch.long, device=device)
            return empty, empty_keep, empty_ids, empty_rows, empty_rows

        b = idx[:, 0]
        x = idx[:, 1]
        y = idx[:, 2]
        z = idx[:, 3]

        Wx, Wy, Wz = self.window_size
        _, Wgy, Wgz = self.window_grid_size
        window_ids = (
            (x // Wx) * (Wgy * Wgz)
            + (y // Wy) * Wgz
            + (z // Wz)
        )
        local_ids = (
            (x % Wx) * (Wy * Wz)
            + (y % Wy) * Wz
            + (z % Wz)
        )

        group_ids = b * self.num_windows + window_ids
        sort_key = group_ids * self.num_window_positions + local_ids
        order = torch.argsort(sort_key)

        tok = tok[order]
        b = b[order]
        window_ids = window_ids[order]
        local_ids = local_ids[order]
        group_ids = group_ids[order]

        _, counts = torch.unique_consecutive(group_ids, return_counts=True)
        n_rows = int(counts.numel())
        n_max = int(counts.max().item())
        starts = torch.cumsum(counts, dim=0) - counts
        row_ids = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.long),
            counts,
        )
        within = torch.arange(group_ids.numel(), device=device) - torch.repeat_interleave(
            starts,
            counts,
        )

        packed = tok.new_zeros((n_rows, n_max, C))
        keep = torch.zeros((n_rows, n_max), dtype=torch.bool, device=device)
        pos_ids = torch.zeros((n_rows, n_max), dtype=torch.long, device=device)
        packed[row_ids, within] = tok
        keep[row_ids, within] = True
        pos_ids[row_ids, within] = local_ids

        row_batch_ids = b[starts]
        row_window_ids = window_ids[starts]
        return packed, keep, pos_ids, row_batch_ids, row_window_ids

    @staticmethod
    def _pack_active_rows_by_batch(seq, keep, row_batch_ids, batch_size):
        """Pack selected tokens from active rows back to one variable-length batch."""
        C = seq.size(-1)
        device = seq.device
        row_ids, tok_ids = torch.nonzero(keep, as_tuple=True)
        if row_ids.numel() == 0:
            packed = seq.new_zeros((batch_size, 0, C))
            keep_out = torch.zeros((batch_size, 0), dtype=torch.bool, device=device)
            return packed, keep_out

        b_ids = row_batch_ids[row_ids]
        _, counts = torch.unique_consecutive(b_ids, return_counts=True)
        n_max = int(counts.max().item())
        starts = torch.cumsum(counts, dim=0) - counts
        within = torch.arange(row_ids.numel(), device=device) - torch.repeat_interleave(
            starts,
            counts,
        )

        packed = seq.new_zeros((batch_size, n_max, C))
        keep_out = torch.zeros((batch_size, n_max), dtype=torch.bool, device=device)
        packed[b_ids, within] = seq[row_ids, tok_ids]
        keep_out[b_ids, within] = True
        return packed, keep_out

    def _cls_keep_from_patchmask(
        self,
        patch_mask,
        num_cls,
        t,
        ensure_one_if_nonempty=True,
        dummy_one_if_empty=False,
    ):
        """Build a boolean keep mask for CLS queries from a patch occupancy mask."""
        num_cls = int(num_cls)
        if num_cls <= 0:
            if patch_mask.dim() == 2:
                return patch_mask.new_zeros(patch_mask.size(0), 0)
            if patch_mask.dim() == 3:
                return patch_mask.new_zeros(patch_mask.size(0), patch_mask.size(1), 0)
            raise ValueError(f"patch_mask must be rank-2 or rank-3, got {patch_mask.shape}")

        t = max(1, int(t))
        v = patch_mask.sum(dim=-1).to(torch.long)
        k = (v + (t - 1)) // t
        k = torch.minimum(k, v).clamp(min=0, max=num_cls)
        if ensure_one_if_nonempty:
            k = torch.where(v > 0, k.clamp_min(1), k)
        if dummy_one_if_empty:
            k = torch.where(v == 0, torch.ones_like(k), k)

        device = patch_mask.device
        if patch_mask.dim() == 3:
            slot = torch.arange(num_cls, device=device).view(1, 1, num_cls)
            return slot < k.unsqueeze(-1)
        slot = torch.arange(num_cls, device=device).view(1, num_cls)
        return slot < k.view(-1, 1)

    def forward_features(self, x_sp):
        """Encode sparse voxels while skipping empty windows entirely."""
        x_sp = self.patch_embed(x_sp)
        B = x_sp.batch_size
        C = self.embed_dim

        x_local, keep_local, local_pos_ids, row_batch_ids, row_window_ids = self._pack_active_windows(x_sp)
        if x_local.size(0) == 0:
            lat = self.cls_token.new_zeros((B, 1, C))
            lat_keep = torch.ones((B, 1), dtype=torch.bool, device=lat.device)
            return lat, lat_keep

        x_local = x_local + self.local_pos_embed(local_pos_ids) * keep_local.unsqueeze(-1).to(x_local.dtype)

        n_rows = x_local.size(0)
        cls = self.cls_token.expand(n_rows, -1, -1)
        x_local = torch.cat([cls, x_local], dim=1)
        x_local = self.pos_drop(x_local)
        cls_keep_local = self._cls_keep_from_patchmask(
            keep_local,
            num_cls=self.num_cls,
            t=self.t_patches_per_cls,
            ensure_one_if_nonempty=True,
            dummy_one_if_empty=False,
        )
        attn_mask_local = torch.cat([cls_keep_local, keep_local], dim=1)
        for blk in self.blocks:
            x_local = blk(x_local, attn_mask=attn_mask_local, q_mask=attn_mask_local)
        x_local = self.norm(x_local)

        cls_win = x_local[:, : self.num_cls, :]
        tok_win = x_local[:, self.num_cls :, :]

        window_bias = self.window_pos_embed(row_window_ids).unsqueeze(1)
        cls_win = cls_win + window_bias
        tok_win = tok_win + window_bias

        kv, kv_keep = self._pack_active_rows_by_batch(tok_win, keep_local, row_batch_ids, B)
        kv = self.tokens_norm(kv)

        lat, lat_keep = self._pack_active_rows_by_batch(cls_win, cls_keep_local, row_batch_ids, B)
        empty_evt = lat_keep.sum(dim=1) == 0
        if lat.size(1) == 0:
            lat = cls_win.new_zeros((B, 1, C))
            lat_keep = torch.zeros((B, 1), dtype=torch.bool, device=lat.device)
            empty_evt = torch.ones((B,), dtype=torch.bool, device=lat.device)
        if empty_evt.any():
            lat_keep = lat_keep.clone()
            lat_keep[empty_evt, 0] = True

        for xa, sa in zip(self.lat_xattn_blocks, self.latent_self_blocks):
            lat = xa(lat, kv, attn_mask=kv_keep, q_mask=lat_keep)
            lat = sa(lat, attn_mask=lat_keep, q_mask=lat_keep)

        return lat, lat_keep

    def pool_latents(self, lat, lat_keep):
        """Pool latent tokens to one embedding per particle."""
        den = lat_keep.sum(dim=1, keepdim=True).clamp_min(1).float()
        return (lat * lat_keep.unsqueeze(-1).float()).sum(dim=1) / den

    def encode_pooled(self, x_sp):
        """Return one pooled embedding per sparse particle input."""
        lat, lat_keep = self.forward_features(x_sp)
        return self.pool_latents(lat, lat_keep)

    def _fuse_particle_meta(self, pooled, particle_meta):
        if self.meta_proj is None:
            return pooled
        if particle_meta is None:
            raise RuntimeError("particle_meta is required when the single-particle metadata branch is enabled.")
        return pooled + self.meta_drop(self.meta_proj(particle_meta))

    def forward(self, x_sp, particle_meta=None):
        lat, lat_keep = self.forward_features(x_sp)
        B = lat.shape[0]

        if self.global_pool:
            pooled = self.pool_latents(lat, lat_keep)
            pooled = self._fuse_particle_meta(pooled, particle_meta)
            out_pid = self.head_pid(pooled)
        else:
            task_q = self.task_tokens.expand(B, -1, -1)
            lat_kv = self.latents_norm(lat)
            outcome = task_q + self.task_cross_attn(task_q, lat_kv, attn_mask=lat_keep) * self.gamma
            outcome = self._fuse_particle_meta(outcome[:, 0, :], particle_meta)
            out_pid = self.head_pid(outcome)

        return {"out_pid": out_pid}


# -----------------------------------------------------------------------
# Model factory functions
# -----------------------------------------------------------------------

def pilarnet_encoder_tiny(**kwargs):
    defaults = dict(
        embed_dim=384, depth=2, io_depth=6, num_heads=12,
        mlp_ratio=4.0, num_cls=2, global_pool=True,
    )
    defaults.update(kwargs)
    return PILArNetEncoder(**defaults)


def pilarnet_encoder_base(**kwargs):
    defaults = dict(
        embed_dim=384, depth=4, io_depth=4, num_heads=12,
        mlp_ratio=4.0, num_cls=2, global_pool=True,
    )
    defaults.update(kwargs)
    return PILArNetEncoder(**defaults)
