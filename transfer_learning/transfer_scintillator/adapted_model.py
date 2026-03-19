"""Scintillator encoder used for PID transfer."""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from spconv.pytorch import SparseConv3d, SparseSequential

from model.utils import (
    BlockWithMask,
    CrossAttention,
    CrossAttnBlock,
    get_3d_sincos_pos_embed,
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


class ScintillatorPIDEncoder(nn.Module):
    """Sparse encoder for particle identification on the scintillator dataset."""

    def __init__(
        self,
        spatial_shape=(120, 120, 120),
        patch_size=(12, 12, 10),
        window_size=(2, 2, 2),
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
        num_pid_classes=4,
        global_feat_dim=31,
        head_dropout=0.0,
        head_init=2e-5,
        global_pool=True,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.spatial_shape = tuple(spatial_shape)
        self.patch_size = tuple(patch_size)
        self.window_size = tuple(window_size)
        self.embed_dim = embed_dim
        self.head_init = head_init
        self.global_pool = global_pool
        self.global_feat_dim = int(global_feat_dim)
        self._pretrained_loaded_keys = set()
        self._patch_embed_loaded = False

        S = self.spatial_shape
        P = self.patch_size
        assert all(s % p == 0 for s, p in zip(S, P)), \
            "spatial_shape must be divisible by patch_size"
        self.grid_size = tuple(s // p for s, p in zip(S, P))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        W = self.window_size
        assert all(w > 0 for w in W), "window_size entries must be positive"
        assert all(w <= g for w, g in zip(W, self.grid_size)), \
            "window_size cannot exceed the patch grid size"
        self.window_grid_size = tuple(
            (g + w - 1) // w for g, w in zip(self.grid_size, W)
        )
        self.num_windows = (
            self.window_grid_size[0]
            * self.window_grid_size[1]
            * self.window_grid_size[2]
        )
        self.num_window_positions = W[0] * W[1] * W[2]

        vol = P[0] * P[1] * P[2]
        if vol > 512:
            mid = embed_dim // 4
            k1, k2 = _choose_k1_k2(P)
            self.patch_embed = SparseSequential(
                SparseConv3d(
                    in_chans,
                    mid,
                    kernel_size=k1,
                    stride=k1,
                    padding=0,
                    bias=False,
                ),
                norm_layer(mid),
                nn.GELU(),
                SparseConv3d(
                    mid,
                    embed_dim,
                    kernel_size=k2,
                    stride=k2,
                    padding=0,
                    bias=True,
                ),
            )
        else:
            self.patch_embed = SparseConv3d(
                in_chans,
                embed_dim,
                kernel_size=P,
                stride=P,
                padding=0,
                bias=True,
            )

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

        self.t_patches_per_cls = max(
            1,
            self.num_window_positions // max(1, int(num_cls)),
        )

        self.num_cls = int(num_cls)
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_cls, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.tokens_norm = norm_layer(embed_dim)

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
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dp_lat_x[i],
                norm_layer=norm_layer,
            )
            for i in range(io_depth)
        ])
        self.latent_self_blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dp_lat_s[i],
                norm_layer=norm_layer,
            )
            for i in range(io_depth)
        ])

        if not global_pool:
            self.latents_norm = norm_layer(embed_dim)
            self.task_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.task_tokens, std=0.02)
            self.task_cross_attn = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=True,
                attn_drop=attn_drop_rate,
            )
            self.gamma = nn.Parameter(torch.ones(1) * 1e-4)

        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.global_feat_dim),
            nn.Linear(self.global_feat_dim, embed_dim),
            nn.GELU(),
            norm_layer(embed_dim),
        )
        self.context_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.context_type_embed, std=0.02)
        self.head_pid = nn.Sequential(
            norm_layer(embed_dim),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, num_pid_classes),
        )

        self._init_weights()

    def _init_weights(self):
        self.apply(self.__init_module)
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
        return {"cls_token", "task_tokens", "context_type_embed"}

    def _make_context_kv(self, global_features, batch_size, device, dtype):
        if global_features is None:
            return None, None
        context_tok = self.context_proj(global_features.to(dtype=dtype)).unsqueeze(1)
        context_tok = context_tok + self.context_type_embed.to(dtype=dtype)
        context_keep = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        return context_tok, context_keep

    def _pack_active_windows(self, x_sp):
        """Pack only occupied patch tokens into active local windows."""
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

    def forward_features(self, x_sp, global_features=None):
        """Encode sparse voxels while skipping empty windows entirely."""
        x_sp = self.patch_embed(x_sp)
        B = x_sp.batch_size
        C = self.embed_dim
        context_tok, context_keep = self._make_context_kv(
            global_features,
            batch_size=B,
            device=x_sp.features.device,
            dtype=x_sp.features.dtype,
        )

        x_local, keep_local, local_pos_ids, row_batch_ids, row_window_ids = self._pack_active_windows(x_sp)
        if x_local.size(0) == 0:
            lat = self.cls_token.new_zeros((B, 1, C))
            lat_keep = torch.ones((B, 1), dtype=torch.bool, device=lat.device)
            if context_tok is not None:
                kv = self.tokens_norm(context_tok)
                for xa, sa in zip(self.lat_xattn_blocks, self.latent_self_blocks):
                    lat = xa(lat, kv, attn_mask=context_keep, q_mask=lat_keep)
                    lat = sa(lat, attn_mask=lat_keep, q_mask=lat_keep)
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
        if context_tok is not None:
            kv = torch.cat([kv, context_tok], dim=1)
            kv_keep = torch.cat([kv_keep, context_keep], dim=1)
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
        """Pool latent tokens to one embedding per sample."""
        den = lat_keep.sum(dim=1, keepdim=True).clamp_min(1).float()
        return (lat * lat_keep.unsqueeze(-1).float()).sum(dim=1) / den

    def forward(self, x_sp, global_features=None):
        lat, lat_keep = self.forward_features(x_sp, global_features)
        B = lat.shape[0]

        if self.global_pool:
            pooled = self.pool_latents(lat, lat_keep)
            logits = self.head_pid(pooled)
        else:
            task_q = self.task_tokens.expand(B, -1, -1)
            lat_kv = self.latents_norm(lat)
            outcome = task_q + self.task_cross_attn(
                task_q,
                lat_kv,
                attn_mask=lat_keep,
            ) * self.gamma
            rep = outcome[:, 0, :]
            logits = self.head_pid(rep)

        return {"logits": logits}


def scintillator_encoder_tiny(**kwargs):
    defaults = dict(
        embed_dim=384,
        depth=2,
        io_depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        num_cls=2,
        global_pool=True,
    )
    defaults.update(kwargs)
    return ScintillatorPIDEncoder(**defaults)


def scintillator_encoder_base(**kwargs):
    defaults = dict(
        embed_dim=384,
        depth=4,
        io_depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        num_cls=2,
        global_pool=True,
    )
    defaults.update(kwargs)
    return ScintillatorPIDEncoder(**defaults)
