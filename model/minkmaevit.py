"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 09.25

Description: PyTorch MAE-ViT model with spconv patching.
"""

import math
import torch
import torch.nn as nn
from spconv.pytorch import SparseConv3d, SparseSequential
from functools import partial
from .utils import (
    get_3d_sincos_pos_embed, choose_k1_k2, BlockWithMask, GlobalFeatureEncoderSimple, 
    CrossAttnBlock, SeparableDCT3D, SharedLatentVoxelHead, LazyIdxMap
)


class MinkMAEViT(nn.Module):
    def __init__(
        self,
        in_chans=1,
        D=3,
        img_size=(48, 48, 200),
        module_depth_voxels=20,
        embed_dim=384,
        patch_size=(16, 16, 4),
        num_module_cls=1,
        depth=8,
        num_global_tokens=1,
        io_depth=4,
        io_decode_depth=4,
        num_heads=16,
        num_modes=(16, 8),
        num_pid_classes=3,
        decoder_embed_dim=192,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        drop_rate_dec=0.,
        attn_drop_rate_dec=0.,
        norm_layer=nn.LayerNorm,
        metadata=None,
    ):
        """
        Args:
            in_chans (int): Number of input channels.
            D (int): Spatial dimension for Minkowski layers.
            img_size (tuple): Input image size (H, W, D).
            args: Namespace with at least a `dataset_path` attribute.
        """
        super().__init__()

        self.metadata = metadata
    
        # patch and grid setup
        H, W, D_img = img_size
        p_h, p_w, p_d = patch_size
        assert H % p_h == 0 and W % p_w == 0 and D_img % p_d == 0, \
            "img_size must be divisible by patch_size"
        self.in_chans = in_chans
        self.grid_size = (H // p_h, W // p_w, D_img // p_d)
        self.num_patches = (self.grid_size[0] * self.grid_size[1] * self.grid_size[2])
        self.patch_voxels = p_h * p_w * p_d
        self.register_buffer('patch_size', torch.tensor(patch_size, dtype=torch.long))

        # module slicing along Z
        assert module_depth_voxels % p_d == 0, "module_depth_voxels must be divisible by patch depth"
        self.module_depth_voxels = module_depth_voxels
        self.module_depth_patches = module_depth_voxels // p_d
        G_h, G_w, G_d = self.grid_size
        assert G_d % self.module_depth_patches == 0, "grid depth must be multiple of module depth (in patches)"
        self.num_modules = G_d // self.module_depth_patches
        self.intra_grid_size = (G_h, G_w, self.module_depth_patches)     # H×W×(depth within module)
        self.num_intra_positions = G_h * G_w * self.module_depth_patches

        # patch embedding
        if self.patch_size.prod().item() > 512:
            # too large -> use two-step conv
            mid = embed_dim // 4
            k1, k2 = choose_k1_k2(patch_size)
            self.patch_embed = SparseSequential(
                SparseConv3d(in_chans, mid, kernel_size=k1, stride=k1, padding=0, bias=False),
                norm_layer(mid),
                nn.GELU(),
                SparseConv3d(mid, embed_dim, kernel_size=k2, stride=k2, padding=0, bias=True)
            )
        else:
            self.patch_embed = SparseConv3d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, 
                padding=0, bias=True,
            )

        # Precompute dense patch templates
        mh = torch.arange(G_h)
        mw = torch.arange(G_w)
        md = torch.arange(G_d)
        HH, WW, DD = torch.meshgrid(mh, mw, md, indexing='ij')

        flat_ids = (HH * (G_w * G_d) + WW * G_d + DD).reshape(-1)         # [Np]
        d_mod    = (DD % self.module_depth_patches)
        module   = (DD // self.module_depth_patches).reshape(-1)          # [Np]
        intra    = (HH * (G_w * self.module_depth_patches)
                   + WW * self.module_depth_patches + d_mod).reshape(-1)  # [Np]

        self.register_buffer('idx_template',       flat_ids.long())       # [Np]
        self.register_buffer('intra_idx_template', intra.long())          # [Np]

        # Per-module token index mapping [M, Lm]
        M = self.num_modules
        module_indices = []
        for m in range(M):
            mask = (module.long() == m)
            intra_m = self.intra_idx_template[mask]
            flat_m  = self.idx_template[mask]
            order   = torch.argsort(intra_m)      # stable intra order
            module_indices.append(flat_m[order])
        self.register_buffer('module_token_indices', torch.stack(module_indices, 0))  # [M, Lm]

        # Encoder: hierarchical ViT
        self.intra_depth = depth
        self.num_module_cls = num_module_cls
        self.module_cls_token = nn.Parameter(torch.zeros(1, num_module_cls, embed_dim))  # per-module CLS (shared weights)
        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim)         # fixed sin-cos per patch
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)                # learned module index for intra-attn

        # Intra-module transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.intra_depth)]
        self.blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(self.intra_depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Perceiver-IO bottleneck (encoder side)
        self.num_global_tokens = num_global_tokens
        self.global_feats_encoder = GlobalFeatureEncoderSimple(embed_dim, dropout=drop_rate)
        self.global_mem = nn.Parameter(torch.zeros(1, self.num_global_tokens, embed_dim))
        self.xattn_blocks = nn.ModuleDict({
            name: nn.ModuleList([
                CrossAttnBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=0., norm_layer=norm_layer
                )
                for _ in range(io_depth)
            ])
            for name in ["lat<-tok", "tok<-lat"]
        })
        self.latent_self_blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=0., norm_layer=norm_layer
            )
            for _ in range(io_depth)
        ])
        self.tokens_norm = norm_layer(embed_dim)

        # Perceiver-IO decoder for reconstruction
        assert decoder_embed_dim % decoder_num_heads == 0, "decoder_embed_dim must be divisible by decoder_num_heads"
        self.decoder_intra_pos_embed = nn.Embedding(self.num_intra_positions, decoder_embed_dim) # frozen sin-cos
        self.module_embed_dec = nn.Embedding(self.num_modules, decoder_embed_dim)
        self.query_tokens = nn.Parameter(torch.zeros(1, decoder_embed_dim))
        self.enc_to_dec = nn.Linear(embed_dim, decoder_embed_dim)
        self.decode_xattn_blocks = nn.ModuleList([
            CrossAttnBlock(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, 
                mlp_ratio=mlp_ratio, qkv_bias=True, drop=drop_rate_dec, 
                attn_drop=attn_drop_rate_dec, drop_path=0., norm_layer=norm_layer
            )
            for _ in range(io_decode_depth)
        ])

        # Heads
        self.sep_basis = SeparableDCT3D(self.patch_size.tolist(), alphas=(0.4, 0.4, 0.6))
        self.shared_voxel_head = nn.ModuleDict({
            name: SharedLatentVoxelHead(
                decoder_embed_dim, self.sep_basis, H=num_modes[i], 
                norm_layer=norm_layer, post_norm=True
            )
            for i, name in enumerate(["rel", "rec"])
        })
        self.head_channels = {
            "gho": 1,
            "hie": 3,
            "dec": 3,
            "pid": num_pid_classes,
            "occ": 1,
            "reg": in_chans,
        }
        self.heads = nn.ModuleDict({
            name: nn.Linear(num_modes[0] if name in ["gho", "hie", "dec", "pid"] 
                            else num_modes[1],
                            self.head_channels[name]) 
            for name in self.head_channels.keys()
        })

        self.initialize_weights()


    def initialize_weights(self):
        # init fixed pos embeddings
        enc_pos = get_3d_sincos_pos_embed(
            self.intra_pos_embed.weight.shape[-1],
            self.intra_grid_size,
            cls_token=False
        )
        with torch.no_grad():
            self.intra_pos_embed.weight.copy_(torch.from_numpy(enc_pos).float())
            self.intra_pos_embed.weight.requires_grad_(False)
            
        dec_pos = get_3d_sincos_pos_embed(
            self.decoder_intra_pos_embed.weight.shape[-1],
            self.intra_grid_size,
            cls_token=False
        )
        with torch.no_grad():
            self.decoder_intra_pos_embed.weight.copy_(torch.from_numpy(dec_pos).float())
            self.decoder_intra_pos_embed.weight.requires_grad_(False)

        # init tokens
        with torch.no_grad():
            nn.init.normal_(self.global_mem, std=.02)
            nn.init.normal_(self.module_cls_token, std=.02)
            nn.init.normal_(self.module_embed_enc.weight, std=0.02)
            nn.init.normal_(self.module_embed_dec.weight, std=0.02)
            nn.init.normal_(self.query_tokens, std=0.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, SparseConv3d):
            # initialize conv like nn.Linear
            w = m.weight
            torch.nn.init.xavier_uniform_(w.view(w.size(0), -1))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def no_weight_decay(self):
        return {
            'module_cls_token',
            'module_embed_enc.weight',
            'module_embed_dec.weight',
            'global_mem',
            'query_tokens',
        }
    

    def densify_patches(self, x_sp):
        B = x_sp.batch_size
        C = x_sp.features.size(1)
        X, Y, Z = x_sp.spatial_shape  # == (H, W, D)

        # Dense is [B, C, X, Y, Z] → flatten H->W->D by permuting to [B, X, Y, Z, C]
        x_dense = x_sp.dense()  # [B, C, X, Y, Z]
        dense_tokens = x_dense.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C)

        # Mask from indices
        idx = x_sp.indices.long()
        b, h, w, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        occ = torch.zeros((B, X, Y, Z), dtype=torch.bool, device=x_sp.features.device)
        occ[b, h, w, d] = True
        attn_mask = occ.view(B, -1)  # H->W->D order matches the permute above

        intra_idx = self.intra_idx_template.unsqueeze(0).expand(B, -1)
        return dense_tokens, attn_mask, intra_idx


    def build_patch_occupancy_map(self, x):
        """
        From the original sparse tensor coordinates, build a [B, N_patches, P] mapping
        with the raw id of the actual hit in that sub‐voxel.
        """
        idx = x.indices.long()  # [N, 4] = [b, x, y, z] == [b, h, w, d]
        b, h, w, d = idx.unbind(-1)

        p_h, p_w, p_d = self.patch_size.tolist()
        Gh, Gw, Gd    = self.grid_size
        P             = self.patch_voxels
        Np            = self.num_patches

        patch_idx = (h // p_h) * (Gw * Gd) + (w // p_w) * Gd + (d // p_d)
        sub_idx   = (h %  p_h) * (p_w * p_d) + (w %  p_w) * p_d + (d %  p_d)

        key   = (b * (Np * P)) + (patch_idx.to(torch.int64) * P) + sub_idx.to(torch.int64)  # [N]
        order = torch.argsort(key)                                                          # [N]
        sorted_keys   = key[order].contiguous()
        sorted_to_raw = order.contiguous()    # raw hit ids (0..N-1)

        return LazyIdxMap(
            sorted_keys=sorted_keys,
            sorted_to_raw=sorted_to_raw,
            patches_per_evt=Np,
            voxels_per_patch=P,
        )
    

    def _group_tokens_by_module(self, x, attn_mask):
        """
        x: [B, Np, C] dense tokens
        attn_mask: [B, Np]  (bool)
        returns:
            x_mod: [B, M, Lm, C] re-ordered by module then intra-order
            attn_mask_mod: [B, M, Lm] (bool)
        """
        B, Np, C = x.shape
        M, Lm = self.num_modules, self.num_intra_positions
        idx = self.module_token_indices.unsqueeze(0).expand(B, -1, -1)  # [B, M, Lm]
        flat_idx = idx.reshape(B, M * Lm)

        x_mod = torch.gather(x, 1, flat_idx.unsqueeze(-1).expand(-1, -1, C)).contiguous()  # [B, M*Lm, C]
        attn_mask_mod = torch.gather(attn_mask, 1, flat_idx).contiguous()                  # [B, M*Lm]
        return x_mod.view(B, M, Lm, C), attn_mask_mod.view(B, M, Lm)


    def _module_random_masking(self, x, attn_mask, mask_ratio, enforce_both=True):
        """
        Occupancy-aware masking with fixed shapes and no wasted compute.

        Args:
            x:          [B, M, Lm, C]   tokens per module
            attn_mask:  [B, M, Lm]      True = real token, False = pad/invalid
            mask_ratio: float           desired mask ratio over *real* tokens
            enforce_both (bool):        if True, for modules with v>=2, keep at least 1 and mask at least 1

        Returns:
            x_keep:       [B, M, Lk, C]      gathered tokens (padded with dummies)
            attn_keep:    [B, M, Lk]         True for real-kept tokens; False = pad/dummy
            rand_mask:    [B, M, Lm]         True for masked real tokens; False elsewhere
            ids_keep_all: [B, M, Lk]         intra indices selected per module (order ~random)
            keep:         [B, M]             #real kept per module (for logging/metrics)
        """
        B, M, Lm, C = x.shape
        device = x.device

        if mask_ratio <= 0.0:
            # keep everything that’s real; preserve shapes (Lk == Lm)
            ids_keep_all = torch.arange(Lm, device=device).view(1, 1, Lm).expand(B, M, Lm)
            x_keep       = x
            attn_keep    = attn_mask.clone()
            rand_mask    = torch.zeros_like(attn_mask)
            keep         = attn_mask.sum(dim=-1).to(torch.long)
            return x_keep, attn_keep, rand_mask, ids_keep_all, keep
    
        valid = attn_mask  # [B,M,Lm], bool

        # fixed gather width (static shape)
        Lk = max(1, int(math.ceil(Lm * (1.0 - mask_ratio))))  # keep constant for efficiency

        # per-module real kept (unbiased stochastic rounding of v*(1-r))
        v = valid.sum(dim=-1)                                   # [B,M], long
        keep_f = v.float() * (1.0 - mask_ratio)                 # [B,M]
        keep   = keep_f.floor().long()
        frac   = keep_f - keep.float()
        keep   = keep + (torch.rand_like(frac) < frac).long()   # unbiased

        if enforce_both:
            # v==0 -> keep=0; v==1 -> keep=1; v>=2 -> clamp to [1, v-1]
            keep = torch.where(v == 0, torch.zeros_like(keep), keep)
            keep = torch.where(v == 1, torch.ones_like(keep), keep)
            keep_capped = torch.minimum(keep.clamp_min(1), v - 1)
            keep = torch.where(v >= 2, keep_capped, keep)
        else:
            keep = keep.clamp_min(0)
            keep = torch.minimum(keep, v)

        # sample Lk candidate positions among valids (invalids set to +inf)
        noise = torch.rand(B, M, Lm, device=device).masked_fill(~valid, float('inf'))
        ids_keep_all = torch.topk(noise, k=Lk, dim=-1, largest=False, sorted=False).indices  # [B, M, Lk]

        # flags telling which gathered slots are real vs invalid
        real_flags = torch.gather(valid, 2, ids_keep_all)                 # [B, M, Lk], bool

        # among gathered real slots, mark only the first 'keep' as actually kept
        real_cum = real_flags.int().cumsum(dim=-1)                        # [B, M, Lk]
        attn_keep = real_flags & (real_cum <= keep.unsqueeze(-1))         # [B, M, Lk], bool

        # gather tokens (padding stays; masked out by attn_keep downstream)
        x_keep = torch.gather(x, 2, ids_keep_all.unsqueeze(-1).expand(-1, -1, -1, C))  # [B,M,Lk,C]

        # build rand_mask over original intra space: real & not kept -> True
        kept_full = torch.zeros_like(valid)                                # [B, M, Lm], bool
        kept_full.scatter_(2, ids_keep_all, attn_keep)                     # mark real-kept only
        rand_mask = valid & (~kept_full)                                   # True = masked real, False else

        # attn_keep is exactly which gathered slots are real-kept
        return x_keep, attn_keep, rand_mask, ids_keep_all, keep
    

    def _pack_by_mask(self, kv: torch.Tensor, mask: torch.Tensor):
        """
        Pack tokens by boolean mask.
        kv:   [B, M, L, C]
        mask: [B, M, L]  (True = real)
        returns:
            packed:  [B, N_max, C]     (padded per batch)
            keep:    [B, N_max]        (bool)
            b_ids, m_ids, l_ids: 1D index lists for real tokens (flattened)
            within:  1D ranks within batch for scattering/gathering
            N_max:   int, max number of real tokens in a batch
        """
        B, M, L, C = kv.shape
        device = kv.device
        b_ids, m_ids, l_ids = torch.nonzero(mask, as_tuple=True)  # [N_real]
        N = b_ids.numel()
        if N == 0:
            N_max = 0
            packed = kv.new_zeros(B, 0, C)
            keep = torch.zeros(B, 0, dtype=torch.bool, device=device)
            within = b_ids
            return packed, keep, b_ids, m_ids, l_ids, within, N_max

        # per-batch ranks
        _, counts = torch.unique_consecutive(b_ids, return_counts=True)
        N_max = int(counts.max().item())
        starts = torch.cumsum(counts, 0) - counts
        within = torch.arange(N, device=device) - torch.repeat_interleave(starts, counts)

        packed = kv.new_zeros(B, N_max, C)
        keep   = torch.zeros(B, N_max, dtype=torch.bool, device=device)
        packed[b_ids, within] = kv[b_ids, m_ids, l_ids, :]
        keep[b_ids, within]   = True
        return packed, keep, b_ids, m_ids, l_ids, within, N_max


    def _unpack_to_modules(self, packed: torch.Tensor,
                        b_ids: torch.Tensor, m_ids: torch.Tensor, l_ids: torch.Tensor,
                        within: torch.Tensor,
                        out_shape: torch.Size):
        """
        Inverse of _pack_by_mask for values. Returns a full [B,M,L,C] tensor where
        only real slots (mask True) are filled and the rest are zero.
        """
        B, M, L, C = out_shape
        out = packed.new_zeros(B, M, L, C)
        out[b_ids, m_ids, l_ids, :] = packed[b_ids, within, :]
        return out    

    
    def _prepare_queries(
        self,
        cls_mod: torch.Tensor,  # [B, M, CLS, C]  per-module CLS
    ):
        """
        Returns: queries        [B, M*CLS, C]
        """
        B, M, CLS, C = cls_mod.shape
        device  = cls_mod.device

        # Anchored CLS queries
        mod_ids   = torch.arange(M, device=device)
        queries  = cls_mod + self.module_embed_enc(mod_ids).view(1, M, 1, C)  # [B, M, CLS, C]
        queries = queries.view(B, M * CLS, C)                                 # [B, M*CLS, C]

        return queries

    
    def forward_encoder(self, x_sparse, x_glob, mask_ratio):
        """
        x_sparse:                   sparse input
        x_glob:       [B, C]        global context
        mask_ratio:   float         masking ratio
        """
        # patchify
        x_sparse = self.patch_embed(x_sparse)
        x, attn_mask, intra_idx = self.densify_patches(x_sparse)

        # add positional embeddings
        x = x + self.intra_pos_embed(intra_idx)

        # group to modules and mask
        x_mod, attn_mask_mod = self._group_tokens_by_module(x, attn_mask)
        x_keep, attn_mask_keep, rand_mask, ids_keep, _ = self._module_random_masking(x_mod, attn_mask_mod, mask_ratio)
        B, M, Lk, C = x_keep.shape

        # Intra-module transformer with per-module CLS
        cls = self.module_cls_token.expand(B*M, self.num_module_cls, C)
        x_intra = x_keep.reshape(B*M, Lk, C)
        x_intra = torch.cat([cls, x_intra], dim=1)
        attn_mask_intra = torch.cat(
            [torch.ones(B*M, self.num_module_cls, dtype=torch.bool, device=x_intra.device),
             attn_mask_keep.reshape(B*M, Lk)], dim=1)
        for blk in self.blocks:
            x_intra = blk(x_intra, attn_mask=attn_mask_intra)
        x_intra = self.norm(x_intra)

        # extract intra-module CLS and patch features
        cls_mod = x_intra[:, :self.num_module_cls, :].view(B, M, self.num_module_cls, C)  # [B, M, CLS, C]
        tok_mod = x_intra[:, self.num_module_cls:, :].reshape(B, M, Lk, C)                # [B, M, Lk, C]
        mod_ids = torch.arange(self.num_modules, device=tok_mod.device)
        tok_mod = tok_mod + self.module_embed_enc(mod_ids).view(1, M, 1, -1)

        # global context
        g_enc   = self.global_feats_encoder(x_glob)                                       # [B, C]
        g_tokens = (g_enc.unsqueeze(1) + self.global_mem).to(tok_mod.dtype)               # [B, G, C]

        # pack kept tokens for cross-attention
        kv_tokens, kv_keep, b_ids, m_ids, lk_ids, within, N_max = \
            self._pack_by_mask(tok_mod, attn_mask_keep)                                   # [B,N_max,C], [B,N_max]

        # append global tokens as real KV
        kv_tokens = torch.cat([kv_tokens, g_tokens], dim=1)                               # [B, Nk+G, C]
        kv_keep   = torch.cat([kv_keep, torch.ones(
            kv_keep.size(0), self.num_global_tokens,
            dtype=torch.bool, device=kv_keep.device)], dim=1)                             # [B, Nk+G]

        # prepare queries and masks
        queries = self._prepare_queries(cls_mod)                                          # [B, M*CLS, C]
        lat = queries
        for xa_lat, sa, xa_tok in zip(
            self.xattn_blocks["lat<-tok"],
            self.latent_self_blocks,
            self.xattn_blocks["tok<-lat"]
        ):
            lat = xa_lat(lat, kv_tokens, attn_mask=kv_keep)                               # cross: lat <- tokens
            lat = sa(lat, attn_mask=None)                                                 # self-attn over latents
            kv_tokens = xa_tok(kv_tokens, lat, attn_mask=None)                            # cross: tokens <- latents

        kv_tokens = self.tokens_norm(kv_tokens)                                           # [B, N_max + G, C]

        tok_enriched = self._unpack_to_modules(
            kv_tokens[:, :N_max, :],  # drop the appended G global rows
            b_ids, m_ids, lk_ids, within,
            out_shape=torch.Size([B, M, Lk, C])
        )

        return (
            tok_enriched, rand_mask, attn_mask_mod, attn_mask_keep, ids_keep
        )
        

    def _compute_within_ranks(self, b_ids: torch.Tensor, N: int) -> torch.Tensor:
        # b_ids must be nondecreasing (true for torch.nonzero over [B, ...]).
        if N == 0:
            return b_ids
        _, counts = torch.unique_consecutive(b_ids, return_counts=True)  # [G]
        starts = torch.cumsum(counts, dim=0) - counts                    # [G]
        return torch.arange(N, device=b_ids.device) - torch.repeat_interleave(starts, counts)


    def forward_relational(self, tok_keep_enriched, attn_mask_keep, ids_keep, idx_map):
        """
        tok_keep_enriched: [B, M, Lk, Cenc]  (enriched token embeddings)
        attn_mask_keep:    [B, M, Lk]        (bool)
        ids_keep:          [B, M, Lk]        (intra indices of gathered slots)
        idx_map:           LazyIdxMap
        """
        # real kept positions
        b_ids, m_ids, lk_ids = torch.nonzero(attn_mask_keep, as_tuple=True)  # [Nk]

        # gather enriched features
        out_flat = tok_keep_enriched[b_ids, m_ids, lk_ids, :]                # [Nk, Cenc]
        out_flat_dec = self.enc_to_dec(out_flat)

        # voxel head (+ heads)
        shared       = self.shared_voxel_head["rel"](out_flat_dec)           # [Nk, P, H]
        preds        = {}
        preds["gho"] = self.heads["gho"](shared).squeeze(-1)                 # [Nk, P]
        preds["hie"] = self.heads["hie"](shared)                             # [Nk, P, 3]
        preds["dec"] = self.heads["dec"](shared)                             # [Nk, P, 3]
        preds["pid"] = self.heads["pid"](shared)                             # [Nk, P, num_pid]

        # targets identical to your current logic
        l_intra   = ids_keep[b_ids, m_ids, lk_ids]                           # [Nk]
        patch_ids = self.module_token_indices[m_ids, l_intra]                # [Nk]
        idx_targets_kept = idx_map[b_ids, patch_ids]                         # [Nk, P]

        return preds, idx_targets_kept


    def forward_reconstruction(self, tok_enriched, attn_mask_mod, attn_mask_keep, rand_mask, idx_map):
        """
        tok_enriched:   [B, M, Lk, Cenc]  enriched kept tokens
        rand_mask:      [B, M, Lm]        True = masked real position (over full intra space)
        attn_mask_mod:  [B, M, Lm]        True = real
        attn_mask_keep: [B, M, Lm]        True = kept
        idx_map:        LazyIdxMap
        """
        B, M, Lk, Cenc = tok_enriched.shape
        Cdec = self.decoder_intra_pos_embed.weight.shape[-1]

        # which positions to predict: masked real
        prediction_mask = rand_mask & attn_mask_mod                          # [B, M, Lm]
        counts = prediction_mask.view(B, -1).sum(-1)
        max_n  = int(counts.max().item())

        # indices for masked positions
        b_ids, m_ids, l_ids = torch.nonzero(prediction_mask, as_tuple=True)  # [Nm]
        Nm = b_ids.numel()

        # build queries in decoder dim for these masked positions
        within = self._compute_within_ranks(b_ids, Nm)
        Q = tok_enriched.new_zeros(B, max_n, Cdec)
        q = ( self.decoder_intra_pos_embed(l_ids) +
            self.module_embed_dec(m_ids) +
            self.query_tokens )                                              # [Nm, Cdec]
        Q[b_ids, within] = q

        # KV: enriched tokens, packed (only real kept)
        kv_tokens, kv_keep, *_ = self._pack_by_mask(
            tok_enriched, attn_mask_keep
        )
        KV = self.enc_to_dec(kv_tokens)                                      # [B, Nk_max, Cdec]

        # Perceiver-IO decode: queries (masked positions) attend to enriched tokens
        X = Q
        for blk in self.decode_xattn_blocks:
            X = blk(X, KV, attn_mask=kv_keep)                                # respect KV mask

        out_flat = X[b_ids, within]                                          # [Nm, Cdec]

        # heads
        shared       = self.shared_voxel_head["rec"](out_flat)               # [Nm, P, H]
        preds        = {}
        preds["occ"] = self.heads["occ"](shared).squeeze(-1)                 # [Nm, P]
        preds["reg"] = self.heads["reg"](shared)                             # [Nm, P, in_chans]

        # targets
        patch_ids = self.module_token_indices[m_ids, l_ids]                  # [Nm]
        idx_targets = idx_map[b_ids, patch_ids]                              # [Nm, P]

        return preds, idx_targets, b_ids, patch_ids
    

    def forward(self, x, x_glob, mask_ratio=0.75):
        idx_map = self.build_patch_occupancy_map(x)

        # Relational path: no masking
        tok_enriched_rel, _, _, attn_mask_rel, ids_rel = \
            self.forward_encoder(x, x_glob, mask_ratio=0.0)
        preds_rel, rel_idx_targets = \
            self.forward_relational(tok_enriched_rel, attn_mask_rel, ids_rel, idx_map)

        # Reconstruction path: with masking
        tok_enriched_rec, rand_mask, attn_mask_rec, attn_keep_rec, _ = \
            self.forward_encoder(x, x_glob, mask_ratio)
        preds_rec, rec_idx_targets, row_event_ids, row_patch_ids = \
            self.forward_reconstruction(tok_enriched_rec, attn_mask_rec, attn_keep_rec, rand_mask, idx_map)

        preds = {**preds_rel, **preds_rec}
        return (preds, rel_idx_targets, rec_idx_targets, row_event_ids, row_patch_ids)



    def print_param_report(self):
        """
        Prints parameter counts by component:
        - patch_embed
        - intra_vit (pos-emb, module token/emb, blocks, norm)
        - perceiver_encoder (global encoder, global_mem, cross/self blocks, norm)
        - perceiver_decoder (dec pos-emb, module emb, query tokens, latents_to_dec, cross blocks)
        - heads_relational
        - heads_reconstruction
        - TOTAL
        """
        from collections import OrderedDict

        def _iter_params(obj):
            """Yield parameters from nn.Modules, nn.Parameters, and containers (list/tuple/dict/ModuleList/etc.)."""
            if obj is None:
                return
            if isinstance(obj, nn.Parameter):
                yield obj
                return
            if isinstance(obj, nn.Module):
                for p in obj.parameters(recurse=True):
                    yield p
                return
            # Common Python containers
            if isinstance(obj, (list, tuple, set)):
                for o in obj:
                    yield from _iter_params(o)
                return 
            if isinstance(obj, dict):
                for o in obj.values():
                    yield from _iter_params(o)
                return
            # Torch container types
            if isinstance(obj, (nn.ModuleList, nn.Sequential, nn.ParameterList, nn.ParameterDict)):
                for o in obj:
                    yield from _iter_params(o)
                return
            # Ignore everything else (buffers / plain tensors that aren't parameters)

        def _count(objs):
            # Collect and de-duplicate by id in case something shows up twice in a group
            ps = list(_iter_params(objs))
            seen = set()
            uniq = []
            for p in ps:
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    uniq.append(p)
            total = sum(p.numel() for p in uniq)
            train = sum(p.numel() for p in uniq if p.requires_grad)
            return total, train

        groups = OrderedDict()

        # --- Patch embed ---
        groups["patch_embed"] = [self.patch_embed] if hasattr(self, "patch_embed") else []

        # --- Intra ViT ---
        intra_list = []
        if hasattr(self, "intra_pos_embed"):   intra_list.append(self.intra_pos_embed)
        if hasattr(self, "module_cls_token"):  intra_list.append(self.module_cls_token)
        if hasattr(self, "module_embed_enc"):  intra_list.append(self.module_embed_enc)
        if hasattr(self, "blocks"):            intra_list.append(self.blocks)   # ModuleList (let Module handle recurse)
        if hasattr(self, "norm"):              intra_list.append(self.norm)
        groups["intra_vit"] = intra_list

        # --- Perceiver encoder ---
        enc_list = []
        if hasattr(self, "global_feats_encoder"): enc_list.append(self.global_feats_encoder)
        if hasattr(self, "global_mem"):           enc_list.append(self.global_mem)     # nn.Parameter
        if hasattr(self, "latent_xattn_blocks"):  enc_list.append(self.latent_xattn_blocks)  # ModuleList
        if hasattr(self, "latent_self_blocks"):   enc_list.append(self.latent_self_blocks)   # ModuleList
        if hasattr(self, "latent_norm"):          enc_list.append(self.latent_norm)
        groups["perceiver_encoder"] = enc_list

        # --- Perceiver decoders ---
        dec_list = []
        if hasattr(self, "decoder_intra_pos_embed"): dec_list.append(self.decoder_intra_pos_embed)
        if hasattr(self, "module_embed_dec"):        dec_list.append(self.module_embed_dec)
        if hasattr(self, "query_tokens"):            dec_list.append(self.query_tokens)
        if hasattr(self, "latents_to_dec"):          dec_list.append(self.latents_to_dec)
        if hasattr(self, "decode_xattn_blocks"):     dec_list.append(self.decode_xattn_blocks)
        groups["perceiver_decoder"] = dec_list

        # --- Heads: relational ---
        heads_rel = []
        if hasattr(self, "shared_voxel_head") and ("rel" in getattr(self, "shared_voxel_head")):
            heads_rel.append(self.shared_voxel_head["rel"])
        if hasattr(self, "track_head"):    heads_rel.append(self.track_head)
        if hasattr(self, "primary_head"):  heads_rel.append(self.primary_head)
        if hasattr(self, "pid_head"):      heads_rel.append(self.pid_head)
        groups["heads_relational"] = [m for m in heads_rel if m is not None]

        # --- Heads: reconstruction ---
        heads_rec = []
        if hasattr(self, "shared_voxel_head") and ("rec" in getattr(self, "shared_voxel_head")):
            heads_rec.append(self.shared_voxel_head["rec"])
        if hasattr(self, "occ_head"):      heads_rec.append(self.occ_head)
        if hasattr(self, "reg_head"):      heads_rec.append(self.reg_head)
        groups["heads_reconstruction"] = [m for m in heads_rec if m is not None]

        # Optional classifier (kept for finetune variants)
        if getattr(self, "classifier", None) is not None:
            groups["classifier"] = [self.classifier]

        grand_total = grand_train = 0
        print("=== Parameter report ===")
        for name, objs in groups.items():
            tot, trn = _count(objs)
            grand_total += tot
            grand_train += trn
            print(f"{name:24s} total={tot/1e6:8.3f}M  trainable={trn/1e6:8.3f}M")

        # Full model sanity check
        tot_all = sum(p.numel() for p in self.parameters())
        trn_all = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("-" * 64)
        print(f"{'ALL (sanity)':24s} total={tot_all/1e6:8.3f}M  trainable={trn_all/1e6:8.3f}M")
        print(f"{'SUM(groups)':24s} total={grand_total/1e6:8.3f}M  trainable={grand_train/1e6:8.3f}M")


def mae_vit_tiny(**kwargs):
    model = MinkMAEViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=528, patch_size=(12, 12, 10),
        depth=4, num_heads=12, num_global_tokens=1,
        io_depth=3, io_decode_depth=2, num_module_cls=1,
        num_modes=(32, 8), decoder_embed_dim=384, decoder_num_heads=12,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model


def mae_vit_base(**kwargs):
    model = MinkMAEViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(16, 16, 4),
        depth=4, num_heads=12, num_global_tokens=1,
        io_depth=8, io_decode_depth=4, num_module_cls=2,
        num_modes=(48, 8), decoder_embed_dim=528, decoder_num_heads=12,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model
    

def mae_vit_large(**kwargs):
    model = MinkMAEViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(16, 16, 4),
        depth=8, num_heads=12, num_global_tokens=2,
        io_depth=16, io_decode_depth=6, num_module_cls=4,
        num_modes=(64, 12), decoder_embed_dim=528, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model


def mae_vit_huge(**kwargs):
    model = MinkMAEViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(16, 16, 4),
        depth=16, num_heads=12, num_global_tokens=4,
        io_depth=32, io_decode_depth=8, num_module_cls=4,
        num_modes=(64, 16), decoder_embed_dim=528, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model
