"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.26

Description: PyTorch MAE-ViT model with spconv patching.
"""

import math
import torch
import torch.nn as nn
from spconv.pytorch import SparseConv3d, SparseSequential
from functools import partial
from .utils import (
    get_3d_sincos_pos_embed, choose_k1_k2, BlockWithMask, 
    CrossAttnBlock, SeparableDCT3D, SharedLatentVoxelHead, LazyIdxMap
)


class SparseMAEViT(nn.Module):
    def __init__(
        self,
        in_chans=1,
        fcal_size=(48, 48, 200),
        module_depth_voxels=20,
        embed_dim=384,
        fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40),
        ahcal_patch_size=(6, 6, 5),
        num_module_cls=1,
        num_ahcal_cls=2,
        depth=4,
        ahcal_depth=2,
        io_depth=3,
        io_decode_depth=2,
        num_heads=12,
        num_modes=(8, 4),
        num_pid_classes=3,
        decoder_embed_dim=256,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        drop_rate_dec=0.,
        attn_drop_rate_dec=0.,
        norm_layer=nn.LayerNorm,
        metadata=None,
    ):
        super().__init__()

        self.metadata = metadata
    
        # patch and grid setup
        H, W, D_img = fcal_size
        p_h, p_w, p_d = fcal_patch_size
        assert H % p_h == 0 and W % p_w == 0 and D_img % p_d == 0, \
            "fcal_size must be divisible by fcal_patch_size"
        self.in_chans = in_chans
        self.grid_size = (H // p_h, W // p_w, D_img // p_d)
        self.num_patches = (self.grid_size[0] * self.grid_size[1] * self.grid_size[2])
        self.patch_voxels = p_h * p_w * p_d
        self.register_buffer('fcal_patch_size', torch.tensor(fcal_patch_size, dtype=torch.long))
        self.register_buffer('ahcal_patch_size', torch.tensor(ahcal_patch_size, dtype=torch.long))

        # FASERCAL module slicing along Z
        assert module_depth_voxels % p_d == 0, "module_depth_voxels must be divisible by patch depth"
        self.module_depth_voxels = module_depth_voxels
        self.module_depth_patches = module_depth_voxels // p_d
        G_h, G_w, G_d = self.grid_size
        assert G_d % self.module_depth_patches == 0, "grid depth must be multiple of module depth (in patches)"
        self.num_modules = G_d // self.module_depth_patches
        self.intra_grid_size = (G_h, G_w, self.module_depth_patches)     # H×W×(depth within module)
        self.num_intra_positions = G_h * G_w * self.module_depth_patches

        # AHCAL grid bookkeeping (post-embedding grid)
        Ah, Aw, Ad = ahcal_size
        ap_h, ap_w, ap_d = self.ahcal_patch_size.tolist()
        assert Ah % ap_h == 0 and Aw % ap_w == 0 and Ad % ap_d == 0, \
            "AHCAL grid must be divisible by ahcal_patch_size"
        self.ahcal_grid_size = (Ah // ap_h, Aw // ap_w, Ad // ap_d)  # e.g., (2,2,4)
        self.num_ahcal_positions = (
            self.ahcal_grid_size[0] * self.ahcal_grid_size[1] * self.ahcal_grid_size[2]
        )

        # sparse patch embedding
        if self.fcal_patch_size.prod().item() > 512:
            # too large -> use two-step conv
            mid = embed_dim // 4
            k1, k2 = choose_k1_k2(fcal_patch_size)
            self.fcal_patch_embed = SparseSequential(
                SparseConv3d(in_chans, mid, kernel_size=k1, stride=k1, padding=0, bias=False),
                norm_layer(mid),
                nn.GELU(),
                SparseConv3d(mid, embed_dim, kernel_size=k2, stride=k2, padding=0, bias=True)
            )
        else:
            self.fcal_patch_embed = SparseConv3d(
                in_chans, embed_dim, kernel_size=fcal_patch_size, stride=fcal_patch_size, 
                padding=0, bias=True,
            )

        if self.ahcal_patch_size.prod().item() > 512:
            # too large -> use two-step conv
            mid = embed_dim // 4
            k1, k2 = choose_k1_k2(ahcal_patch_size)
            self.ahcal_patch_embed = SparseSequential(
                SparseConv3d(in_chans, mid, kernel_size=k1, stride=k1, padding=0, bias=False),
                norm_layer(mid),
                nn.GELU(),
                SparseConv3d(mid, embed_dim, kernel_size=k2, stride=k2, padding=0, bias=True)
            )
        else:
            self.ahcal_patch_embed = SparseConv3d(
                in_chans, embed_dim, kernel_size=ahcal_patch_size, stride=ahcal_patch_size, 
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

        # =========================
        # Encoder: hierarchical ViT
        # =========================
        self.intra_depth = depth
        self.num_module_cls = num_module_cls

        self.module_cls_token = nn.Parameter(torch.zeros(1, num_module_cls, embed_dim))  # per-module CLS (shared weights)
        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim)         # fixed sin-cos per patch
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)                # learned module index for intra-attn

        self.ahcal_pos_embed = nn.Embedding(self.num_ahcal_positions, embed_dim)         # fixed sin-cos per patch
        self.kv_src_embed = nn.Embedding(3, embed_dim)                                   # 0: AHCAL, 1: ECAL, 2: MUON_SPEC

        # FASERCAL intra-module transformer blocks
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

        # AHCAL short self-attention + K CLS
        self.num_ahcal_cls = int(num_ahcal_cls)
        self.ahcal_depth = int(ahcal_depth)
        self.ahcal_cls_token = nn.Parameter(torch.zeros(1, self.num_ahcal_cls, embed_dim))
        dpr_ahcal = [x.item() for x in torch.linspace(0, drop_path_rate, self.ahcal_depth)]
        self.ahcal_blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_ahcal[i], norm_layer=norm_layer
            )
            for i in range(self.ahcal_depth)
        ])
        self.ahcal_norm = norm_layer(embed_dim)

        # ==========================
        # Perceiver-IO bottleneck (encoder side): lat <- tok + latent self
        # ==========================
        self.ecal_embed = nn.Linear(25, embed_dim)
        self.muon_spec_count_encoder = nn.Linear(1, embed_dim)
        self.muon_spec_embed = nn.Linear(5, embed_dim)
        self.muon_spec_xattn = CrossAttnBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=0., norm_layer=norm_layer
        )
        self.lat_xattn_blocks = nn.ModuleList([
            CrossAttnBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=0., norm_layer=norm_layer
            )
            for _ in range(io_depth)
        ])
        self.latent_self_blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=0., norm_layer=norm_layer
            )
            for _ in range(io_depth)
        ])
        self.tokens_norm = norm_layer(embed_dim)

        # ==========================
        # Perceiver-IO decoder (reconstruction)
        # ==========================
        assert decoder_embed_dim % decoder_num_heads == 0, "decoder_embed_dim must be divisible by decoder_num_heads"
        self.decoder_intra_pos_embed = nn.Embedding(self.num_intra_positions, decoder_embed_dim)  # frozen sin-cos
        self.module_embed_dec = nn.Embedding(self.num_modules, decoder_embed_dim)
        self.query_tokens = nn.Parameter(torch.zeros(1, decoder_embed_dim))

        self.enc_to_dec = nn.Linear(embed_dim, decoder_embed_dim)
        
        # masked queries attend to LATENTS
        self.decode_lat_xattn_blocks = nn.ModuleList([
            CrossAttnBlock(
                dim=decoder_embed_dim, num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=True, drop=drop_rate_dec,
                attn_drop=attn_drop_rate_dec, drop_path=0., norm_layer=norm_layer
            )
            for _ in range(io_decode_depth)
        ])

        self.decoder_ahcal_pos_embed = nn.Embedding(self.num_ahcal_positions, decoder_embed_dim)
        self.ahcal_query_tokens = nn.Parameter(torch.zeros(1, decoder_embed_dim))

        # Heads
        self.fasercal_sep_basis = SeparableDCT3D(
            self.fcal_patch_size.tolist(), alphas=(0.4, 0.4, 0.6)
        )
        self.fasercal_shared_voxel_head = SharedLatentVoxelHead(
            decoder_embed_dim, self.fasercal_sep_basis, H=num_modes[0],
            norm_layer=norm_layer, post_norm=True
        )
        self.ahcal_sep_basis = SeparableDCT3D(
            self.ahcal_patch_size.tolist(), alphas=(0.4, 0.4, 0.6)
        )
        self.ahcal_voxel_head = SharedLatentVoxelHead(
            decoder_embed_dim, self.ahcal_sep_basis, H=num_modes[1],
            norm_layer=norm_layer, post_norm=True,
        )
        self.head_channels = {
            "gho": 1,
            "hie": 3,
            "dec": 3,
            "pid": num_pid_classes,
            "occ": 1,
            "reg": in_chans,
            "occ_ahcal": 1,
            "reg_ahcal": in_chans,
        }
        
        self.heads = nn.ModuleDict()
        for name, out_channels in self.head_channels.items():
            in_dim = num_modes[1] if name in ["occ_ahcal", "reg_ahcal"] else num_modes[0]
            self.heads[name] = nn.Linear(in_dim, out_channels)

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

        ahcal_pos = get_3d_sincos_pos_embed(
            self.ahcal_pos_embed.weight.shape[-1],
            self.ahcal_grid_size,
            cls_token=False
        )
        with torch.no_grad():
            self.ahcal_pos_embed.weight.copy_(torch.from_numpy(ahcal_pos).float())
            self.ahcal_pos_embed.weight.requires_grad_(False)

        dec_pos = get_3d_sincos_pos_embed(
            self.decoder_intra_pos_embed.weight.shape[-1],
            self.intra_grid_size,
            cls_token=False
        )
        with torch.no_grad():
            self.decoder_intra_pos_embed.weight.copy_(torch.from_numpy(dec_pos).float())
            self.decoder_intra_pos_embed.weight.requires_grad_(False)

        dec_ahcal_pos = get_3d_sincos_pos_embed(
            self.decoder_ahcal_pos_embed.weight.shape[-1],
            self.ahcal_grid_size,
            cls_token=False
        )
        with torch.no_grad():
            self.decoder_ahcal_pos_embed.weight.copy_(torch.from_numpy(dec_ahcal_pos).float())
            self.decoder_ahcal_pos_embed.weight.requires_grad_(False)

        # init tokens
        with torch.no_grad():
            nn.init.normal_(self.module_cls_token, std=.02)
            nn.init.normal_(self.ahcal_cls_token, std=.02)
            nn.init.normal_(self.module_embed_enc.weight, std=0.02)
            nn.init.normal_(self.kv_src_embed.weight, std=0.02)
            nn.init.normal_(self.module_embed_dec.weight, std=0.02)
            nn.init.normal_(self.query_tokens, std=0.02)
            nn.init.normal_(self.ahcal_query_tokens, std=0.02)

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
            'ahcal_cls_token',
            'module_embed_enc.weight',
            'kv_src_embed.weight',
            'module_embed_dec.weight',
            'query_tokens',
            'ahcal_query_tokens',
        }
    

    def densify_patches(self, x_sp, is_fasercal: bool = True):
        """
        Convert a SparseConvTensor to dense tokens + occupancy mask + positional indices.

        Args:
            x_sp: SparseConvTensor after patch embedding.
            is_fasercal: 
                True  -> uses self.intra_idx_template (module-aware intra indexing)
                False -> uses a simple [0..Np-1] indexing (generic/AHCAL-style)

        Returns:
            dense_tokens: [B, Np, C]
            attn_mask:    [B, Np]    (True where a patch token is real/non-empty)
            intra_idx:    [B, Np]    indices for positional embedding lookup
        """
        B = x_sp.batch_size
        C = x_sp.features.size(1)
        X, Y, Z = x_sp.spatial_shape
        Np = X * Y * Z

        # Dense: [B, C, X, Y, Z] -> tokens [B, Np, C] with flatten order matching occ.view(B, -1)
        x_dense = x_sp.dense()  # [B, C, X, Y, Z]
        dense_tokens = x_dense.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C)

        # Occupancy mask from sparse indices
        idx = x_sp.indices.long()  # [N, 4] = [b, x, y, z]
        b, h, w, d = idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]
        occ = torch.zeros((B, X, Y, Z), dtype=torch.bool, device=x_sp.features.device)
        occ[b, h, w, d] = True
        attn_mask = occ.view(B, -1)  # [B, Np]

        # Positional indices
        if is_fasercal:
            # Must match the*main grid's intra indexing
            assert Np == self.intra_idx_template.numel(), (
                f"FASERCal densify expected Np={self.intra_idx_template.numel()} but got {Np}. "
                "Call with is_fasercal=False for non-main grids."
            )
            intra_idx = self.intra_idx_template.unsqueeze(0).expand(B, -1)
        else:
            intra_idx = torch.arange(Np, device=x_sp.features.device, dtype=torch.long).view(1, -1).expand(B, -1)

        return dense_tokens, attn_mask, intra_idx


    def build_patch_occupancy_map(self, x, patch_size, grid_size):
        """
        From the original sparse tensor coordinates, build a [B, N_patches, P] mapping
        with the raw id of the actual hit in that sub‐voxel.
        """
        idx = x.indices.long()             # [N, 4] = [b, x, y, z] == [b, h, w, d]
        b, h, w, d = idx.unbind(-1)

        p_h, p_w, p_d = patch_size.tolist() if torch.is_tensor(patch_size) else patch_size
        G_h, G_w, G_d = grid_size
        P             = p_h * p_w * p_d    # voxels per patch
        Np            = G_h * G_w * G_d    # patches per event

        patch_idx = (h // p_h) * (G_w * G_d) + (w // p_w) * G_d + (d // p_d)
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


    def _prepare_latent_queries(
        self,
        cls_mod: torch.Tensor,   # [B, M, CLS, C]
        ah_cls: torch.Tensor,    # [B, K, C]
    ):
        B, M, CLS, C = cls_mod.shape
        device = cls_mod.device

        # FASERCal CLS queries anchored by module id
        mod_ids = torch.arange(M, device=device)
        q_fas = cls_mod + self.module_embed_enc(mod_ids).view(1, M, 1, C)   # [B, M, CLS,C]
        q_fas = q_fas.view(B, M * CLS, C)                                   # [B, M*CLS, C]

        # AHCAL CLS queries tagged as AHCAL (type embedding)
        q_ah = ah_cls + self.kv_src_embed.weight[0].view(1, 1, C)           # [B, K, C]

        return torch.cat([q_fas, q_ah], dim=1)                              # [B, M*CLS+K, C]


    def forward_encoder(self, x_sparse, x_glob, mask_ratio):
        """
        x_sparse:                   sparse input
        x_glob:                     global context
        mask_ratio:   float         masking ratio
        """
        # retrieve global features
        ahcal_sparse, ecal_hits, muspec_feats, muspec_attn_mask, muspec_counts = x_glob

        # FASERCAL patchify + mask + intra-attn
        x_sparse_emb = self.fcal_patch_embed(x_sparse)
        x, attn_mask, intra_idx = self.densify_patches(x_sparse_emb, is_fasercal=True)
        x = x + self.intra_pos_embed(intra_idx)

        x_mod, attn_mask_mod = self._group_tokens_by_module(x, attn_mask)
        x_keep, attn_mask_keep, rand_mask, ids_keep, _ = self._module_random_masking(
            x_mod, attn_mask_mod, mask_ratio
        )
        B, M, Lk, C = x_keep.shape

        cls = self.module_cls_token.expand(B*M, self.num_module_cls, C)
        x_intra = x_keep.reshape(B*M, Lk, C)
        x_intra = torch.cat([cls, x_intra], dim=1)
        attn_mask_intra = torch.cat(
            [torch.ones(B*M, self.num_module_cls, dtype=torch.bool, device=x_intra.device),
             attn_mask_keep.reshape(B*M, Lk)], dim=1)
        for blk in self.blocks:
            x_intra = blk(x_intra, attn_mask=attn_mask_intra)
        x_intra = self.norm(x_intra)

        cls_mod = x_intra[:, :self.num_module_cls, :].view(B, M, self.num_module_cls, C)    # [B, M, CLS, C]
        tok_mod = x_intra[:, self.num_module_cls:, :].reshape(B, M, Lk, C)                  # [B, M, Lk, C]
        mod_ids = torch.arange(self.num_modules, device=tok_mod.device)
        tok_mod = tok_mod + self.module_embed_enc(mod_ids).view(1, M, 1, C)

        # AHCAL patchify + mask + self-attn
        # Count actual input voxels per batch element (before patch embedding)
        ahcal_batch_ids = ahcal_sparse.indices[:, 0].long()                                # [N_hits]
        ahcal_hit_counts = torch.bincount(ahcal_batch_ids, minlength=B)                    # [B]
        degenerate = (ahcal_hit_counts < 2)                                                # True if 0 or 1 voxel (likely dummy)
        
        ahcal_sparse = self.ahcal_patch_embed(ahcal_sparse)
        ah_tokens, ah_mask, ah_idx = self.densify_patches(ahcal_sparse, is_fasercal=False)  # [B, Na, C], [B, Na], [B, Na]
        ah_tokens = ah_tokens + self.ahcal_pos_embed(ah_idx) \
            + self.kv_src_embed.weight[0].view(1, 1, -1)                                    # tag as AHCAL
        if degenerate.any():
            ah_mask = ah_mask.clone()
            ah_mask[degenerate] = False

        ah_tokens_mod = ah_tokens.unsqueeze(1)                                              # [B, 1, Na, C]
        ah_mask_mod   = ah_mask.unsqueeze(1)                                                # [B, 1, Na]
        ah_keep_mod, ah_attn_keep_mod, ah_rand_mask_mod, ah_ids_keep_mod, _ = \
            self._module_random_masking(ah_tokens_mod, ah_mask_mod, mask_ratio
        )
        tok_ah_keep = ah_keep_mod.squeeze(1)                                                # [B, Lk_ah, C]
        ah_attn_keep = ah_attn_keep_mod.squeeze(1)                                          # [B, Lk_ah]
        ah_rand_mask = ah_rand_mask_mod.squeeze(1)                                          # [B, Na]

        # self-attn on AHCAL kept tokens with K CLS
        K = self.num_ahcal_cls
        cls_ah = self.ahcal_cls_token.expand(B, K, C)                                       # [B, K, C]
        x_ah = torch.cat([cls_ah, tok_ah_keep], dim=1)                                      # [B, K+Lk_ah, C]
        ah_attn_intra = torch.cat(
            [torch.ones(B, K, dtype=torch.bool, device=x_ah.device),
                ah_attn_keep], dim=1
        )
        for blk in self.ahcal_blocks:
            x_ah = blk(x_ah, attn_mask=ah_attn_intra)
        x_ah = self.ahcal_norm(x_ah)

        ah_cls = x_ah[:, :K, :]                                                             # [B, K, C]
        tok_ah_keep = x_ah[:, K:, :]                                                        # [B, Lk_ah, C]

        # ECAL token
        ecal_tok = self.ecal_embed(ecal_hits.view(B, -1)).unsqueeze(1)                      # [B, 1, C]
        ecal_tok = ecal_tok + self.kv_src_embed.weight[1].view(1, 1, -1)                    # tag as ECAL

        # muon spectrometer token
        muspec_count_emb = self.muon_spec_count_encoder(muspec_counts).unsqueeze(1)         # [B, 1, C]
        muon_spec_emb = self.muon_spec_embed(muspec_feats)                                  # [B, N_muspec, C]
        has_tracks = (muspec_counts > 0)                                                    # [B, 1] bool
        safe_mask = muspec_attn_mask.clone()
        safe_mask[~has_tracks.squeeze(-1), 0] = True
        muon_tok = self.muon_spec_xattn(
            muspec_count_emb,
            muon_spec_emb,
            attn_mask=safe_mask
        )
        muon_tok = muon_tok * has_tracks.view(B, 1, 1).float()                              # [B, 1, C]
        muon_tok = muon_tok + self.kv_src_embed.weight[2].view(1, 1, -1)                    # tag as MUON_SPEC

        # randomly drop global tokens (consistent with masking policy)
        keep_ecal = (torch.rand(B, device=ecal_tok.device) > mask_ratio)                    # [B]
        keep_muon = (torch.rand(B, device=muon_tok.device) > mask_ratio)                    # [B]
        ecal_kv_mask = keep_ecal.view(B, 1)                                                 # [B, 1]
        muon_kv_mask = keep_muon.view(B, 1)                                                 # [B, 1]

        # build KV for lat <- tok
        kv_fas, kv_fas_keep, *_ = self._pack_by_mask(tok_mod, attn_mask_keep)               # [B, Nmax, C], [B, Nmax]
        kv_tokens = torch.cat([kv_fas, tok_ah_keep, ecal_tok, muon_tok], dim=1)
        kv_keep   = torch.cat([kv_fas_keep, ah_attn_keep, ecal_kv_mask, muon_kv_mask], dim=1)
        kv_tokens = self.tokens_norm(kv_tokens)

        # latents: FASER CLS + AHCAL CLS
        lat = self._prepare_latent_queries(cls_mod, ah_cls)                                 # [B, N_lat, C]

        # Perceiver-IO encoder loop
        for xa_lat, sa in zip(self.lat_xattn_blocks, self.latent_self_blocks):
            lat = xa_lat(lat, kv_tokens, attn_mask=kv_keep)                                 # lat <- tokens
            lat = sa(lat, attn_mask=None)                                                   # latent self-attn

        return (
            lat,                   # [B, N_lat, C]
            rand_mask,             # [B, M, Lm]     FASERcal masked real positions
            attn_mask_mod,         # [B, M, Lm]     FASERcal real positions
            ah_mask,               # [B, Na]        AHCAL real positions
            ah_rand_mask,          # [B, Na]        AHCAL masked real positions
        )


    def _compute_within_ranks(self, b_ids: torch.Tensor, N: int) -> torch.Tensor:
        # b_ids must be nondecreasing (true for torch.nonzero over [B, ...]).
        if N == 0:
            return b_ids
        _, counts = torch.unique_consecutive(b_ids, return_counts=True)  # [G]
        starts = torch.cumsum(counts, dim=0) - counts                    # [G]
        return torch.arange(N, device=b_ids.device) - torch.repeat_interleave(starts, counts)


    def forward_reconstruction_fcal(
        self,
        lat: torch.Tensor,            # [B, N_lat, Cenc]
        attn_mask_mod: torch.Tensor,  # [B, M, Lm]
        rand_mask: torch.Tensor,      # [B, M, Lm]
        idx_map,
    ):
        """
        FASERCal reconstruction from bottleneck latents only.
        """
        B = lat.size(0)
        Cdec = self.decoder_intra_pos_embed.weight.shape[-1]

        # which FASERCAL patches to predict: masked real
        prediction_mask = rand_mask & attn_mask_mod                            # [B, M, Lm]
        counts = prediction_mask.view(B, -1).sum(-1)
        max_n  = int(counts.max().item())

        # indices for masked positions
        b_ids, m_ids, l_ids = torch.nonzero(prediction_mask, as_tuple=True)    # [Nm]
        Nm = b_ids.numel()

        # build queries in decoder dim for these masked positions
        within = self._compute_within_ranks(b_ids, Nm)
        Q = self.query_tokens.new_zeros(B, max_n, Cdec)

        q = ( self.decoder_intra_pos_embed(l_ids)
            + self.module_embed_dec(m_ids)
            + self.query_tokens )                                              # [Nm, Cdec]
        Q[b_ids, within] = q

        # query -> latents
        LAT = self.enc_to_dec(lat)                                             # [B, N_lat, Cdec] (no mask needed)
        X = Q
        for blk in self.decode_lat_xattn_blocks:
            X = blk(X, LAT, attn_mask=None)

        out_flat = X[b_ids, within]                                            # [Nm, Cdec]

        # Heads
        preds = {}
        shared = self.fasercal_shared_voxel_head(out_flat)                     # [Nm, P, H]
        preds["occ"] = self.heads["occ"](shared).squeeze(-1)                   # [Nm, P]
        preds["reg"] = self.heads["reg"](shared)                               # [Nm, P, in_chans]
        preds["gho"] = self.heads["gho"](shared).squeeze(-1)                   # [Nm, P]
        preds["hie"] = self.heads["hie"](shared)                               # [Nm, P, 3]
        preds["dec"] = self.heads["dec"](shared)                               # [Nm, P, 3]
        preds["pid"] = self.heads["pid"](shared)                               # [Nm, P, num_pid]

        # targets (same for all predictions on masked patches)
        patch_ids = self.module_token_indices[m_ids, l_ids]                    # [Nm]
        idx_targets = idx_map[b_ids, patch_ids]                                # [Nm, P]

        return preds, idx_targets
    

    def forward_reconstruction_ahcal(
        self,
        lat: torch.Tensor,            # [B, N_lat, Cenc]
        ah_mask: torch.Tensor,        # [B, Na]
        ah_rand_mask: torch.Tensor,   # [B, Na]
        ah_idx_map,
    ):
        """
        AHCAL reconstruction from bottleneck latents only.
        """
        B = lat.size(0)
        Cdec = self.decoder_ahcal_pos_embed.weight.shape[-1]

        # which AHCAL patches to predict: masked real
        prediction_mask = ah_rand_mask & ah_mask                             # [B, Na]
        counts = prediction_mask.sum(dim=-1)
        max_n = int(counts.max().item()) if B > 0 else 0

        # indices for masked AHCAL patches
        b_ids, l_ids = torch.nonzero(prediction_mask, as_tuple=True)         # [Nm]
        Nm = b_ids.numel()

        # build queries in decoder dim for these masked AHCAL positions
        within = self._compute_within_ranks(b_ids, Nm)                       # [Nm]
        Q = self.ahcal_query_tokens.new_zeros(B, max_n, Cdec)                # [B, max_n, Cdec]
        q = self.decoder_ahcal_pos_embed(l_ids) + self.ahcal_query_tokens    # [Nm, Cdec]
        Q[b_ids, within] = q

        # query -> latents
        LAT = self.enc_to_dec(lat)  # [B,N_lat,Cdec]
        X = Q
        for blk in self.decode_lat_xattn_blocks:
            X = blk(X, LAT, attn_mask=None)

        out_flat = X[b_ids, within]                                          # [Nm, Cdec]

        # Heads
        shared_ah       = self.ahcal_voxel_head(out_flat)                    # [Nm, P_ah, H]
        preds_ah        = {}
        preds_ah["occ_ah"] = self.heads["occ_ahcal"](shared_ah).squeeze(-1)  # [Nm, P_ah]
        preds_ah["reg_ah"] = self.heads["reg_ahcal"](shared_ah)              # [Nm, P_ah, in_chans]

        # targets: patch id is just l_ids (no modules here)
        patch_ids_ah = l_ids                                                 # [Nm]
        idx_targets_ah = ah_idx_map[b_ids, patch_ids_ah]                     # [Nm, P_ah]

        return preds_ah, idx_targets_ah
    

    def forward(self, x, x_glob, mask_ratio=0.75):
        """
        Single encoder pass with masking (mask_ratio), then:
          - FASERCAL reconstruction + semantic segmentation on masked patches,
          - AHCAL reconstruction on masked patches.
        """
        # occupancy maps
        idx_map = self.build_patch_occupancy_map(x, self.fcal_patch_size, self.grid_size)
        ahcal_sparse, *_ = x_glob
        ah_idx_map = self.build_patch_occupancy_map(ahcal_sparse, self.ahcal_patch_size, self.ahcal_grid_size)

        # single encoder pass
        (
            lat,
            rand_mask,
            attn_mask_mod,
            ah_mask,
            ah_rand_mask,
        ) = self.forward_encoder(x, x_glob, mask_ratio)

        # FASERCAL: both reconstruction and semantic segmentation on masked patches
        preds_fas, idx_targets_fas = self.forward_reconstruction_fcal(
            lat=lat,
            attn_mask_mod=attn_mask_mod,
            rand_mask=rand_mask,
            idx_map=idx_map,
        )

        # AHCAL reconstruction on masked patches
        preds_ah, idx_targets_ah = self.forward_reconstruction_ahcal(
            lat=lat,
            ah_mask=ah_mask,
            ah_rand_mask=ah_rand_mask,
            ah_idx_map=ah_idx_map,
        )

        # merge all predictions
        preds = {**preds_fas, **preds_ah}

        return (
            preds,
            idx_targets_fas,        # FASERcal idx_targets (for all tasks: occ, reg, gho, hie, dec, pid)
            idx_targets_ah,         # AHCAL idx_targets (for occ_ahcal, reg_ahcal)
        )


def mae_vit_tiny(**kwargs):
    model = SparseMAEViT(
        in_chans=1, embed_dim=384, 
        fcal_size=(48, 48, 200), fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40), ahcal_patch_size=(6, 6, 5),
        depth=4, ahcal_depth=2, num_heads=12, io_depth=3, io_decode_depth=2, 
        num_module_cls=1, num_ahcal_cls=2,
        num_modes=(8, 4), decoder_embed_dim=256, decoder_num_heads=8,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model

def mae_vit_base(**kwargs):
    model = SparseMAEViT(
        in_chans=1, embed_dim=528, 
        fcal_size=(48, 48, 200), fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40), ahcal_patch_size=(6, 6, 5),
        depth=4, ahcal_depth=2, num_heads=12, io_depth=3, io_decode_depth=2, 
        num_module_cls=1, num_ahcal_cls=2,
        num_modes=(8, 4), decoder_embed_dim=384, decoder_num_heads=12,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model

