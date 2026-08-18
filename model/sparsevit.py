"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.26

Description: PyTorch ViT model with spconv patching.
"""

import torch
import torch.nn as nn
import timm.models.vision_transformer as vit
from spconv.pytorch import SparseConv3d, SparseSequential
from functools import partial
from timm.layers import trunc_normal_
from .utils import (
    get_3d_sincos_pos_embed, choose_k1_k2, BlockWithMask, 
    CrossAttnBlock, CrossAttention, CylindricalHeadNormalized,
    make_parallel_then_merge_dpr, MUSPEC_FEATURE_DIM,
)


class SparseViT(vit.VisionTransformer):
    """ 
    Vision Transformer with spconv patching
    and support for global average pooling
    """
    def __init__(
        self,
        D=3,
        fcal_size=(48, 48, 200),
        module_depth_voxels=20,
        fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40),
        ahcal_patch_size=(6, 6, 5),
        ecal_size=(18, 18, 40),
        ecal_patch_size=(6, 6, 5),
        sparse_ecal=False,
        num_module_cls=1,
        num_ahcal_cls=2,
        io_depth=4,
        ahcal_depth=2,
        head_init=2e-5,
        global_pool=False,
        metadata=None,
        head_dropout_cls=0.0,
        head_dropout_reg=0.0,
        **kwargs
    ):
        super(SparseViT, self).__init__(**kwargs)
        
        self.metadata = metadata
        self.head_init = head_init
        self.sparse_ecal = sparse_ecal

        depth = kwargs['depth']
        num_heads = kwargs['num_heads']
        mlp_ratio = kwargs['mlp_ratio']
        attn_drop_rate = kwargs['attn_drop_rate']
        drop_path_rate = kwargs['drop_path_rate']
        norm_layer = kwargs['norm_layer']
        in_chans = kwargs['in_chans']
        drop_rate = kwargs['drop_rate']
        embed_dim = kwargs['embed_dim']

        # patch and grid setup
        H, W, D_img = fcal_size
        p_h, p_w, p_d = fcal_patch_size
        assert H % p_h == 0 and W % p_w == 0 and D_img % p_d == 0, \
            "fcal_size must be divisible by fcal_patch_size"
        self.grid_size = (H // p_h, W // p_w, D_img // p_d)
        self.num_patches = (self.grid_size[0] * self.grid_size[1] * self.grid_size[2])
        self.patch_voxels = p_h * p_w * p_d
        self.register_buffer('fcal_patch_size', torch.tensor(fcal_patch_size, dtype=torch.long))
        self.register_buffer('ahcal_patch_size', torch.tensor(ahcal_patch_size, dtype=torch.long))

        # module slicing along Z
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

        # ECAL grid bookkeeping (sparse ECAL only)
        if sparse_ecal:
            Eh, Ew, Ed = ecal_size
            ep_h, ep_w, ep_d = ecal_patch_size
            assert Eh % ep_h == 0 and Ew % ep_w == 0 and Ed % ep_d == 0
            self.ecal_grid_size = (Eh // ep_h, Ew // ep_w, Ed // ep_d)
            self.num_ecal_positions = (
                self.ecal_grid_size[0] * self.ecal_grid_size[1] * self.ecal_grid_size[2]
            )
            self.register_buffer('ecal_patch_size_buf', torch.tensor(ecal_patch_size, dtype=torch.long))

        # remove original ViT patch_embed / cls_token
        del self.cls_token, self.patch_embed, self.pos_embed, self.norm_pre, self.fc_norm, self.head

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

        # ECAL sparse patch embedding
        if sparse_ecal:
            if torch.tensor(ecal_patch_size).prod().item() > 512:
                mid = embed_dim // 4
                k1, k2 = choose_k1_k2(ecal_patch_size)
                self.ecal_patch_embed = SparseSequential(
                    SparseConv3d(in_chans, mid, kernel_size=k1, stride=k1, padding=0, bias=False),
                    norm_layer(mid),
                    nn.GELU(),
                    SparseConv3d(mid, embed_dim, kernel_size=k2, stride=k2, padding=0, bias=True)
                )
            else:
                self.ecal_patch_embed = SparseConv3d(
                    in_chans, embed_dim, kernel_size=ecal_patch_size, stride=ecal_patch_size,
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

        # drop path schedule
        dp_fas, dp_ah, dp_muon, dp_lat_x, dp_lat_s = make_parallel_then_merge_dpr(
            drop_path_rate, depth, ahcal_depth, io_depth,
            xattn_scale=0.5, include_muon=True, first_lat_xattn_zero=True
        )

        def _set_dp(m, p):
            for attr in ("drop_path", "drop_path1", "drop_path2"):
                if hasattr(m, attr):
                    getattr(m, attr).drop_prob = p
        for idx, blk in enumerate(self.blocks):
            _set_dp(blk, dp_fas[idx])
                
        self.num_module_cls = int(num_module_cls)
        self.t_patches_per_module_cls = max(1, self.num_intra_positions // max(1, self.num_module_cls))
        
        self.module_cls_token = nn.Parameter(torch.zeros(1, num_module_cls, embed_dim))  # per-module CLS (shared weights)
        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim)         # fixed sin-cos per-module
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)                # learned module index
        
        self.ahcal_pos_embed = nn.Embedding(self.num_ahcal_positions, embed_dim)         # fixed sin-cos per patch
        self.kv_src_embed = nn.Embedding(3, embed_dim)                                   # 0: AHCAL, 1: ECAL, 2: MUON_SPEC

        self.num_ahcal_cls = int(num_ahcal_cls)
        self.t_patches_per_ahcal_cls = max(1, self.num_ahcal_positions // max(1, self.num_ahcal_cls))
        self.ahcal_depth = int(ahcal_depth)
        self.ahcal_cls_token = nn.Parameter(torch.zeros(1, self.num_ahcal_cls, embed_dim))
        self.ahcal_blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dp_ah[i], norm_layer=norm_layer
            )
            for i in range(self.ahcal_depth)
        ])
        self.ahcal_norm = norm_layer(embed_dim)

        # ECAL: sparse pipeline or dense single-token
        if sparse_ecal:
            self.num_ecal_cls = int(num_ahcal_cls)  # same as AHCAL
            self.t_patches_per_ecal_cls = max(1, self.num_ecal_positions // max(1, self.num_ecal_cls))
            self.ecal_cls_token = nn.Parameter(torch.zeros(1, self.num_ecal_cls, embed_dim))
            self.ecal_pos_embed = nn.Embedding(self.num_ecal_positions, embed_dim)
            self.ecal_blocks = nn.ModuleList([
                BlockWithMask(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=True, proj_drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dp_ah[i], norm_layer=norm_layer
                )
                for i in range(self.ahcal_depth)
            ])
            self.ecal_norm = norm_layer(embed_dim)
        else:
            self.ecal_embed = nn.Linear(25, embed_dim)

        # Perceiver-IO bottleneck: lat <- tok + latent self
        self.muon_state_embed = nn.Embedding(2, embed_dim)  # 0=abstain (no tracks), 1=present
        self.muon_spec_count_encoder = nn.Linear(1, embed_dim)
        self.muon_spec_embed = nn.Linear(MUSPEC_FEATURE_DIM, embed_dim)
        self.muon_spec_xattn = CrossAttnBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dp_muon, norm_layer=norm_layer
        )
        self.lat_xattn_blocks = nn.ModuleList([
            CrossAttnBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dp_lat_x[i], norm_layer=norm_layer
            )
            for i in range(io_depth)
        ])
        self.latent_self_blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dp_lat_s[i], norm_layer=norm_layer
            )
            for i in range(io_depth)
        ])

        self.tokens_norm = norm_layer(embed_dim)

        # Task specifics
        self.global_pool = global_pool
        self.head_channels = {
            "flavour": 6,
            "charm": 4,
            "vis": 3,
            "jet": 3,
            "vertex": 3,
        }
        self.num_tasks = len(self.head_channels)

        if not self.global_pool:
            # Task tokens and cross-attention
            self.latents_norm = norm_layer(embed_dim)
            self.task_tokens = nn.Parameter(torch.zeros(1, self.num_tasks, embed_dim))
            self.task_cross_attn = CrossAttention(
                dim=embed_dim, num_heads=num_heads, qkv_bias=True,
                attn_drop=attn_drop_rate,
            )
            self.gamma = nn.Parameter(torch.ones(1) * 1e-4) # scaling after cross-attention

        # Allow separate dropout rates for classification/vertex and regression heads.
        cls_tasks = {"flavour", "charm", "vertex"}
        self.heads = nn.ModuleDict()
        for name in self.head_channels.keys():
            hd = head_dropout_cls if name in cls_tasks else head_dropout_reg
            self.heads[name] = nn.Sequential(
                norm_layer(embed_dim),
                nn.Dropout(hd),
                nn.Linear(embed_dim, self.head_channels[name]) 
                if name not in metadata 
                else self.make_head_from_stats(metadata[name], hidden=embed_dim)
            )
            
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

        if self.sparse_ecal:
            ecal_pos = get_3d_sincos_pos_embed(
                self.ecal_pos_embed.weight.shape[-1],
                self.ecal_grid_size,
                cls_token=False
            )
            with torch.no_grad():
                self.ecal_pos_embed.weight.copy_(torch.from_numpy(ecal_pos).float())
                self.ecal_pos_embed.weight.requires_grad_(False)

        # init tokens
        with torch.no_grad():
            nn.init.normal_(self.module_cls_token, std=0.02)
            nn.init.normal_(self.ahcal_cls_token, std=0.02)
            if self.sparse_ecal:
                nn.init.normal_(self.ecal_cls_token, std=0.02)
            nn.init.normal_(self.module_embed_enc.weight, std=0.02)
            nn.init.normal_(self.kv_src_embed.weight, std=0.02)
            nn.init.normal_(self.muon_state_embed.weight, std=0.02)
            if not self.global_pool:
                nn.init.normal_(self.task_tokens, std=0.02)

        self.apply(self._init_weights)

        # init heads
        for name, head in self.heads.items():
            lin = head[2]
            if name in self.metadata and hasattr(lin, "mlp"):
                lin = lin.mlp
            trunc_normal_(lin.weight, std=self.head_init)
            if lin.bias is not None:
                lin.bias.data.fill_(0)


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
        nwd = {
            'module_cls_token',
            'ahcal_cls_token',
            'module_embed_enc.weight',
            'kv_src_embed.weight',
            'task_tokens',
            'muon_state_embed.weight',
        }
        if self.sparse_ecal:
            nwd.add('ecal_cls_token')
        return nwd
            

    def make_head_from_stats(self, stats_entry, hidden=128):
        """
        stats_entry: a dict like stats['vis'] or stats['lep']
                    with keys: mu_logpT, sigma_logpT, mu_pz, sigma_pz, eps_T
        """
        return CylindricalHeadNormalized(
            k_T=stats_entry["k_T"],
            mu_uT=stats_entry["mu_uT"],
            sigma_uT=stats_entry["sigma_uT"],
            k_Z=stats_entry["k_Z"],
            mu_uZ=stats_entry["mu_uZ"],
            sigma_uZ=stats_entry["sigma_uZ"],
            hidden=hidden,
        )


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
    

    def _cls_keep_from_patchmask(
        self,
        patch_mask: torch.Tensor,
        num_cls: int,
        t: int,
        ensure_one_if_nonempty: bool = True,
        dummy_one_if_empty: bool = False,
    ) -> torch.Tensor:
        """
        Returns boolean CLS keep mask:
        - if patch_mask is [B, M, L] -> returns [B, M, CLS]
        - if patch_mask is [B, L]   -> returns [B, CLS]
        """
        num_cls = int(num_cls)
        if num_cls <= 0:
            # preserve rank: [B,0] or [B,M,0]
            if patch_mask.dim() == 2:
                return patch_mask.new_zeros(patch_mask.size(0), 0)
            elif patch_mask.dim() == 3:
                return patch_mask.new_zeros(patch_mask.size(0), patch_mask.size(1), 0)
            else:
                raise ValueError(f"patch_mask must be rank-2 or rank-3, got {patch_mask.shape}")

        t = max(1, int(t))
        v = patch_mask.sum(dim=-1).to(torch.long)  # [B,M] or [B]

        k = (v + (t - 1)) // t                     # ceil(v/t)
        k = torch.minimum(k, v)                    # never more CLS than patches
        k = k.clamp(min=0, max=num_cls)

        if ensure_one_if_nonempty:
            k = torch.where(v > 0, k.clamp_min(1), k)

        if dummy_one_if_empty:
            k = torch.where(v == 0, torch.ones_like(k), k)

        device = patch_mask.device
        if patch_mask.dim() == 3:
            # [B, M, CLS]
            slot = torch.arange(num_cls, device=device).view(1, 1, num_cls)
            return slot < k.unsqueeze(-1)
        else:
            # [B, CLS]
            slot = torch.arange(num_cls, device=device).view(1, num_cls)
            return slot < k.view(-1, 1)


    def _build_latent_keep(self, attn_mask_mod: torch.Tensor, ah_mask: torch.Tensor,
                           ecal_mask: torch.Tensor = None) -> torch.Tensor:
        B, M, _ = attn_mask_mod.shape

        fas_cls_keep = self._cls_keep_from_patchmask(
            attn_mask_mod,
            num_cls=self.num_module_cls,
            t=self.t_patches_per_module_cls,
            ensure_one_if_nonempty=True,
            dummy_one_if_empty=False,
        )  # [B, M, CLS]

        ah_cls_keep = self._cls_keep_from_patchmask(
            ah_mask,
            num_cls=self.num_ahcal_cls,
            t=self.t_patches_per_ahcal_cls,
            ensure_one_if_nonempty=True,
            dummy_one_if_empty=False,
        )  # [B, K]

        parts = [fas_cls_keep.reshape(B, M * int(self.num_module_cls)), ah_cls_keep]

        if self.sparse_ecal and ecal_mask is not None:
            ecal_cls_keep = self._cls_keep_from_patchmask(
                ecal_mask,
                num_cls=self.num_ecal_cls,
                t=self.t_patches_per_ecal_cls,
                ensure_one_if_nonempty=True,
                dummy_one_if_empty=False,
            )
            parts.append(ecal_cls_keep)

        lat_keep = torch.cat(parts, dim=1)

        empty_evt = (lat_keep.sum(dim=1) == 0)
        if empty_evt.any():
            lat_keep = lat_keep.clone()
            lat_keep[empty_evt, 0] = True

        return lat_keep
    

    def _prepare_latent_queries(
        self,
        cls_mod: torch.Tensor,   # [B, M, CLS, C]
        ah_cls: torch.Tensor,    # [B, K, C]
        ecal_cls: torch.Tensor = None,
    ):
        B, M, CLS, C = cls_mod.shape
        device = cls_mod.device

        mod_ids = torch.arange(M, device=device)
        q_fas = cls_mod + self.module_embed_enc(mod_ids).view(1, M, 1, C)
        q_fas = q_fas.view(B, M * CLS, C)

        q_ah = ah_cls + self.kv_src_embed.weight[0].view(1, 1, C)

        parts = [q_fas, q_ah]
        if self.sparse_ecal and ecal_cls is not None:
            q_ecal = ecal_cls + self.kv_src_embed.weight[1].view(1, 1, C)
            parts.append(q_ecal)

        return torch.cat(parts, dim=1)

    
    def forward_features(self, x_sparse, x_glob):
        # retrieve global features
        ahcal_sparse, ecal_hits, muspec_feats, muspec_attn_mask, muspec_counts = x_glob

        # FASERCAL patchify + intra-attn
        fcal_batch_ids = x_sparse.indices[:, 0].long()
        fcal_hit_counts = torch.bincount(fcal_batch_ids, minlength=x_sparse.batch_size)
        fcal_degenerate = (fcal_hit_counts < 2)
        x_sparse_emb = self.fcal_patch_embed(x_sparse)
        x, attn_mask, intra_idx = self.densify_patches(x_sparse_emb)
        if fcal_degenerate.any():
            attn_mask = attn_mask.clone()
            attn_mask[fcal_degenerate] = False
        x = x + self.intra_pos_embed(intra_idx)

        x_mod, attn_mask_mod = self._group_tokens_by_module(x, attn_mask)
        B, M, L, C = x_mod.shape

        cls = self.module_cls_token.expand(B*M, self.num_module_cls, C)
        x_intra = x_mod.reshape(B*M, L, C)
        x_intra = torch.cat([cls, x_intra], dim=1)
        x_intra = self.pos_drop(x_intra)
        
        CLS = int(self.num_module_cls)
        cls_keep_mask_flat = self._cls_keep_from_patchmask(
            attn_mask_mod,
            num_cls=CLS,
            t=self.t_patches_per_module_cls,
            ensure_one_if_nonempty=True,
            dummy_one_if_empty=True,
        ).reshape(B * M, CLS)                                                              # [B*M, CLS]
        
        attn_mask_intra = torch.cat(
            [cls_keep_mask_flat, attn_mask_mod.reshape(B*M, L)], dim=1)
        for blk in self.blocks:
            x_intra = blk(x_intra, attn_mask=attn_mask_intra, q_mask=attn_mask_intra)
        x_intra = self.norm(x_intra)

        # extract intra-module CLS and patch features
        cls_mod = x_intra[:, :self.num_module_cls, :].view(B, M, self.num_module_cls, C)    # [B, M, C]
        tok_mod = x_intra[:, self.num_module_cls:, :].reshape(B, M, L, C)                   # [B, M, L, C]
        mod_ids = torch.arange(self.num_modules, device=tok_mod.device)
        tok_mod = tok_mod + self.module_embed_enc(mod_ids).view(1, M, 1, -1)

        # ahcal embedding
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

        # self-attn on AHCAL kept tokens with K CLS
        K = self.num_ahcal_cls
        cls_ah = self.ahcal_cls_token.expand(B, K, C)                                       # [B, K, C]
        x_ah = torch.cat([cls_ah, ah_tokens], dim=1)                                        # [B, K+Na, C]
        
        ah_cls_keep = self._cls_keep_from_patchmask(
            ah_mask,
            num_cls=K,
            t=self.t_patches_per_ahcal_cls,
            ensure_one_if_nonempty=True,
            dummy_one_if_empty=True,
        )                                                                                   # [B, K]
        
        ah_attn_intra = torch.cat([ah_cls_keep, ah_mask], dim=1)
        for blk in self.ahcal_blocks:
            x_ah = blk(x_ah, attn_mask=ah_attn_intra, q_mask=ah_attn_intra)
        x_ah = self.ahcal_norm(x_ah)

        ah_cls = x_ah[:, :K, :]                                                             # [B, K, C]
        tok_ah = x_ah[:, K:, :]                                                             # [B, Na, C]

        # ECAL: sparse pipeline or dense single-token
        ecal_cls = None
        ecal_mask = None
        ecal_tok = None
        tok_ecal = None
        if self.sparse_ecal:
            ecal_sparse = ecal_hits
            ecal_batch_ids = ecal_sparse.indices[:, 0].long()
            ecal_hit_counts = torch.bincount(ecal_batch_ids, minlength=B)
            ecal_degenerate = (ecal_hit_counts < 2)

            ecal_sparse = self.ecal_patch_embed(ecal_sparse)
            ec_tokens, ecal_mask, ec_idx = self.densify_patches(ecal_sparse, is_fasercal=False)
            ec_tokens = ec_tokens + self.ecal_pos_embed(ec_idx) \
                + self.kv_src_embed.weight[1].view(1, 1, -1)
            if ecal_degenerate.any():
                ecal_mask = ecal_mask.clone()
                ecal_mask[ecal_degenerate] = False

            K_ec = self.num_ecal_cls
            cls_ec = self.ecal_cls_token.expand(B, K_ec, C)
            x_ec = torch.cat([cls_ec, ec_tokens], dim=1)
            ec_cls_keep = self._cls_keep_from_patchmask(
                ecal_mask, num_cls=K_ec, t=self.t_patches_per_ecal_cls,
                ensure_one_if_nonempty=True, dummy_one_if_empty=True,
            )
            ec_attn_intra = torch.cat([ec_cls_keep, ecal_mask], dim=1)
            for blk in self.ecal_blocks:
                x_ec = blk(x_ec, attn_mask=ec_attn_intra, q_mask=ec_attn_intra)
            x_ec = self.ecal_norm(x_ec)
            ecal_cls = x_ec[:, :K_ec, :]
            tok_ecal = x_ec[:, K_ec:, :]
        else:
            ecal_tok = self.ecal_embed(ecal_hits.view(B, -1)).unsqueeze(1)
            ecal_tok = ecal_tok + self.kv_src_embed.weight[1].view(1, 1, -1)

        # muon spectrometer token
        muspec_count_emb = self.muon_spec_count_encoder(muspec_counts).unsqueeze(1)         # [B, 1, C]
        muon_spec_emb = self.muon_spec_embed(muspec_feats)                                  # [B, N_muspec, C]
        has_tracks = muspec_attn_mask.any(dim=1, keepdim=True)                              # [B, 1] bool
        safe_mask = muspec_attn_mask.clone()
        safe_mask[~has_tracks.squeeze(-1), 0] = True
        muon_tok_present = self.muon_spec_xattn(
            muspec_count_emb, muon_spec_emb, attn_mask=safe_mask
        )                                                                                   # [B,1,C]
        muon_tok = torch.where(has_tracks.view(B,1,1), muon_tok_present, muspec_count_emb)
        muon_state = has_tracks.long().squeeze(-1)  # [B]
        muon_tok = muon_tok + self.muon_state_embed(muon_state).unsqueeze(1)                # 0=abstain, 1=present
        muon_tok = muon_tok + self.kv_src_embed.weight[2].view(1, 1, -1)                    # tag as MUON_SPEC

        # build KV for lat <- tok
        kv_fas, kv_fas_keep, *_ = self._pack_by_mask(tok_mod, attn_mask_mod)
        kv_parts = [kv_fas, tok_ah]
        kv_keep_parts = [kv_fas_keep, ah_mask]

        if self.sparse_ecal:
            kv_parts.append(tok_ecal)
            kv_keep_parts.append(ecal_mask)
        else:
            kv_parts.append(ecal_tok)
            kv_keep_parts.append(torch.ones(B, 1, dtype=torch.bool, device=kv_fas_keep.device))

        kv_parts.append(muon_tok)
        kv_keep_parts.append(torch.ones(B, 1, dtype=torch.bool, device=kv_fas_keep.device))

        kv_tokens = torch.cat(kv_parts, dim=1)
        kv_keep   = torch.cat(kv_keep_parts, dim=1)
        kv_tokens = self.tokens_norm(kv_tokens)

        # latents: FASER CLS + AHCAL CLS + (optionally) ECAL CLS
        lat = self._prepare_latent_queries(cls_mod, ah_cls, ecal_cls)
        lat_keep = self._build_latent_keep(attn_mask_mod=attn_mask_mod, ah_mask=ah_mask, ecal_mask=ecal_mask)

        # Perceiver encoder loop (NO tok<-lat)
        for xa_lat, sa in zip(self.lat_xattn_blocks, self.latent_self_blocks):
            lat = xa_lat(lat, kv_tokens, attn_mask=kv_keep, q_mask=lat_keep)                # lat <- tokens
            lat = sa(lat, attn_mask=lat_keep, q_mask=lat_keep)                              # latent self-attn

        # Task-specific heads
        if self.global_pool:
            den = lat_keep.sum(dim=1, keepdim=True).clamp_min(1).type_as(lat)               # [B, 1]
            outcome = (lat * lat_keep.unsqueeze(-1).type_as(lat)).sum(dim=1) / den          # [B, C]
        else:
            task_q  = self.task_tokens.expand(B, -1, -1)                                    # [B, T, C]
            lat_kv = self.latents_norm(lat)                                                 # [B, N_lat, C]
            outcome = task_q + self.task_cross_attn(task_q, lat_kv, attn_mask=lat_keep) * self.gamma

        return outcome
    

    def forward_head(self, x):
        outputs = {}
        for i, (name, head) in enumerate(self.heads.items()):
            if self.global_pool:
                output = head(x)   
            else:
                output = head(x[:, i, :])
            outputs[f"out_{name}"] = output

        return outputs
        
        
    def forward(self, x, x_glob):
        x = self.forward_features(x, x_glob)
        x = self.forward_head(x)
        return x


def vit_tiny(**kwargs):
    head_dropout_cls = kwargs.pop('head_dropout_cls', 0.0)
    head_dropout_reg = kwargs.pop('head_dropout_reg', 0.0)
    model = SparseViT(
        in_chans=1, D=3, embed_dim=384,
        fcal_size=(48, 48, 200), fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40), ahcal_patch_size=(6, 6, 5),
        depth=2, ahcal_depth=2, num_heads=12, io_depth=6,
        num_module_cls=2, num_ahcal_cls=2,
        mlp_ratio=4.0, global_pool=True,
        head_dropout_cls=head_dropout_cls,
        head_dropout_reg=head_dropout_reg,
        block_fn=BlockWithMask,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(**kwargs):
    head_dropout_cls = kwargs.pop('head_dropout_cls', 0.0)
    head_dropout_reg = kwargs.pop('head_dropout_reg', 0.0)
    model = SparseViT(
        in_chans=1, D=3, embed_dim=384, 
        fcal_size=(48, 48, 200), fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40), ahcal_patch_size=(6, 6, 5),
        depth=4, ahcal_depth=4, num_heads=12, io_depth=4,
        num_module_cls=2, num_ahcal_cls=2,
        mlp_ratio=4.0, global_pool=True,
        head_dropout_cls=head_dropout_cls,
        head_dropout_reg=head_dropout_reg,
        block_fn=BlockWithMask,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
