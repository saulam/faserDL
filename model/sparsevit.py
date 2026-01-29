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
    make_parallel_then_merge_dpr,
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
        num_module_cls=1,
        num_ahcal_cls=2,
        io_depth=4,
        ahcal_depth=2,
        head_init=2e-5,
        global_pool=False,
        metadata=None,
        **kwargs
    ):
        super(SparseViT, self).__init__(**kwargs)
        
        self.metadata = metadata
        self.head_init = head_init

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
                
        self.num_module_cls = num_module_cls
        
        self.module_cls_token = nn.Parameter(torch.zeros(1, num_module_cls, embed_dim))  # per-module CLS (shared weights)
        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim)         # fixed sin-cos per-module
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)                # learned module index
        
        self.ahcal_pos_embed = nn.Embedding(self.num_ahcal_positions, embed_dim)         # fixed sin-cos per patch
        self.kv_src_embed = nn.Embedding(3, embed_dim)                                   # 0: AHCAL, 1: ECAL, 2: MUON_SPEC

        self.num_ahcal_cls = int(num_ahcal_cls)
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

        # Perceiver-IO bottleneck: lat <- tok + latent self
        self.ecal_embed = nn.Linear(25, embed_dim)
        self.muon_state_embed = nn.Embedding(2, embed_dim)  # 0=abstain (no tracks), 1=present
        self.muon_spec_count_encoder = nn.Linear(1, embed_dim)
        self.muon_spec_embed = nn.Linear(5, embed_dim)
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

        self.heads = nn.ModuleDict()
        for name in self.head_channels.keys():
            self.heads[name] = nn.Sequential(
                norm_layer(embed_dim),
                nn.Dropout(drop_rate),
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

        # init tokens
        with torch.no_grad():
            nn.init.normal_(self.module_cls_token, std=0.02)
            nn.init.normal_(self.ahcal_cls_token, std=0.02)
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
        return {
            'module_cls_token',
            'ahcal_cls_token',
            'module_embed_enc.weight',
            'kv_src_embed.weight',
            'task_tokens',
            'muon_state_embed.weight',
        }
            

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
    

    def _build_latent_keep(self, attn_mask_mod: torch.Tensor, ah_mask: torch.Tensor) -> torch.Tensor:
        """
        Build a [B, N_lat] boolean mask for Perceiver latents:
        - FASER latents valid iff that module has >=1 real patch (attn_mask_mod any True)
        - AHCAL latents valid iff AHCAL has >=1 real patch (ah_mask any True)

        Also ensures at least one latent is kept per batch element (numerical safety).
        """
        # attn_mask_mod: [B, M, Lm]
        # ah_mask:       [B, Na]
        B, M, _ = attn_mask_mod.shape
        device = attn_mask_mod.device

        module_valid = attn_mask_mod.any(dim=-1)  # [B, M]
        lat_fas_keep = module_valid.repeat_interleave(self.num_module_cls, dim=1)  # [B, M*CLS]

        ah_valid = ah_mask.any(dim=-1)  # [B]
        lat_ah_keep = ah_valid.view(B, 1).expand(B, self.num_ahcal_cls)  # [B, K]

        lat_keep = torch.cat([lat_fas_keep, lat_ah_keep], dim=1).to(device=device)  # [B, N_lat]

        # If an event is totally empty everywhere, keep one dummy latent to avoid "all keys masked" SDPA edge cases.
        empty_evt = (lat_keep.sum(dim=1) == 0)
        if empty_evt.any():
            lat_keep = lat_keep.clone()
            lat_keep[empty_evt, 0] = True

        return lat_keep
    

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

    
    def forward_features(self, x_sparse, x_glob):
        # retrieve global features
        ahcal_sparse, ecal_hits, muspec_feats, muspec_attn_mask, muspec_counts = x_glob

        # FASERCAL patchify + intra-attn
        x_sparse_emb = self.fcal_patch_embed(x_sparse)
        x, attn_mask, intra_idx = self.densify_patches(x_sparse_emb)
        x = x + self.intra_pos_embed(intra_idx)

        x_mod, attn_mask_mod = self._group_tokens_by_module(x, attn_mask)
        B, M, L, C = x_mod.shape

        cls = self.module_cls_token.expand(B*M, self.num_module_cls, C)
        x_intra = x_mod.reshape(B*M, L, C)
        x_intra = torch.cat([cls, x_intra], dim=1)
        x_intra = self.pos_drop(x_intra)
        attn_mask_intra = torch.cat(
            [torch.ones(B*M, self.num_module_cls, dtype=torch.bool, device=x_intra.device),
             attn_mask_mod.reshape(B*M, L)], dim=1)
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
        ah_attn_intra = torch.cat(
            [torch.ones(B, K, dtype=torch.bool, device=x_ah.device),
                ah_mask], dim=1
        )
        for blk in self.ahcal_blocks:
            x_ah = blk(x_ah, attn_mask=ah_attn_intra, q_mask=ah_attn_intra)
        x_ah = self.ahcal_norm(x_ah)

        ah_cls = x_ah[:, :K, :]                                                             # [B, K, C]
        tok_ah = x_ah[:, K:, :]                                                             # [B, Na, C]

        # ecal token
        ecal_tok = self.ecal_embed(ecal_hits.view(B, -1)).unsqueeze(1)                      # [B, 1, C]
        ecal_tok = ecal_tok + self.kv_src_embed.weight[1].view(1, 1, -1)                    # tag as ECAL

        # muon spectrometer token
        muspec_count_emb = self.muon_spec_count_encoder(muspec_counts).unsqueeze(1)         # [B, 1, C]
        muon_spec_emb = self.muon_spec_embed(muspec_feats)                                  # [B, N_muspec, C]
        has_tracks = (muspec_counts > 0)                                                    # [B, 1] bool
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
        kv_fas, kv_fas_keep, *_ = self._pack_by_mask(tok_mod, attn_mask_mod)                # [B, N_max, C], [B, N_max]
        kv_tokens = torch.cat([kv_fas, tok_ah, ecal_tok, muon_tok], dim=1)                  # [B, Nmax + Na + 2, C]
        kv_keep   = torch.cat([
            kv_fas_keep, ah_mask, torch.ones(B, 2, dtype=torch.bool, device=kv_fas_keep.device)
        ], dim=1)                                                                           # [B, Nmax + Na + 2]
        kv_tokens = self.tokens_norm(kv_tokens)

        # latents: FASER CLS + AHCAL CLS
        lat = self._prepare_latent_queries(cls_mod, ah_cls)                                 # [B, N_lat, C]
        lat_keep = self._build_latent_keep(attn_mask_mod=attn_mask_mod, ah_mask=ah_mask)    # [B, N_lat]

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
    model = SparseViT(
        in_chans=1, D=3, embed_dim=384,
        fcal_size=(48, 48, 200), fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40), ahcal_patch_size=(6, 6, 5),
        depth=2, ahcal_depth=2, num_heads=12, io_depth=6,
        num_module_cls=2, num_ahcal_cls=4,
        mlp_ratio=4.0, global_pool=True,
        block_fn=BlockWithMask,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(**kwargs):
    model = SparseViT(
        in_chans=1, D=3, embed_dim=384, 
        fcal_size=(48, 48, 200), fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40), ahcal_patch_size=(6, 6, 5),
        depth=4, ahcal_depth=4, num_heads=12, io_depth=4,
        num_module_cls=2, num_ahcal_cls=4,
        mlp_ratio=4.0, global_pool=True,
        block_fn=BlockWithMask,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
