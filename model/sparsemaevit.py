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
    CrossAttnBlock, MultiRankSeparableBasis3D, MultiRankSharedLatentVoxelHead, LazyIdxMap,
    muon_summary_target, make_parallel_then_merge_dpr
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
        ecal_size=(18, 18, 40),
        ecal_patch_size=(6, 6, 5),
        sparse_ecal=False,
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
        self.sparse_ecal = sparse_ecal
    
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

        # ECAL grid bookkeeping (sparse mode only)
        if self.sparse_ecal:
            self.register_buffer('ecal_patch_size', torch.tensor(ecal_patch_size, dtype=torch.long))
            Eh, Ew, Ed = ecal_size
            ep_h, ep_w, ep_d = ecal_patch_size
            assert Eh % ep_h == 0 and Ew % ep_w == 0 and Ed % ep_d == 0, \
                "ECAL grid must be divisible by ecal_patch_size"
            self.ecal_grid_size = (Eh // ep_h, Ew // ep_w, Ed // ep_d)
            self.num_ecal_positions = (
                self.ecal_grid_size[0] * self.ecal_grid_size[1] * self.ecal_grid_size[2]
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

        # ECAL sparse patch embedding (mirrors AHCAL)
        if self.sparse_ecal:
            if self.ecal_patch_size.prod().item() > 512:
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

        # =========================
        # Encoder: hierarchical ViT
        # =========================
        self.intra_depth = depth
        self.num_module_cls = int(num_module_cls)
        self.t_patches_per_module_cls = max(1, self.num_intra_positions // max(1, self.num_module_cls))

        self.module_cls_token = nn.Parameter(torch.zeros(1, self.num_module_cls, embed_dim))  # per-module CLS (shared weights)
        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim)              # fixed sin-cos per patch
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)                     # learned module index for intra-attn

        self.ahcal_pos_embed = nn.Embedding(self.num_ahcal_positions, embed_dim)              # fixed sin-cos per patch
        self.kv_src_embed = nn.Embedding(3, embed_dim)                                        # 0: AHCAL, 1: ECAL, 2: MUON_SPEC

        # drop path schedule
        dp_fas, dp_ah, dp_muon, dp_lat_x, dp_lat_s = make_parallel_then_merge_dpr(
            drop_path_rate, depth, ahcal_depth, io_depth,
            xattn_scale=0.5, include_muon=True, first_lat_xattn_zero=True
        )

        # FASERCAL intra-module transformer blocks
        self.blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, proj_drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dp_fas[i], norm_layer=norm_layer
            )
            for i in range(self.intra_depth)
        ])
        self.norm = norm_layer(embed_dim)

        # AHCAL short self-attention + K CLS
        self.ahcal_depth = int(ahcal_depth)
        self.num_ahcal_cls = int(num_ahcal_cls)
        self.t_patches_per_ahcal_cls = max(1, self.num_ahcal_positions // max(1, self.num_ahcal_cls))

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

        # ECAL sparse self-attention (mirrors AHCAL) or dense embedding
        if self.sparse_ecal:
            self.num_ecal_cls = int(num_ahcal_cls)
            self.t_patches_per_ecal_cls = max(1, self.num_ecal_positions // max(1, self.num_ecal_cls))
            self.ecal_pos_embed = nn.Embedding(self.num_ecal_positions, embed_dim)
            self.ecal_cls_token = nn.Parameter(torch.zeros(1, self.num_ecal_cls, embed_dim))
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

        # ==========================
        # Perceiver-IO bottleneck (encoder side): lat <- tok + latent self
        # ==========================
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

        # ECAL sparse decoder components
        if self.sparse_ecal:
            self.decoder_ecal_pos_embed = nn.Embedding(self.num_ecal_positions, decoder_embed_dim)
            self.ecal_query_tokens = nn.Parameter(torch.zeros(1, decoder_embed_dim))

        # ==========================
        # Heads
        # ==========================
        self.fasercal_sep_basis = MultiRankSeparableBasis3D(
            self.fcal_patch_size.tolist(), alphas=(0.4, 0.4, 0.6), R=2,
        )
        self.fasercal_shared_voxel_head = MultiRankSharedLatentVoxelHead(
            decoder_embed_dim, self.fasercal_sep_basis, H=num_modes[0],
            norm_layer=norm_layer, post_norm=True
        )
        self.ahcal_sep_basis = MultiRankSeparableBasis3D(
            self.ahcal_patch_size.tolist(), alphas=(0.4, 0.4, 0.6), R=2,
        )
        self.ahcal_voxel_head = MultiRankSharedLatentVoxelHead(
            decoder_embed_dim, self.ahcal_sep_basis, H=num_modes[1],
            norm_layer=norm_layer, post_norm=True,
        )
        self.head_channels = {
            "gho": 1,
            "hie": 3,
            "pid": num_pid_classes,
            "occ": 1,
            "reg": in_chans,
            "occ_ahcal": 1,
            "reg_ahcal": in_chans,
        }
        if self.sparse_ecal:
            self.ecal_sep_basis = MultiRankSeparableBasis3D(
                self.ecal_patch_size.tolist(), alphas=(0.4, 0.4, 0.6), R=2,
            )
            self.ecal_voxel_head = MultiRankSharedLatentVoxelHead(
                decoder_embed_dim, self.ecal_sep_basis, H=num_modes[1],
                norm_layer=norm_layer, post_norm=True,
            )
            self.head_channels["occ_ecal"] = 1
            self.head_channels["reg_ecal"] = in_chans
        
        self.heads = nn.ModuleDict()
        for name, out_channels in self.head_channels.items():
            in_dim = num_modes[1] if name in ["occ_ahcal", "reg_ahcal", "occ_ecal", "reg_ecal"] else num_modes[0]
            self.heads[name] = nn.Linear(in_dim, out_channels)

        # ==========================
        # Global token reconstruction heads
        # ==========================
        if self.sparse_ecal:
            self.global_query = nn.Parameter(torch.zeros(1, decoder_embed_dim))  # MUON only
        else:
            self.global_query = nn.Parameter(torch.zeros(2, decoder_embed_dim))  # 0=ECAL, 1=MUON
            self.ecal_recon_head = nn.Linear(decoder_embed_dim, 25)
        self.muon_recon_head = nn.Linear(decoder_embed_dim, 5)

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

        # ECAL sparse positional embeddings
        if self.sparse_ecal:
            ecal_pos = get_3d_sincos_pos_embed(
                self.ecal_pos_embed.weight.shape[-1],
                self.ecal_grid_size,
                cls_token=False
            )
            with torch.no_grad():
                self.ecal_pos_embed.weight.copy_(torch.from_numpy(ecal_pos).float())
                self.ecal_pos_embed.weight.requires_grad_(False)

            dec_ecal_pos = get_3d_sincos_pos_embed(
                self.decoder_ecal_pos_embed.weight.shape[-1],
                self.ecal_grid_size,
                cls_token=False
            )
            with torch.no_grad():
                self.decoder_ecal_pos_embed.weight.copy_(torch.from_numpy(dec_ecal_pos).float())
                self.decoder_ecal_pos_embed.weight.requires_grad_(False)

        # init tokens
        with torch.no_grad():
            nn.init.normal_(self.module_cls_token, std=.02)
            nn.init.normal_(self.ahcal_cls_token, std=.02)
            nn.init.normal_(self.module_embed_enc.weight, std=0.02)
            nn.init.normal_(self.kv_src_embed.weight, std=0.02)
            nn.init.normal_(self.module_embed_dec.weight, std=0.02)
            nn.init.normal_(self.query_tokens, std=0.02)
            nn.init.normal_(self.ahcal_query_tokens, std=0.02)
            nn.init.normal_(self.muon_state_embed.weight, std=0.02)
            nn.init.normal_(self.global_query, std=0.02)
            if self.sparse_ecal:
                nn.init.normal_(self.ecal_cls_token, std=.02)
                nn.init.normal_(self.ecal_query_tokens, std=0.02)

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
        nwd = {
            'module_cls_token',
            'ahcal_cls_token',
            'module_embed_enc.weight',
            'kv_src_embed.weight',
            'module_embed_dec.weight',
            'query_tokens',
            'ahcal_query_tokens',
            'muon_state_embed.weight',
            'global_query',
        }
        if self.sparse_ecal:
            nwd.update({'ecal_cls_token', 'ecal_query_tokens'})
        return nwd
    

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

    
    def _cls_keep_from_patchmask(
        self,
        patch_mask: torch.Tensor,   # [B,M,L] or [B,L]
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
        """
        Build a [B, N_lat] boolean mask for Perceiver latents with occupancy-based CLS gating.

        - For each module: activate k = ceil(v / t_mod) CLS slots, where v=#real patches in that module.
        - For AHCAL:       activate k = ceil(v / t_ah) CLS slots, where v=#real patches in AHCAL.
        - For ECAL (sparse): activate k = ceil(v / t_ecal) CLS slots.
        """
        # attn_mask_mod: [B, M, Lm] bool
        # ah_mask:       [B, Na] bool
        B, M, _ = attn_mask_mod.shape

        fas_cls_keep = self._cls_keep_from_patchmask(
            attn_mask_mod,
            num_cls=self.num_module_cls,
            t=self.t_patches_per_module_cls,
            ensure_one_if_nonempty=True,
            dummy_one_if_empty=False,   # IMPORTANT: no dummy latents here
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
            )  # [B, K_ecal]
            parts.append(ecal_cls_keep)

        lat_keep = torch.cat(parts, dim=1)

        # global numerical safety: if totally empty everywhere, keep one dummy latent
        empty_evt = (lat_keep.sum(dim=1) == 0)
        if empty_evt.any():
            lat_keep = lat_keep.clone()
            lat_keep[empty_evt, 0] = True

        return lat_keep


    def _prepare_latent_queries(
        self,
        cls_mod: torch.Tensor,   # [B, M, CLS, C]
        ah_cls: torch.Tensor,    # [B, K, C]
        ecal_cls: torch.Tensor = None,  # [B, K_ecal, C] (sparse ECAL only)
    ):
        B, M, CLS, C = cls_mod.shape
        device = cls_mod.device

        # FASERCal CLS queries anchored by module id
        mod_ids = torch.arange(M, device=device)
        q_fas = cls_mod + self.module_embed_enc(mod_ids).view(1, M, 1, C)   # [B, M, CLS,C]
        q_fas = q_fas.view(B, M * CLS, C)                                   # [B, M*CLS, C]

        # AHCAL CLS queries tagged as AHCAL (type embedding)
        q_ah = ah_cls + self.kv_src_embed.weight[0].view(1, 1, C)           # [B, K, C]

        parts = [q_fas, q_ah]
        if self.sparse_ecal and ecal_cls is not None:
            q_ecal = ecal_cls + self.kv_src_embed.weight[1].view(1, 1, C)   # [B, K_ecal, C]
            parts.append(q_ecal)

        return torch.cat(parts, dim=1)


    def forward_encoder(self, x_sparse, x_glob, mask_ratio):
        """
        x_sparse:                   sparse input
        x_glob:                     global context
        mask_ratio:   float         masking ratio
        """
        # retrieve global features
        ahcal_sparse, ecal_hits, muspec_feats, muspec_attn_mask, muspec_counts = x_glob

        # FASERCAL patchify + mask + intra-attn
        fcal_batch_ids = x_sparse.indices[:, 0].long()
        fcal_hit_counts = torch.bincount(fcal_batch_ids, minlength=x_sparse.batch_size)
        fcal_degenerate = (fcal_hit_counts < 2)
        x_sparse_emb = self.fcal_patch_embed(x_sparse)
        x, attn_mask, intra_idx = self.densify_patches(x_sparse_emb, is_fasercal=True)
        if fcal_degenerate.any():
            attn_mask = attn_mask.clone()
            attn_mask[fcal_degenerate] = False
        x = x + self.intra_pos_embed(intra_idx)

        x_mod, attn_mask_mod = self._group_tokens_by_module(x, attn_mask)
        x_keep, attn_mask_keep, rand_mask, ids_keep, _ = self._module_random_masking(
            x_mod, attn_mask_mod, mask_ratio
        )
        B, M, Lk, C = x_keep.shape

        cls = self.module_cls_token.expand(B*M, self.num_module_cls, C)
        x_intra = x_keep.reshape(B*M, Lk, C)
        x_intra = torch.cat([cls, x_intra], dim=1)

        CLS = int(self.num_module_cls)
        cls_keep_mask_flat = self._cls_keep_from_patchmask(
            attn_mask_mod,
            num_cls=CLS,
            t=self.t_patches_per_module_cls,
            ensure_one_if_nonempty=True,
            dummy_one_if_empty=True,
        ).reshape(B * M, CLS)                                                              # [B*M, CLS]

        attn_mask_intra = torch.cat(
            [cls_keep_mask_flat, attn_mask_keep.reshape(B * M, Lk)], dim=1
        )
        for blk in self.blocks:
            x_intra = blk(x_intra, attn_mask=attn_mask_intra, q_mask=attn_mask_intra)
        x_intra = self.norm(x_intra)

        cls_mod = x_intra[:, :self.num_module_cls, :].view(B, M, self.num_module_cls, C)   # [B, M, CLS, C]
        tok_mod = x_intra[:, self.num_module_cls:, :].reshape(B, M, Lk, C)                 # [B, M, Lk, C]
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

        ah_cls_keep = self._cls_keep_from_patchmask(
            ah_mask,
            num_cls=K,
            t=self.t_patches_per_ahcal_cls,
            ensure_one_if_nonempty=True,
            dummy_one_if_empty=True,
        )                                                                                   # [B, K]

        ah_attn_intra = torch.cat([ah_cls_keep, ah_attn_keep], dim=1)
        for blk in self.ahcal_blocks:
            x_ah = blk(x_ah, attn_mask=ah_attn_intra, q_mask=ah_attn_intra)
        x_ah = self.ahcal_norm(x_ah)

        ah_cls = x_ah[:, :K, :]                                                             # [B, K, C]
        tok_ah_keep = x_ah[:, K:, :]                                                        # [B, Lk_ah, C]

        # ECAL: sparse pipeline (mirrors AHCAL) or dense single-token
        ecal_cls = None
        ecal_mask = None
        ecal_rand_mask = None
        ecal_tok = None
        ecal_attn_keep = None
        tok_ecal_keep = None
        if self.sparse_ecal:
            ecal_sparse = ecal_hits  # this is a SparseConvTensor when sparse_ecal=True
            ecal_batch_ids = ecal_sparse.indices[:, 0].long()
            ecal_hit_counts = torch.bincount(ecal_batch_ids, minlength=B)
            ecal_degenerate = (ecal_hit_counts < 2)

            ecal_sparse = self.ecal_patch_embed(ecal_sparse)
            ec_tokens, ecal_mask, ec_idx = self.densify_patches(ecal_sparse, is_fasercal=False)
            ec_tokens = ec_tokens + self.ecal_pos_embed(ec_idx) \
                + self.kv_src_embed.weight[1].view(1, 1, -1)                                # tag as ECAL
            if ecal_degenerate.any():
                ecal_mask = ecal_mask.clone()
                ecal_mask[ecal_degenerate] = False

            ec_tokens_mod = ec_tokens.unsqueeze(1)
            ec_mask_mod = ecal_mask.unsqueeze(1)
            ec_keep_mod, ec_attn_keep_mod, ec_rand_mask_mod, _, _ = \
                self._module_random_masking(ec_tokens_mod, ec_mask_mod, mask_ratio)
            tok_ecal_keep = ec_keep_mod.squeeze(1)
            ecal_attn_keep = ec_attn_keep_mod.squeeze(1)
            ecal_rand_mask = ec_rand_mask_mod.squeeze(1)

            # self-attn on ECAL kept tokens with CLS
            K_ec = self.num_ecal_cls
            cls_ec = self.ecal_cls_token.expand(B, K_ec, C)
            x_ec = torch.cat([cls_ec, tok_ecal_keep], dim=1)

            ec_cls_keep = self._cls_keep_from_patchmask(
                ecal_mask, num_cls=K_ec, t=self.t_patches_per_ecal_cls,
                ensure_one_if_nonempty=True, dummy_one_if_empty=True,
            )
            ec_attn_intra = torch.cat([ec_cls_keep, ecal_attn_keep], dim=1)
            for blk in self.ecal_blocks:
                x_ec = blk(x_ec, attn_mask=ec_attn_intra, q_mask=ec_attn_intra)
            x_ec = self.ecal_norm(x_ec)

            ecal_cls = x_ec[:, :K_ec, :]
            tok_ecal_keep = x_ec[:, K_ec:, :]
        else:
            ecal_tok = self.ecal_embed(ecal_hits.view(B, -1)).unsqueeze(1)                  # [B, 1, C]
            ecal_tok = ecal_tok + self.kv_src_embed.weight[1].view(1, 1, -1)                # tag as ECAL

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
        muon_state = has_tracks.long().squeeze(-1)                                          # [B]
        muon_tok = muon_tok + self.muon_state_embed(muon_state).unsqueeze(1)                # 0=abstain, 1=present
        muon_tok = muon_tok + self.kv_src_embed.weight[2].view(1, 1, -1)                    # tag as MUON_SPEC

        # randomly drop global tokens (consistent with masking policy)
        keep_muon = (torch.rand(B, device=muon_tok.device) > (mask_ratio * 0.5))
        muon_kv_mask = keep_muon.view(B, 1)

        # build KV for lat <- tok
        kv_fas, kv_fas_keep, *_ = self._pack_by_mask(tok_mod, attn_mask_keep)               # [B, Nmax, C], [B, Nmax]
        kv_parts = [kv_fas, tok_ah_keep]
        kv_keep_parts = [kv_fas_keep, ah_attn_keep]

        if self.sparse_ecal:
            kv_parts.append(tok_ecal_keep)
            kv_keep_parts.append(ecal_attn_keep)
        else:
            keep_ecal = (torch.rand(B, device=ecal_tok.device) > (mask_ratio * 0.5))
            ecal_kv_mask = keep_ecal.view(B, 1)
            kv_parts.append(ecal_tok)
            kv_keep_parts.append(ecal_kv_mask)

        kv_parts.append(muon_tok)
        kv_keep_parts.append(muon_kv_mask)

        kv_tokens = torch.cat(kv_parts, dim=1)
        kv_keep   = torch.cat(kv_keep_parts, dim=1)
        kv_tokens = self.tokens_norm(kv_tokens)

        # latents: FASER CLS + AHCAL CLS + (optionally) ECAL CLS
        lat = self._prepare_latent_queries(cls_mod, ah_cls, ecal_cls)
        lat_keep = self._build_latent_keep(
            attn_mask_mod=attn_mask_mod, ah_mask=ah_mask, ecal_mask=ecal_mask
        )

        # Perceiver-IO encoder loop
        for xa_lat, sa in zip(self.lat_xattn_blocks, self.latent_self_blocks):
            lat = xa_lat(lat, kv_tokens, attn_mask=kv_keep, q_mask=lat_keep)                # lat <- tokens
            lat = sa(lat, attn_mask=lat_keep, q_mask=lat_keep)                              # latent self-attn

        result = (
            lat,                   # [B, N_lat, C]
            lat_keep,              # [B, N_lat]     which latents are valid
            rand_mask,             # [B, M, Lm]     FASERcal masked real positions
            attn_mask_mod,         # [B, M, Lm]     FASERcal real positions
            ah_mask,               # [B, Na]        AHCAL real positions
            ah_rand_mask,          # [B, Na]        AHCAL masked real positions
        )
        if self.sparse_ecal:
            result += (
                ecal_mask,         # [B, Ne]        ECAL real positions
                ecal_rand_mask,    # [B, Ne]        ECAL masked real positions
                ~keep_muon,        # [B]            MUON dropped events
            )
        else:
            keep_ecal_local = keep_ecal  # defined in the else branch above
            result += (
                ~keep_ecal_local,  # [B]            ECAL dropped events
                ~keep_muon,        # [B]            MUON dropped events
            )
        return result


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
        lat_keep: torch.Tensor,       # [B, N_lat]
        attn_mask_mod: torch.Tensor,  # [B, M, Lm]
        rand_mask: torch.Tensor,      # [B, M, Lm]
        idx_map,
    ):
        """
        FASERCal reconstruction from bottleneck latents only.
        """
        B = lat.size(0)
        Cdec = self.decoder_intra_pos_embed.weight.shape[-1]

        # masked real positions only
        prediction_mask = rand_mask & attn_mask_mod                            # [B, M, Lm]
        counts = prediction_mask.view(B, -1).sum(-1)
        max_n  = int(counts.max().item())

        # indices for masked positions
        b_ids, m_ids, l_ids = torch.nonzero(prediction_mask, as_tuple=True)    # [Nm]
        Nm = b_ids.numel()
        if Nm == 0:
            P = int(self.fcal_patch_size.prod().item())
            preds = {
                "occ": lat.new_zeros((0, P)),
                "reg": lat.new_zeros((0, P, self.heads["reg"].out_features)),
            }
            idx_targets = torch.empty((0, P), dtype=torch.long, device=lat.device)
            return preds, idx_targets

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
            X = blk(X, LAT, attn_mask=lat_keep)

        out_flat = X[b_ids, within]                                            # [Nm, Cdec]

        # Heads
        preds = {}
        shared = self.fasercal_shared_voxel_head(out_flat)                     # [Nm, P, H]
        preds["occ"] = self.heads["occ"](shared).squeeze(-1)                   # [Nm, P]
        preds["reg"] = self.heads["reg"](shared)                               # [Nm, P, in_chans]

        # targets (same for all predictions on masked patches)
        patch_ids = self.module_token_indices[m_ids, l_ids]                    # [Nm]
        idx_targets = idx_map[b_ids, patch_ids]                                # [Nm, P]

        return preds, idx_targets
    

    def forward_reconstruction_ahcal(
        self,
        lat: torch.Tensor,            # [B, N_lat, Cenc]
        lat_keep: torch.Tensor,       # [B, N_lat]
        ah_mask: torch.Tensor,        # [B, Na]
        ah_rand_mask: torch.Tensor,   # [B, Na]
        ah_idx_map,
    ):
        """
        AHCAL reconstruction from bottleneck latents only.
        """
        B = lat.size(0)
        Cdec = self.decoder_ahcal_pos_embed.weight.shape[-1]

        # masked real positions only
        prediction_mask = ah_rand_mask & ah_mask                             # [B, Na]
        counts = prediction_mask.sum(dim=-1)
        max_n = int(counts.max().item()) if B > 0 else 0

        # indices for masked AHCAL patches
        b_ids, l_ids = torch.nonzero(prediction_mask, as_tuple=True)         # [Nm]
        Nm = b_ids.numel()
        if Nm == 0:
            P_ah = int(self.ahcal_patch_size.prod().item())
            preds_ah = {
                "occ_ah": lat.new_zeros((0, P_ah)),
                "reg_ah": lat.new_zeros((0, P_ah, self.heads["reg_ahcal"].out_features)),
            }
            idx_targets_ah = torch.empty((0, P_ah), dtype=torch.long, device=lat.device)
            return preds_ah, idx_targets_ah

        # build queries in decoder dim for these masked AHCAL positions
        within = self._compute_within_ranks(b_ids, Nm)                       # [Nm]
        Q = self.ahcal_query_tokens.new_zeros(B, max_n, Cdec)                # [B, max_n, Cdec]
        q = self.decoder_ahcal_pos_embed(l_ids) + self.ahcal_query_tokens    # [Nm, Cdec]
        Q[b_ids, within] = q

        # query -> latents
        LAT = self.enc_to_dec(lat)  # [B,N_lat,Cdec]
        X = Q
        for blk in self.decode_lat_xattn_blocks:
            X = blk(X, LAT, attn_mask=lat_keep)

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
    

    def forward_reconstruction_ecal(
        self,
        lat: torch.Tensor,            # [B, N_lat, Cenc]
        lat_keep: torch.Tensor,       # [B, N_lat]
        ecal_mask: torch.Tensor,      # [B, Ne]
        ecal_rand_mask: torch.Tensor, # [B, Ne]
        ecal_idx_map,
    ):
        """
        Sparse ECAL reconstruction from bottleneck latents (mirrors AHCAL reconstruction).
        """
        B = lat.size(0)
        Cdec = self.decoder_ecal_pos_embed.weight.shape[-1]

        prediction_mask = ecal_rand_mask & ecal_mask                          # [B, Ne]
        counts = prediction_mask.sum(dim=-1)
        max_n = int(counts.max().item()) if B > 0 else 0

        b_ids, l_ids = torch.nonzero(prediction_mask, as_tuple=True)
        Nm = b_ids.numel()
        if Nm == 0:
            P_ec = int(self.ecal_patch_size.prod().item())
            preds_ec = {
                "occ_ecal": lat.new_zeros((0, P_ec)),
                "reg_ecal": lat.new_zeros((0, P_ec, self.heads["reg_ecal"].out_features)),
            }
            idx_targets_ec = torch.empty((0, P_ec), dtype=torch.long, device=lat.device)
            return preds_ec, idx_targets_ec

        within = self._compute_within_ranks(b_ids, Nm)
        Q = self.ecal_query_tokens.new_zeros(B, max_n, Cdec)
        q = self.decoder_ecal_pos_embed(l_ids) + self.ecal_query_tokens
        Q[b_ids, within] = q

        LAT = self.enc_to_dec(lat)
        X = Q
        for blk in self.decode_lat_xattn_blocks:
            X = blk(X, LAT, attn_mask=lat_keep)

        out_flat = X[b_ids, within]

        shared_ec = self.ecal_voxel_head(out_flat)
        preds_ec = {}
        preds_ec["occ_ecal"] = self.heads["occ_ecal"](shared_ec).squeeze(-1)
        preds_ec["reg_ecal"] = self.heads["reg_ecal"](shared_ec)

        patch_ids_ec = l_ids
        idx_targets_ec = ecal_idx_map[b_ids, patch_ids_ec]

        return preds_ec, idx_targets_ec


    def forward_reconstruction_global(
        self,
        lat: torch.Tensor,              # [B, N_lat, Cenc]
        lat_keep: torch.Tensor,         # [B, N_lat] bool
        ecal_hits,                       # [B, 5, 5] or None (sparse ECAL uses separate method)
        muspec_feats: torch.Tensor,     # [B, N, 5]
        muspec_attn_mask: torch.Tensor, # [B, N] bool
        ecal_drop: torch.Tensor = None, # [B] bool (None when sparse ECAL)
        muon_drop: torch.Tensor = None, # [B] bool
    ):
        """
        Predict ECAL and muon-summary only (loss-masked) when the corresponding token was dropped.
        When sparse_ecal, ECAL is handled separately; this only does muon.
        """
        B = lat.size(0)
        Cdec = self.global_query.size(-1)

        LAT = self.enc_to_dec(lat) # [B, N_lat, Cdec]

        preds = {}
        targets = {}
        masks = {}

        if not self.sparse_ecal:
            # ECAL query -> latents
            ecal_tgt = ecal_hits.view(B, -1)  # [B, 25]
            q_ecal = self.global_query[0].view(1, 1, Cdec).expand(B, 1, Cdec)
            x = q_ecal
            for blk in self.decode_lat_xattn_blocks:
                x = blk(x, LAT, attn_mask=lat_keep)
            preds["ecal_rec"] = self.ecal_recon_head(x.squeeze(1))
            targets["ecal_tgt"] = ecal_tgt
            masks["ecal_drop"] = ecal_drop

        # MUON query -> latents
        muon_tgt = muon_summary_target(muspec_feats, muspec_attn_mask)  # [B, 5]
        q_idx = 0 if self.sparse_ecal else 1
        q_muon = self.global_query[q_idx].view(1, 1, Cdec).expand(B, 1, Cdec)
        x = q_muon
        for blk in self.decode_lat_xattn_blocks:
            x = blk(x, LAT, attn_mask=lat_keep)
        preds["muon_rec"] = self.muon_recon_head(x.squeeze(1))
        targets["muon_tgt"] = muon_tgt
        masks["muon_drop"] = muon_drop

        return preds, targets, masks
    

    def forward_relational_fcal(
        self,
        lat: torch.Tensor,            # [B, N_lat, Cenc]
        lat_keep: torch.Tensor,       # [B, N_lat]
        attn_mask_mod: torch.Tensor,  # [B, M, Lm]  True = real patch
        rand_mask: torch.Tensor,      # [B, M, Lm]  True = masked real patch
        idx_map,
    ):
        """
        Relational/semantic heads (gho/hie/pid) on KEPT tokens.
        """
        B = lat.size(0)
        Cdec = self.decoder_intra_pos_embed.weight.shape[-1]

        keep_mask = attn_mask_mod & (~rand_mask)                                # [B, M, Lm]
        counts = keep_mask.view(B, -1).sum(-1)
        max_n  = int(counts.max().item()) if B > 0 else 0

        b_ids, m_ids, l_ids = torch.nonzero(keep_mask, as_tuple=True)           # [Nr]
        Nr = b_ids.numel()
        if Nr == 0:
            P = int(self.fcal_patch_size.prod().item())
            preds = {
                "gho": lat.new_zeros((0, P)),
                "hie": lat.new_zeros((0, P, self.heads["hie"].out_features)),
                "pid": lat.new_zeros((0, P, self.heads["pid"].out_features)),
            }
            idx_targets = torch.empty((0, P), dtype=torch.long, device=lat.device)
            return preds, idx_targets

        within = self._compute_within_ranks(b_ids, Nr)
        Q = self.query_tokens.new_zeros(B, max_n, Cdec)

        q = ( self.decoder_intra_pos_embed(l_ids)
            + self.module_embed_dec(m_ids)
            + self.query_tokens )                                              # [Nr, Cdec]
        Q[b_ids, within] = q

        LAT = self.enc_to_dec(lat)                                             # [B, N_lat, Cdec]
        X = Q
        for blk in self.decode_lat_xattn_blocks:
            X = blk(X, LAT, attn_mask=lat_keep)

        out_flat = X[b_ids, within]                                            # [Nr, Cdec]
        shared = self.fasercal_shared_voxel_head(out_flat)                     # [Nr, P, H]

        preds = {
            "gho": self.heads["gho"](shared).squeeze(-1),                      # [Nr, P]
            "hie": self.heads["hie"](shared),                                  # [Nr, P, 3]
            "pid": self.heads["pid"](shared),                                  # [Nr, P, num_pid]
        }

        patch_ids = self.module_token_indices[m_ids, l_ids]                    # [Nr]
        idx_targets = idx_map[b_ids, patch_ids]                                # [Nr, P]

        return preds, idx_targets
    

    def forward(self, x, x_glob, mask_ratio=0.75, do_relational = True, relational_mask_ratio=0.0):
        """
        Two-pass forward:

        Pass A (masked, MAE):
          - FASERCAL reconstruction + relational tasks on masked patches,
          - AHCAL reconstruction on masked patches.
          - ECAL reconstruction on masked patches (sparse_ecal only).

        Pass B (optional):
          - FASERCal relational: hie/pid on all real (kept) patches.

        Returns:
          preds:                  merged dict of predictions
          idx_targets_fas_masked: for occ/reg/gho losses (masked patches)
          idx_targets_fas_sem:    for hie/pid losses (real/kept patches from full pass)
          idx_targets_ah:         for AHCAL losses
          idx_targets_ecal:       for ECAL losses (sparse_ecal only, else None)
          glob_tgts, glob_masks:  for global losses
          aux:                    dict with run flags
        """
        # occupancy maps (same for both passes)
        idx_map = self.build_patch_occupancy_map(x, self.fcal_patch_size, self.grid_size)
        ahcal_sparse, ecal_hits, *_ = x_glob
        ah_idx_map = self.build_patch_occupancy_map(ahcal_sparse, self.ahcal_patch_size, self.ahcal_grid_size)

        ecal_idx_map = None
        if self.sparse_ecal:
            ecal_idx_map = self.build_patch_occupancy_map(ecal_hits, self.ecal_patch_size, self.ecal_grid_size)

        # ==========================
        # Pass A: masked MAE
        # ==========================
        enc_out = self.forward_encoder(x, x_glob, mask_ratio)

        if self.sparse_ecal:
            (lat, lat_keep, rand_mask, attn_mask_mod,
             ah_mask, ah_rand_mask,
             ecal_mask, ecal_rand_mask, muon_drop) = enc_out
            ecal_drop = None
        else:
            (lat, lat_keep, rand_mask, attn_mask_mod,
             ah_mask, ah_rand_mask,
             ecal_drop, muon_drop) = enc_out
            ecal_mask = ecal_rand_mask = None

        # FASERCAL: masked recon
        preds_fas_masked, idx_targets_fas_masked = self.forward_reconstruction_fcal(
            lat=lat,
            lat_keep=lat_keep,
            attn_mask_mod=attn_mask_mod,
            rand_mask=rand_mask,
            idx_map=idx_map,
        )

        # AHCAL masked recon
        preds_ah, idx_targets_ah = self.forward_reconstruction_ahcal(
            lat=lat,
            lat_keep=lat_keep,
            ah_mask=ah_mask,
            ah_rand_mask=ah_rand_mask,
            ah_idx_map=ah_idx_map,
        )

        # ECAL sparse masked recon
        idx_targets_ecal = None
        preds_ecal = {}
        if self.sparse_ecal:
            preds_ecal, idx_targets_ecal = self.forward_reconstruction_ecal(
                lat=lat,
                lat_keep=lat_keep,
                ecal_mask=ecal_mask,
                ecal_rand_mask=ecal_rand_mask,
                ecal_idx_map=ecal_idx_map,
            )

        # global recon (ECAL dense + MUON when dropped, or MUON only when sparse_ecal)
        _, ecal_hits_raw, muspec_feats, muspec_attn_mask, _ = x_glob
        preds_glob, glob_tgts, glob_masks = self.forward_reconstruction_global(
            lat=lat,
            lat_keep=lat_keep,
            ecal_hits=None if self.sparse_ecal else ecal_hits_raw,
            muspec_feats=muspec_feats,
            muspec_attn_mask=muspec_attn_mask,
            ecal_drop=ecal_drop,
            muon_drop=muon_drop,
        )

        preds = {**preds_fas_masked, **preds_ah, **preds_ecal, **preds_glob}

        # ==========================
        # Pass B: relational (optional)
        # ==========================
        idx_targets_fas_rel = None
        if do_relational:
            enc_out_full = self.forward_encoder(x, x_glob, mask_ratio=relational_mask_ratio)
            if self.sparse_ecal:
                (lat_full, lat_keep_full, rand_mask_full, attn_mask_mod_full,
                 *_rest) = enc_out_full
            else:
                (lat_full, lat_keep_full, rand_mask_full, attn_mask_mod_full,
                 *_rest) = enc_out_full

            preds_rel, idx_targets_fas_rel = self.forward_relational_fcal(
                lat=lat_full,
                lat_keep=lat_keep_full,
                attn_mask_mod=attn_mask_mod_full,
                rand_mask=rand_mask_full,
                idx_map=idx_map,
            )
            preds.update(preds_rel)

        aux = {
            "did_relational": bool(do_relational),
            "relational_mask_ratio": float(relational_mask_ratio),
        }

        return (
            preds,
            idx_targets_fas_masked,  # for occ/reg (masked patches)
            idx_targets_fas_rel,     # for gho/hie/pid (kept patches)
            idx_targets_ah,          # for AHCAL occ_ah/reg_ah (masked patches)
            idx_targets_ecal,        # for ECAL occ_ecal/reg_ecal (masked patches) or None
            glob_tgts,               # ecal_tgt, muon_tgt (dense ECAL only)
            glob_masks,              # ecal_drop, muon_drop
            aux,
        )


def mae_vit_tiny(**kwargs):
    model = SparseMAEViT(
        in_chans=1, embed_dim=384, 
        fcal_size=(48, 48, 200), fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40), ahcal_patch_size=(6, 6, 5),
        depth=2, ahcal_depth=2, num_heads=12, io_depth=6, io_decode_depth=2, 
        num_module_cls=2, num_ahcal_cls=2,
        num_modes=(8, 4), decoder_embed_dim=256, decoder_num_heads=8,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model


def mae_vit_base(**kwargs):
    model = SparseMAEViT(
        in_chans=1, embed_dim=384, 
        fcal_size=(48, 48, 200), fcal_patch_size=(12, 12, 10),
        ahcal_size=(18, 18, 40), ahcal_patch_size=(6, 6, 5),
        depth=4, ahcal_depth=4, num_heads=12, io_depth=4, io_decode_depth=2, 
        num_module_cls=2, num_ahcal_cls=2,
        num_modes=(8, 4), decoder_embed_dim=256, decoder_num_heads=8,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs,
    )
    return model
