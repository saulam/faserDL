"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 07.25

Description: PyTorch MAE-ViT model with MinkowskiEngine patching.
"""

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from functools import partial
from MinkowskiEngine import MinkowskiConvolution
from .utils import get_3d_sincos_pos_embed, BlockWithMask, GlobalFeatureEncoderSimple, CrossAttnBlock, SeparableDCT3D, SharedLatentVoxelHead
        

class MinkMAEViT(nn.Module):
    def __init__(
        self,
        in_chans=1, 
        out_chans=4, 
        D=3,
        img_size=(48, 48, 200),
        module_depth_voxels=20,
        embed_dim=384,
        patch_size=(16, 16, 4),
        depth=8,
        num_global_tokens=2,
        latent_tokens=32,
        io_depth=4,
        io_decode_depth=4,
        num_heads=16,
        num_modes=8,
        contrastive_embed_dim=64,
        decoder_embed_dim=192,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        drop_rate_dec=0.,
        attn_drop_rate_dec=0.,
        norm_layer=nn.LayerNorm,
    ):
        """
        Args:
            in_chans (int): Number of input channels.
            out_chans (int): Number of output channels.
            D (int): Spatial dimension for Minkowski layers.
            img_size (tuple): Input image size (H, W, D).
            args: Namespace with at least a `dataset_path` attribute.
        """
        super().__init__()
    
        # patch and grid setup
        H, W, D_img = img_size
        p_h, p_w, p_d = patch_size
        assert H % p_h == 0 and W % p_w == 0 and D_img % p_d == 0, \
            "img_size must be divisible by patch_size"
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.grid_size = (H // p_h, W // p_w, D_img // p_d)
        self.num_patches = (
            self.grid_size[0] 
            * self.grid_size[1] 
            * self.grid_size[2]
        )
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
        self.patch_embed = MinkowskiConvolution(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, 
            bias=True, dimension=D,
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
        self.register_buffer('module_id_template', module.long())         # [Np]

        # Per-module token index mapping [M, Lm]
        M = self.num_modules
        module_indices = []
        for m in range(M):
            mask = (self.module_id_template == m)
            intra_m = self.intra_idx_template[mask]
            flat_m  = self.idx_template[mask]
            order   = torch.argsort(intra_m)      # stable intra order
            module_indices.append(flat_m[order])
        self.register_buffer('module_token_indices', torch.stack(module_indices, 0))  # [M, Lm]

        # Encoder: hierarchical ViT
        self.intra_depth = depth
        self.module_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))        # per-module CLS (shared weights)
        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim)  # fixed sin-cos per-module
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)         # learned module index for intra-attn

        # Intra-module transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.intra_depth)]
        self.blocks = nn.ModuleList([
            BlockWithMask(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(self.intra_depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Perceiver-IO bottleneck (encoder side)
        self.num_global_tokens = num_global_tokens
        self.global_feats_encoder = GlobalFeatureEncoderSimple(embed_dim)
        self.global_mem = nn.Parameter(torch.zeros(1, self.num_global_tokens, embed_dim))
        self.latent_tokens = latent_tokens
        self.latents = nn.Parameter(torch.zeros(1, latent_tokens, embed_dim))
        self.latent_xattn_blocks = nn.ModuleList([
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
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=0., norm_layer=norm_layer
            )
            for _ in range(io_depth)
        ])
        self.latent_norm = norm_layer(embed_dim)

        # Perceiver-IO decoder (queries -> latents)
        self.decoder_intra_pos_embed = nn.Embedding(self.num_intra_positions, decoder_embed_dim) # frozen sin-cos
        self.latents_to_dec = nn.Linear(embed_dim, decoder_embed_dim)
        self.module_embed_dec = nn.Embedding(self.num_modules, decoder_embed_dim)
        self.keep_query_token = nn.Parameter(torch.zeros(1, decoder_embed_dim))
        self.mask_query_token = nn.Parameter(torch.zeros(1, decoder_embed_dim))
        self.decode_xattn_blocks = nn.ModuleList([
            CrossAttnBlock(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate_dec, attn_drop=attn_drop_rate_dec,
                drop_path=0., norm_layer=norm_layer
            )
            for _ in range(io_decode_depth)
        ])

        # separable basis
        def _choose_Kxyz(patch_size, alphas=(0.4, 0.4, 0.75), max_per_axis=16):
            ph, pw, pd = patch_size
            Ks = []
            for dim, alpha in zip((ph, pw, pd), alphas):
                K = min(int(round(alpha * dim)), max_per_axis, dim)
                if dim >= 2:
                    K = max(K, 2)  # at least 2 modes if dimension has >=2 voxels
                Ks.append(K)
            return tuple(Ks)
        self.sep_basis = SeparableDCT3D(
            self.patch_size.tolist(),
            Kxyz=_choose_Kxyz(self.patch_size.tolist())
        )

        # Contrastive heads
        H, E = num_modes, contrastive_embed_dim
        self.shared_voxel_head_con = SharedLatentVoxelHead(
            decoder_embed_dim, self.sep_basis, H=H*3, norm_layer=norm_layer
        )
        self.track_head   = nn.Linear(H*3, E)
        self.primary_head = nn.Linear(H*3, E)
        self.pid_head     = nn.Linear(H*3, E)

        # Reconstruction heads        
        self.shared_voxel_head_dec = SharedLatentVoxelHead(
            decoder_embed_dim, self.sep_basis, H=H, norm_layer=norm_layer
        )
        self.occ_head = nn.Linear(H, 1)
        self.reg_head = nn.Linear(H, self.in_chans)

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
            nn.init.normal_(self.latents, std=.02)
            nn.init.normal_(self.module_cls_token, std=.02)
            nn.init.normal_(self.keep_query_token, std=0.02)
            nn.init.normal_(self.mask_query_token, std=0.02)
            nn.init.normal_(self.module_embed_enc.weight, std=0.02)
            nn.init.normal_(self.module_embed_dec.weight, std=0.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            # initialize conv like nn.Linear
            w = m.kernel.data
            torch.nn.init.xavier_uniform_(w.view(-1, w.size(2)))
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
            'latents',
            'keep_query_token',
            'mask_query_token',
            'patch_embed.bias',
        }


    def densify_patches(self, x_sparse):
        coords = x_sparse.C.long()  # [N,4]
        feats  = x_sparse.F

        B      = int(coords[:, 0].max().item()) + 1
        G_h, G_w, G_d = self.grid_size
        Np     = self.num_patches
        C      = feats.size(1)

        # convert from voxel coords to patch-grid coords
        h = coords[:, 1] // self.patch_size[0]
        w = coords[:, 2] // self.patch_size[1]
        d = coords[:, 3] // self.patch_size[2]

        patch_ids = (h * (G_w * G_d)) + (w * G_d) + d
        batch_ids = coords[:, 0]
        
        # scatter into dense tensor
        dense = feats.new_zeros(B, Np, C)
        dense[batch_ids, patch_ids, :] = feats

        attn_mask = torch.zeros(B, Np, dtype=torch.bool, device=feats.device)
        attn_mask[batch_ids, patch_ids] = True
        intra_idx = self.intra_idx_template.unsqueeze(0).expand(B, -1)
        module_id = self.module_id_template.unsqueeze(0).expand(B, -1)

        return dense, attn_mask, intra_idx, module_id


    def build_patch_occupancy_map(self, x):
        """
        From the original sparse tensor coordinates, build a [B, N_patches, P] mapping
        with the raw id of the actual hit in that sub‐voxel.
        """
        coords     = x.C
        device     = coords.device
        event_ids  = coords[:, 0].long()
        x_, y_, z_ = coords[:, 1:].unbind(-1)
    
        p_h, p_w, p_d  = self.patch_size.tolist()
        G_h, G_w, G_d  = self.grid_size
        P              = self.patch_voxels
        Np             = self.num_patches
        B              = int(event_ids.max().item()) + 1
        N              = coords.shape[0]
    
        patch_idx = (x_ // p_h) * (G_w * G_d) \
                  + (y_ // p_w) * G_d \
                  + (z_ // p_d)
        sub_idx   = (x_ % p_h) * (p_w * p_d) \
                  + (y_ % p_w) * p_d \
                  + (z_ % p_d)

        raw_inds = torch.arange(N, device=device)
        idx_map = torch.full ((B, Np, P), -1, dtype=torch.long, device=device)
        idx_map[event_ids, patch_idx, sub_idx] = raw_inds
    
        return idx_map
    

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


    def _module_random_masking(self, x, attn_mask, mask_ratio):
        """
        x: [B, M, Lm, C]
        attn_mask: [B, M, Lm]
        returns:
            x_keep: [B, M, Lk, C]
            attn_mask_keep: [B, M, Lk]
            rand_mask: [B, M, Lm]  True = masked (in original order)
            ids_restore: [B, M, Lm]  (identity; unused by the rest of the model)
            ids_keep: [B, M, Lk]
        """
        B, M, Lm, C = x.shape
        Lk = max(1, int(Lm * (1.0 - mask_ratio)))
        device = x.device

        # Sample scores and push invalids to +inf so they won't be selected unless forced
        scores = torch.rand(B, M, Lm, device=device)
        scores = scores.masked_fill(~attn_mask, float('inf'))

        # Take k smallest (valid first) (cheaper than sorting the whole axis)
        keep_scores, ids_keep = torch.topk(scores, k=Lk, dim=-1, largest=False, sorted=False)  # [B, M, Lk]

        # Gather kept tokens and their mask
        x_keep = torch.gather(x, 2, ids_keep.unsqueeze(-1).expand(-1, -1, -1, C))              # [B, M, Lk, C]
        attn_mask_keep = torch.gather(attn_mask, 2, ids_keep)                                  # [B, M, Lk]

        # Build rand_mask in original order: start all True (masked), then unmask kept ids
        rand_mask = torch.ones(B, M, Lm, dtype=torch.bool, device=device)
        rand_mask.scatter_(2, ids_keep, False)                                                 # kept -> not masked

        # ids_restore is not used downstream
        ids_restore = torch.arange(Lm, device=device).view(1, 1, Lm).expand(B, M, Lm)

        return x_keep, attn_mask_keep, rand_mask, ids_restore, ids_keep


    def compute_within_ranks(self, b_ids: torch.Tensor, N: int) -> torch.Tensor:
        # b_ids must be nondecreasing (true for torch.nonzero over [B, ...]).
        if N == 0:
            return b_ids
        _, counts = torch.unique_consecutive(b_ids, return_counts=True)  # [G]
        starts = torch.cumsum(counts, dim=0) - counts                    # [G]
        return torch.arange(N, device=b_ids.device) - torch.repeat_interleave(starts, counts)

    
    def forward_encoder(self, x_sparse, x_glob, mask_ratio):
        """
        x_sparse:                   sparse input
        x_glob:       [B, C]        global context
        mask_ratio:   float         masking ratio
        """
        # patchify
        x_sparse = self.patch_embed(x_sparse)
        x, attn_mask, intra_idx, _ = self.densify_patches(x_sparse)

        # add positional embeddings
        x = x + self.intra_pos_embed(intra_idx)

        # group to modules and mask
        x_mod, attn_mask_mod = self._group_tokens_by_module(x, attn_mask)
        x_keep, attn_mask_keep, rand_mask, _, ids_keep = self._module_random_masking(x_mod, attn_mask_mod, mask_ratio)
        B, M, Lk, C = x_keep.shape

        # Intra-module transformer with per-module CLS
        cls = self.module_cls_token.expand(B*M, 1, C)
        x_intra = x_keep.reshape(B*M, Lk, C)
        x_intra = torch.cat([cls, x_intra], dim=1)
        attn_mask_intra = torch.cat([torch.ones(B*M, 1, dtype=torch.bool, device=x.device), attn_mask_keep.reshape(B*M, Lk)], dim=1)
        for blk in self.blocks:
            x_intra = blk(x_intra, attn_mask=attn_mask_intra)
        x_intra = self.norm(x_intra)

        # discard intra-module CLS and add module embedding
        tok_mod = x_intra[:, 1:, :].reshape(B, M, Lk, C)   # [B, M, Lk, C]
        mod_ids = torch.arange(self.num_modules, device=tok_mod.device)
        tok_mod = tok_mod + self.module_embed_enc(mod_ids).view(1, M, 1, -1)

        # global input
        g_enc   = self.global_feats_encoder(x_glob)                  # [B, C]
        g_tokens = g_enc.unsqueeze(1) + self.global_mem              # [B, G, C]
        g_tokens = g_tokens.to(tok_mod.dtype)

        # Perceiver-IO encoder: latents attend to all kept tokens
        kv_tokens = tok_mod.reshape(B, M*Lk, C)                       # [B, Nk, C]
        kv_tokens = torch.cat([kv_tokens, g_tokens], dim=1)           # [B, Nk+G, C]
        kv_keep  = attn_mask_keep.reshape(B, M*Lk)                    # [B, Nk]
        kv_keep   = torch.cat([kv_keep, torch.ones(
            kv_keep.size(0), self.num_global_tokens, 
            dtype=torch.bool, device=kv_keep.device)], dim=1)         # [B, Nk+G]
        lat = self.latents.expand(B, -1, -1)                          # [B, K, C]
        for xa, sa in zip(self.latent_xattn_blocks, self.latent_self_blocks):
            lat = xa(lat, kv_tokens, attn_mask=kv_keep)               # cross: lat <- tokens
            lat = sa(lat, attn_mask=None)                             # self-attn over latents
        lat = self.latent_norm(lat)                                   # [B, K, C]

        return (
            lat, rand_mask, attn_mask_mod, attn_mask_keep, ids_keep
        )
        

    def forward_contrastive(self, latents, attn_mask_keep, ids_keep, idx_map):
        """
        latents:        [B, K, Cenc]
        attn_mask_keep: [B, M, Lk]
        ids_keep:       [B, M, Lk] 
        idx_map:        [B, Np, P]
        """
        B, K, Cenc = latents.shape
        Cdec = self.decoder_intra_pos_embed.weight.shape[-1]
        
        # masked REAL positions
        keep_mask = attn_mask_keep                                         # [B, M, Lk], True=keep
        counts = keep_mask.view(B, -1).sum(-1)                             # [B]
        max_n = int(counts.max().item())
        
        # flatten selected (b,m,l)
        b_ids, m_ids, lk_ids = torch.nonzero(keep_mask, as_tuple=True)     # [Nk]
        l_intra = ids_keep[b_ids, m_ids, lk_ids]                           # [Nk]
        Nk = b_ids.numel()

        # build queries for all selected positions
        q = ( self.decoder_intra_pos_embed(l_intra) +
              self.module_embed_dec(m_ids) +
              self.keep_query_token )                                      # [Nk, Cdec]

        # pack by batch into [B, max_n, Cdec] using one-hot cumsum trick
        within = self.compute_within_ranks(b_ids, Nk)
        Q = latents.new_zeros(B, max_n, Cdec)                              # padded queries
        Q[b_ids, within] = q                                               # place queries by (b, rank)

        # decode with latents
        KV = self.latents_to_dec(latents)                                  # [B, K, Cdec]
        X  = Q
        for blk in self.decode_xattn_blocks:
            X = blk(X, KV, attn_mask=None)
        out_flat = X[b_ids, within]                                        # [Nk, Cdec]

        # heads
        shared = self.shared_voxel_head_con(out_flat)                      # [Nk, P, H*3]
        pred_track   = self.track_head(shared)                             # [Nk, P, E]
        pred_primary = self.primary_head(shared)                           # [Nk, P, E]
        pred_pid     = self.pid_head(shared)                               # [Nk, P, E]

        # targets
        patch_ids = self.module_token_indices[m_ids, l_intra]              # [Nk]
        idx_targets_kept = idx_map[b_ids, patch_ids]                       # [Nk, P]

        return pred_track, pred_primary, pred_pid, idx_targets_kept


    def forward_reconstruction(self, latents, rand_mask, attn_mask_mod, idx_map):
        """
        latents:       [B, K, Cenc]
        rand_mask:     [B, M, Lm]
        attn_mask_mod: [B, M, Lm]
        idx_map:       [B, Np, P]
        """
        B, K, Cenc = latents.shape
        Cdec = self.decoder_intra_pos_embed.weight.shape[-1]

        # masked REAL positions
        prediction_mask = rand_mask & attn_mask_mod                          # [B, M, Lm]
        counts = prediction_mask.view(B, -1).sum(-1)                         # [B]
        max_n = int(counts.max().item())

        # flatten selected (b,m,l)
        b_ids, m_ids, l_ids = torch.nonzero(prediction_mask, as_tuple=True)  # [Nm]
        Nm = b_ids.numel()

        # build queries for all selected positions
        q = ( self.decoder_intra_pos_embed(l_ids) +
            self.module_embed_dec(m_ids) +
            self.mask_query_token )                                          # [Nm, Cdec]

        # pack by batch into [B, max_n, Cdec] using one-hot cumsum trick
        within = self.compute_within_ranks(b_ids, Nm)
        Q = latents.new_zeros(B, max_n, Cdec)                                # padded queries
        Q[b_ids, within] = q                                                 # place queries by (b, rank)

        # decode with latents
        KV = self.latents_to_dec(latents)                                    # [B, K, Cdec]
        X  = Q
        for blk in self.decode_xattn_blocks:
            X = blk(X, KV, attn_mask=None)
        out_flat = X[b_ids, within]                                          # [Nm, Cdec]

        # heads
        shared = self.shared_voxel_head_dec(out_flat)                        # [Nm, P, H]
        pred_occ = self.occ_head(shared).squeeze(-1)                         # [Nm, P]
        pred_reg = self.reg_head(shared)                                     # [Nm, P, in_chans]

        # targets
        patch_ids = self.module_token_indices[m_ids, l_ids]                  # [Nm]
        idx_targets = idx_map[b_ids, patch_ids]                              # [Nm, P]

        return pred_occ, pred_reg, idx_targets, b_ids, patch_ids


    def forward(self, x, x_glob, mask_ratio=0.75):
        idx_map = self.build_patch_occupancy_map(x)

        # Encoder
        lat, rand_mask, attn_mask_mod, attn_mask_keep, ids_keep = \
            self.forward_encoder(x, x_glob, mask_ratio)

        # Contrastive path
        pred_track, pred_primary, pred_pid, enc_idx_targets = \
            self.forward_contrastive(lat, attn_mask_keep, ids_keep, idx_map)
        
        # Reconstruction path
        pred_occ, pred_reg, idx_targets, row_event_ids, row_patch_ids = \
            self.forward_reconstruction(lat, rand_mask, attn_mask_mod, idx_map)

        return (
            pred_track, pred_primary, pred_pid,
            pred_occ, pred_reg, idx_targets, enc_idx_targets,
            row_event_ids, row_patch_ids
        )

        
    def replace_depthwise_with_channelwise(self):
        """
        Replace all MinkowskiDepthwiseConvolution modules with
        MinkowskiChannelwiseConvolution modules preserving the parameters.
        """
        for name, module in self.named_modules():
            if isinstance(module, ME.MinkowskiDepthwiseConvolution):
                # Retrieve parameters of the current depthwise convolution
                in_channels = module.in_channels
                kernel_size = module.kernel_generator.kernel_size
                stride = module.kernel_generator.kernel_stride
                dilation = module.kernel_generator.kernel_dilation
                bias = module.bias is not None
                dimension = module.dimension
                
                # Create a new channelwise convolution with the same parameters
                new_conv = ME.MinkowskiChannelwiseConvolution(
                    in_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    dimension=dimension
                )
                new_conv.kernel = module.kernel
                if bias:
                    new_conv.bias = module.bias
                
                # Replace the old module with the new one
                parent_module, attr_name = self._get_parent_module(name)
                setattr(parent_module, attr_name, new_conv)
        
        return

    
    def _get_parent_module(self, layer_name):
        """
        Retrieve the parent module and attribute name for a given layer.
        
        Args:
            layer_name (str): Dot-separated module path.
        
        Returns:
            Tuple of (parent_module, attribute_name)
        """
        components = layer_name.split('.')
        parent = self
        for comp in components[:-1]:
            parent = getattr(parent, comp)
        return parent, components[-1]
    

    def print_param_report(self):
        """
        Prints parameter counts by component:
        - patch_embed
        - intra_vit (pos-emb, module token/emb, blocks, norm)
        - perceiver_encoder (global encoder, global_mem, latents, cross/self blocks, norm)
        - perceiver_decoder (dec pos-emb, module emb, query tokens, cross blocks)
        - heads_contrastive
        - heads_reconstruction
        - TOTAL
        """
        import torch
        import torch.nn as nn

        def _iter_params(obj):
            """Yield parameters from nn.Modules, nn.Parameters, and containers (list/tuple/dict/ModuleList/etc.)."""
            if obj is None:
                return
            if isinstance(obj, nn.Parameter):
                # Yield the loose parameter itself
                yield obj
                return
            if isinstance(obj, nn.Module):
                # Yield all parameters from the module
                for p in obj.parameters(recurse=True):
                    yield p
                return
            # Handle common containers
            if isinstance(obj, (list, tuple, set)):
                for o in obj:
                    yield from _iter_params(o)
                return
            if isinstance(obj, dict):
                for o in obj.values():
                    yield from _iter_params(o)
                return
            # Torch container types
            if isinstance(obj, (nn.ModuleList, nn.Sequential, nn.ParameterList)):
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

        # Build groups with hasattr/getattr guards so finetune variants don't break
        groups = {}

        if hasattr(self, "patch_embed"):
            groups["patch_embed"] = [self.patch_embed]
        else:
            groups["patch_embed"] = []

        intra_list = []
        if hasattr(self, "intra_pos_embed"):   intra_list.append(self.intra_pos_embed)
        if hasattr(self, "module_cls_token"):  intra_list.append(self.module_cls_token)
        if hasattr(self, "module_embed_enc"):  intra_list.append(self.module_embed_enc)
        if hasattr(self, "norm"):              intra_list.append(self.norm)
        if hasattr(self, "blocks"):            intra_list.extend(list(self.blocks))
        groups["intra_vit"] = intra_list

        enc_list = []
        if hasattr(self, "global_feats_encoder"): enc_list.append(self.global_feats_encoder)
        if hasattr(self, "global_mem"):           enc_list.append(self.global_mem)
        if hasattr(self, "latents"):              enc_list.append(self.latents)
        if hasattr(self, "latent_norm"):          enc_list.append(self.latent_norm)
        if hasattr(self, "latent_xattn_blocks"):  enc_list.extend(list(self.latent_xattn_blocks))
        if hasattr(self, "latent_self_blocks"):   enc_list.extend(list(self.latent_self_blocks))
        groups["perceiver_encoder"] = enc_list

        dec_list = []
        if hasattr(self, "decoder_intra_pos_embed"): dec_list.append(self.decoder_intra_pos_embed)
        if hasattr(self, "latents_to_dec"):          dec_list.append(self.latents_to_dec)
        if hasattr(self, "module_embed_dec"):        dec_list.append(self.module_embed_dec)
        if hasattr(self, "mask_query_token"):        dec_list.append(self.mask_query_token)
        if hasattr(self, "decode_xattn_blocks"):     dec_list.extend(list(self.decode_xattn_blocks))
        groups["perceiver_decoder"] = dec_list

        heads_con = []
        if hasattr(self, "shared_voxel_head_con"): heads_con.append(self.shared_voxel_head_con)
        if hasattr(self, "track_head"):            heads_con.append(self.track_head)
        if hasattr(self, "primary_head"):          heads_con.append(self.primary_head)
        if hasattr(self, "pid_head"):              heads_con.append(self.pid_head)
        groups["heads_contrastive"] = [m for m in heads_con if m is not None]

        heads_rec = []
        if hasattr(self, "shared_voxel_head_dec"): heads_rec.append(self.shared_voxel_head_dec)
        if hasattr(self, "occ_head"):              heads_rec.append(self.occ_head)
        if hasattr(self, "reg_head"):              heads_rec.append(self.reg_head)
        groups["heads_reconstruction"] = [m for m in heads_rec if m is not None]

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
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        embed_dim=528, patch_size=(16, 16, 4),
        depth=2, num_heads=12, num_global_tokens=2,
        latent_tokens=8, io_depth=2, io_decode_depth=2,
        num_modes=8, contrastive_embed_dim=32,
        decoder_embed_dim=384, decoder_num_heads=12,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model


def mae_vit_base(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(16, 16, 4),
        depth=4, num_heads=12, num_global_tokens=2,
        latent_tokens=16, io_depth=4, io_decode_depth=4,
        num_modes=16, contrastive_embed_dim=64,
        decoder_embed_dim=384, decoder_num_heads=12,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model
    

def mae_vit_large(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(16, 16, 4),
        depth=8, num_heads=12, num_global_tokens=2,
        latent_tokens=32, io_depth=8, io_decode_depth=8,
        num_modes=32, contrastive_embed_dim=64,
        decoder_embed_dim=528, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model


def mae_vit_huge(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(16, 16, 4),
        depth=16, num_heads=12, num_global_tokens=4,
        latent_tokens=64, io_depth=16, io_decode_depth=16,
        num_modes=64, contrastive_embed_dim=128,
        decoder_embed_dim=528, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model
