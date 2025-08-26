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
from torch.nn.utils.rnn import pad_sequence
from MinkowskiEngine import MinkowskiConvolution
from timm.models.vision_transformer import Block
from .utils import get_3d_sincos_pos_embed, GlobalFeatureEncoderSimple, SeparableDCT3D, SharedLatentVoxelHead
        

class MinkMAEViT(nn.Module):
    def __init__(
        self,
        in_chans=1, 
        out_chans=4, 
        D=3,
        img_size=(48, 48, 200),
        module_depth_voxels=20,
        embed_dim=384,
        patch_size=(16, 16, 10),
        depth=24,
        inter_depth=2,
        num_heads=16,
        decoder_embed_dim=192,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        drop_rate_dec=0.,
        attn_drop_rate_dec=0.,
        drop_path_rate_dec=0.,
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
        self.register_buffer('patch_size', torch.tensor(patch_size))

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
        print("Module token indices shape:", self.module_token_indices.shape)

        # Encoder: hierarchical ViT
        self.intra_depth = depth
        self.inter_depth = inter_depth
        self.module_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))        # per-module CLS (shared weights)
        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim)  # fixed sin-cos per-module
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)         # learned module index

        # Intra-module transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.intra_depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(self.intra_depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Inter-module transformer over M CLS tokens
        self.global_token_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_feats_encoder = GlobalFeatureEncoderSimple(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.inter_depth)]
        self.inter_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(self.inter_depth)
        ])
        self.inter_norm = norm_layer(embed_dim)
    
        # MAE decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_intra_pos_embed = nn.Embedding(self.num_intra_positions, decoder_embed_dim)  # frozen sin-cos
        self.module_embed_dec = nn.Embedding(self.num_modules, decoder_embed_dim)                 # learned
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate_dec, decoder_depth)]
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate_dec, attn_drop=attn_drop_rate_dec,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(decoder_depth)
        ])

        # Reconstruction heads
        def _choose_Kxyz(patch_size, alpha=0.5, max_per_axis=16):
            ph, pw, pd = patch_size
            Ks = []
            for dim in (ph, pw, pd):
                K = min(int(round(alpha * dim)), max_per_axis, dim)
                if dim >= 2:
                    K = max(K, 2)  # at least 2 modes if dimension has >=2 voxels
                Ks.append(K)
            return tuple(Ks)
        
        self.sep_basis = SeparableDCT3D(
            self.patch_size.tolist(),
            Kxyz=_choose_Kxyz(self.patch_size.tolist())
        )

        self.shared_voxel_head = SharedLatentVoxelHead(
            decoder_embed_dim, self.sep_basis, H=16,
            out_ch_reg=self.in_chans, out_ch_cls=self.out_chans,
            norm_layer=norm_layer
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
            
        dec_pos = get_3d_sincos_pos_embed(
            self.decoder_intra_pos_embed.weight.shape[-1],
            self.intra_grid_size,
            cls_token=False
        )
        with torch.no_grad():
            self.decoder_intra_pos_embed.weight.copy_(torch.from_numpy(dec_pos).float())
            self.decoder_intra_pos_embed.weight.requires_grad_(False)

        # init tokens
        nn.init.normal_(self.module_cls_token, std=.02)
        nn.init.normal_(self.global_token_embed, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

        with torch.no_grad():
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

        intra_idx = self.intra_idx_template.unsqueeze(0).expand(B, -1)
        module_id = self.module_id_template.unsqueeze(0).expand(B, -1)

        return dense, intra_idx, module_id


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
    

    def _group_tokens_by_module(self, x):
        """
        x: [B, Np, C] dense tokens
        returns:
            x_mod: [B, M, Lm, C] re-ordered by module then intra-order
        """
        B, Np, C = x.shape
        M, Lm = self.num_modules, self.num_intra_positions
        idx = self.module_token_indices.unsqueeze(0).expand(B, -1, -1)  # [B, M, Lm]
        x_flat = torch.gather(
            x, 1,
            idx.reshape(B, M*Lm).unsqueeze(-1).expand(-1, -1, C)
        )  # [B, M*Lm, C]
        return x_flat.view(B, M, Lm, C)
    

    def _module_random_masking(self, x_mod, mask_ratio):
        """
        x_mod: [B, M, Lm, C]
        returns:
            x_keep: [B, M, Lk, C] (Lk is per-module, same across modules by floor)
            rand_mask: [B, M, Lm] (1=masked)
            ids_restore: [B, M, Lm] to recover original intra order
        """
        B, M, Lm, C = x_mod.shape
        Lk = max(1, int(Lm * (1 - mask_ratio)))  # keep at least 1
        device = x_mod.device
        noise = torch.rand(B, M, Lm, device=device) 
        ids_shuffle = torch.argsort(noise, dim=-1)         # [B, M, Lm]
        ids_restore = torch.argsort(ids_shuffle, dim=-1)   # [B, M, Lm]
        ids_keep = ids_shuffle[..., :Lk]                   # [B, M, Lk]
        x_keep = torch.gather(
            x_mod, 2, ids_keep.unsqueeze(-1).expand(-1, -1, -1, C)
        )  # [B, M, Lk, C]
        rand_mask = torch.ones(B, M, Lm, device=device)
        rand_mask[..., :Lk] = 0
        rand_mask = torch.gather(rand_mask, 2, ids_restore)

        return x_keep, rand_mask.bool(), ids_restore

    
    def forward_encoder(self, x_sparse, x_glob, mask_ratio):
        # patchify
        x_sparse = self.patch_embed(x_sparse)
        x, intra_idx, module_id = self.densify_patches(x_sparse)

        # add positional embeddings
        x = x + self.intra_pos_embed(intra_idx) + self.module_embed_enc(module_id)

        # group to modules and mask
        x_mod = self._group_tokens_by_module(x)
        x_keep, rand_mask, ids_restore = self._module_random_masking(x_mod, mask_ratio)
        B, M, Lk, C = x_keep.shape

        # Intra-module transformer with per-module CLS
        cls = self.module_cls_token.expand(B*M, 1, C)
        x_intra = x_keep.reshape(B*M, Lk, C)
        x_intra = torch.cat([cls, x_intra], dim=1)
        for blk in self.blocks:
            x_intra = blk(x_intra)
        x_intra = self.norm(x_intra)

        # split CLS and tokens
        cls_mod = x_intra[:, 0, :].reshape(B, M, C)        # [B, M, C]
        tok_mod = x_intra[:, 1:, :].reshape(B, M, Lk, C)   # [B, M, Lk, C]

        # Inter-module transformer over CLS tokens
        g = self.global_feats_encoder(x_glob)                                  # [B, C]
        g_tok = g.unsqueeze(1) + self.global_token_embed                       # [B, 1, C]
        mod_pos = self.module_embed_enc.weight.unsqueeze(0).expand(B, -1, -1)
        x_inter = torch.cat([g_tok, cls_mod + mod_pos], dim=1)                 # [B, 1+M, C]
        for blk in self.inter_blocks:
            x_inter = blk(x_inter)
        x_inter = self.inter_norm(x_inter)
        x_inter = x_inter[:, 1:, :]  # remove global token

        return tok_mod, x_inter, rand_mask, ids_restore

    
    def forward_decoder(self, tok_mod, x_inter, rand_mask, ids_restore, idx_map):
        """
        tok_mod:     [B, M, Lk, C]  encoder tokens per module
        rand_mask:   [B, M, Lm]     1=masked
        ids_restore: [B, M, Lm]     per-module restore to intra order
        idx_map:     [B, Np, P]     occupancy map from build_patch_occupancy_map
        """
        B, M, Lk, C = tok_mod.shape
        Lm = rand_mask.shape[2]
        Cdec = self.decoder_intra_pos_embed.weight.shape[-1]

        # project to decoder dim
        tok_mod = self.decoder_embed(tok_mod)    # [B, M, Lk, Cdec]

        # per-module restore with mask tokens
        num_mask = Lm - Lk
        mask_tok = self.mask_token.expand(B*M, num_mask, Cdec)           # [B*M, num_mask, Cdec]
        keep = tok_mod.reshape(B*M, Lk, Cdec)
        x_full = torch.cat([keep, mask_tok], dim=1)                      # [B*M, Lm, Cdec]
        x_full = torch.gather(
            x_full, 1,
            ids_restore.reshape(B*M, Lm).unsqueeze(-1).expand(-1, -1, Cdec)
        )                                                                # [B*M, Lm, Cdec]

        # add decoder intra-pos and module embeddings
        intra_positions = torch.arange(Lm, device=tok_mod.device)
        pos_intra = self.decoder_intra_pos_embed(intra_positions).unsqueeze(0).unsqueeze(0)  # [1, 1, Lm, Cdec]
        pos_intra = pos_intra.expand(B, M, -1, -1)                                           # [B, M, Lm, Cdec]
        mod_pos = self.module_embed_dec(
            torch.arange(M, device=tok_mod.device)
        ).unsqueeze(0).unsqueeze(2).expand(B, -1, Lm, -1)                                    # [B, M, Lm, Cdec]
        x_full = x_full.view(B, M, Lm, Cdec) + pos_intra + mod_pos   

        # add back CLS
        cls_dec = self.decoder_embed(x_inter)                                                # [B, M, Cdec]
        x_full = torch.cat([cls_dec.unsqueeze(2), x_full], dim=2).reshape(B*M, 1+Lm, Cdec)   # [B*M, 1+Lm, Cdec]
        
        # run transformer blocks
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        
        x_tokens = x_full[:, 1:, :].reshape(B, M, Lm, Cdec)                                  # [B, M, Lm, Cdec]
        prediction_mask = rand_mask                                                          # [B, M, Lm] (bool)
        flat_out = x_tokens[prediction_mask]                                                 # [N_masked, Cdec]

        pred_occ, pred_reg, pred_cls = self.shared_voxel_head(flat_out)                      # shapes same as beforxe

        # map (b, m, l) -> global patch id -> targets
        bm_l = torch.nonzero(prediction_mask, as_tuple=False)                                # [N_masked, 3] (b,m,l)
        b_ids  = bm_l[:, 0]
        m_ids  = bm_l[:, 1]
        l_ids  = bm_l[:, 2]

        # module_token_indices: [M, Lm] -> global flat patch id
        patch_ids = self.module_token_indices[m_ids, l_ids]                                  # [N_masked]
        idx_targets = idx_map[b_ids, patch_ids]   

        return pred_occ, pred_reg, pred_cls, idx_targets, b_ids, patch_ids
        

    def forward(self, x, x_glob, mask_ratio=0.75):
        """
        Forward pass through the encoder-decoder network.

        Args:
            x: Input sparse tensor.
            x_glob: Global feature tensors.
            cls_labels: Labels for classification task.
            mask_ratio: mask probability.

        Returns:
            Voxel predictions.
        """
        idx_map = self.build_patch_occupancy_map(x)
        tok_mod, x_inter, rand_mask, ids_restore = self.forward_encoder(x, x_glob, mask_ratio)
        pred_occ, pred_reg, pred_cls, idx_targets, row_event_ids, row_patch_ids = \
            self.forward_decoder(tok_mod, x_inter, rand_mask, ids_restore, idx_map)

        return pred_occ, pred_reg, pred_cls, idx_targets, row_event_ids, row_patch_ids

        
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


def mae_vit_tiny(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        embed_dim=528, patch_size=(16, 16, 4),
        depth=10, inter_depth=2, num_heads=12,
        decoder_embed_dim=384, decoder_depth=6, decoder_num_heads=12,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model


def mae_vit_base(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(16, 16, 4),
        depth=10, inter_depth=2, num_heads=12,
        decoder_embed_dim=528, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model
    

def mae_vit_large(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(16, 16, 4),
        depth=24, num_heads=16,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model


def mae_vit_huge(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        eembed_dim=768, patch_size=(16, 16, 4),
        depth=32, num_heads=16,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model
