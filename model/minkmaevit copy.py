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
from .utils import BlockWithMask, get_3d_sincos_pos_embed, GlobalFeatureEncoderSimple, SeparableDCT3D, SharedLatentVoxelHead
        

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
        num_heads=16,
        decoder_embed_dim=192,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.,
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
    
        # patch & grid setup
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
        assert module_depth_voxels % p_d == 0, "module_depth_voxels must be divisible by patch depth"
        self.module_depth_voxels = module_depth_voxels
        self.module_depth_patches = module_depth_voxels // p_d
        G_h, G_w, G_d = self.grid_size
        assert G_d % self.module_depth_patches == 0, "grid depth must be multiple of module depth (in patches)"
        self.num_modules = G_d // self.module_depth_patches
        self.intra_grid_size = (G_h, G_w, self.module_depth_patches)     # H×W×(depth within module)
        self.num_intra_positions = G_h * G_w * self.module_depth_patches

        self.patch_embed = MinkowskiConvolution(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, 
            bias=True, dimension=D,
        )

        # MAE encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.global_feats_encoder = GlobalFeatureEncoderSimple(embed_dim)
        self.intra_pos_embed = nn.Embedding(self.num_intra_positions, embed_dim) # frozen sin-cos
        self.module_embed_enc = nn.Embedding(self.num_modules, embed_dim)        # learned
        self.blocks = nn.ModuleList([
            BlockWithMask(
                embed_dim, num_heads, mlp_ratio,
                qkv_bias=True, drop=drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
    
        # MAE decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_intra_pos_embed = nn.Embedding(self.num_intra_positions, decoder_embed_dim)  # frozen sin-cos
        self.module_embed_dec = nn.Embedding(self.num_modules, decoder_embed_dim)                 # learned
        self.decoder_blocks = nn.ModuleList([
            BlockWithMask(
                decoder_embed_dim, decoder_num_heads, mlp_ratio,
                qkv_bias=True, drop=drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])

        # Reconstruction heads
        def _choose_Kxyz(patch_size, alpha=0.5, max_per_axis=16):
            ph, pw, pd = patch_size
            Ks = []
            for dim in (ph, pw, pd):
                K = min(int(round(alpha * dim)), max_per_axis, dim)
                if dim >= 2:
                    K = max(K, 2)   # at least 2 modes if dimension has >=2 voxels
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
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

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

    
    def group_patches_by_event(
        self,
        sparse_tensor,
    ):
        """
        Buckets patches into feats and (flattened) positions to padded dense tensors,
        and calculates the corresponding attention mask.
    
        Args:
            sparse_tensor: MinkowskiEngine sparse tensor with
                           .C of shape [N, 4] (coords, with C[:,0]=event_id)
                           .F of shape [N, C] (features).
    
        Returns:
            padded_feats:  [B, L_max, C] float, zero‑padded features
            padded_idx:    [B, L_max] int, flat position ids.
            attn_mask:     [B, L_max] bool, True = real voxel.
        """
        coords = sparse_tensor.C
        feats  = sparse_tensor.F
    
        event_ids             = coords[:, 0].long()
        spatial_coords        = coords[:, 1:] // self.patch_size
        sorted_eids, perm     = event_ids.sort()
        sorted_feats          = feats[perm]
        sorted_spatial_coords = spatial_coords[perm]
    
        uniq_ids, counts = torch.unique_consecutive(sorted_eids,
                                                    return_counts=True)
        counts_list   = counts.tolist()    
        feat_groups   = torch.split(sorted_feats, counts_list, dim=0)
        coord_groups  = torch.split(sorted_spatial_coords, counts_list, dim=0)
        
        padded_feats  = pad_sequence(feat_groups, batch_first=True, padding_value=0.0)
        padded_coords = pad_sequence(coord_groups, batch_first=True, padding_value=0)
    
        L_max     = int(counts.max().item())
        arange    = torch.arange(L_max, device=feats.device)
        attn_mask = arange.unsqueeze(0) < counts.unsqueeze(1)

        G_h, G_w, G_d = self.grid_size
        h_idx = padded_coords[..., 0]
        w_idx = padded_coords[..., 1]
        d_idx = padded_coords[..., 2]
        padded_idx = h_idx * (G_w * G_d) + w_idx * G_d + d_idx

        d_mod      = (d_idx % self.module_depth_patches).long()
        module_id  = (d_idx // self.module_depth_patches).long()
        intra_idx  = (h_idx * (G_w * self.module_depth_patches)
                    +  w_idx * self.module_depth_patches
                    +  d_mod).long()
    
        return padded_feats, padded_idx, attn_mask, intra_idx, module_id


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
    

    def random_masking(self, x, attn_mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, L, D], sequence
        attn_mask: [B, L]

        Source: https://github.com/facebookresearch/mae/blob/main/models_mae.py
        """
        B, L, C = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, C))
        attn_mask_masked = torch.gather(attn_mask, 1, ids_keep)
        rand_mask = torch.ones(B, L, device=x.device)
        rand_mask[:, :len_keep] = 0
        rand_mask = torch.gather(rand_mask, 1, ids_restore)
        return x_masked, attn_mask_masked, rand_mask, ids_restore

    
    def forward_encoder(self, x_sparse, glob, mask_ratio):
        # patchify
        x_sparse = self.patch_embed(x_sparse)
        x, idx, orig_attn_mask, intra_idx, module_id = self.group_patches_by_event(x_sparse)

        # add positional embeddings
        x = x + self.intra_pos_embed(intra_idx) + self.module_embed_enc(module_id)

        # masking
        x, attn_mask, rand_mask, ids_restore = self.random_masking(
            x, orig_attn_mask, mask_ratio
        )

        # add cls token
        #glob_emb = self.global_feats_encoder(glob).unsqueeze(1)  # (B, 1, D)
        #cls = self.cls_token + glob_emb          
        cls = self.cls_token.expand(x.size(0), -1, -1)                  
        cls_attn = attn_mask.new_ones((x.size(0), 1))
        x = torch.cat((cls, x), dim=1)
        attn_mask = torch.cat([cls_attn, attn_mask], dim=1)

        # run transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.norm(x)
        return x, idx, orig_attn_mask, rand_mask, ids_restore, intra_idx, module_id

    
    def forward_decoder(self, x, idx, attn_mask, rand_mask, ids_restore, idx_map, intra_idx, module_id):
        assert intra_idx.shape[:2] == ids_restore.shape[:2], \
            "Decoder intra_idx must match restored token grid"

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        B, L, C = x.shape
        num_mask = ids_restore.shape[1] + 1 - L
        mask_tokens = self.mask_token.repeat(B, num_mask, 1)
        x_ = torch.cat((x[:, 1:, :], mask_tokens), dim=1)
        x_ = torch.gather(
            x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, C)
        )

        # add positional embeddings
        dec_pos = self.decoder_intra_pos_embed(intra_idx) + self.module_embed_dec(module_id)
        x_ = x_ + dec_pos

        # add back CLS
        x = torch.cat([x[:, :1, :], x_], dim=1)
        cls_attn = attn_mask.new_ones((x.size(0), 1))
        attn_mask = torch.cat([cls_attn, attn_mask], dim=1)

        # run transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask=attn_mask)
        x = x[:, 1:]

        attn_mask       = attn_mask[:, 1:]
        prediction_mask = rand_mask.bool() & attn_mask
        flat_out        = x[prediction_mask]

        pred_occ, pred_reg, pred_cls = self.shared_voxel_head(flat_out)

        event_ids, slot_ids = torch.nonzero(prediction_mask, as_tuple=True)
        patch_ids = idx[event_ids, slot_ids]
        idx_targets = idx_map[event_ids, patch_ids]

        return pred_occ, pred_reg, pred_cls, idx_targets, event_ids, patch_ids
        

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
        latent, idx, attn_mask, rand_mask, ids_restore, intra_idx, module_id = self.forward_encoder(x, x_glob, mask_ratio)
        pred_occ, pred_reg, pred_cls, idx_targets, row_event_ids, row_patch_ids = \
            self.forward_decoder(latent, idx, attn_mask, rand_mask, ids_restore, idx_map, intra_idx, module_id)

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
        embed_dim=384, patch_size=(48, 48, 2),
        depth=12, num_heads=12,
        decoder_embed_dim=288, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model


def mae_vit_base(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        embed_dim=768, patch_size=(48, 48, 2),
        depth=12, num_heads=12,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model
    

def mae_vit_large(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        embed_dim=1008, patch_size=(48, 48, 2),
        depth=24, num_heads=16,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model


def mae_vit_huge(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        embed_dim=1296, patch_size=(48, 48, 2),
        depth=32, num_heads=16,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model
