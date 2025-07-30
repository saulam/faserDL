"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 07.25

Description: PyTorch MAE-ViT model with MinkowskiEngine patching.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiGELU,
)
from .utils import BlockWithMask, get_3d_sincos_pos_embed, GlobalFeatureEncoderSimple, MinkowskiLayerNorm
        

class MinkMAEViT(nn.Module):
    def __init__(
        self,
        in_chans=1, 
        out_chans=4, 
        D=3,
        img_size=(48, 48, 200),
        encoder_dims=[192, 256, 384],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        depth=24,
        num_heads=16,
        decoder_embed_dim=192,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        loss_weights={'occ': 1.0, 'reg': 1.0, 'cls': 1.0},
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
        patch_size = np.prod(np.array(kernel_size), axis=0).tolist()
        H, W, D_img = img_size
        p_h, p_w, p_d = patch_size
    
        assert H % p_h == 0 and W % p_w == 0 and D_img % p_d == 0, \
            "img_size must be divisible by patch_size"

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.loss_weights = loss_weights
        self.grid_size = (H // p_h, W // p_w, D_img // p_d)
        self.num_patches = (
            self.grid_size[0] 
            * self.grid_size[1] 
            * self.grid_size[2]
        )
        self.patch_voxels = p_h * p_w * p_d
        self.register_buffer('patch_size', torch.tensor(patch_size))
    
        # downsample blocks
        def _down_blk(in_c, out_c, ks):
            return nn.Sequential(
                MinkowskiConvolution(
                    in_c, out_c, kernel_size=ks, stride=ks, 
                    bias=True, dimension=D,
                ),
                MinkowskiLayerNorm(out_c, eps=1e-6),
                MinkowskiGELU(),
            )

        self.downsample_layers = nn.Sequential(
            _down_blk(in_chans, encoder_dims[0], kernel_size[0]),
            _down_blk(encoder_dims[0], encoder_dims[1], kernel_size[1]),
            _down_blk(encoder_dims[1], encoder_dims[2], kernel_size[2]),
        )
    
        # MAE encoder
        embed_dim = encoder_dims[-1]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_feats_encoder = GlobalFeatureEncoderSimple(embed_dim)
        self.pos_embed = nn.Embedding(self.num_patches, embed_dim)
        self.blocks = nn.ModuleList([
            BlockWithMask(
                embed_dim, num_heads, mlp_ratio,
                qkv_bias=True, norm_layer=norm_layer
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
    
        # MAE decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Embedding(self.num_patches, decoder_embed_dim)
        self.decoder_blocks = nn.ModuleList([
            BlockWithMask(
                decoder_embed_dim, decoder_num_heads, mlp_ratio,
                qkv_bias=True, norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_pred_occ = nn.Sequential(
            norm_layer(decoder_embed_dim),
            nn.Linear(decoder_embed_dim, self.patch_voxels, bias=True),
        )
        self.decoder_pred_reg = nn.Sequential(
            norm_layer(decoder_embed_dim),
            nn.Linear(decoder_embed_dim, self.patch_voxels * in_chans, bias=True),
        )
        self.decoder_pred_cls = nn.Sequential(
            norm_layer(decoder_embed_dim),
            nn.Linear(decoder_embed_dim, self.patch_voxels * out_chans, bias=True),
        )
    
        self.initialize_weights()


    def initialize_weights(self):
        # init fixed pos embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.weight.shape[-1], self.grid_size, cls_token=False
        )
        with torch.no_grad():
            self.pos_embed.weight.copy_(torch.from_numpy(pos_embed).float())
            self.pos_embed.weight.requires_grad_(False)
        
        decoder_pos = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.weight.shape[-1], self.grid_size, cls_token=False
        )
        with torch.no_grad():
            self.decoder_pos_embed.weight.copy_(torch.from_numpy(decoder_pos).float())
            self.decoder_pos_embed.weight.requires_grad_(False)

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
            padded_coords: [B, L_max] int, flat position ids.
            attn_mask:     [B, L_max] bool, True = real voxel.
        """
        coords = sparse_tensor.C
        feats  = sparse_tensor.F
        N, C  = feats.shape
    
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
    
        return padded_feats, padded_idx, attn_mask


    def build_patch_occupancy_map(self, x):
        """
        From the original sparse tensor coordinates, build a [B, N_patches, P] mapping
        with the raw id of the actual hit in that sub‐voxel.
        """
        coords     = x.C
        device     = coords.device
        event_ids  = coords[:, 0].long()
        x_, y_, z_ = coords[:, 1:].unbind(-1)
    
        p_h, p_w, p_d        = self.patch_size.tolist()
        G_h, G_w, G_d        = self.grid_size
        P                    = self.patch_voxels
        Np                   = self.num_patches
        B                    = int(event_ids.max().item()) + 1
        N                    = coords.shape[0]
    
        patch_idx = (x_ // p_h) * (G_w * G_d) \
                  + (y_ // p_w) * G_d \
                  + (z_ // p_d)
        sub_idx   = (x_ % p_h) * (p_w * p_d) \
                  + (y_ % p_w) * p_d \
                  + (z_ % p_d)

        raw_inds = torch.arange(N, device=device)
        idx_map = torch.full ((B, Np, P), -1, dtype=torch.long, device=device)
        idx_map [event_ids, patch_idx, sub_idx] = raw_inds
    
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
        # patchify and mask
        x_sparse = self.downsample_layers(x_sparse)
        x, idx, orig_attn_mask = self.group_patches_by_event(x_sparse)
        x = x + self.pos_embed(idx)
        x, attn_mask, rand_mask, ids_restore = self.random_masking(
            x, orig_attn_mask, mask_ratio
        )

        # add cls token
        glob_emb = self.global_feats_encoder(glob).unsqueeze(1)  # (B, 1, D)
        cls = self.cls_token + glob_emb                            
        cls_attn = attn_mask.new_ones((x.size(0), 1))
        x = torch.cat((cls, x), dim=1)
        attn_mask = torch.cat([cls_attn, attn_mask], dim=1)

        # run transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.norm(x)
        return x, x_sparse, idx, orig_attn_mask, rand_mask, ids_restore

    
    def forward_decoder(self, x, idx, attn_mask, rand_mask, ids_restore, idx_map):
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
        x_ = x_ + self.decoder_pos_embed(idx)                 # add pos emb
        x = torch.cat([x[:, :1, :], x_], dim=1)               # add cls token
        cls_attn = attn_mask.new_ones((x.size(0), 1))
        attn_mask = torch.cat([cls_attn, attn_mask], dim=1)

        # run transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask=attn_mask)
        x = x[:, 1:]

        attn_mask       = attn_mask[:, 1:]
        prediction_mask = rand_mask.bool() & attn_mask
        flat_out        = x[prediction_mask]

        pred_occ = self.decoder_pred_occ(flat_out)
        pred_reg = self.decoder_pred_reg(flat_out)
        pred_cls = self.decoder_pred_cls(flat_out)

        event_ids, slot_ids = torch.nonzero(prediction_mask, as_tuple=True)
        patch_ids = idx[event_ids, slot_ids]
        idx_targets = idx_map[event_ids, patch_ids]

        return pred_occ, pred_reg, pred_cls, idx_targets


    def forward_loss(self, targ_reg, targ_cls, pred_occ, pred_reg, pred_cls, idx_targets, smooth=0.1):
        mask_targets = (idx_targets >= 0)
        mask_flat    = mask_targets.view(-1)
        idx_flat     = idx_targets.view(-1)[mask_flat]

        targ_occ = mask_targets.float()
        if self.training and smooth > 0:
            targ_occ = targ_occ * (1.0 - smooth) + 0.5 * smooth
        loss_occ = F.binary_cross_entropy_with_logits(pred_occ, targ_occ)

        pred_reg   = pred_reg.view(-1, self.in_chans)[mask_flat]
        targ_reg   = targ_reg[idx_flat]
        loss_reg   = F.mse_loss(pred_reg, targ_reg)

        pred_cls   = pred_cls.view(-1, self.out_chans)[mask_flat]
        targ_cls   = targ_cls[idx_flat]
        loss_cls   = F.cross_entropy(pred_cls, targ_cls)

        total_loss = (
            self.loss_weights['occ'] * loss_occ +
            self.loss_weights['reg'] * loss_reg +
            self.loss_weights['cls'] * loss_cls
        )
        return total_loss, dict(occ=loss_occ, reg=loss_reg, cls=loss_cls)
        

    def forward(self, x, x_glob, cls_labels, mask_ratio=0.75):
        """
        Forward pass through the encoder-decoder network.

        Args:
            x: Input sparse tensor.
            x_glob: Global feature tensors.
            cls_labels: Labels for classification task.
            mask_ratio: mask probability.

        Returns:
            A dictionary with voxel predictions.
        """
        idx_map = self.build_patch_occupancy_map(x)
        latent, x_sparse, idx, attn_mask, rand_mask, ids_restore = self.forward_encoder(x, x_glob, mask_ratio)
        pred_occ, pred_reg, pred_cls, idx_targets = self.forward_decoder(latent, idx, attn_mask, rand_mask, ids_restore, idx_map)
        total_loss, individual_losses = self.forward_loss(x.F, cls_labels, pred_occ, pred_reg, pred_cls, idx_targets)
        
        return total_loss, individual_losses, pred_occ, pred_reg, pred_cls, idx_targets

        
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


def mae_vit_base(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        encoder_dims=[192, 384, 768],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        depth=12, num_heads=12,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        loss_weights={'occ': 1.0, 'reg': 1.0, 'cls': 1.0}, **kwargs)
    return model
    

def mae_vit_large(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        encoder_dims=[252, 504, 1008],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        depth=24, num_heads=16,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        loss_weights={'occ': 1.0, 'reg': 1.0, 'cls': 1.0}, **kwargs)
    return model


def mae_vit_huge(**kwargs):
    model = MinkMAEViT(
        in_chans=1, out_chans=4, D=3, img_size=(48, 48, 200),
        encoder_dims=[324, 648, 1296],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        depth=32, num_heads=16,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        loss_weights={'occ': 1.0, 'reg': 1.0, 'cls': 1.0}, **kwargs)
    return model
