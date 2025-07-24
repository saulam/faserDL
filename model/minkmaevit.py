"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 07.25

Description: PyTorch model - stage 1: pretraining.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from timm.models.vision_transformer import Attention, Block

import MinkowskiEngine as ME
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiGELU,
    MinkowskiConvolutionTranspose,
    MinkowskiLinear,
    MinkowskiGlobalAvgPooling,
    MinkowskiGlobalMaxPooling,
    MinkowskiReLU,
    MinkowskiGELU,
)

from .utils import get_3d_sincos_pos_embed, GlobalFeatureEncoder, MinkowskiLayerNorm


# NOTE: patch works for timm version 0.6.13
class MaskableAttention(Attention):
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            # if key padding mask (B, N) → expand to (B, 1, 1, N)
            if attn_mask.ndim == 2:
                mask = attn_mask[:, None, None, :].to(torch.bool)
            # if full mask (B, N, N) → expand to (B, 1, N, N)
            else:
                mask = attn_mask[:, None, :, :].to(torch.bool)
            attn = attn.masked_fill(~mask, float("-1e9"))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithMask(Block):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.attn = MaskableAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
        

class MinkMAEViT(nn.Module):
    def __init__(
        self,
        in_channels=1, 
        out_channels=4, 
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
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        loss_weights={'reg': 1.0, 'cls': 1.0},
        args=None
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
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
                    bias=True, dimension=D
                ),
                MinkowskiLayerNorm(out_c, eps=1e-6),
                MinkowskiGELU(),
            )
    
        self.downsample_layers = nn.Sequential(
            _down_blk(in_channels, encoder_dims[0], kernel_size[0]),
            _down_blk(encoder_dims[0], encoder_dims[1], kernel_size[1]),
            _down_blk(encoder_dims[1], encoder_dims[2], kernel_size[2]),
        )
    
        embed_dim = encoder_dims[-1]
    
        # MAE encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.global_feats_encoder = GlobalFeatureEncoder(embed_dim)
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
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.final_embed = nn.Linear(decoder_embed_dim, embed_dim) 
    
        # upsample blocks
        def _up_blk(in_c, out_c, ks):
            return nn.Sequential(
                MinkowskiConvolutionTranspose(
                    in_c, out_c, kernel_size=ks, stride=ks,
                    bias=True, dimension=D
                ),
                MinkowskiLayerNorm(out_c, eps=1e-6),
                MinkowskiGELU(),
            )

        up_out_dim = encoder_dims[0]//2
        self.upsample_layers = nn.Sequential(
            _up_blk(encoder_dims[2], encoder_dims[1], kernel_size[2]),
            _up_blk(encoder_dims[1], encoder_dims[0], kernel_size[1]),
            _up_blk(encoder_dims[0], up_out_dim, kernel_size[0]),
        )
    
        # heads
        self.reg_head = MinkowskiConvolution(
            up_out_dim, in_channels, kernel_size=1,
            stride=1, bias=True, dimension=D
        )
        self.cls_head = MinkowskiConvolution(
            up_out_dim, out_channels, kernel_size=1,
            stride=1, bias=True, dimension=D
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
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        # we use xavier_uniform following official JAX ViT:
        if isinstance(m, MinkowskiConvolution) or isinstance(m, MinkowskiConvolutionTranspose):
            torch.nn.init.xavier_uniform_(m.kernel)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            w = m.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias_ih' in name:
                    # zero then set forget gate bias
                    param.data.zero_()
                    hidden_size = m.hidden_size
                    param.data[hidden_size:2*hidden_size].fill_(1)
                elif 'bias_hh' in name:
                    param.data.zero_()

    
    def group_voxels_by_event(
        self,
        sparse_tensor,
    ):
        """
        Buckets feats by event ID, pads to a dense tensor,
        and returns an index‐map you can use to scatter back.
    
        Args:
            sparse_tensor: MinkowskiEngine sparse tensor with
                           .C of shape [N, 4] (coords, with C[:,0]=event_id)
                           .F of shape [N, C] (features).
    
        Returns:
            padded_feats: [B, L_max, C] float, zero‑padded features
            mask:         [B, L_max] bool, True = real voxel
            lengths:      [B] long, number of voxels per event
            idx_map:      [B, L_max] long, original-voxel indices (−1=pad)
        """
        coords = sparse_tensor.C
        feats  = sparse_tensor.F
        N, C  = feats.shape
    
        event_ids             = coords[:, 0].long()
        spatial_coords        = coords[:, 1:] // self.patch_size
        sorted_eids, perm     = event_ids.sort()
        sorted_feats          = feats[perm]
        sorted_spatial_coords = spatial_coords[perm]
        sorted_idx            = perm  # remembers original row indices
    
        uniq_ids, counts = torch.unique_consecutive(sorted_eids,
                                                    return_counts=True)
        counts_list = counts.tolist()    
        feat_groups   = torch.split(sorted_feats, counts_list, dim=0)
        coord_groups  = torch.split(sorted_spatial_coords, counts_list, dim=0)
        idx_groups    = torch.split(sorted_idx, counts_list, dim=0)
        
        padded_feats  = pad_sequence(feat_groups, batch_first=True, padding_value=0.0)
        padded_coords = pad_sequence(coord_groups, batch_first=True, padding_value=0)
        idx_map       = pad_sequence(idx_groups, batch_first=True, padding_value=-1)
    
        lengths = counts
        L_max   = int(lengths.max().item())
        arange  = torch.arange(L_max, device=feats.device)
        mask    = arange.unsqueeze(0) < lengths.unsqueeze(1)

        Gh, Gw, Gd = self.grid_size
        h_idx = padded_coords[..., 0]
        w_idx = padded_coords[..., 1]
        d_idx = padded_coords[..., 2]
        padded_coords = h_idx * (Gw * Gd) + w_idx * Gd + d_idx
    
        return padded_feats, padded_coords, mask, lengths, idx_map

    
    def random_masking(self, x, attn_mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, L, D], sequence
        attn_mask: [B, L]

        Source: https://github.com/facebookresearch/mae/blob/main/models_mae.py
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        attn_mask_masked = torch.gather(attn_mask, 1, ids_keep)
        rand_mask = torch.ones(B, L, device=x.device)
        rand_mask[:, :len_keep] = 0
        rand_mask = torch.gather(rand_mask, 1, ids_restore)
        return x_masked, attn_mask_masked, rand_mask, ids_restore


    def _patch_key(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute a 1‑D patch key: batch_index * num_patches + flat_patch_index.
        """
        batch_idx = coords[:, 0]
        spatial = coords[:, 1:] // self.patch_size
        h, w, d = spatial.unbind(-1)
        flat = (
            h * (self.grid_size[1] * self.grid_size[2])
            + w * self.grid_size[2]
            + d
        )
        return batch_idx * self.num_patches + flat

    
    def forward_encoder(self, x_sparse, glob, mask_ratio):
        # patchify and mask
        x_sparse = self.downsample_layers(x_sparse)
        x, pos, orig_attn_mask, lengths, idx_map = self.group_voxels_by_event(x_sparse)
        x = x + self.pos_embed(pos)
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
        return x, x_sparse, pos, orig_attn_mask, rand_mask, idx_map, ids_restore

    
    def forward_decoder(self, x, x_sparse, pos, attn_mask, rand_mask, idx_map, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        B, L, D = x.shape
        num_mask = ids_restore.shape[1] + 1 - L
        mask_tokens = self.mask_token.repeat(B, num_mask, 1)
        x_ = torch.cat((x[:, 1:, :], mask_tokens), dim=1)
        x_ = torch.gather(
            x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D)
        )
        x_ = x_ + self.decoder_pos_embed(pos)                 # add pos emb
        x = torch.cat([x[:, :1, :], x_], dim=1)               # add cls token
        cls_attn = attn_mask.new_ones((x.size(0), 1))
        attn_mask = torch.cat([cls_attn, attn_mask], dim=1)

        # run transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.decoder_norm(x)[:, 1:]

        # keep valid voxels
        attn_mask = attn_mask[:, 1:]
        keep = rand_mask.bool() & attn_mask
        flat_out  = x[keep]
        flat_idx  = idx_map[keep]

        # embed back to original dimension
        out_feats = self.final_embed(flat_out).to(x_sparse.F.dtype)

        # embeddings of masked voxels to sparse tensor
        new_F = x_sparse.F.clone()
        new_F[flat_idx] = out_feats
        x_sparse = ME.SparseTensor(
            features=new_F,
            coordinate_manager=x_sparse.coordinate_manager,
            coordinate_map_key=x_sparse.coordinate_map_key,
        )

        # upsample and get predictions
        x_sparse = self.upsample_layers(x_sparse)
        pred_reg = self.reg_head(x_sparse)
        pred_cls = self.cls_head(x_sparse)
        preds_C  = pred_reg.C

        # build a mask in the upsampled lattice
        masked_coords      = x_sparse.C[flat_idx]
        key_masked         = self._patch_key(masked_coords)
        key_pred           = self._patch_key(preds_C)
        mask_up            = torch.isin(key_pred, key_masked.unique())
    
        return pred_reg, pred_cls, mask_up


    def forward_loss(self, targ_reg, targ_cls, pred_reg, pred_cls, mask):
        targ_reg_masked = targ_reg[mask]
        targ_cls_masked = targ_cls[mask]
        pred_reg_masked = pred_reg[mask]
        pred_cls_masked = pred_cls[mask]

        total_loss = 0.
        losses = {}
        
        loss_reg = F.mse_loss(pred_reg_masked, targ_reg_masked, reduction='mean')
        losses['reg'] = loss_reg
        total_loss += loss_reg * self.loss_weights['reg']
        
        loss_cls = F.cross_entropy(pred_cls_masked, targ_cls_masked, reduction='mean')
        losses['cls'] = loss_cls
        total_loss += loss_cls * self.loss_weights['cls']

        return total_loss, losses
        

    def forward(self, x, x_glob, cls_labels, mask_ratio=0.5):
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
        latent, x_sparse, pos, attn_mask, rand_mask, idx_map, ids_restore = self.forward_encoder(x, x_glob, mask_ratio)
        preds_reg, preds_cls, mask_up = self.forward_decoder(latent, x_sparse, pos, attn_mask, rand_mask, idx_map, ids_restore)
        total_loss, individual_losses = self.forward_loss(x.F, cls_labels, preds_reg.F, preds_cls.F, mask_up)
        
        return total_loss, individual_losses, preds_reg, preds_cls, mask_up

        
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
        in_channels=1, out_channels=4, D=3, img_size=(48, 48, 200),
        encoder_dims=[192, 384, 768],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        depth=12, num_heads=12,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        loss_weights={'reg': 1.0, 'cls': 1.0}, **kwargs)
    return model
    

def mae_vit_large(**kwargs):
    model = MinkMAEViT(
        in_channels=1, out_channels=4, D=3, img_size=(48, 48, 200),
        encoder_dims=[252, 504, 1008],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        depth=24, num_heads=16,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        loss_weights={'reg': 1.0, 'cls': 1.0}, **kwargs)
    return model


def mae_vit_huge(**kwargs):
    model = MinkMAEViT(
        in_channels=1, out_channels=4, D=3, img_size=(48, 48, 200),
        encoder_dims=[324, 648, 1296],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        depth=32, num_heads=16,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        loss_weights={'reg': 1.0, 'cls': 1.0}, **kwargs)
    return model
