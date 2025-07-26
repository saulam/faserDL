"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 07.25

Description: PyTorch ViT model with MinkowskiEngine patching.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import timm.models.vision_transformer as vit
from functools import partial
from torch.nn.utils.rnn import pad_sequence 
from timm.models.layers import trunc_normal_
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiGELU,
)
from .utils import BlockWithMask, get_3d_sincos_pos_embed, GlobalFeatureEncoder, MinkowskiLayerNorm


class MinkViT(vit.VisionTransformer):
    """ 
    Vision Transformer with MinkowskiEngine patching 
    and support for global average pooling
    """
    def __init__(
        self,
        D=3,
        img_size=(48, 48, 200),
        encoder_dims=[192, 256, 384],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        global_pool=False,
        **kwargs
    ):
        super(MinkViT, self).__init__(**kwargs)
        
        norm_layer = kwargs['norm_layer']
        in_chans = kwargs['in_chans']
        drop_rate = kwargs['drop_rate']

        # patch & grid setup
        patch_size = np.prod(np.array(kernel_size), axis=0).tolist()
        H, W, D_img = img_size
        p_h, p_w, p_d = patch_size
    
        assert H % p_h == 0 and W % p_w == 0 and D_img % p_d == 0, \
            "img_size must be divisible by patch_size"

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
            _down_blk(in_chans, encoder_dims[0], kernel_size[0]),
            _down_blk(encoder_dims[0], encoder_dims[1], kernel_size[1]),
            _down_blk(encoder_dims[1], encoder_dims[2], kernel_size[2]),
        )

        embed_dim = encoder_dims[-1]
        del self.pos_embed, self.patch_embed, self.norm_pre, self.fc_norm, self.head
        self.global_feats_encoder = GlobalFeatureEncoder(embed_dim)
        self.pos_embed = nn.Embedding(self.num_patches, embed_dim)

        self.global_pool = global_pool
        if self.global_pool:
            embed_dim = kwargs['embed_dim']
            del self.norm  # remove the original norm

        self.branch_tokens = {
            "flavour": 0,
            "charm": 0,
            "e_vis": 0,
            "pt_miss": 0,
            "lepton_momentum_mag": 0,
            "lepton_momentum_dir": 0,
            "jet_momentum_mag": 0,
            "jet_momentum_dir": 0,
        }
        branch_out_channels = {
            "flavour": 4,
            "charm": 1,
            "e_vis": 1,
            "pt_miss": 1,
            "lepton_momentum_mag": 1,
            "lepton_momentum_dir": 3,
            "jet_momentum_mag": 1,
            "jet_momentum_dir": 3,
        }
        self.num_cls = 1 if global_pool else max(self.branch_tokens.values()) + 1
        self.cls_tokens = nn.Parameter(torch.zeros(1, self.num_cls, embed_dim))
        self.heads = nn.ModuleDict()
        for name in self.branch_tokens.keys():
            self.heads[name] = nn.Sequential(
                norm_layer(embed_dim),
                nn.Dropout(drop_rate),
                nn.Linear(embed_dim, branch_out_channels[name])
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

        # init cls tokens
        nn.init.normal_(self.cls_tokens, std=0.02)

        self.apply(self._init_weights)

        # init heads
        for name in self.branch_tokens.keys():
            lin = self.heads[name][2]
            trunc_normal_(lin.weight, std=2e-5)
            if lin.bias is not None:
                nn.init.constant_(lin.bias, 0)

    def _init_weights(self, m):
        # we use xavier_uniform following official JAX ViT:
        if isinstance(m, MinkowskiConvolution):
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
        Buckets feats by event ID, pads to a dense tensor.
    
        Args:
            sparse_tensor: MinkowskiEngine sparse tensor with
                           .C of shape [N, 4] (coords, with C[:,0]=event_id)
                           .F of shape [N, C] (features).
    
        Returns:
            padded_feats:  [B, L_max, C] float, zero‑padded features
            padded_coords: [B, L_max] int, flat position ids.
            mask:          [B, L_max] bool, True = real voxel.
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
        counts_list = counts.tolist()    
        feat_groups   = torch.split(sorted_feats, counts_list, dim=0)
        coord_groups  = torch.split(sorted_spatial_coords, counts_list, dim=0)
        
        padded_feats  = pad_sequence(feat_groups, batch_first=True, padding_value=0.0)
        padded_coords = pad_sequence(coord_groups, batch_first=True, padding_value=0)
    
        lengths = counts
        L_max   = int(lengths.max().item())
        arange  = torch.arange(L_max, device=feats.device)
        mask    = arange.unsqueeze(0) < lengths.unsqueeze(1)

        Gh, Gw, Gd = self.grid_size
        h_idx = padded_coords[..., 0]
        w_idx = padded_coords[..., 1]
        d_idx = padded_coords[..., 2]
        padded_coords = h_idx * (Gw * Gd) + w_idx * Gd + d_idx
    
        return padded_feats, padded_coords, mask
        

    def forward_features(self, x_sparse, glob):
        x_sparse = self.downsample_layers(x_sparse)
        x, pos, attn_mask = self.group_voxels_by_event(x_sparse)
        x = x + self.pos_embed(pos)

        # add cls token
        glob_emb = self.global_feats_encoder(glob).unsqueeze(1)  # (B, 1, D)
        cls = self.cls_tokens + glob_emb                            
        cls_attn = attn_mask.new_ones((x.size(0), self.num_cls))
        x = torch.cat((cls, x), dim=1)
        attn_mask = torch.cat([cls_attn, attn_mask], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        if self.global_pool:
            outcome = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            x = self.norm(x)
            outcome = x[:, :self.num_cls]

        return outcome

    
    def forward_head(self, x):
        outputs = {}
        for i, (name, head) in enumerate(self.heads.items()):
            if self.global_pool:
                output = head(x)
            else:
                output = head(x[:, self.branch_tokens[name]])
            if name == "flavour" or name == "charm":
                outputs[f"out_{name}"] = output
            elif "_dir" in name:
                outputs[f"out_{name}"] = F.normalize(output, dim=-1)
            else:
                outputs[f"out_{name}"] = F.softplus(output)
        return outputs
        
        
    def forward(self, x, x_glob):
        x = self.forward_features(x, x_glob)
        x = self.forward_head(x)
        return x

        
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


def vit_base(**kwargs):
    model = MinkViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        encoder_dims=[192, 384, 768],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        embed_dim=768, depth=12, num_heads=12, drop_rate=0.,
        mlp_ratio=4.0, qkv_bias=True, 
        block_fn=BlockWithMask, drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    

def vit_large(**kwargs):
    model = MinkViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        encoder_dims=[252, 504, 1008],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        embed_dim=1008, depth=24, num_heads=16, drop_rate=0.,
        mlp_ratio=4.0, qkv_bias=True, 
        block_fn=BlockWithMask, drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(**kwargs):
    model = MinkViT(
        in_chans=1, D=3, img_size=(48, 48, 200),
        encoder_dims=[324, 648, 1296],
        kernel_size=[(4, 4, 5), (2, 2, 2), (2, 2, 1)],
        embed_dim=1296, depth=32, num_heads=16, drop_rate=0.,
        mlp_ratio=4.0, qkv_bias=True,
        block_fn=BlockWithMask, drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
