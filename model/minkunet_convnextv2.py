"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch model - stage 1: semantic segmentation.
"""


import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .utils import (
    Block,
    LayerNorm,
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath
)

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGlobalMaxPooling,
    MinkowskiReLU,
    MinkowskiGELU,
)


# Custom weight initialization function
def _init_weights(m):
    if isinstance(m, MinkowskiConvolution):
        trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiConvolutionTranspose):
        trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiDepthwiseConvolution):
        trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiLinear):
        trunc_normal_(m.linear.weight, std=.02)
        if m.linear.bias is not None:
            nn.init.constant_(m.linear.bias, 0)
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MinkUNetConvNeXtV2(nn.Module):
    def __init__(self, in_channels, out_channels, D=3, args=None):
        nn.Module.__init__(self)
        self.is_v5 = True if 'v5' in args.dataset_path else False 

        """Encoder"""
        #depths=[2, 4, 4, 8, 8, 8]
        #dims = (16, 32, 64, 128, 256, 512)
        depths=[3, 3, 9, 3]
        dims=[96, 192, 384, 768]     
        #dims = (64, 128, 256, 512, 512)
        #depths = (3, 3, 3, 9, 3)
        #kernel_size = (2, 2, 3)
        kernel_size = 2
        drop_path_rate=0.

        assert len(depths) == len(dims)

        self.nb_elayers = len(dims)

        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # Linear transformation for glibal features
        self.global_mlp = nn.Sequential(
            nn.Linear(1 + 1 + 1 + 1 + 9 + (10 if self.is_v5 else 15), dims[0]),
        )

        # stem
        self.stem = MinkowskiConvolution(in_channels, dims[0], kernel_size=1, stride=1, dimension=D)
        self.stem_ln = MinkowskiLayerNorm(dims[0], eps=1e-6)

        for i in range(self.nb_elayers):
            encoder_layer = nn.Sequential(
                *[Block(dim=dims[i], kernel_size=5, drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.encoder_layers.append(encoder_layer)
            cur += depths[i]

            if i < self.nb_elayers - 1:  
                downsample_layer = nn.Sequential(
                    MinkowskiLayerNorm(dims[i], eps=1e-6),                
                    MinkowskiConvolution(dims[i], dims[i+1], kernel_size=kernel_size, stride=kernel_size, bias=True, dimension=D),
                )
                self.downsample_layers.append(downsample_layer)

        """Decoder"""
        last_enc_depth = depths[-1]
        depths = [2] * (len(depths) + 1)
        dims = dims[::-1]
        decoder_embed_dim = 32

        self.nb_dlayers = len(dims) - 1

        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(dp_rates[-last_enc_depth], 0, sum(depths))]
        cur = 0

        for i in range(self.nb_dlayers):
            upsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6), 
                MinkowskiConvolutionTranspose(dims[i], dims[i+1], kernel_size=kernel_size, stride=kernel_size, bias=True, dimension=D),
            )
            self.upsample_layers.append(upsample_layer)

            decoder_layer = nn.Sequential(
                *[Block(dim=dims[i+1], kernel_size=5, drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.decoder_layers.append(decoder_layer)
            cur += depths[i] 

        """Semantic-segmentation layers"""
        self.primlepton_layer = nn.Sequential(
            MinkowskiLayerNorm(dims[-1], eps=1e-6),
            Block(dim=dims[-1], kernel_size=5, drop_path=0., D=D),
            MinkowskiConvolution(dims[-1], 1, kernel_size=1, stride=1, dimension=D),
        )
        self.seg_layer = nn.Sequential(
            MinkowskiLayerNorm(dims[-1], eps=1e-6),
            Block(dim=dims[-1], kernel_size=5, drop_path=0., D=D),
            MinkowskiConvolution(dims[-1], 3, kernel_size=1, stride=1, dimension=D),
        )

        """ Initialise weights """
        self.apply(_init_weights)

    
    def print_tensor(self, x, label=""):
        print(label + " "*(20-len(label)),
              np.array(x.shape),
              x.coordinates.min(dim=0)[0].detach().cpu().numpy(),
              x.coordinates.max(dim=0)[0].detach().cpu().numpy())
        print("\tx:", torch.unique(x.coordinates[:, 1]).detach().cpu().numpy())
        print("\ty:", torch.unique(x.coordinates[:, 2]).detach().cpu().numpy())
        print("\tz:", torch.unique(x.coordinates[:, 3]).detach().cpu().numpy())
        print("\n")


    def forward(self, x, x_glob):
        """Encoder"""
        # stem
        x = self.stem(x)
        x_glob = self.global_mlp(x_glob)

        # add global to voxel features
        batch_indices = x.C[:, 0].long()  # batch idx
        x_glob_expanded = x_glob[batch_indices]
        new_feats = x.F + x_glob_expanded
        x = ME.SparseTensor(features=new_feats, coordinates=x.C)

        # layer norm
        x = self.stem_ln(x)

        # encoder layers
        x_enc = []
        for i in range(self.nb_elayers):
            x = self.encoder_layers[i](x)
            if i < self.nb_elayers - 1:
                x_enc.append(x)
                x = self.downsample_layers[i](x)

        """Decoder"""
        x_enc = x_enc[::-1]
        
        # decoder layers
        out_cls = []
        for i in range(self.nb_dlayers):
            x = self.upsample_layers[i](x)
            x = x + x_enc[i]
            x = self.decoder_layers[i](x)

        # voxel predictions
        out_primlepton = self.primlepton_layer(x)
        out_seg = self.seg_layer(x)

        output = {"out_primlepton": out_primlepton,
                  "out_seg": out_seg,
                  }
        
        return output

    def replace_depthwise_with_channelwise(self):
        for name, module in self.named_modules():
            if isinstance(module, ME.MinkowskiDepthwiseConvolution):
                # Get the parameters of the current depthwise convolution
                in_channels = module.in_channels
                kernel_size = module.kernel_generator.kernel_size
                stride = module.kernel_generator.kernel_stride
                dilation = module.kernel_generator.kernel_dilation
                bias = module.bias is not None
                dimension = module.dimension
                
                # Create a new MinkowskiChannelwiseConvolution with the same parameters
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
                
                # Replace the old depthwise convolution with the new channelwise convolution
                parent_module, attr_name = self._get_parent_module(name)
                setattr(parent_module, attr_name, new_conv)
        
        return
    
    def _get_parent_module(self, layer_name):
        components = layer_name.split('.')
        parent = self
        for comp in components[:-1]:
            parent = getattr(parent, comp)
        return parent, components[-1]

