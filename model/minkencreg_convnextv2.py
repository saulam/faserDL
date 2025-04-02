"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch model - stage 2: regression tasks.
"""


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
    MinkowskiLeakyReLU,
    MinkowskiSoftplus,
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


class MinkEncRegConvNeXtV2(nn.Module):
    def __init__(self, in_channels, out_channels, D=3, args=None):
        nn.Module.__init__(self)
        self.is_v5 = True if 'v5' in args.dataset_path else False 

        """Encoder"""
        #depths=[2, 4, 4, 8, 8, 8]
        #dims = (16, 32, 64, 128, 256, 512)
        depths=[3, 3, 9, 3]
        dims=[96, 192, 384, 768]     
        kernel_size = 5
        drop_path_rate=0.

        assert len(depths) == len(dims)

        self.nb_elayers = len(dims)

        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # stem
        self.stem = MinkowskiConvolution(in_channels, dims[0], kernel_size=1, stride=1, dimension=D)
        self.stem_ln = MinkowskiLayerNorm(dims[0], eps=1e-6)

        # Linear transformation for glibal features
        self.global_mlp = nn.Sequential(
            nn.Linear(1 + 1 + 1 + 1 + 9 + (10 if self.is_v5 else 15), dims[0]),
        )

        for i in range(self.nb_elayers):
            encoder_layer = nn.Sequential(
                *[Block(dim=dims[i], kernel_size=kernel_size, drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.encoder_layers.append(encoder_layer)
            cur += depths[i]

            if i < self.nb_elayers - 1:  
                downsample_layer = nn.Sequential(
                    MinkowskiLayerNorm(dims[i], eps=1e-6),                
                    MinkowskiConvolution(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True, dimension=D),
                )
                self.downsample_layers.append(downsample_layer)

        """Classification/regression layers"""
        self.global_pool = MinkowskiGlobalMaxPooling()
        self.e_vis_head = nn.Sequential(
            MinkowskiLinear(dims[-1], dims[-1]//2),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//2, dims[-1]//4),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//4, 1),
            MinkowskiSoftplus(beta=1, threshold=20),
        ) 
        self.pt_miss_head = nn.Sequential(
            MinkowskiLinear(dims[-1], dims[-1]//2),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//2, dims[-1]//4),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//4, 1),
            MinkowskiSoftplus(beta=1, threshold=20),
        )
        self.out_lepton_momentum_mag_head = nn.Sequential(
            MinkowskiLinear(dims[-1], dims[-1]//2),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//2, dims[-1]//4),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//4, 1),
            MinkowskiSoftplus(beta=1, threshold=20),
        )
        self.out_lepton_momentum_dir_head = nn.Sequential(
            MinkowskiLinear(dims[-1], dims[-1]//2),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//2, dims[-1]//4),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//4, 3),
        )
        self.jet_momentum_mag_head = nn.Sequential(
            MinkowskiLinear(dims[-1], dims[-1]//2),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//2, dims[-1]//4),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//4, 1),
            MinkowskiSoftplus(beta=1, threshold=20),
        )
        self.jet_momentum_dir_head = nn.Sequential(
            MinkowskiLinear(dims[-1], dims[-1]//2),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//2, dims[-1]//4),
            MinkowskiLeakyReLU(0.1),
            MinkowskiLinear(dims[-1]//4, 3),
        )


        """ Initialise weights """
        self.apply(_init_weights)

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

        # encoder layers
        x_enc = []
        for i in range(self.nb_elayers):
            x = self.encoder_layers[i](x)
            if i < self.nb_elayers - 1:
                x_enc.append(x)
                x = self.downsample_layers[i](x)
        
        # event predictions
        x_pooled = self.global_pool(x)
        out_e_vis = self.e_vis_head(x_pooled)
        out_pt_miss = self.pt_miss_head(x_pooled)
        out_lepton_momentum_mag = self.out_lepton_momentum_mag_head(x_pooled)
        out_lepton_momentum_dir = self.out_lepton_momentum_dir_head(x_pooled)
        out_jet_momentum_mag = self.jet_momentum_mag_head(x_pooled)
        out_jet_momentum_dir = self.jet_momentum_dir_head(x_pooled)

        output = {"out_e_vis": out_e_vis.F,
                  "out_pt_miss": out_pt_miss.F,
                  "out_lepton_momentum_mag": out_lepton_momentum_mag.F,
                  "out_lepton_momentum_dir": out_lepton_momentum_dir.F,
                  "out_jet_momentum_mag": out_jet_momentum_mag.F,
                  "out_jet_momentum_dir": out_jet_momentum_dir.F,
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

