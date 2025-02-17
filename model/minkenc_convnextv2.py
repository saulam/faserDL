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


class MinkEncConvNeXtV2(nn.Module):
    def __init__(self, in_channels, out_channels, D=3, args=None):
        nn.Module.__init__(self)

        """Encoder"""
        #depths=[2, 4, 4, 8, 8, 8]
        #dims = (16, 32, 64, 128, 256, 512)
        depths=[3, 3, 9, 3]
        dims=[96, 192, 384, 768]     
        kernel_size = 3
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

        # Linear transformation for glibal features
        self.global_mlp = nn.Sequential(
            nn.Linear(1 + 1 + 1 + 1 + 9 + 15, dims[-1]),
            #nn.GELU(),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(dims[-1], dims[-1]),
            #nn.GELU(),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(dims[-1], 64),
            #nn.GELU(),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )

        """Classification/regression layers"""
        self.global_pool = MinkowskiGlobalMaxPooling()
        self.flavour_layer = nn.Sequential(
            nn.Linear(dims[-1] + 64, dims[-1]),
            nn.GELU(),
            nn.Linear(dims[-1], 4)
        ) 
        self.evis_layer = nn.Sequential(
            nn.Linear(dims[-1] + 64, dims[-1]),
            nn.LeakyReLU(0.1),
            nn.Linear(dims[-1], 1)
        ) 
        self.ptmiss_layer = nn.Sequential(
            nn.Linear(dims[-1] + 64, dims[-1]),
            nn.LeakyReLU(0.1),
            nn.Linear(dims[-1], 1)
        )
        self.out_lepton_momentum_layer = nn.Sequential(
            nn.Linear(dims[-1] + 64, dims[-1]),
            nn.LeakyReLU(0.1),
            nn.Linear(dims[-1], 3),
        )
        self.jet_momentum_layer = nn.Sequential(
            nn.Linear(dims[-1] + 64, dims[-1]),
            nn.LeakyReLU(0.1),
            nn.Linear(dims[-1], 3),
        )

        """ Initialise weights """
        self.apply(_init_weights)

    def forward(self, x, x_glob):
        """Encoder"""
        # stem
        x = self.stem(x)
        x = self.stem_ln(x)

        # encoder layers
        x_enc = []
        for i in range(self.nb_elayers):
            x = self.encoder_layers[i](x)
            if i < self.nb_elayers - 1:
                x_enc.append(x)
                x = self.downsample_layers[i](x)
        
        # global params
        x_glob = self.global_mlp(x_glob)
        
        # event predictions
        x_pooled = self.global_pool(x)
        #x_pooled = x_pooled.F + x_glob 
        x_pooled = torch.cat((x_pooled.F, x_glob), dim=1) 

        out_flavour = self.flavour_layer(x_pooled)
        out_evis = self.evis_layer(x_pooled)
        out_ptmiss = self.ptmiss_layer(x_pooled)
        out_lepton_momentum = self.out_lepton_momentum_layer(x_pooled)
        out_jet_momentum = self.jet_momentum_layer(x_pooled)

        output = {"out_flavour": out_flavour,
                  "out_evis": out_evis,
                  "out_ptmiss": out_ptmiss,
                  "out_lepton_momentum": out_lepton_momentum,
                  "out_jet_momentum": out_lepton_momentum,
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

