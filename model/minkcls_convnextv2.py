import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
import torch.nn.functional as F
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
    MinkowskiGlobalMaxPooling,
    MinkowskiLinear,
    MinkowskiReLU,
    MinkowskiGELU,
)


# Custom weight initialization function
def _init_weights(m):
    if isinstance(m, ME.MinkowskiConvolution):
        nn.init.trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
        nn.init.trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, ME.MinkowskiDepthwiseConvolution):
        nn.init.trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, ME.MinkowskiLinear):
        nn.init.trunc_normal_(m.linear.weight, std=.02)
        if m.linear.bias is not None:
            nn.init.constant_(m.linear.bias, 0)
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MinkClsConvNeXtV2(nn.Module):
    def __init__(self, in_channels, out_channels, D=3, args=None):
        nn.Module.__init__(self)
        
        """Encoder"""
        depths=[3, 3, 9, 3]
        dims=[96, 192, 384, 768]     
        decoder_embed_dim = 256
        drop_path_rate=0.
        self.contrastive = args.contrastive
        self.finetuning = args.finetuning

        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            MinkowskiConvolution(in_channels, dims[0], kernel_size=4, stride=4, dimension=3),
            MinkowskiLayerNorm(dims[0], eps=1e-6),
        ) 
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6),
                MinkowskiConvolution(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True, dimension=3)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages_enc = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.stages_enc.append(stage)
            cur += depths[i]
        

        """Cls layers"""
        if self.finetuning:
            self.cls_layer = nn.Sequential(
                MinkowskiGlobalMaxPooling(),
                MinkowskiConvolution(dims[-1], out_channels, kernel_size=1, stride=1, dimension=3)
            )
        elif self.contrastive:
            self.cls_layer = nn.Sequential(
                MinkowskiGlobalMaxPooling(),
                MinkowskiConvolution(dims[-1], decoder_embed_dim, kernel_size=1, stride=1, dimension=3),
                MinkowskiReLU(),
                MinkowskiLayerNorm(decoder_embed_dim, eps=1e-6),
                MinkowskiConvolution(decoder_embed_dim, decoder_embed_dim, kernel_size=1, stride=1, dimension=3),
                MinkowskiReLU(),
                MinkowskiLayerNorm(decoder_embed_dim, eps=1e-6),     
                MinkowskiConvolution(decoder_embed_dim, decoder_embed_dim, kernel_size=1, stride=1, dimension=3),
            ) 
        else:
            self.cls_layer = nn.Sequential(
                MinkowskiGlobalMaxPooling(),
                    MinkowskiConvolution(dims[i], decoder_embed_dim, kernel_size=1, stride=1, dimension=3),
                    Block(dim=decoder_embed_dim, drop_path=0., D=3),
                    MinkowskiConvolution(decoder_embed_dim, out_channels, kernel_size=1, stride=1, dimension=3),
                ) 

        """ Initialise weights """
        self.apply(_init_weights)

    def forward(self, x):
        """Encoder"""
        x = self.downsample_layers[0](x)
        for i in range(4):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages_enc[i](x)
        
        out_cls =  self.cls_layer(x)
       
        return out_cls

