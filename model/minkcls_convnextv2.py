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
      
        """MLP for global features"""
        self.global_mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )

        """Global poolin"""
        self.global_pool = MinkowskiGlobalMaxPooling()

        """Cls layers"""
        if self.contrastive:
            self.cls_layer = nn.Sequential(
                torch.nn.Linear(dims[-1], decoder_embed_dim),
                nn.ReLU(),
                torch.nn.Linear(decoder_embed_dim, decoder_embed_dim)
            ) 
        else:
            self.cls_layer = nn.Sequential(
                torch.nn.Linear(dims[-1], out_channels)
            ) 

        """ Initialise weights """
        self.apply(_init_weights)

    def forward(self, x, glob_x):
        """Encoder"""
        x = self.downsample_layers[0](x)
        for i in range(4):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages_enc[i](x)
        x = self.global_pool(x)

        # add global features
        glob_x = self.global_mlp(glob_x)
        x = x.F + glob_x

        out_cls =  self.cls_layer(x)
       
        return out_cls

