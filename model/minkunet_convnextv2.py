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
    MinkowskiLinear,
    MinkowskiGELU
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


class MinkUNetConvNeXtV2(nn.Module):
    def __init__(self, in_channels, out_channels, D=3, args=None):
        nn.Module.__init__(self)
        
        """Encoder"""
        depths=[3, 3, 9, 3]
        dims=[96, 192, 384, 768]     
        #depths=[2, 2, 6, 2]
        #dims=(32, 64, 128, 256)
        drop_path_rate=0.

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
            
        """Decoder"""
        depths=depths[::-1]
        dims=dims[::-1]
        decoder_embed_dim=64
        
        self.upsample_layers = nn.ModuleList()
        
        for i in range(3):
            upsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i-1]+dims[i] if i>0 else dims[i], eps=1e-6),
                MinkowskiConvolutionTranspose(dims[i-1]+dims[i] if i>0 else dims[i], dims[i], kernel_size=2, stride=2, bias=True, dimension=3)
            )
            self.upsample_layers.append(upsample_layer)
            
        stem = nn.Sequential(
            MinkowskiConvolutionTranspose(dims[2]+dims[3], dims[3], kernel_size=4, stride=4, dimension=3),
            MinkowskiLayerNorm(dims[3], eps=1e-6),
        )
        self.upsample_layers.append(stem)

        self.stages_dec = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.stages_dec.append(stage)
            cur += depths[i]
        
        """Cls layers"""
        self.cls_layers = nn.ModuleList()
        for i in range(4):
            cls_layer = nn.Sequential(
                MinkowskiConvolution(dims[i], decoder_embed_dim, kernel_size=1, stride=1, dimension=3),
                Block(dim=decoder_embed_dim, drop_path=0., D=3),
                MinkowskiConvolution(decoder_embed_dim, out_channels, kernel_size=1, stride=1, dimension=3),
            )
            self.cls_layers.append(cls_layer)
            
        """ Max pool just for generating downsampled labels """        
        self.avg_pool2x2 = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3) 
        self.avg_pool4x4 = ME.MinkowskiAvgPooling(kernel_size=4, stride=4, dimension=3) 

        """ Initialise weights """
        self.apply(_init_weights)

    def forward(self, x, y):
        """ Generate labels for deep supervision """
        ys = []
        for i in range(5):
            if i==0:
                y_aux = y.detach()
            elif i==1:
                y_aux = self.avg_pool4x4(y_aux)
            else:
                y_aux = self.avg_pool2x2(y_aux)
            ys.append(y_aux)
  
        """Encoder"""
        x = self.downsample_layers[0](x)
        x_enc = []
        for i in range(4):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages_enc[i](x)
            x_enc.append(x)
        x_enc = x_enc[::-1]
        
        """Decoder"""
        out_cls = []
        for i in range(4):
            if i > 0:
                x = ME.cat(x, x_enc[i])
            x = self.upsample_layers[i](x)
            x = self.stages_dec[i](x)
            out_cl = self.cls_layers[i](x)
            out_cls.insert(0, out_cl)
        
        return out_cls, ys

