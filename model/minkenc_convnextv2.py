"""
Author: Dr. Saul Alonso-Monsalve (modified)
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 04.25

Description: PyTorch model - stage 2: event-level classification and regression tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

import MinkowskiEngine as ME
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiGlobalMaxPooling,
    MinkowskiGlobalAvgPooling,
    MinkowskiLinear,
    MinkowskiSoftplus,
    MinkowskiGELU,
)

from .utils import GlobalFeatureEncoder, Block, MinkowskiLayerNorm, MinkowskiSE


def _init_weights(m):
    """Custom weight initialization for supported layers."""
    if isinstance(m, MinkowskiConvolution):
        trunc_normal_(m.kernel, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, ME.MinkowskiDepthwiseConvolution):
        trunc_normal_(m.kernel, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, MinkowskiLinear):
        trunc_normal_(m.linear.weight, std=0.02)
        if m.linear.bias is not None:
            nn.init.constant_(m.linear.bias, 0)
    elif isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MinkEncConvNeXtV2(nn.Module):
    def __init__(self, in_channels, out_channels, D=3, args=None):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            D (int): Spatial dimension.
            args: Arguments namespace with at least a `dataset_path` attribute.
        """
        super().__init__()
        # Determine the version flag based on the dataset_path in args.
        self.is_v5 = "v5" in args.dataset_path if args is not None and hasattr(args, "dataset_path") else False
        self.module_size = 24 if self.is_v5 else 20
        
        # Encoder configuration
        encoder_depths = [3, 3, 9, 3]
        encoder_dims = [96, 192, 384, 768]
        se_block_reds = [16, 16, 16, 16]
        kernel_size_ds = (2, 2, 2)
        dilation_ds = (1, 1, 2)
        block_kernel = (5, 5, 5)
        drop_path_rate = 0.1
        assert len(encoder_depths) == len(encoder_dims)
        self.nb_bblayers = len(encoder_dims) - 1
        total_depth = sum(encoder_depths)
        dp_rates_enc = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dp_cur = 0
        
        ###############################################
        # Shared Backbone
        ###############################################
        
        # Stem
        self.stem_ch = nn.Linear(in_channels, encoder_dims[0])
        self.stem_mod = nn.Embedding(self.module_size, encoder_dims[0]) 
        self.stem_ln = MinkowskiLayerNorm(encoder_dims[0], eps=1e-6)

        # Global features
        self.register_buffer("global_weight", torch.tensor(1.0))  # Scalar controlling global parameter contribution
        self.global_feats_encoder = GlobalFeatureEncoder(encoder_dim=encoder_dims[0], dropout=0.3)

        # Backbone
        self.shared_encoders = nn.ModuleList()
        self.shared_downsamples = nn.ModuleList()
        self.shared_se_layers = nn.ModuleList()
        for i in range(self.nb_bblayers):
            # Encoder layer
            encoder_layer = nn.Sequential(
                *[Block(dim=encoder_dims[i], kernel_size=block_kernel, dilation=dilation_ds,
                        drop_path=dp_rates_enc[dp_cur + j], D=D)
                  for j in range(encoder_depths[i])]
            )
            self.shared_encoders.append(encoder_layer)
            dp_cur += encoder_depths[i]
            
            # SE layer
            se_layer = MinkowskiSE(channels=encoder_dims[i], glob_dim=encoder_dims[0], reduction=se_block_reds[i])
            self.shared_se_layers.append(se_layer)

            # Downsampling layer
            if i < self.nb_bblayers - 1:
                downsample_layer = nn.Sequential(
                    MinkowskiLayerNorm(encoder_dims[i], eps=1e-6),
                    MinkowskiConvolution(
                        encoder_dims[i],
                        encoder_dims[i + 1],
                        kernel_size=kernel_size_ds, 
                        stride=kernel_size_ds,
                        bias=True,
                        dimension=D
                    ),
                )
                self.shared_downsamples.append(downsample_layer)

        ###############################################
        # Branch-Specific Modules
        ###############################################
        
        branch_names = [
            "flavour",
            "e_vis",
            "pt_miss",
            "lepton_momentum_mag",
            "lepton_momentum_dir",
            "jet_momentum_mag",
            "jet_momentum_dir"
        ]
        branch_out_channels = {
            "flavour": 4,
            "e_vis": 1,
            "pt_miss": 1,
            "lepton_momentum_mag": 1,
            "lepton_momentum_dir": 3,
            "jet_momentum_mag": 1,
            "jet_momentum_dir": 3,
        }
        
        self.branches = nn.ModuleDict()
        for name in branch_names:
            branch = {}
            branch["downsample"] = nn.Sequential(
                MinkowskiLayerNorm(encoder_dims[i], eps=1e-6),
                MinkowskiConvolution(
                    encoder_dims[i],
                    encoder_dims[i+1],
                    kernel_size=kernel_size_ds, 
                    stride=kernel_size_ds,
                    bias=True,
                    dimension=D),
            )
            
            branch["encoder"] = nn.Sequential(
                *[Block(dim=encoder_dims[i+1], kernel_size=block_kernel, dilation=dilation_ds,
                        drop_path=dp_rates_enc[dp_cur + j], D=D) for j in range(encoder_depths[i+1])]
            )
            branch["se"] = MinkowskiSE(channels=encoder_dims[i+1], glob_dim=encoder_dims[0], reduction=se_block_reds[i+1])
            branch["global_pool"] = (
                MinkowskiGlobalMaxPooling() if name == "flavour" else MinkowskiGlobalAvgPooling()
            )
            branch["head"] = nn.Sequential(
                MinkowskiLinear(encoder_dims[i+1], branch_out_channels[name]),
            )
            self.branches[name] = nn.ModuleDict(branch)

        # Initialise weights
        self.apply(_init_weights)
    
    def forward(self, x, x_glob):
        """
        Forward pass through the shared backbone and branch-specific modules.
        
        Args:
            x: Input sparse tensor.
            x_glob: Global feature tensor.
            
        Returns:
            A dictionary mapping branch names prefixed with 'out_' to their outputs.
        """
        coords, charge = x.C, x.F    # feats: [N,2]
        mod_id = (coords[:, 3] % self.module_size).long()  # [N]
        
        # Stem and global feature MLP transformation.
        charge_emb   = self.stem_ch(charge)  # [N, module_emb_dim]
        mod_id_emb   = self.stem_mod(mod_id) # [N, module_emb_dim]
        new_feats = charge_emb + mod_id_emb  # [N, module_emb_dim]
        x = ME.SparseTensor(
            new_feats, 
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key,
        )        
        x = self.stem_ln(x)
        x_glob = self.global_feats_encoder(x_glob)

        # Process each backbone stage in a loop.
        for i in range(len(self.shared_encoders)):
            x = self.shared_encoders[i](x)
            x = self.shared_se_layers[i](x, x_glob, self.global_weight)
            if i < len(self.shared_downsamples):
                x = self.shared_downsamples[i](x)
        
        # Branch-specific processing
        outputs = {}
        for name, branch in self.branches.items():
            xb = x 
            xb = branch["downsample"](xb)
            xb = branch["encoder"](xb)
            xb = branch["se"](xb, x_glob, self.global_weight)
            xb_gp = branch["global_pool"](xb)
            xb_head = branch["head"](xb_gp)
            outputs[f"out_{name}"] = xb_head.F
        return outputs

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
