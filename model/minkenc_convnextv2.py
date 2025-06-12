"""
Author: Dr. Saul Alonso-Monsalve (modified)
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 04.25

Description: PyTorch model - stage 2: event-level classification and regression tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import MinkowskiEngine as ME
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiLinear,
    MinkowskiGlobalAvgPooling,
    MinkowskiGlobalMaxPooling,
    MinkowskiReLU,
    MinkowskiGELU,
)

from .utils import _init_weights, GlobalFeatureEncoder, RelPosTransformer, Block, MinkowskiLayerNorm


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
        self.num_modules = 10 if self.is_v5 else 15
        
        # Encoder configuration
        encoder_depths = [3, 3, 9, 3]
        #encoder_dims = [96, 192, 384, 768]
        encoder_dims = [32, 64, 128, 256]
        kernel_size_ds = (2, 2, 2)
        block_kernel = (3, 3, 3)
        
        drop_path_rate = 0.2
        
        assert len(encoder_depths) == len(encoder_dims)
        self.nb_elayers = len(encoder_dims)
        total_depth = sum(encoder_depths)
        dp_rates_enc = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dp_cur = 0

        # Stem
        self.stem_conv = MinkowskiConvolution(in_channels, encoder_dims[0], kernel_size=1, stride=1, dimension=D)
        self.stem_norm = MinkowskiLayerNorm(encoder_dims[0], eps=1e-6)

        # Build encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i in range(self.nb_elayers):
            # Encoder layer
            encoder_layer = nn.Sequential(
                *[Block(dim=encoder_dims[i], kernel_size=block_kernel, drop_path=dp_rates_enc[dp_cur + j], D=D)
                  for j in range(encoder_depths[i])]
            )
            self.encoder_layers.append(encoder_layer)
            dp_cur += encoder_depths[i]

            # Downsampling layer
            if i < self.nb_elayers - 1:
                downsample_layer = nn.Sequential(
                    MinkowskiLayerNorm(encoder_dims[i], eps=1e-6),
                    MinkowskiConvolution(
                        encoder_dims[i], encoder_dims[i + 1],
                        kernel_size=kernel_size_ds, stride=kernel_size_ds,
                        bias=True, dimension=D
                    ),
                )
                self.downsample_layers.append(downsample_layer)

        # Module-level real-pos transformer
        d_mod = encoder_dims[-1]
        heads_m = 8
        self.mod_transformer = RelPosTransformer(d_model=d_mod, nhead=heads_m, num_special_tokens=1, 
                                                 num_layers=2, dropout=args.dropout)
        self.cls_mod = nn.Parameter(torch.zeros(1, 1, d_mod))

        # Global features encoder and empty module embedding
        self.global_feats_encoder = GlobalFeatureEncoder(d_model=d_mod, dropout=args.dropout)
        self.empty_mod_emb = nn.Parameter(torch.zeros(d_mod))

        # Event-level transformer (across modules)
        heads_e = 8
        evt_layer = nn.TransformerEncoderLayer(d_model=d_mod, nhead=heads_e, batch_first=True, dropout=args.dropout)
        self.event_transformer = nn.TransformerEncoder(evt_layer, num_layers=3)
        self.pos_embed = nn.Embedding(1 + self.num_modules, d_mod)
        self.dropout = nn.Dropout(args.dropout)

        # Task branches and corresponding token
        self.branch_tokens = {
            "flavour": 0,
            "charm": 1,
            "e_vis_cc": 2,
            "e_vis_nc": 3,
            "pt_miss": 4,
            "lepton_momentum_mag": 5,
            "lepton_momentum_dir": 5,
            "jet_momentum_mag": 6,
            "jet_momentum_dir": 6,
        }
        branch_out_channels = {
            "flavour": 4,
            "charm": 1,
            "e_vis_cc": 1,
            "e_vis_nc": 1,
            "pt_miss": 1,
            "lepton_momentum_mag": 1,
            "lepton_momentum_dir": 3,
            "jet_momentum_mag": 1,
            "jet_momentum_dir": 3,
        }
        self.num_tasks = max(self.branch_tokens.values()) + 1
        self.cls_task = nn.Parameter(torch.zeros(1, self.num_tasks, d_mod))
        self.branches = nn.ModuleDict()
        for name in self.branch_tokens.keys():
            self.branches[name] = nn.Linear(d_mod, branch_out_channels[name])
                
        # Initialise weights
        self.apply(_init_weights)
    
    def forward(self, x, x_glob, module_to_event, module_pos, mask=None):
        """
        Forward pass through the shared backbone and branch-specific modules.
        
        Args:
            x: Input sparse tensor.
            x_glob: Global feature tensor.
            
        Returns:
            A dictionary mapping branch names prefixed with 'out_' to their outputs.
        """
        # Encoder
        x = self.stem_conv(x)
        x = self.stem_norm(x)
        for i in range(self.nb_elayers):
            x = self.encoder_layers[i](x)
            if i < self.nb_elayers - 1:
                x = self.downsample_layers[i](x)

        # Module-level Transformer
        coords_all = x.C[:, 1:].float()            # [N, 3]
        feats_all = x.F                            # [N, d]
        batch_ids = x.C[:, 0].long()               # [N]
        grouped_feats = [feats_all[batch_ids == m] for m in range(len(module_to_event))]
        grouped_coords = [coords_all[batch_ids == m] for m in range(len(module_to_event))]
        padded_feats = pad_sequence(grouped_feats, batch_first=True)   # [M, L, d]
        padded_coords = pad_sequence(grouped_coords, batch_first=True) # [M, L, 3]
        lengths = torch.tensor([g.size(0) for g in grouped_feats], device=padded_feats.device)
        L = padded_feats.size(1)
        pad_mask = torch.arange(L, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        cls_mod = self.cls_mod.expand(len(grouped_feats), -1, -1)          # [M, 1, d]
        seq_mod = torch.cat([cls_mod, padded_feats], dim=1)                # [M, 1 + L, d]
        seq_mod = self.dropout(seq_mod)                                    # [M, 1 + L, d]
        cls_pad = torch.zeros((len(grouped_feats), 1), dtype=torch.bool, device=pad_mask.device)
        key_mask_mod = torch.cat([cls_pad, pad_mask], dim=1)               # [M, 1 + L]
        zero_coord = torch.zeros((len(grouped_feats), 1, 3), device=padded_coords.device)
        seq_coords = torch.cat([zero_coord, padded_coords], dim=1)         # [M, 1 + L, 3]
        mod_out = self.mod_transformer(seq_mod, seq_coords, key_mask_mod)
        mod_emb = mod_out[:, 0, :]                                         # [M, d] 

        # Event-level Transformer
        B = module_to_event.max().item() + 1
        d = mod_emb.size(1)
        device = mod_emb.device
        pad_evt = self.empty_mod_emb.unsqueeze(0).unsqueeze(0).expand(B, self.num_modules, d).clone()  # [B, num_modules, d]
        event_idx = module_to_event.long()                                   # [M]
        module_idx = module_pos.long()                                       # [M]
        pad_evt[event_idx, module_idx, :] = mod_emb
        cls_tokens = self.cls_task.expand(B, self.num_tasks, -1)             # [B, K, d]
        glob_emb = self.global_feats_encoder(x_glob).unsqueeze(1)            # [B, 1, d]
        seq_evt = torch.cat([cls_tokens, glob_emb, pad_evt], dim=1)          # [B, K + 1 + num_modules, d]
        pos_indices = torch.arange(1 + self.num_modules, device=device)      # [1 + num_modules]
        pos_emb = self.pos_embed(pos_indices).unsqueeze(0)                   # [1, 1 + num_modules, d]
        seq_evt[:, self.num_tasks:] = seq_evt[:, self.num_tasks:] + pos_emb  
        seq_evt = self.dropout(seq_evt)                                      # [B, K + 1 + num_modules, d]
        evt_out = self.event_transformer(seq_evt, mask=mask)                 # [B, K + 1 + num_modules, d]
        evt_emb = evt_out[:, :self.num_tasks, :]                             # [B, K, d]
        #evt_emb = self.dropout(evt_emb)                                      # [B, K, d]
        
        # Branch-specific processing
        outputs = {}
        for i, (name, branch) in enumerate(self.branches.items()):
            output = branch(evt_emb[:, self.branch_tokens[name]])
            if name == "flavour" or name == "charm":
                outputs[f"out_{name}"] = output
            if "_dir" in name:
                outputs[f"out_{name}"] = F.normalize(output, dim=-1)
            else:
                outputs[f"out_{name}"] = F.softplus(output)
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
