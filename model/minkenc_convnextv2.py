"""
Author: Dr. Saul Alonso-Monsalve (modified)
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 04.25

Description: PyTorch model - stage 2: event-level classification and regression tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.layers import trunc_normal_
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

from .utils import RelPosEncoderLayer, GlobalFeatureEncoder, Block, MinkowskiLayerNorm


def _init_weights(m):
    """Custom weight initialization for various layers."""
    if isinstance(m, MinkowskiConvolution):
        trunc_normal_(m.kernel, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, MinkowskiConvolutionTranspose):
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
    elif isinstance(m, nn.LSTM):
        # For each parameter tensor in the LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                # input-to-hidden weights: Xavier uniform
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # hidden-to-hidden (recurrent) weights: orthogonal
                init.orthogonal_(param.data)
            elif 'bias' in name:
                # biases: zero, except forget-gate bias = 1
                param.data.fill_(0)
                # each bias is [b_ii | b_if | b_ig | b_io], so
                hidden_size = m.hidden_size
                # forget-gate slice is the 2nd quarter
                param.data[hidden_size:2*hidden_size].fill_(1)


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
        encoder_dims = [96, 192, 384, 768]
        kernel_size_ds = (2, 2, 2)
        block_kernel = (3, 3, 3)
        
        drop_path_rate = 0.2
        
        assert len(encoder_depths) == len(encoder_dims)
        self.nb_elayers = len(encoder_dims)
        total_depth = sum(encoder_depths)
        dp_rates_enc = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dp_cur = 0

        # Stem
        self.stem = nn.Sequential(
             MinkowskiConvolution(in_channels, encoder_dims[0], kernel_size=1, stride=1, dimension=D),
             MinkowskiLayerNorm(encoder_dims[0], eps=1e-6),
        )

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
        heads_m = 12
        self.mod_layer = RelPosEncoderLayer(d_model=d_mod, nhead=heads_m, num_special_tokens=1)
        self.cls_mod = nn.Parameter(torch.zeros(1, 1, d_mod))

        # Global features encoder
        self.global_feats_encoder = GlobalFeatureEncoder(encoder_dim=encoder_dims[-1])

        # Event-level transformer (across modules)
        heads_e = 12
        evt_layer = nn.TransformerEncoderLayer(d_model=d_mod, nhead=heads_e, batch_first=True)
        self.event_transformer = nn.TransformerEncoder(evt_layer, num_layers=5)
        self.event_pos = nn.Embedding(self.num_modules, d_mod)  # positional embedding for modules

        # Task branches (across modules)
        self.branch_names = [
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
        self.num_tasks = len(self.branch_names)
        self.cls_task = nn.Parameter(torch.zeros(1, self.num_tasks, d_mod))
        self.branches = nn.ModuleDict()
        for name in self.branch_names:
            if name == "flavour":
                self.branches[name] = nn.Linear(d_mod, branch_out_channels[name])
            else:
                self.branches[name] = nn.Linear(d_mod + 4, branch_out_channels[name])
                
        # Initialise weights
        self.apply(_init_weights)
    
    def forward(self, x, x_glob, module_to_event, module_pos):
        """
        Forward pass through the shared backbone and branch-specific modules.
        
        Args:
            x: Input sparse tensor.
            x_glob: Global feature tensor.
            
        Returns:
            A dictionary mapping branch names prefixed with 'out_' to their outputs.
        """
        # Encoder
        x = self.stem(x)
        for i in range(self.nb_elayers):
            x = self.encoder_layers[i](x)
            if i < self.nb_elayers - 1:
                x = self.downsample_layers[i](x)

        # Module-level Transformer
        coords_all = x.C[:, 1:].float()            # [N, 3]
        feats_all = x.F                            # [N, d]
        batch_ids = x.C[:, 0]                      # [N]
        grouped_feats = [feats_all[batch_ids == m] for m in range(len(module_to_event))]
        grouped_coords = [coords_all[batch_ids == m] for m in range(len(module_to_event))]
        padded_feats = pad_sequence(grouped_feats, batch_first=True)   # [M, L, d]
        padded_coords = pad_sequence(grouped_coords, batch_first=True) # [M, L, 3]
        lengths = torch.tensor([g.size(0) for g in grouped_feats], device=padded_feats.device)
        L = padded_feats.size(1)
        pad_mask = torch.arange(L, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        cls_mod = self.cls_mod.expand(len(grouped_feats), -1, -1)      # [M, 1, d]
        seq_mod = torch.cat([cls_mod, padded_feats], dim=1)            # [M, L+1, d]
        cls_pad = torch.zeros((len(grouped_feats), 1), dtype=torch.bool, device=pad_mask.device)
        key_mask_mod = torch.cat([cls_pad, pad_mask], dim=1)          # [M, L+1]
        zero_coord = torch.zeros((len(grouped_feats), 1, 3), device=padded_coords.device)
        seq_coords = torch.cat([zero_coord, padded_coords], dim=1)    # [M, L+1, 3]
        mod_out = self.mod_layer(seq_mod, seq_coords, key_mask_mod)
        mod_emb = mod_out[:, 0, :]                                    # [M, d]     

        # Event-level Transformer
        B = module_to_event.max().item() + 1
        evt_lists = [mod_emb[module_to_event == e] for e in range(B)]
        pos_lists = [module_pos[module_to_event == e] for e in range(B)]
        pad_evt = pad_sequence(evt_lists, batch_first=True)             # [B, M_max, d]
        pad_pos = pad_sequence(pos_lists, batch_first=True)             # [B, M_max]
        pos_emb = self.event_pos(pad_pos)                               # [B, M_max, d]
        seq_evt = pad_evt + pos_emb
        lengths_evt = torch.tensor([g.size(0) for g in evt_lists], device=seq_evt.device)
        M_max = seq_evt.size(1)
        evt_mask = torch.arange(M_max, device=lengths_evt.device).unsqueeze(0) >= lengths_evt.unsqueeze(1)
        glob_emb = self.global_feats_encoder(x_glob).unsqueeze(1)   # [B, 1, d]
        cls_tokens = self.cls_task.expand(B, self.num_tasks, -1)  # [B,K,d]
        seq2 = torch.cat([cls_tokens, glob_emb, seq_evt], dim=1)    # [B,K+1+M_max,d]
        spec_mask = torch.zeros((B, self.num_tasks+1), dtype=torch.bool, device=evt_mask.device)
        key_mask = torch.cat([spec_mask, evt_mask], dim=1)          # [B,K+1+M_max]
        evt_out = self.event_transformer(seq2, src_key_padding_mask=key_mask)
        evt_emb = evt_out[:, :self.num_tasks, :]
        
        # Branch-specific processing
        outputs = {}
        flavour_index = self.branch_names.index("flavour")
        flav_logits = self.branches["flavour"](evt_emb[:, flavour_index]) 
        outputs["out_flavour"] = flav_logits 
        for i, (name, branch) in enumerate(self.branches.items()):
            if name == "flavour":
                continue  # already handled
            reg_input = torch.cat([evt_emb[:, i], flav_logits], dim=1)  # [B,d+4]
            if "_dir" in name:
                outputs[f"out_{name}"] = F.normalize(branch(reg_input), dim=-1)
            else:
                outputs[f"out_{name}"] = F.softplus(branch(reg_input))
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
