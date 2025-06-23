"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 04.25

Description: PyTorch model - stage 1: semantic segmentation using an autoencoder + transformer.
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

from .utils import _init_weights, FourierPosEmb, GlobalFeatureEncoder, Block, MinkowskiLayerNorm


class MinkAEConvNeXtV2(nn.Module):
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
        #encoder_dims = [64, 128, 256, 384]
        encoder_dims = [32, 64, 128, 256]
        kernel_size_ds = (2, 2, 2)
        block_kernel = (3, 3, 3)
        
        drop_path_rate = 0.0
        
        assert len(encoder_depths) == len(encoder_dims)
        self.nb_elayers = len(encoder_dims)
        total_depth = sum(encoder_depths)
        dp_rates_enc = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dp_cur = 0

        # Stem
        self.stem_conv = MinkowskiConvolution(in_channels, encoder_dims[0], kernel_size=1, stride=1, dimension=D)
        self.stem_norm = MinkowskiLayerNorm(encoder_dims[0], eps=1e-6)
        self.mask_voxel_emb = nn.Parameter(torch.zeros(encoder_dims[0]))  # embedding for masked voxels
        
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
        num_bands_m = 6
        mod_layer = nn.TransformerEncoderLayer(d_model=d_mod, nhead=heads_m, batch_first=True, dropout=args.dropout)
        self.mod_transformer = nn.TransformerEncoder(mod_layer, num_layers=2)
        self.pos_emb_mod = FourierPosEmb(num_bands=num_bands_m, max_freq=10.)
        self.pos_emb_mod_proj = nn.Linear(2*num_bands_m*3, d_mod)
        self.cls_mod = nn.Parameter(torch.zeros(1, 1, d_mod))
        self.cls_mod_extra_emb = nn.Linear(2, d_mod)

        # Event-level transformer (across modules)
        heads_e = 8
        num_bands_e = 6
        self.global_feats_encoder = GlobalFeatureEncoder(d_model=d_mod, dropout=args.dropout)
        evt_layer = nn.TransformerEncoderLayer(d_model=d_mod, nhead=heads_e, batch_first=True, dropout=args.dropout)
        self.event_transformer = nn.TransformerEncoder(evt_layer, num_layers=3)
        self.pos_emb_evt = FourierPosEmb(num_bands=num_bands_e, max_freq=10.)
        self.pos_emb_evt_proj = nn.Linear(2*num_bands_e, d_mod)
        self.cls_evt = nn.Parameter(torch.zeros(1, 1, d_mod))
        self.dropout = nn.Dropout(args.dropout)
        
        # Decoder configuration (reversing the encoder dimensions)
        decoder_depths = [2] * len(encoder_depths)
        decoder_dims = list(reversed(encoder_dims))
        kernel_size_us = kernel_size_ds
        self.nb_dlayers = len(decoder_dims) - 1
        total_depth_dec = sum(decoder_depths)
        dp_rates_dec = [x.item() for x in torch.linspace(dp_rates_enc[-encoder_depths[-1]], 0, total_depth_dec)]
        dp_cur_dec = 0

        # Build decoder
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i in range(self.nb_dlayers):
            upsample_layer = nn.Sequential(
                MinkowskiLayerNorm(decoder_dims[i], eps=1e-6),
                MinkowskiConvolutionTranspose(decoder_dims[i], decoder_dims[i + 1],
                                              kernel_size=kernel_size_us, stride=kernel_size_us,
                                              bias=True, dimension=D),
            )
            self.upsample_layers.append(upsample_layer)

            decoder_layer = nn.Sequential(
                *[Block(dim=decoder_dims[i + 1], kernel_size=block_kernel, drop_path=dp_rates_dec[dp_cur_dec + j], D=D)
                  for j in range(decoder_depths[i])]
            )
            self.decoder_layers.append(decoder_layer)
            dp_cur_dec += decoder_depths[i]

        # Prediction layers
        self.primlepton_layer = nn.Sequential(
            MinkowskiLayerNorm(decoder_dims[-1], eps=1e-6),
            Block(dim=decoder_dims[-1], kernel_size=block_kernel, drop_path=0.0, D=D),
            MinkowskiConvolution(decoder_dims[-1], 1, kernel_size=1, stride=1, dimension=D),
        )
        self.seg_layer = nn.Sequential(
            MinkowskiLayerNorm(decoder_dims[-1], eps=1e-6),
            Block(dim=decoder_dims[-1], kernel_size=block_kernel, drop_path=0.0, D=D),
            MinkowskiConvolution(decoder_dims[-1], 3, kernel_size=1, stride=1, dimension=D),
        )
        self.charge_layer = nn.Sequential(
            MinkowskiLayerNorm(decoder_dims[-1], eps=1e-6),
            Block(dim=decoder_dims[-1], kernel_size=block_kernel, drop_path=0.0, D=D),
            MinkowskiConvolution(decoder_dims[-1], 1, kernel_size=1, stride=1, dimension=D),
        )
        self.iscc_layer = nn.Linear(d_mod, 1)

        # Initialise weights
        self.apply(_init_weights)

    def forward(self, x, x_glob, module_to_event, module_pos, mask_bool=None):
        """
        Forward pass through the encoder-decoder network.

        Args:
            x: Input sparse tensor.
            x_glob: Global feature tensors.
            module_to_event: mapping module index to event index tensor
            module_pos: mapping module index to module position tensor
            mask_bool: mask for voxels (masked autoencoder).

        Returns:
            A dictionary with voxel predictions.
        """        
        device = x.device
        module_hits, faser_cal_modules = x_glob[:2]
        params_glob = x_glob[2:] 
    
        # Encoder
        x = self.stem_conv(x)
        if mask_bool is not None:
            num_masked = mask_bool.sum().item()
            masked_feats = x.F.clone()
            replacement = self.mask_voxel_emb.unsqueeze(0).expand(num_masked, -1).to(masked_feats.dtype)
            masked_feats[mask_bool] = replacement
            x = ME.SparseTensor(
                features=masked_feats,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
        x = self.stem_norm(x)
        for i in range(self.nb_elayers):
            x = self.encoder_layers[i](x)
            if i < self.nb_elayers - 1:
                x = self.downsample_layers[i](x)

        # Module-level transformer
        # N_vox = total number of voxels in the batch
        # M = number of modules in the batch
        # L_vox = max number of voxels in a module
        # D_mod = model dimension
        initial_voxel_feats = x.F                                                      # [N_vox, D_mod]               
        voxel_coords = x.C[:, 1:].float()                                              # [N_vox, 3]        
        voxel_to_module_map = x.C[:, 0].long()                                         # [N_vox]
        M = len(module_to_event)                
        grouped_voxel_feats = [initial_voxel_feats[voxel_to_module_map == m] for m in range(M)]
        grouped_voxel_coords = [voxel_coords[voxel_to_module_map == m] for m in range(M)]
        module_lengths = torch.tensor([g.size(0) for g in grouped_voxel_feats], device=device)  # [M]
        max_voxels_per_module = module_lengths.max().item() if len(module_lengths) > 0 else 0   # L_vox
        padded_voxel_feats = pad_sequence(grouped_voxel_feats, batch_first=True)       # [M, L_vox, D_mod]
        padded_voxel_coords = pad_sequence(grouped_voxel_coords, batch_first=True)     # [M, L_vox, 3]
        hits_per_module = module_hits[module_to_event.long(), module_pos.long()]       # [M]
        ergy_per_module = faser_cal_modules[module_to_event.long(), module_pos.long()] # [M]
        ergy_and_hits = torch.stack([ergy_per_module, hits_per_module], dim=1)         # [M, 2]
        ergy_and_hits_emb = self.cls_mod_extra_emb(ergy_and_hits).unsqueeze(1)         # [M, 1, D_mod]
        module_cls_token = self.cls_mod.expand(M, -1, -1)                              # [M, 1, D_mod]
        module_cls_token = module_cls_token + ergy_and_hits_emb                        # [M, 1, D_mod]
        padded_voxel_coords = self.norm_coords(padded_voxel_coords, 48+1.)             # [M, L_vox, 3] (48+1 due to augmentations)
        pos_emb_mod = self.pos_emb_mod(padded_voxel_coords)                            # [M, L_vox, 2*num_bands*3]
        pos_emb_mod_proj = self.pos_emb_mod_proj(pos_emb_mod)                          # [M, L_vox, D_mod]
        padded_voxel_feats = padded_voxel_feats + pos_emb_mod_proj                     # [M, L_vox, D_mod]
        module_seq_in = torch.cat([module_cls_token, padded_voxel_feats], dim=1)       # [M, 1 + L_vox, D_mod]
        module_seq_in = self.dropout(module_seq_in)                                    # [M, 1 + L_vox, D_mod]
        module_key_padding_mask = torch.ones((M, 1 + max_voxels_per_module), 
                                             dtype=torch.bool, device=device)          # [M, 1 + L_vox]
        module_key_padding_mask[:, 0] = False
        module_key_padding_mask[:, 1:] = (
            torch.arange(max_voxels_per_module, device=device).unsqueeze(0)
            >= module_lengths.unsqueeze(1)
        )
        module_seq_out = self.mod_transformer(
            module_seq_in, src_key_padding_mask=module_key_padding_mask)               # [M, 1 + L_vox, D_mod]
        module_summary_emb = module_seq_out[:, 0, :]                                   # [M, D_mod]

        # Event-level Transformer
        # B = number of events in the batch
        # L_mod = max number of modules in an event
        B = module_to_event.max().item() + 1
        d = module_summary_emb.size(1)                                                   # D_mod
        grouped_module_emb = [module_summary_emb[module_to_event == b] for b in range(B)]
        grouped_module_pos = [module_pos[module_to_event == b] for b in range(B)]
        evt_lengths = torch.tensor([g.size(0) for g in grouped_module_emb], device=device) # [B]
        max_modules_per_event = int(evt_lengths.max().item()) if len(evt_lengths) > 0 else 0    # L_mod
        padded_module_emb = pad_sequence(grouped_module_emb, batch_first=True)           # [B, L_mod, D_mod]
        padded_module_pos = pad_sequence(grouped_module_pos, batch_first=True, padding_value=0) # [B, L_mod]
        glob_emb = self.global_feats_encoder(params_glob).unsqueeze(1)                   # [B, 1, D_mod]
        event_cls_token = self.cls_evt.expand(B, 1, d)                                   # [B, 1, D_mod]
        #event_cls_token = event_cls_token + glob_emb
        padded_module_pos = self.norm_coords(
            padded_module_pos.unsqueeze(-1), self.num_modules)                           # [B, L_mod, 1]
        pos_emb_evt = self.pos_emb_evt(padded_module_pos)                                # [B, L_mod, 2*num_bands]
        pos_emb_evt_proj = self.pos_emb_evt_proj(pos_emb_evt)                            # [B, L_mod, D_mod]
        padded_module_emb = padded_module_emb + pos_emb_evt_proj                         # [B, L_mod, D_mod]        
        event_seq_in = torch.cat([event_cls_token, padded_module_emb], dim=1)            # [B, 1 + L_mod, D_mod]
        event_seq_in = self.dropout(event_seq_in)                                        # [B, 1 + L_mod, D_mod]
        event_key_padding_mask = torch.ones((B, 1 + max_modules_per_event),
                                            dtype=torch.bool, device=device)             # [B, 1 + L_mod]
        event_key_padding_mask[:, 0] = False
        event_key_padding_mask[:, 1:] = (
            torch.arange(max_modules_per_event, device=device).unsqueeze(0)
            >= evt_lengths.unsqueeze(1)
        )
        event_seq_out = self.event_transformer(
            event_seq_in, src_key_padding_mask=event_key_padding_mask)                  # [B, 1 + L_mod, D_mod]
        event_summary_emb = event_seq_out[:, 0, :]                                      # [B, D_mod]
        #event_summary_emb = self.dropout(event_summary_emb)                           # [B, D_mod]
        contextualized_module_emb_list = [
            event_seq_out[b, 1:1 + evt_lengths[b], :]
            for b in range(B)
        ]
        contextualized_module_emb = torch.cat(contextualized_module_emb_list, dim=0)  # [M, D_mod]

        # Broadcast context from both transformer levels back to each voxel
        intra_module_ctx_all = torch.zeros_like(initial_voxel_feats)                  # [N_vox, D_mod]
        for m in range(M):
            indices = (voxel_to_module_map == m).nonzero(as_tuple=False).squeeze(1)
            intra_module_ctx_all[indices] = module_seq_out[m, 1 : 1 + module_lengths[m], :]
        inter_module_ctx_all = contextualized_module_emb[voxel_to_module_map]         # [N_vox, D_mod]
        final_voxel_feat = intra_module_ctx_all + inter_module_ctx_all                # [N_vox, D_mod]
        x = ME.SparseTensor(
            features=final_voxel_feat,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key
        )

        # Decoder
        for i in range(self.nb_dlayers):
            x = self.upsample_layers[i](x)
            x = self.decoder_layers[i](x)

        out_primlepton = self.primlepton_layer(x)
        out_seg = self.seg_layer(x)
        out_charge = self.charge_layer(x)
        out_iscc = self.iscc_layer(event_summary_emb)
        return {
            "out_primlepton": out_primlepton,
            "out_seg": out_seg,
            "out_charge": out_charge,
            "out_iscc": out_iscc,
        }

    def norm_coords(self, coords, max_value=10.):
        norm = coords / (max_value - 1)
        return norm * 2 - 1

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

