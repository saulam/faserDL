"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 06.25

Description: PyTorch model - stage 1: pretraining.
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

from .utils import _init_weights, ScaledFourierPosEmb3D, RelPosTransformer, GlobalFeatureEncoder, Block, MinkowskiLayerNorm


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
        self.module_size = 20
        self.num_modules = 10 if self.is_v5 else 15
        
        # Encoder configuration
        encoder_depths = [3, 3, 9, 3]
        encoder_dims = [96, 192, 384, 768]
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
        self.d_mod = encoder_dims[-1]
        heads_m = 12
        self.mod_transformer = RelPosTransformer(
            d_model=self.d_mod, nhead=heads_m, num_special_tokens=1, num_layers=3, num_dims=3, dropout=args.dropout)
        self.pos_emb_mod = ScaledFourierPosEmb3D(num_features=32, d_model=self.d_mod)
        self.cls_mod = nn.Parameter(torch.zeros(1, 1, self.d_mod))
        self.cls_mod_extra_emb = nn.Linear(2, self.d_mod)

        # Sub-tokens
        self.K = 2
        assert self.d_mod % self.K == 0, "d_mod must be divisible by K subtokens"
        self.d_evt = self.d_mod // self.K
        self.subtoken_proj = nn.Linear(self.d_mod, self.K * self.d_evt)
        self.pos_emb_sub = nn.Embedding(self.K, self.d_evt)
        self.subtokens_to_module = nn.Linear(self.K * self.d_evt, self.d_mod)

        # Event-level transformer (across modules)
        heads_e = 8
        self.global_feats_encoder = GlobalFeatureEncoder(d_model=self.d_evt, dropout=args.dropout)
        self.event_transformer = RelPosTransformer(
            d_model=self.d_evt, nhead=heads_e, num_special_tokens=1, num_layers=3, num_dims=1, dropout=args.dropout)
        self.pos_emb = nn.Embedding(self.num_modules, self.d_evt)
        self.cls_evt = nn.Parameter(torch.zeros(1, 1, self.d_evt))
        self.dropout = nn.Dropout(args.dropout)

        # Context fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.d_mod * 2, self.d_mod),
            nn.GELU(),
            nn.Linear(self.d_mod, self.d_mod),
        )
        
        # Decoder configuration (reversing the encoder dimensions)
        decoder_depths = [2] * len(encoder_depths)
        decoder_dims = list(reversed(encoder_dims))
        kernel_size_us = kernel_size_ds
        self.nb_dlayers = len(decoder_dims) - 1
        total_depth_dec = sum(decoder_depths)
        dp_rates_dec = [x.item() for x in torch.linspace(drop_path_rate, 0, total_depth_dec)]
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
        self.iscc_layer = nn.Linear(self.d_evt, 1)

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

        # ----------------- Encoder -----------------
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

        # ----------------- Module-level Transformer -----------------
        # N_vox = total number of voxels in the batch
        # M = number of modules in the batch
        # L_vox = max number of voxels in a module
        # D_mod = model dimension
        initial_voxel_feats = x.F                                                         # [N_vox, D_mod]
        voxel_coords = x.C[:, 1:].float()                                                 # [N_vox, 3]
        voxel_to_module_map = x.C[:, 0].long()                                            # [N_vox]
        M = int(voxel_to_module_map.max().item()) + 1
        assert M == module_to_event.size(0)       
        grouped_feats = [initial_voxel_feats[voxel_to_module_map == m] for m in range(M)]
        grouped_coords = [voxel_coords[voxel_to_module_map == m] for m in range(M)]
        grouped_idxs = [
            (voxel_to_module_map == m).nonzero(as_tuple=False).squeeze(1)
            for m in range(M)
        ]
        module_lengths = torch.tensor([g.size(0) for g in grouped_feats], device=device)  # [M]
        L_vox = int(module_lengths.max().item()) if len(module_lengths) > 0 else 0
        padded_feats  = pad_sequence(grouped_feats,  batch_first=True)                    # [M, L_vox, D_mod]
        padded_coords = pad_sequence(grouped_coords, batch_first=True) / 48.              # [M, L_vox, 3]
        padded_idxs   = pad_sequence(grouped_idxs,   batch_first=True, padding_value=-1)  # [M, L_vox]
        hits_per_module = module_hits[module_to_event.long(), module_pos.long()]          # [M]
        ergy_per_module = faser_cal_modules[module_to_event.long(), module_pos.long()]    # [M]
        ergy_and_hits = torch.stack([ergy_per_module, hits_per_module], dim=1)            # [M, 2]
        ergy_and_hits_emb = self.cls_mod_extra_emb(ergy_and_hits).unsqueeze(1)            # [M, 1, D_mod]
        pos_emb_mod = self.pos_emb_mod(padded_coords)                                     # [M, L_vox, D_mod]
        cls_tokens_mod = self.cls_mod.expand(M, -1, -1)                                   # [M, 1, D_mod]
        cls_tokens_mod = cls_tokens_mod + ergy_and_hits_emb                               # [M, 1, D_mod]
        mod_in = torch.cat([cls_tokens_mod, padded_feats + pos_emb_mod], dim=1)           # [M, 1+L_vox, D_mod]
        mod_in = self.dropout(mod_in)                                                     # [M, 1+L_vox, D_mod]
        module_key_padding_mask = torch.ones((M, 1 + L_vox),
                                             dtype=torch.bool, device=device)             # [M, 1+L_vox]
        cls_coords = torch.zeros((M, 1, padded_coords.size(-1)), device=device)           # [M, 1, 3]
        mod_coords = torch.cat([cls_coords, padded_coords], dim=1)                        # [M, 1+L_vox, 3]
        module_key_padding_mask[:, 0] = False
        module_key_padding_mask[:, 1:] = (
            torch.arange(L_vox, device=device).unsqueeze(0)
            >= module_lengths.unsqueeze(1)
        )
        module_seq_out = self.mod_transformer(
            mod_in, coords=mod_coords, key_padding_mask=module_key_padding_mask)          # [M, 1+L_vox, D_mod]

        # ----------------- Scatter back to voxels -----------------
        vox_out = module_seq_out[:, 1:, :].reshape(-1, self.d_mod)                        # [M*L_vox, D_mod]
        idxs_flat = padded_idxs.reshape(-1)                                               # [M*L_vox]
        valid = idxs_flat >= 0
        intra_module_ctx = torch.zeros_like(initial_voxel_feats)                          # [N_vox, D_mod]
        intra_module_ctx[idxs_flat[valid]] = vox_out[valid]

        # ----------------- Subtoken expansion -----------------
        module_cls = module_seq_out[:, 0, :]                                              # [M, D_mod]
        subtoks = self.subtoken_proj(module_cls).view(M, self.K, self.d_evt)              # [M, K, D_evt]
        mod_pos_exp = module_pos.unsqueeze(1).expand(-1, self.K)                          # [M, K]
        pe_mod2 = self.pos_emb(mod_pos_exp)                                               # [M, K, D_evt]
        sub_ids = torch.arange(self.K, device=device).unsqueeze(0).expand(M, -1)          # [M, K]
        pe_sub = self.pos_emb_sub(sub_ids)                                                # [M, K, D_evt]
        subtoks = subtoks + pe_mod2 + pe_sub                                              # [M, K, D_evt]
        flat_subtoks = subtoks.view(-1, self.d_evt)                                       # [M*K, D_evt]
        flat_mod2evt = module_to_event.repeat_interleave(self.K)                          # [M*K]

        # ----------------- Event-level Transformer -----------------
        # B = number of events in the batch
        # L_sub = max number of modules in an event * K
        B = int(module_to_event.max().item()) + 1
        glob_emb = self.global_feats_encoder(params_glob).unsqueeze(1)                    # [B, 1, D_evt]
        cls_tokens_evt = self.cls_evt.expand(B, 1, self.d_evt)                            # [B, 1, D_evt]
        #cls_tokens_evt = cls_tokens_evt + glob_emb                                       # [B, 1, D_evt]
        grouped_subs = [flat_subtoks[flat_mod2evt == b] for b in range(B)]

        padded_subs = pad_sequence(grouped_subs, batch_first=True)                        # [B, L_sub, D_evt]
        evt_in = self.dropout(torch.cat([cls_tokens_evt, padded_subs], dim=1))            # [B, 1+L_sub, D_evt]
        flat_mod_pos = mod_pos_exp.reshape(-1, 1)                                         # [M*K,1]
        grouped_pos = [flat_mod_pos[flat_mod2evt == b] for b in range(B)]
        padded_pos = pad_sequence(grouped_pos, batch_first=True)                          # [B, L_sub, 1]
        cls_pos = torch.zeros((B, 1, 1), device=device)                                   # [B, 1, 1]
        evt_coords = torch.cat([cls_pos, padded_pos], dim=1)                              # [B, 1+L_sub,1]
        event_key_padding_mask = torch.ones(
            (B, 1 + padded_subs.size(1)), dtype=torch.bool, device=device)                # [B, 1+L_sub]
        event_key_padding_mask[:, 0] = False
        event_key_padding_mask[:, 1:] = (
            torch.arange(padded_subs.size(1), device=device).unsqueeze(0)
            >= torch.tensor([g.size(0) for g in grouped_subs], device=device).unsqueeze(1)
        )
        event_seq_out = self.event_transformer(
            evt_in, coords=evt_coords, key_padding_mask=event_key_padding_mask)           # [B, 1+L_sub, D_evt]
        event_cls = event_seq_out[:, 0, :]                                                # [B, D_evt]
        event_cls = self.dropout(event_cls)                                               # [B, D_evt]

        # ----------------- Collapse and fusion -----------------
        mod_out = event_seq_out[:, 1:, :].reshape(-1, self.d_evt)                         # [B*L_sub, D_evt]
        idxs_evt_flat = pad_sequence(
            [(flat_mod2evt == b).nonzero(as_tuple=False).squeeze(1) for b in range(B)],
            batch_first=True,
            padding_value=-1
        ).reshape(-1)                                                                     # [B*L_sub]
        valid2 = idxs_evt_flat >= 0
        flat_ctx = torch.zeros_like(flat_subtoks)                                         # [M*K, D_evt]
        flat_ctx[idxs_evt_flat[valid2]] = mod_out[valid2]                                 # [M*K, D_evt]      
        subtok_out = flat_ctx.view(M, self.K, self.d_evt)                                 # [M, K, D_evt]
        module_ctx = self.subtokens_to_module(subtok_out.reshape(M, -1))                  # [M, D_mod]
        inter_module_ctx = module_ctx[voxel_to_module_map]                                # [N_vox, D_mod]
        combined_context = torch.cat([intra_module_ctx, inter_module_ctx], dim=1)         # [N_vox, D_mod*2]
        gate = torch.sigmoid(self.fusion_gate(combined_context))                          # [N_vox, D_mod]
        final_voxel_feat = gate * intra_module_ctx + (1 - gate) * inter_module_ctx        # [N_vox, D_mod]        
        x = ME.SparseTensor(
            features=final_voxel_feat,
            coordinate_manager=x.coordinate_manager,
            coordinate_map_key=x.coordinate_map_key
        )

        # ----------------- Decoder -----------------
        for i in range(self.nb_dlayers):
            x = self.upsample_layers[i](x)
            x = self.decoder_layers[i](x)

        out_primlepton = self.primlepton_layer(x)
        out_seg = self.seg_layer(x)
        out_charge = self.charge_layer(x)
        out_iscc = self.iscc_layer(event_cls)
        return {
            "out_primlepton": out_primlepton,
            "out_seg": out_seg,
            "out_charge": out_charge,
            "out_iscc": out_iscc,
        }
        
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

