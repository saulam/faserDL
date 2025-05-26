"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 04.25

Description: PyTorch model - stage 1: semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.nn.utils.rnn import pad_sequence

import MinkowskiEngine as ME
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiLinear,
    MinkowskiGlobalMaxPooling,
    MinkowskiReLU,
    MinkowskiGELU,
)

from .utils import GlobalFeatureEncoder, Block, MinkowskiLayerNorm, MinkowskiSE


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



'''Model'''
class RelPosSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.scale = (d_model // nhead) ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model)
        
        self.coord_dim = 3  # Match the dimension of the coordinates
        self.bias_mlp = nn.Sequential(
            nn.Linear(self.coord_dim, nhead*4),
            nn.GELU(), 
            nn.Linear(nhead*4, nhead)
        )
        
        # For debugging
        self.attn_weights = None
        self.bias = None
        self.dots = None
    
    def forward(self, x, coords, src_key_padding_mask=None):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.nhead, -1).transpose(1, 2) for t in qkv]
        
        # coords: [B,N,3] as (x,y,z)
        ci, cj = coords.unsqueeze(2), coords.unsqueeze(1)  # [B,N,1,3], [B,1,N,3]
        dpos = ci - cj  # [B,N,N,3]
        bias = self.bias_mlp(dpos).permute(0, 3, 1, 2)  # [B,heads,N,N]
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        logits = dots + bias
        self.bias = bias 
        self.dots = dots 
        
        if src_key_padding_mask is not None:
            key_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(key_mask, float('-inf'))
        
        attn = torch.softmax(logits, dim=-1)
        self.attn_weights = attn  # Store for debugging
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out(out)


class RelPosEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = RelPosSelfAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model*4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 
        self.act = nn.ReLU()
    
    def forward(self, src, coords, src_key_padding_mask=None):
        src2 = self.self_attn(src, coords, src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout(src2)
        return self.norm2(src)


class SparseRelPosAttentionAdapter(nn.Module):
    """
    Adapter class that converts SparseTensor to dense tensors for RelPosSelfAttention
    and converts the result back to SparseTensor
    """
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        print("SparseRelPosAttentionAdapter initialized")
        self.layers = nn.ModuleList([                     
            RelPosEncoderLayer(d_model, nhead, dropout),
            # RelPosEncoderLayer(d_model, nhead, dropout),
        ])
    
    def forward(self, x: ME.SparseTensor):
        features = x.F
        coordinates = x.C  # Get coordinates including batch index

        batch_indices = coordinates[:, 0]  # Extract batch indices

        
        # Group features by batch
        unique_batches = batch_indices.unique(sorted=True)
        grouped_features = [features[batch_indices == b] for b in unique_batches]
        grouped_coords = [coordinates[batch_indices == b, 1:].float() for b in unique_batches]  # Spatial coords only
        
        # Get batch sizes
        batch_sizes = [f.shape[0] for f in grouped_features]
        
        # Create padding mask for attention
        max_len = max(batch_sizes)
        B = len(unique_batches)
        key_padding_mask = torch.zeros((B, max_len), dtype=torch.bool, device=features.device)
        for i, size in enumerate(batch_sizes):
            key_padding_mask[i, size:] = True
        
        # Pad sequences for batch processing
        padded_features = pad_sequence(grouped_features, batch_first=True)  # [B, N_max, D]
        padded_coords = pad_sequence(grouped_coords, batch_first=True)  # [B, N_max, 3]
        
        # Process through the layers
        out = padded_features
        for layer in self.layers:
            out = layer(out, padded_coords, key_padding_mask)
        
        # Unpad to recover sparse structure
        unpadded = torch.cat([out[i, :size] for i, size in enumerate(batch_sizes)], dim=0)
        
        # Return as SparseTensor with original coordinate structure

        result = ME.SparseTensor(
            features=unpadded,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager
        )

        return result
    


class MinkAttUNetConvNeXtV2(nn.Module):
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
        dilation_ds = (2, 2, 2)
        block_kernel = (3, 3, 3)
        drop_path_rate = 0.1
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
        #self.stem_ch = nn.Linear(in_channels, encoder_dims[0])
        #self.stem_mod = nn.Embedding(self.module_size, encoder_dims[0]) 
        #self.stem_ln = MinkowskiLayerNorm(encoder_dims[0], eps=1e-6)

        # Global features
        self.register_buffer("global_weight", torch.tensor(1.0))  # Scalar controlling global parameter contribution
        self.global_feats_encoder = GlobalFeatureEncoder(encoder_dim=encoder_dims[0], dropout=0.3)

        # Build encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.se_layers = nn.ModuleList()
        for i in range(self.nb_elayers):
            # Encoder layer
            encoder_layer = nn.Sequential(
                *[Block(dim=encoder_dims[i], kernel_size=block_kernel, dilation=dilation_ds,
                        drop_path=dp_rates_enc[dp_cur + j], D=D)
                  for j in range(encoder_depths[i])]
            )
            self.encoder_layers.append(encoder_layer)
            dp_cur += encoder_depths[i]

            # SE layer
            se_layer = MinkowskiSE(channels=encoder_dims[i], glob_dim=encoder_dims[0], reduction=se_block_reds[i])
            self.se_layers.append(se_layer)

            # Downsampling layer
            if i < self.nb_elayers - 1:
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
                self.downsample_layers.append(downsample_layer)

        """Layer Normalization"""
        self.layer_norm_ba = MinkowskiLayerNorm(encoder_dims[-1], eps=1e-6)

        """Offset Attention module"""
        self.offset_attn = SparseRelPosAttentionAdapter(d_model=encoder_dims[-1])


        # Decoder configuration (reversing the encoder dimensions)
        decoder_depths = [2] * len(encoder_depths)
        decoder_dims = list(reversed(encoder_dims))
        kernel_size_us = kernel_size_ds
        dilation_us = dilation_ds
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
                *[Block(dim=decoder_dims[i + 1], kernel_size=block_kernel, dilation=dilation_us,
                        drop_path=dp_rates_dec[dp_cur_dec + j], D=D)
                  for j in range(decoder_depths[i])]
            )
            self.decoder_layers.append(decoder_layer)
            dp_cur_dec += decoder_depths[i]

        # Semantic segmentation prediction layers.
        self.primlepton_layer = nn.Sequential(
            MinkowskiLayerNorm(decoder_dims[-1], eps=1e-6),
            Block(dim=decoder_dims[-1], kernel_size=block_kernel, dilation=dilation_us, drop_path=0.0, D=D),
            MinkowskiConvolution(decoder_dims[-1], 1, kernel_size=1, stride=1, dimension=D),
        )
        self.seg_layer = nn.Sequential(
            MinkowskiLayerNorm(decoder_dims[-1], eps=1e-6),
            Block(dim=decoder_dims[-1], kernel_size=block_kernel, dilation=dilation_us, drop_path=0.0, D=D),
            MinkowskiConvolution(decoder_dims[-1], 3, kernel_size=1, stride=1, dimension=D),
        )

        # Initialise weights
        self.apply(_init_weights)

    def forward(self, x, x_glob):
        """
        Forward pass through the encoder-decoder network.

        Args:
            x: Input sparse tensor.
            x_glob: Global feature tensor.

        Returns:
            A dictionary with voxel predictions.
        """        
        # Stem and global feature MLP transformation.
        #coords, charge = x.C, x.F    # feats: [N,2]
        #mod_id = (coords[:, 3] % self.module_size).long()  # [N]
        #charge_emb   = self.stem_ch(charge)  # [N, module_emb_dim]
        #mod_id_emb   = self.stem_mod(mod_id) # [N, module_emb_dim]
        #new_feats = charge_emb + mod_id_emb  # [N, module_emb_dim]
        #x = ME.SparseTensor(
        #    new_feats, 
        #    coordinate_manager=x.coordinate_manager,
        #    coordinate_map_key=x.coordinate_map_key,
        #)        
        #x = self.stem_ln(x)
        x = self.stem(x)
        x_glob = self.global_feats_encoder(x_glob)

        # Encoder path with SE block integration.
        x_enc = []
        for i in range(self.nb_elayers):
            x = self.encoder_layers[i](x)
            x = self.se_layers[i](x, x_glob, self.global_weight)
            if i < self.nb_elayers - 1:
                x_enc.append(x)
                x = self.downsample_layers[i](x)
        
        # Layer Nomr
        x = self.layer_norm_ba(x)
        
        # # offset attention
        x = self.offset_attn(x)

        # Decoder path with skip connections
        x_enc = x_enc[::-1]
        out_cls = []
        for i in range(self.nb_dlayers):
            x = self.upsample_layers[i](x)
            x = x + x_enc[i]
            x = self.decoder_layers[i](x)

        out_primlepton = self.primlepton_layer(x)
        out_seg = self.seg_layer(x)
        return {"out_primlepton": out_primlepton, "out_seg": out_seg}

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

