# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# URL: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py

import numpy.random as random

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from MinkowskiEngine import (
    SparseTensor,
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU,
    MinkowskiGlobalAvgPooling,
)


import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()

        # Precompute [max_len × d_model] sinusoidal table once
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)   # shape = [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]
        

class RelPosSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, num_special_tokens=1, coord_scales=(12.0, 12.0, 5.0)):
        super().__init__()
        self.nhead = nhead
        self.scale = (d_model // nhead) ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.bias_alpha = nn.Parameter(torch.zeros(1))
        self.bias_mlp = nn.Sequential(
            nn.Linear(4, nhead * 4),
            nn.GELU(),
            nn.Linear(nhead * 4, nhead)
        )
        self.num_special_tokens = num_special_tokens
        #self.register_buffer('coord_scales', torch.tensor(coord_scales))  # [3]
        self.coord_scales = nn.Parameter(torch.tensor(coord_scales))

    def forward(self, x, coords, key_padding_mask=None):
        B, N, _ = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.nhead, -1).transpose(1, 2) for t in qkv]
        c = coords / self.coord_scales

        # relative positional bias
        ci = c.unsqueeze(2)  # [B,N,1,3]
        cj = c.unsqueeze(1)  # [B,1,N,3]
        dpos = ci - cj       # [B,N,N,3]
        dist = torch.norm(dpos, dim=-1, keepdim=True)             # [B, N, N, 1]
        dpos_with_norm = torch.cat([dpos, dist], dim=-1)          # [B, N, N, 4]
        bias = self.bias_mlp(dpos_with_norm).permute(0, 3, 1, 2)  # [B,heads,N,N]
        
        # zero out bias for special tokens
        ns = self.num_special_tokens
        if ns > 0:
            bias[:, :, :ns, :] = 0
            bias[:, :, :, :ns] = 0

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        logits = dots + self.bias_alpha * bias
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(mask, float('-inf'))

        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out(out)


class RelPosEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_special_tokens=1, dropout=0.1):
        super().__init__()
        self.self_attn = RelPosSelfAttention(d_model, nhead, num_special_tokens)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, src, coords, key_padding_mask=None):
        attn_out = self.self_attn(src, coords, key_padding_mask)
        src = src + self.dropout(attn_out)
        src = self.norm1(src)
        ffn = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout(ffn)
        return self.norm2(src)


class RelPosTransformer(nn.Module):
    """
    A stack of K RelPosEncoderLayer modules, each performing
    relative‐positional self‐attention + MLP. 

    Args:
      d_model (int): hidden dimension of each layer
      nhead (int): number of attention heads 
      num_special_tokens (int): how many “CLS‐like” tokens to zero out in bias
      depth (int): number of layers (K)
      dropout (float): dropout rate inside each RelPosEncoderLayer
    """
    def __init__(self, d_model: int, nhead: int, num_special_tokens: int = 1, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RelPosEncoderLayer(d_model=d_model, 
                               nhead=nhead, 
                               num_special_tokens=num_special_tokens, 
                               dropout=dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, coords: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, coords, key_padding_mask)
        return x


'''
class GlobalFeatureEncoder(nn.Module):
    """
    Encodes a set of global features consisting of four scalar energies and two small sequences.

    Inputs:
      - x: Tensor of shape (batch_size, 23) or (batch_size, 28)
        Order:
          [0]    rear_cal_energy
          [1]    rear_hcal_energy
          [2]    rear_mucal_energy
          [3]    faser_cal_energy
          [4:13] 9-module energy sequence (rear_hcal_modules)
          [13:]  10-or-15-module energy sequence (faser_cal_modules)

    Args:
      encoder_dim: int, dimensionality of the final embedding
      hidden_dim: int, hidden size for intermediate heads (defaults to encoder_dim)
    """
    def __init__(self, encoder_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or encoder_dim

        self.scalar_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, encoder_dim),
        )
        self.seqA_proj = nn.Linear(hidden_dim, encoder_dim)
        self.seqB_proj = nn.Linear(hidden_dim, encoder_dim)
        self.seqA_lstm = nn.LSTM(1, hidden_dim, batch_first=True, bidirectional=True)
        self.seqB_lstm = nn.LSTM(1, hidden_dim, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(encoder_dim)

    def forward(self, x):
        scalars = x[:, :4]
        seqA = x[:, 4:13].unsqueeze(-1)
        seqB = x[:, 13:].unsqueeze(-1)

        # Scalar path
        emb_s = self.scalar_mlp(scalars)              # (batch, encoder_dim)

        # Seq A
        _, (hA, _) = self.seqA_lstm(seqA)
        hA = hA.sum(0)                                # (batch, hidden_dim)
        emb_A = self.seqA_proj(hA)                    # (batch, encoder_dim)

        # Seq B
        _, (hB, _) = self.seqB_lstm(seqB)
        hB = hB.sum(0)                                # (batch, hidden_dim)
        emb_B = self.seqB_proj(hB)                    # (batch, encoder_dim)

        # Fuse into a single “token” and normalise
        global_token = emb_s + emb_A + emb_B
        return self.norm(global_token)
'''
class GlobalFeatureEncoder(nn.Module):
    """
    Encodes a set of global features consisting of four scalar energies and two small sequences.

    Inputs:
      - x: Tensor of shape (batch_size, 23) or (batch_size, 28)
        Order:
          [0]    rear_cal_energy
          [1]    rear_hcal_energy
          [2]    rear_mucal_energy
          [3]    faser_cal_energy
          [4:13] 9-module energy sequence (rear_hcal_modules)
          [13:]  10-or-15-module energy sequence (faser_cal_modules)

    Args:
      hidden_dim: int, hidden size for intermediate heads
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.scalar_mlp = nn.Linear(4, hidden_dim)
        self.seqA_lstm = nn.LSTM(1, hidden_dim, batch_first=True, bidirectional=True)
        self.seqB_lstm = nn.LSTM(1, hidden_dim, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        scalars = x[:, :4]
        seqA = x[:, 4:13].unsqueeze(-1)
        seqB = x[:, 13:].unsqueeze(-1)

        # Scalar path
        emb_s = self.scalar_mlp(scalars)   # (batch, hidden_dim)

        # Seq A
        _, (hA, _) = self.seqA_lstm(seqA)
        emb_A = hA.sum(0)                  # (batch, hidden_dim)

        # Seq B
        _, (hB, _) = self.seqB_lstm(seqB)
        emb_B = hB.sum(0)                  # (batch, hidden_dim)

        # Fuse into a single “token” and normalise
        global_token = emb_s + emb_A + emb_B
        return self.norm(global_token)


class MinkowskiSE(nn.Module):
    def __init__(self, channels, glob_dim, reduction=16):
        """
        Squeeze-and-excitation block which combines voxel with global features
        Args:
            channels (int): The number of channels in the voxel features.
            glob_dim (int): The dimension of the global feature vector.
            reduction (int): Reduction ratio in the SE module.
        """
        super(MinkowskiSE, self).__init__()
        self.glob_transform = nn.Linear(glob_dim, channels)
        # The SE MLP: takes concatenated pooled branch features and transformed global features.
        self.fc = nn.Sequential(
            nn.Linear(2 * channels, channels // reduction, bias=True),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid()
        )
        self.global_pool = MinkowskiGlobalAvgPooling()

    def forward(self, voxel_feature, global_feature, global_weight):
        """
        Args:
            voxel_feature (ME.SparseTensor): Voxel pecific features.
            global_feature (torch.Tensor): Dense tensor from global features, shape [B, glob_dim].
        Returns:
            (ME.SparseTensor): SE-modulated features.
        """
        # Pool voxel features to get per-sample statistics.
        pooled = self.global_pool(voxel_feature)  # [B, channels]
        pooled_dense = pooled.F  # [B, channels]
        global_transformed = self.glob_transform(global_feature)  # [B, channels]
        global_transformed = global_weight * global_transformed
        combined = torch.cat([pooled_dense, global_transformed], dim=1)  # [B, 2 * channels]
        
        # Compute the channel scaling factors.
        scaling = self.fc(combined)  # [B, channels] with values in (0,1)
        batch_indices = voxel_feature.C[:, 0].long()  # shape [N_voxels]
        scaling_expanded = scaling[batch_indices]  # [N_voxels, channels]
        new_features = voxel_feature.F * scaling_expanded
        
        return SparseTensor(
            new_features, 
            coordinate_manager=voxel_feature.coordinate_manager,
            coordinate_map_key=voxel_feature.coordinate_map_key,
        )


class Block(nn.Module):
    """ Sparse ConvNeXtV2 Block. 

    Args:
        dim (int): Number of input channels.
        kernel_size (int): Size of input kernel.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=7, dilation=1, drop_path=0., D=3):
        super().__init__()
        
        self.dwconv = MinkowskiDepthwiseConvolution(
            dim, 
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
            dimension=D)
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim)   
        self.act = MinkowskiGELU()
        self.grn = MinkowskiGRN(4  * dim)
        self.pwconv2 = MinkowskiLinear(4 * dim, dim)
        self.drop_path = MinkowskiDropPath(drop_path)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)

        return x


class MinkowskiGRN(nn.Module):
    """ GRN layer for sparse tensors.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key

        Gx = torch.norm(x.F, p=2, dim=0, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return SparseTensor(
                self.gamma * (x.F * Nx) + self.beta + x.F,
                coordinate_map_key=in_key,
                coordinate_manager=cm)


class MinkowskiDropPath(nn.Module):
    """ Drop Path for sparse tensors.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(MinkowskiDropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        cm = x.coordinate_manager
        in_key = x.coordinate_map_key
        keep_prob = 1 - self.drop_prob
        mask = torch.cat([
            torch.ones(len(_)) if random.uniform(0, 1) > self.drop_prob
            else torch.zeros(len(_)) for _ in x.decomposed_coordinates
        ]).view(-1, 1).to(x.device)
        if keep_prob > 0.0 and self.scale_by_keep:
            mask.div_(keep_prob)
        return SparseTensor(
                x.F * mask,
                coordinate_map_key=in_key,
                coordinate_manager=cm)


class MinkowskiLayerNorm(nn.Module):
    """ Channel-wise layer normalization for sparse tensors.
    """
    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, input):
        output = self.ln(input.F)
        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

