# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# URL: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py

import numpy.random as random

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
      dropout: float, dropout probability
    """
    def __init__(self,
                 encoder_dim: int,
                 hidden_dim: int = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = encoder_dim

        self.scalar_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.seqA_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.seqB_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        fused_dim = hidden_dim * 3
        self.global_mlp = nn.Sequential(
            nn.Linear(fused_dim, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, total_dim)
        batch_size, total_dim = x.shape
        
        # Split scalars and sequences
        scalars = x[:, :4]
        seqA = x[:, 4:13].unsqueeze(-1)  # (batch, 9, 1)
        seqB = x[:, 13:].unsqueeze(-1)  # (batch, 10 or 15, 1)

        # Embeddings
        emb_scalars = self.scalar_mlp(scalars)  # (batch, hidden_dim)
        outA, (hA, _) = self.seqA_lstm(seqA)
        embA = hA.squeeze(0)                    # (batch, hidden_dim)
        outB, (hB, _) = self.seqB_lstm(seqB)
        embB = hB.squeeze(0)                    # (batch, hidden_dim)

        # Fuse and project
        fused = torch.cat([emb_scalars, embA, embB], dim=-1)  # (batch, hidden_dim*3)
        out = self.global_mlp(fused)                          # (batch, encoder_dim)
        return out


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

