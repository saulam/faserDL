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
import torch.nn.init as init
from timm.models.layers import trunc_normal_
from MinkowskiEngine import (
    SparseTensor,
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU,
    MinkowskiGlobalAvgPooling,
)


def _init_weights(m):
    """Custom weight initialization for various layers."""
    if isinstance(m, MinkowskiConvolution):
        trunc_normal_(m.kernel, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, MinkowskiDepthwiseConvolution):
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
    elif isinstance(m, nn.Embedding):
        nn.init.trunc_normal_(m.weight, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                hidden_size = m.hidden_size
                param.data[hidden_size:2*hidden_size].fill_(1)
    elif hasattr(m, 'mask_voxel_emb'):
        trunc_normal_(m.mask_voxel_emb, std=0.02)
    elif hasattr(m, 'empty_mod_emb'):
        trunc_normal_(m.empty_mod_emb, std=0.02)
    elif hasattr(m, 'cls_mod'):
        trunc_normal_(m.cls_mod, std=0.02)
    elif hasattr(m, 'cls_task'):
        trunc_normal_(m.cls_task, std=0.02)
    elif hasattr(m, 'iscc_token'):
        trunc_normal_(m.iscc_token, std=0.02)   
    elif hasattr(m, 'special_token_embeddings'):
        trunc_normal_(m.special_token_embeddings, std=0.02)   


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


class PositionalEncoding3D(nn.Module):
    """
    3D sinusoidal positional encoding.

    Args:
        L (int): Number of frequency bands per axis. Produces 2*L features per axis, total out_dim = 6 * L.
    """
    def __init__(self, L: int):
        super(PositionalEncoding3D, self).__init__()
        self.L = L
        freqs = 2. ** torch.arange(L).float() * torch.pi
        self.register_buffer('freqs', freqs)

    @property
    def out_dim(self) -> int:
        """Total output dimension of the encoding (6 * L)."""
        return 6 * self.L

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x, y, z = coords.unbind(-1)
        xp = x.unsqueeze(-1) * self.freqs
        yp = y.unsqueeze(-1) * self.freqs
        zp = z.unsqueeze(-1) * self.freqs

        # Compute sin/cos for each
        sin_x, cos_x = torch.sin(xp), torch.cos(xp)
        sin_y, cos_y = torch.sin(yp), torch.cos(yp)
        sin_z, cos_z = torch.sin(zp), torch.cos(zp)

        # Concatenate along feature dimension
        pos_enc = torch.cat([sin_x, cos_x, sin_y, cos_y, sin_z, cos_z], dim=-1)
        return pos_enc
        

class RelPosSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, num_dims, num_special_tokens=1):
        super().__init__()
        self.bias_alpha = nn.Parameter(torch.zeros(1))
        self.bias_mlp = nn.Sequential(
            nn.Linear(num_dims + 1 if num_dims > 1 else num_dims, nhead * 4),
            nn.GELU(),
            nn.Linear(nhead * 4, nhead)
        )
        self.nhead = nhead
        self.scale = (d_model // nhead) ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.num_special_tokens = num_special_tokens

    def forward(self, x, coords, key_padding_mask=None):
        B, N, _ = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.nhead, -1).transpose(1, 2) for t in qkv]
        c = coords

        # relative positional bias
        ci = c.unsqueeze(2)  # [B,N,1,C]
        cj = c.unsqueeze(1)  # [B,1,N,C]
        dpos = ci - cj       # [B,N,N,C]
        if dpos.size(-1) > 1:
            dist = torch.norm(dpos, dim=-1, keepdim=True)  # [B,N,N,1]
            dpos = torch.cat([dpos, dist], dim=-1)         # [B,N,N,C+1]
        bias = self.bias_mlp(dpos).permute(0, 3, 1, 2)     # [B,heads,N,N]
        
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
        return self.out(out), attn


class RelPosEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_special_tokens=1, num_dims=1, dropout=0.1):
        super().__init__()
        self.self_attn = RelPosSelfAttention(d_model, nhead, num_dims, num_special_tokens)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, src, coords, key_padding_mask=None):
        attn_out, _ = self.self_attn(src, coords, key_padding_mask)
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
    def __init__(self, d_model: int, nhead: int, num_special_tokens: int = 1, num_layers: int = 6, 
                 num_dims: int = 3, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead==0, "d_model not divisible by number of heads"
        self.layers = nn.ModuleList([
            RelPosEncoderLayer(d_model=d_model, 
                               nhead=nhead, 
                               num_special_tokens=num_special_tokens,
                               num_dims=num_dims,
                               dropout=dropout)
            for _ in range(num_layers)
        ])
        self.num_special_tokens = num_special_tokens
        self.special_token_embeddings = nn.Parameter(torch.zeros(num_special_tokens, d_model))
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, coords: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, N_total, d = x.size()
        N_regular = N_total - self.num_special_tokens
        special_tokens_emb = self.special_token_embeddings.unsqueeze(0).expand(B, -1, -1)
        x[:, :self.num_special_tokens, :] = x[:, :self.num_special_tokens, :] + special_tokens_emb
        for layer in self.layers:
            x = layer(x, coords, key_padding_mask)
        return x


class GlobalFeatureEncoder(nn.Module):
    """
    Encodes global detector information (Rear ECal, Rear HCal, scalars)
    into a single embedding for use as a [GLOBAL] token.

    Args:
        ecal_hidden_dim (int): Channels for the ECal Conv2D.
        hcal_hidden_dim (int): Hidden size per direction for HCal BiLSTM.
        scalar_hidden_dim (int): Hidden size for MuTag projection.
        d_model (int): Final embedding dimension (transformer hidden size).
    """
    def __init__(self,
                 d_model: int = 512,
                 dropout: float = 0.1,
                ):
        super().__init__()

        # ECal conv encoder
        self.ecal_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d_model, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (batch, ecal_hidden_dim, 1, 1)
            nn.Flatten()              # (batch, ecal_hidden_dim)
        )
        
        # HCal lstm encoder
        self.hcal_lstm = nn.LSTM(
            input_size=1,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Scalars: single scalar → project → mu_hidden_dim
        self.scalars_proj = nn.Linear(5, d_model)
        
        # Fuse all three → project to d_model
        self.dropout = nn.Dropout(0.1)

    def forward(self,
                x_glob) -> torch.Tensor:
        """
        Args (list):
            [
             ecal: Tensor of shape (batch, 5, 5),
             hcal: Tensor of shape (batch, 9),
             scalars: Tensor of shape (batch, 5),
            ]

        Returns:
            Tensor of shape (batch, d_model): the [GLOBAL] embedding
        """
        ecal, hcal, scalars_in = x_glob
        
        # ECal path
        x_ecal = ecal.unsqueeze(1)
        ecal_feat = self.ecal_encoder(x_ecal)
        
        # HCal path
        x_hcal = hcal.unsqueeze(-1)
        outputs, (h_n, _) = self.hcal_lstm(x_hcal)
        hcal_feat = h_n[-2] + h_n[-1]

        # Scalars path
        scalars_feat = self.scalars_proj(scalars_in)
        
        # Fuse and project
        global_embed = ecal_feat + hcal_feat + scalars_feat
        return self.dropout(global_embed)


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

