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
from torch.cuda.amp import autocast
from timm.models.layers import trunc_normal_, DropPath
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
        trunc_normal_(m.kernel)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, MinkowskiLinear):
        trunc_normal_(m.linear.weight)
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
    elif isinstance(m, nn.Conv2d):
        w = m.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv3d):
        w = m.weight.data
        trunc_normal_(w.view([w.shape[0], -1]), std=0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose3d):
        w = m.weight.data
        trunc_normal_(w.view(w.shape[0], -1))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
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
    else:
        for attr in ('mask_voxel_emb', 'mask_mod_emb', 'empty_mod_emb',
                     'cls_mod', 'cls_evt'):
            if hasattr(m, attr):
                init.normal_(getattr(m, attr), std=0.02)


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


class TransposeUpsampleDecoder(nn.Module):
    def __init__(self, input_dim, hidden_channels=16):
        super().__init__()
        
        self.init_dims = (6, 6, 5)
        self.hidden = hidden_channels
        self.fc = nn.Linear(input_dim,
                            hidden_channels * self.init_dims[0] * self.init_dims[1] * self.init_dims[2])
        self.deconv1 = nn.ConvTranspose3d(hidden_channels,
                                          hidden_channels,
                                          kernel_size=(4, 4, 2),
                                          stride=(4, 4, 2))
        self.deconv2 = nn.ConvTranspose3d(hidden_channels,
                                          hidden_channels,
                                          kernel_size=(2, 2, 2),
                                          stride=(2, 2, 2))
        self.conv_out = nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        B = x.size(0)
        x = self.fc(x)                                 # [B, hidden*6*6*5]
        x = x.view(B, self.hidden, *self.init_dims)    # [B, hidden, 6, 6, 5}
        x = self.act(self.deconv1(x))                  # [B, hidden, 24, 24, 10]
        x = self.act(self.deconv2(x))                  # [B, hidden, 48, 48, 20]
        x = self.conv_out(x)                           # [B, 1, 48, 48, 20]
        return x


class Upsample3DDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 decoder_dim: int = 64,
                 depth: int = 1,
                 init_size=(6,6,5),
                 patch_size=(8,8,4)
                ):
        super().__init__()
        H0, W0, D0 = init_size
        ph, pw, pd = patch_size
        pv = ph * pw * pd

        self.proj = nn.Linear(latent_dim, decoder_dim * H0 * W0 * D0)
        self.blocks = nn.Sequential(*[
            BlockDense(decoder_dim)
            for _ in range(depth)
        ])
        self.pred = nn.Conv3d(decoder_dim, pv, kernel_size=1)
        self.init_size = init_size
        self.patch_size = patch_size

    def forward(self, z):
        """
        z: [M_masked, latent_dim]
        returns: [M_masked, 1, 48, 48, 20]
        """
        M = z.size(0)
        H0, W0, D0 = self.init_size
        ph, pw, pd = self.patch_size

        x = self.proj(z)                                # [B, dec*H0*W0*D0]
        x = x.view(M, -1, H0, W0, D0)                   # [B, dec, 6, 6, 5]
        x = self.blocks(x)                              # [B, dec, 6, 6, 5]
        x = self.pred(x)                                # [B, ph*pw*pd, 6, 6, 5]
        x = x.permute(0, 2, 3, 4, 1)                    # [B, 6, 6, 5, pv]
        x = x.view(M, H0, W0, D0, ph, pw, pd)
        x = x.permute(0, 4, 1, 5, 2, 6, 3)              # [B, ph, 6, pw, 6, pd, 5]
        x = x.contiguous().view(M, 1,
                                ph*H0,
                                pw*W0,
                                pd*D0)                  # [B, 1, 48, 48, 20]
        return x


class MultiTaskUpsample3DDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 decoder_dim: int = 64,
                 depth: int = 1,
                 init_size=(6,6,5),
                 patch_size=(8,8,4),
                ):
        super().__init__()
        H0, W0, D0 = init_size
        ph, pw, pd = patch_size
        pv = ph * pw * pd  # Channels needed for voxel shuffling
        self.num_seg_classes = 3

        # --- Shared Backbone ---
        self.proj = nn.Linear(latent_dim, decoder_dim * H0 * W0 * D0)
        self.blocks = nn.Sequential(*[
            BlockDense(decoder_dim)
            for _ in range(depth)
        ])
        
        # --- Four Separate Prediction Heads ---
        # Head 1: Voxel occupancy (binary classification)
        self.pred_occupancy = nn.Conv3d(decoder_dim, pv, kernel_size=1)
        
        # Head 2: Voxel charge (regression)
        self.pred_charge = nn.Conv3d(decoder_dim, pv, kernel_size=1)

        # Head 3: Primary lepton (binary classification)
        self.pred_lepton = nn.Conv3d(decoder_dim, pv, kernel_size=1)

        # Head 4: Electromagnetic/hadronic/ghost (multi-class classification)
        self.pred_particle_type = nn.Conv3d(decoder_dim, self.num_seg_classes * pv, kernel_size=1)

        self.init_size = init_size
        self.patch_size = patch_size
        self.decoder_dim = decoder_dim

    def forward(self, z):
        """
        z: [M_masked, latent_dim]
        returns: dict with four tensors
        """
        M = z.size(0)
        H0, W0, D0 = self.init_size
        ph, pw, pd = self.patch_size

        # Shared backbone
        x_shared = self.proj(z)                                            # [M, C_in*H0*W0*D0]
        x_shared = x_shared.view(M, self.decoder_dim, H0, W0, D0)          # [M, C_in, H0, W0, D0]
        x_shared = self.blocks(x_shared)                                   # [M, C_in, H0, W0, D0]

        # Process all heads in parallel
        x_occ = self.pred_occupancy(x_shared)                              # [M, C_out, H0, W0, D0]
        x_chg = self.pred_charge(x_shared)                                 # [M, C_out, H0, W0, D0]
        x_lep = self.pred_lepton(x_shared)                                 # [M, C_out, H0, W0, D0]
        x_part = self.pred_particle_type(x_shared)                         # [M, C_out, H0, W0, D0]

        # Utility function for unpatching
        def unpatch(tensor, channels_per_voxel=1):
            out = tensor.permute(0, 2, 3, 4, 1)                            # [M, H0, W0, D0, C_out]
            out = out.view(M, H0, W0, D0, ph, pw, pd, channels_per_voxel)  # [M, H0, W0, D0, ph, pw, pd, C_voxel]
            out = out.permute(0, 7, 4, 1, 5, 2, 6, 3)                      # [M, C_voxel, ph, H0, pw, W0, pd, D0]
            final_shape = (M, channels_per_voxel, ph*H0, pw*W0, pd*D0)     # [M, C_voxel, ph*H0, pw*W0, pd*D0]
            return out.contiguous().view(final_shape)                      # [M, C_voxel, 48, 48, 20]

        # Head's output
        occ_logits = unpatch(x_occ, channels_per_voxel=1)
        pred_charge = unpatch(x_chg, channels_per_voxel=1)
        lepton_logits = unpatch(x_lep, channels_per_voxel=1)
        particle_type_logits = unpatch(x_part, channels_per_voxel=self.num_seg_classes)

        return {
            "occupancy_logits": occ_logits,         # Shape: [M, 1, 48, 48, 20]
            "pred_charge": pred_charge,             # Shape: [M, 1, 48, 48, 20]
            "primlepton_logits": lepton_logits,     # Shape: [M, 1, 48, 48, 20]
            "seg_logits": particle_type_logits      # Shape: [M, 3, 48, 48, 20]
        }
        

class ScaledFourierPosEmb3D(nn.Module):
    def __init__(self, num_features, d_model, init_scale=5):
        super().__init__()
        self.B = nn.Parameter(torch.randn(num_features, 3) * init_scale)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.proj = nn.Linear(2 * num_features, d_model)

    def forward(self, coords):
        proj_feats = coords @ self.B.t()
        pe = torch.cat([proj_feats.sin(), proj_feats.cos()], dim=-1)
        pe = self.alpha * pe
        return self.proj(pe)
        

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
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, coords: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, N_total, d = x.size()
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
            nn.GELU(),
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

        
class BlockDense(nn.Module):
    """ Dense ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        # small workaround (conv_depthwise3d not implemented for bf16)
        with autocast(enabled=False):
            x_fp32 = self.dwconv(x.float())
        x = x_fp32.to(input.dtype)
        
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)

        x = input + self.drop_path(x)
        return x
        

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

