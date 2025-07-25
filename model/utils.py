import numpy as np
import torch
import torch.nn as nn
from MinkowskiEngine import SparseTensor
from timm.models.vision_transformer import Attention, Block


# NOTE: patch works for timm version 0.6.13
class MaskableAttention(Attention):
    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            # if key padding mask (B, N) → expand to (B, 1, 1, N)
            if attn_mask.ndim == 2:
                mask = attn_mask[:, None, None, :].to(torch.bool)
            # if full mask (B, N, N) → expand to (B, 1, N, N)
            else:
                mask = attn_mask[:, None, :, :].to(torch.bool)
            attn = attn.masked_fill(~mask, float("-1e9"))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BlockWithMask(Block):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.attn = MaskableAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
        

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int (for cubic grid) or tuple of ints (H, W, D)
    returns:
      pos_embed: [H*W*D, embed_dim] or [cls_tokens+H*W*D, embed_dim] if cls_tokens > 0
    """
    if isinstance(grid_size, int):
        H = W = D = grid_size
    else:
        H, W, D = grid_size

    h = np.arange(H, dtype=np.float32)
    w = np.arange(W, dtype=np.float32)
    d = np.arange(D, dtype=np.float32)
    grid_h, grid_w, grid_d = np.meshgrid(h, w, d, indexing='ij')
    grid = np.stack([grid_h, grid_w, grid_d], axis=0)
    grid = grid.reshape(3, 1, H, W, D).reshape(3, -1)

    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        # prepend a zero-vector for class tokens
        cls = np.zeros((1, embed_dim), dtype=np.float32)
        pos_embed = np.vstack((cls, pos_embed))
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    grid: np.ndarray of shape (3, N)
    returns: (N, embed_dim)
    """
    assert embed_dim % 3 == 0, "embed_dim must be divisible by 3"
    dim_each = embed_dim // 3
    # each dim_each must be even so that it can be split sin/cos
    assert dim_each % 2 == 0, "embed_dim/3 must be even"

    emb_h = get_1d_sincos_pos_embed_from_grid(dim_each, grid[0])  # (N, dim_each)
    emb_w = get_1d_sincos_pos_embed_from_grid(dim_each, grid[1])  # (N, dim_each)
    emb_d = get_1d_sincos_pos_embed_from_grid(dim_each, grid[2])  # (N, dim_each)
    
    return np.concatenate([emb_h, emb_w, emb_d], axis=1)  # (N, 3*dim_each = embed_dim)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: must be even
    pos: 1D array of positions, shape (N,)
    returns: (N, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)              # (dim/2,)

    pos = pos.reshape(-1)                      # (N,)
    angles = np.einsum('n,d->nd', pos, omega)  # (N, dim/2)

    emb_sin = np.sin(angles)                   # (N, dim/2)
    emb_cos = np.cos(angles)                   # (N, dim/2)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class GlobalFeatureEncoder(nn.Module):
    """
    Encodes global detector information (FASERCal, Rear ECal, Rear HCal, scalars)
    into a single embedding for use as a [GLOBAL] token.

    Args:
        embed_dim (int): hidden size.
        dropout (float): dropout probability.
    """
    def __init__(self,
                 embed_dim: int = 384,
                 dropout: float = 0.1,
                ):
        super().__init__()

        # ECal conv encoder
        self.ecal_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # (batch, ecal_hidden_dim, 1, 1)
            nn.Flatten()              # (batch, ecal_hidden_dim)
        )

        # FASERCal lstm encoder
        self.fcal_lstm = nn.LSTM(
            input_size=1,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # HCal lstm encoder
        self.hcal_lstm = nn.LSTM(
            input_size=1,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Scalars: single scalar → project → mu_hidden_dim
        self.scalars_proj = nn.Linear(5, embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x_glob) -> torch.Tensor:
        """
        Args (list):
            [
             fasercal: Tensor of shape (batch, 10),
             ecal: Tensor of shape (batch, 5, 5),
             hcal: Tensor of shape (batch, 9),
             scalars: Tensor of shape (batch, 5),
            ]

        Returns:
            Tensor of shape (batch, embed_dim): the [GLOBAL] embedding
        """
        fcal, ecal, hcal, scalars_in = x_glob

        # FASERCal path
        x_fcal = fcal.unsqueeze(-1)
        outputs, (h_n, _) = self.fcal_lstm(x_fcal)
        fcal_feat = h_n[-2] + h_n[-1]
        
        # ECal path
        x_ecal = ecal.unsqueeze(1)
        ecal_feat = self.ecal_encoder(x_ecal)
        
        # HCal path
        x_hcal = hcal.unsqueeze(-1)
        outputs, (h_n, _) = self.hcal_lstm(x_hcal)
        hcal_feat = h_n[-2] + h_n[-1]

        # Scalars path
        scalars_feat = self.scalars_proj(scalars_in)
        
        # Combine
        global_embed = fcal_feat + ecal_feat + hcal_feat + scalars_feat
        return self.dropout(global_embed)


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
