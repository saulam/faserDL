import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Block
from MinkowskiEngine import (
    SparseTensor,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU,
)

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
        

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., init_gamma=1e-4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gamma = nn.Parameter(torch.ones(dim) * init_gamma)

    def forward(self, q_in, kv_in):  # q_in: [B, Nt, C], kv_in: [B, Nc, C]
        B, Nt, C = q_in.shape
        _, Nc, _ = kv_in.shape
        q = self.q(q_in).reshape(B, Nt, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k(kv_in).reshape(B, Nc, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v(kv_in).reshape(B, Nc, self.num_heads, C // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale              # [B, h, Nt, Nc]
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, Nt, C)           # [B, Nt, C]
        x = self.proj(x)
        return self.proj_drop(x) * self.gamma
    

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


class GlobalFeatureEncoderSimple(nn.Module):
    """
    Flattens, concatenates, and projects the input features to a specified
    embedding dimension.
    """
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        hidden: bool = False,
        norm_layer: nn.Module = None,
    ):
        """
        Args:
            embed_dim (int): The target embedding dimension of the output token,
                             which should match the ViT's embedding dimension.
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # fasercal (10) + ecal (5*5) + hcal (9) + scalars (5)
        global_feature_dim = 10 + 25 + 9 + 5

        layers = []
        if hidden:
            layers += [
                nn.Linear(global_feature_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
            ]
        else:
            layers += [
                nn.Linear(global_feature_dim, embed_dim),
                nn.Dropout(dropout),
            ]

        if norm_layer is not None:
            layers.append(norm_layer(embed_dim))

        self.projector = nn.Sequential(*layers)

    def forward(self, x_glob) -> torch.Tensor:
        """
        Args (list):
            [
             fcal: Tensor of shape (batch, 10),
             ecal: Tensor of shape (batch, 5, 5),
             hcal: Tensor of shape (batch, 9),
             scalars: Tensor of shape (batch, 5),
            ]

        Returns:
            Tensor of shape (batch, embed_dim): the [GLOBAL] embedding
        """
        fcal, ecal, hcal, scalars = x_glob
        batch_size = fcal.shape[0]
        ecal_flat = ecal.view(batch_size, -1)
        combined_globals = torch.cat([fcal, ecal_flat, hcal, scalars], dim=1)
        global_token = self.projector(combined_globals)        
        return global_token

        
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

    def forward(self, x_glob) -> torch.Tensor:
        """
        Args (list):
            [
             fcal: Tensor of shape (batch, 10),
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
    

class SeparableDCT3D(nn.Module):
    def __init__(self, patch_size, Kxyz):
        super().__init__()
        p_h, p_w, p_d = patch_size
        Kx, Ky, Kz = Kxyz
        def dct_1d(L, K, device=None, dtype=None):
            x = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(1)  # [L,1]
            k = torch.arange(K, dtype=torch.float32, device=device).unsqueeze(0)  # [1,K]
            M = torch.cos(torch.pi * (x + 0.5) * k / L)
            M[:, 0] /= torch.sqrt(torch.tensor(L, dtype=M.dtype, device=M.device))
            if K > 1:
                scale = torch.sqrt(torch.tensor(2.0 / L, dtype=M.dtype, device=M.device))
                M[:, 1:] *= scale
            return M
        self.register_buffer('Bx', dct_1d(p_h, Kx))  # [p_h, Kx]
        self.register_buffer('By', dct_1d(p_w, Ky))  # [p_w, Ky]
        self.register_buffer('Bz', dct_1d(p_d, Kz))  # [p_d, Kz]
        self.Kx, self.Ky, self.Kz = Kx, Ky, Kz
        self.P = p_h * p_w * p_d
        self.patch_size = (p_h, p_w, p_d)

    @property
    def K_total(self):
        return self.Kx * self.Ky * self.Kz

    def expand(self, coeff):  # coeff: [N, H, Kx, Ky, Kz]
        t = torch.einsum('nhkyz,ik->nhiyz', coeff, self.Bx)  # [N,H,p_h,Ky,Kz]
        t = torch.einsum('nhiyz,jy->nhijz', t, self.By)      # [N,H,p_h,p_w,Kz]
        t = torch.einsum('nhijz,kz->nhijk', t, self.Bz)      # [N,H,p_h,p_w,p_d]
        return t.reshape(t.size(0), t.size(1), -1)           # [N,H,P]


class SharedLatentVoxelHead(nn.Module):
    """
    One projection -> H×(Kx*Ky*Kz) coeffs
    Expand once to V ∈ R^{P×H}
    Then tiny 1x1 'convs' (linear) produce occ/cls/reg.
    """
    def __init__(self, in_dim, basis: SeparableDCT3D,
                 H=16, out_ch_reg=1, out_ch_cls=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.basis = basis
        self.H = H
        self.norm = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim, H * basis.K_total)

        # 1x1 "convs" from latent H to outputs at each voxel
        self.to_occ = nn.Linear(H, 1)
        self.to_reg = nn.Linear(H, out_ch_reg)
        self.to_cls = nn.Linear(H, out_ch_cls)

    def forward(self, token_emb):  # [N_tokens, D]
        x = self.norm(token_emb)
        coef = self.proj(x)                     # [N, H*Ktot]
        N = coef.size(0)
        coef = coef.view(N, self.H,
                         self.basis.Kx, self.basis.Ky, self.basis.Kz)
        V = self.basis.expand(coef)             # [N, H, P]

        # per-voxel linear maps (vectorized)
        Vt = V.transpose(1, 2)                  # [N, P, H]
        occ = self.to_occ(Vt).squeeze(-1)       # [N, P]
        reg = self.to_reg(Vt)                   # [N, P, out_ch_reg]
        cls = self.to_cls(Vt)                   # [N, P, out_ch_cls]
        return occ, reg, cls


class CylindricalHeadNormalized(nn.Module):
    """
    pT:  uT = mu_uT + sigma_uT*zT,  pT = k_T * expm1(uT)   (>=0)
    pz:  uZ = mu_uZ + sigma_uZ*zz,  pz = k_Z * expm1(uZ)   (>=0)
    phi: via normalised (cos_phi, sin_phi)
    """
    def __init__(self, k_T, mu_uT, sigma_uT, k_Z, mu_uZ, sigma_uZ, hidden=128):
        super().__init__()
        self.k_T, self.mu_uT, self.sigma_uT = float(k_T), float(mu_uT), float(max(sigma_uT,1e-8))
        self.k_Z, self.mu_uZ, self.sigma_uZ = float(k_Z), float(mu_uZ), float(max(sigma_uZ,1e-8))
        self.mlp = nn.Linear(hidden, 4)

    def forward(self, x, eps=1e-8):
        zT, a, b, zz = self.mlp(x).unbind(-1)

        # angle
        norm = torch.sqrt(a*a + b*b + eps)
        cos_phi = a / norm
        sin_phi = b / norm

        # pT via log1p/expm1
        uT = self.mu_uT + self.sigma_uT * zT
        pT = self.k_T * torch.expm1(uT)
        pT = torch.clamp(pT, min=0.0)

        # pz via log1p/expm1
        uZ = self.mu_uZ + self.sigma_uZ * zz
        pz = self.k_Z * torch.expm1(uZ)
        pz = torch.clamp(pz, min=0.0)

        px, py = pT * cos_phi, pT * sin_phi
        p_cart = torch.stack([px, py, pz], dim=-1)
        latents = torch.stack([zT, zz], dim=-1)

        return {
            "p_cart": p_cart,   # (B,3)
            "pT": pT,
            "cos_phi": cos_phi,
            "sin_phi": sin_phi,
            "pz": pz,
            "latents": latents
        }
    
    
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


class BlockDense(nn.Module):
    """ Dense ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, kernel_size=7, padding=3, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
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
        

class BlockSparse(nn.Module):
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
        