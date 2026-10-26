"""
Description: model helpers with no spconv or timm dependency, shared by the
spconv models (via model.utils) and the spconv-free flash-attention models.
"""

import numpy as np
import torch
import torch.nn as nn


def _split_even_3(embed_dim):
    """
    Split embed_dim into 3 even integers that sum to embed_dim.
    If embed_dim is odd, we'll use embed_dim-1 and pad 1 dim later.
    """
    pad = embed_dim % 2
    E = embed_dim - pad  # make even

    base = (E // 3)
    base -= base % 2     # make base even

    dims = [base, base, base]
    remaining = E - sum(dims)  # remaining is even

    # distribute remaining in steps of 2, round-robin
    i = 0
    while remaining > 0:
        dims[i] += 2
        remaining -= 2
        i = (i + 1) % 3

    return dims, pad  # pad is 0 or 1


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
    grid = grid.reshape(3, -1)
    
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
    dims, pad = _split_even_3(embed_dim)
    dh, dw, dd = dims

    emb_h = get_1d_sincos_pos_embed_from_grid(dh, grid[0])  # (N, dh)
    emb_w = get_1d_sincos_pos_embed_from_grid(dw, grid[1])  # (N, dw)
    emb_d = get_1d_sincos_pos_embed_from_grid(dd, grid[2])  # (N, dd)

    out = np.concatenate([emb_h, emb_w, emb_d], axis=1)     # (N, dh+dw+dd)

    if pad:  # embed_dim was odd; add a zero channel
        out = np.concatenate([out, np.zeros((out.shape[0], 1), dtype=out.dtype)], axis=1)

    return out  # (N, embed_dim)


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


class CylindricalHeadNormalized(nn.Module):
    """
    pT:  uT = mu_uT + sigma_uT*zT,  pT = k_T * expm1(uT)
    pz:  uZ = mu_uZ + sigma_uZ*zz,  pz = k_Z * expm1(uZ)
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

        # pz via log1p/expm1
        uZ = self.mu_uZ + self.sigma_uZ * zz
        pz = self.k_Z * torch.expm1(uZ)

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


def choose_k1_k2(patch_size, max_k1_vol=32):
    # pick small per-axis factors so ∏k1 <= max_k1_vol and k1|patch
    k1 = []
    vol = 1
    for p in patch_size:
        f = 1
        for cand in (4, 3, 2):
            if p % cand == 0 and vol * cand <= max_k1_vol:
                f = cand
                break
        k1.append(f)
        vol *= f
    k2 = [p // f for p, f in zip(patch_size, k1)]
    return tuple(k1), tuple(k2)
