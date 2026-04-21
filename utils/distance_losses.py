"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.26

Description:
    Distance-aware loss functions for sparse 3D neutrino interactions.

Public API:
    - unified_reconstruction_loss(...)
    - unified_semantic_segmentation_loss(...)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Optional, Sequence, Union

import torch
from torch.nn import functional as F

try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except Exception:
    distance_transform_edt = None
    _HAS_SCIPY = False

import numpy as np


# =========================
# Helpers
# =========================

def _as_bool(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.bool)

def _safe_ghost_mask_lookup(
    idx_targets: torch.Tensor,  # [...], -1 for empty
    ghost_mask: torch.Tensor,   # [N_hits] bool/int
) -> torch.Tensor:
    """
    Returns a boolean tensor same shape as idx_targets.
    Only indexes ghost_mask where idx_targets >= 0.
    """
    device = idx_targets.device
    ghost_mask = ghost_mask.to(device=device)
    out = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
    valid = idx_targets >= 0
    if ghost_mask.numel() == 0:
        return out
    valid = valid & (idx_targets < ghost_mask.numel())
    if valid.any():
        out[valid] = ghost_mask[idx_targets[valid]].to(dtype=torch.bool)
    return out

def _true_occupancy_from_targets(
    idx_targets: torch.Tensor,   # [M, P]
    ghost_mask: torch.Tensor,    # [N_hits]
    patch_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Builds true occupancy mask excluding ghosts.
    Returns float32 spatial tensor [M, H, W, D] with {0,1}.
    """
    device = idx_targets.device
    M, P = idx_targets.shape
    p_h, p_w, p_d = patch_shape
    if P != p_h * p_w * p_d:
        raise ValueError(f"P={P} does not match patch_shape product={p_h*p_w*p_d}")

    is_occ = idx_targets >= 0
    is_ghost = _safe_ghost_mask_lookup(idx_targets, ghost_mask)
    true_occ = (is_occ & ~is_ghost).to(dtype=torch.float32, device=device)
    return true_occ.view(M, p_h, p_w, p_d)


# =========================
# Distance transforms
# =========================

def compute_distance_transform_3d(
    occupancy_mask: torch.Tensor,  # [M, P] or [M, H, W, D]
    patch_shape: Tuple[int, int, int],
    max_distance: float = 10.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Exact Euclidean Distance Transform (EDT) using SciPy (CPU, non-differentiable).

    For each voxel, returns distance to nearest occupied voxel (occupied => distance 0).

    Notes:
      - SciPy's distance_transform_edt computes distance to the nearest *zero*.
        We therefore pass an array with zeros at occupied voxels and ones elsewhere.

    Returns:
        distance_map with same shape as occupancy_mask (float32).
    """
    if not _HAS_SCIPY:
        raise ImportError("SciPy is required for compute_distance_transform_3d but is not available.")

    device = occupancy_mask.device
    p_h, p_w, p_d = patch_shape
    P = p_h * p_w * p_d

    if occupancy_mask.dim() == 2:
        M, P_in = occupancy_mask.shape
        if P_in != P:
            raise ValueError(f"occupancy_mask.shape[1]={P_in} != P={P}")
        occ_spatial = occupancy_mask.view(M, p_h, p_w, p_d)
    elif occupancy_mask.dim() == 4:
        occ_spatial = occupancy_mask
        M = occ_spatial.shape[0]
    else:
        raise ValueError("occupancy_mask must be [M,P] or [M,H,W,D]")

    occ_np = occ_spatial.detach().cpu().numpy().astype(bool)  # True where occupied

    distance_maps = []
    for i in range(M):
        # input: 0 at occupied, 1 at empty
        inp = (~occ_np[i]).astype(np.uint8)
        dt = distance_transform_edt(inp)
        distance_maps.append(dt)

    distance_maps = np.stack(distance_maps, axis=0)
    dist = torch.from_numpy(distance_maps).to(device=device, dtype=torch.float32)

    dist = torch.clamp(dist, max=float(max_distance))
    if normalize:
        dist = dist / float(max_distance)

    if occupancy_mask.dim() == 2:
        dist = dist.view(M, P)

    return dist


def compute_distance_transform_conv3d(
    occupancy_mask: torch.Tensor,  # [M, H, W, D] float/bool
    patch_shape: Tuple[int, int, int],
    max_iterations: int = 5,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Fast DT-like approximation using iterative neighborhood propagation.
    Approximates a chamfer/Chebyshev-style distance in integer steps.

    Returns:
        dist: [M, H, W, D] float32 in [0, max_iterations] (or normalized).
    """
    device = occupancy_mask.device
    occ = occupancy_mask > 0.5

    dist = torch.full(occ.shape, float(max_iterations), device=device, dtype=torch.float32)
    dist = torch.where(occ, torch.zeros_like(dist), dist)

    dist = dist.unsqueeze(1)  # [M,1,H,W,D]
    occ1 = occ.unsqueeze(1)

    for _ in range(int(max_iterations)):
        padded = F.pad(dist, (1, 1, 1, 1, 1, 1), mode="replicate")
        neighbors_min = -F.max_pool3d(-padded, kernel_size=3, stride=1, padding=0)
        dist = torch.where(occ1, torch.zeros_like(dist), torch.minimum(dist, neighbors_min + 1.0))

    dist = dist.squeeze(1)
    if normalize and max_iterations > 0:
        dist = dist / float(max_iterations)
    return dist


def soft_distance_transform_conv3d(
    source_prob: torch.Tensor,  # [M,H,W,D] in [0,1]
    patch_shape: Tuple[int, int, int],
    max_iterations: int = 5,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Differentiable DT-like approximation from a soft occupancy field.

    dist ≈ sum_{t=0..T-1} (1 - coverage_t)
    where coverage_0 = source_prob, coverage_{t+1} = dilate(coverage_t) via max-pool.

    Returns:
        dist: [M,H,W,D] float32 in [0, max_iterations] (or normalized).
    """
    cov = source_prob.clamp(0.0, 1.0).to(dtype=torch.float32).unsqueeze(1)  # [M,1,H,W,D]
    dist = torch.zeros_like(cov)

    for _ in range(int(max_iterations)):
        dist = dist + (1.0 - cov)
        padded = F.pad(cov, (1, 1, 1, 1, 1, 1), mode="replicate")
        cov = F.max_pool3d(padded, kernel_size=3, stride=1, padding=0)

    dist = dist.squeeze(1)
    if normalize and max_iterations > 0:
        dist = dist / float(max_iterations)
    return dist


# =========================
# Distance-aware losses (internal pieces)
# =========================

def soft_chamfer_occupancy_loss(
    pred_occ_logits: torch.Tensor,   # [M,P] logits
    true_occ_spatial: torch.Tensor,  # [M,H,W,D] float {0,1}
    patch_shape: Tuple[int, int, int],
    *,
    temperature: float = 1.0,
    max_distance: float = 5.0,
    gamma_distance: float = 2.0,
    dt_from_true: Optional[torch.Tensor] = None,  # [M,H,W,D] distances
    use_conv_dt: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Symmetric Chamfer-style occupancy loss (trainable in both directions).

    pred->true uses dt_from_true (constant).
    true->pred uses dt_from_pred which is differentiable if use_conv_dt=True.

    distance penalty:
        w(d) = exp(-(d^g)/(max_d^g))
        penalty(d) = 1 - w(d)
    """
    device = pred_occ_logits.device
    M, P = pred_occ_logits.shape
    p_h, p_w, p_d = patch_shape
    max_d = float(max_distance)
    g = float(gamma_distance)
    eps = 1e-6

    pred_prob = torch.sigmoid(pred_occ_logits / max(float(temperature), 1e-6)).view(M, p_h, p_w, p_d)
    true_occ = true_occ_spatial.to(device=device, dtype=torch.float32)

    dt_steps = int(math.ceil(max_d))

    if dt_from_true is None:
        if use_conv_dt:
            dt_from_true = compute_distance_transform_conv3d(true_occ, patch_shape, max_iterations=dt_steps, normalize=False)
        else:
            dt_from_true = compute_distance_transform_3d(true_occ, patch_shape, max_distance=max_d, normalize=False)

    if use_conv_dt:
        dt_from_pred = soft_distance_transform_conv3d(pred_prob, patch_shape, max_iterations=dt_steps, normalize=False)
    else:
        # non-differentiable fallback
        dt_from_pred = compute_distance_transform_3d((pred_prob > 0.5).to(true_occ.dtype), patch_shape, max_distance=max_d, normalize=False)

    pred_prob_flat = pred_prob.view(M, P)
    true_occ_flat = true_occ.view(M, P)

    d_pt = dt_from_true.view(M, P).clamp(0.0, max_d)
    w_pt = torch.exp(-(d_pt ** g) / (max_d ** g + eps))
    loss_pred = (pred_prob_flat * (1.0 - w_pt)).sum() / (pred_prob_flat.sum() + eps)

    d_tp = dt_from_pred.view(M, P).clamp(0.0, max_d)
    w_tp = torch.exp(-(d_tp ** g) / (max_d ** g + eps))
    loss_true = (true_occ_flat * (1.0 - w_tp)).sum() / (true_occ_flat.sum() + eps)

    loss = 0.5 * (loss_pred + loss_true)

    pred_hard = pred_prob_flat > 0.5
    metrics = {
        "occ_chamfer/pred_to_true": loss_pred.detach(),
        "occ_chamfer/true_to_pred": loss_true.detach(),
        "occ_chamfer/mean_pred_dist": d_pt[pred_hard].mean().detach() if pred_hard.any() else torch.tensor(0.0, device=device),
        "occ_chamfer/mean_true_dist": d_tp[true_occ_flat > 0.5].mean().detach() if (true_occ_flat > 0.5).any() else torch.tensor(0.0, device=device),
    }
    return loss, metrics


def distance_weighted_regression_loss(
    pred_reg: torch.Tensor,        # [M, P*C] or [M,P,C]
    targ_reg: torch.Tensor,        # [N_hits, C]
    idx_targets: torch.Tensor,     # [M, P]
    ghost_mask: torch.Tensor,      # [N_hits]
    distance_map_flat: torch.Tensor,  # [M,P] distance to nearest true hit
    *,
    max_distance: float = 5.0,
    gamma_distance: float = 2.0,
    huber_delta: float = 1.0,
    reg_empty: Optional[torch.Tensor] = None,
    min_neg_weight: float = 0.0,  # set >0 only if you want always some pressure
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Distance-weighted regression loss.

    Intended behavior:
      - positives (true occupied, non-ghost): full weight 1
      - negatives: weight increases with distance from nearest true hit:
            w_neg = clamp(d/max_d, 0..1)^gamma
        so empties near hits are penalized less (encourages spatial tolerance).
    """
    device = pred_reg.device
    ghost_mask = ghost_mask.to(device=device)

    C = targ_reg.shape[1]
    M, P = idx_targets.shape
    max_d = float(max_distance)
    g = float(gamma_distance)
    eps = 1e-6

    if pred_reg.dim() == 2:
        pred_flat = pred_reg.view(-1, C)
    else:
        pred_flat = pred_reg.reshape(-1, C)

    idx_flat = idx_targets.reshape(-1)
    dist_flat = distance_map_flat.reshape(-1).to(device=device, dtype=torch.float32)

    is_occ = idx_flat >= 0
    is_ghost = torch.zeros_like(idx_flat, dtype=torch.bool, device=device)
    if is_occ.any():
        is_ghost[is_occ] = ghost_mask[idx_flat[is_occ]].to(dtype=torch.bool)
    pos_mask = is_occ & ~is_ghost

    if reg_empty is None:
        reg_empty = targ_reg.amin(dim=0)
    reg_empty = reg_empty.to(device=device, dtype=pred_flat.dtype)

    targ_flat = reg_empty.unsqueeze(0).expand(M * P, -1).clone()
    if pos_mask.any():
        targ_flat[pos_mask] = targ_reg[idx_flat[pos_mask]].to(device=device, dtype=pred_flat.dtype)

    reg_elem = F.smooth_l1_loss(pred_flat, targ_flat, beta=float(huber_delta), reduction="none")
    reg_row = reg_elem.sum(dim=1)

    d_norm = (dist_flat / max(max_d, eps)).clamp(0.0, 1.0)
    w_neg = d_norm.pow(g)
    if min_neg_weight > 0:
        w_neg = w_neg.clamp(min=float(min_neg_weight))

    weights = torch.where(pos_mask, torch.ones_like(w_neg), w_neg)

    loss = (reg_row * weights).sum() / (weights.sum() + eps)

    metrics = {
        "reg_dist/total": loss.detach(),
        "reg_dist/pos": (reg_row[pos_mask] * weights[pos_mask]).mean().detach() if pos_mask.any() else torch.tensor(0.0, device=device),
        "reg_dist/neg": (reg_row[~pos_mask] * weights[~pos_mask]).mean().detach() if (~pos_mask).any() else torch.tensor(0.0, device=device),
        "reg_dist/mean_weight": weights.mean().detach(),
    }
    return loss, metrics


def focal_distance_transform_occ_loss(
    pred_occ_logits: torch.Tensor,   # [M,P]
    true_occ_spatial: torch.Tensor,  # [M,H,W,D]
    patch_shape: Tuple[int, int, int],
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    max_distance: float = 5.0,
    distance_gamma: float = 2.0,
    dt_from_true: Optional[torch.Tensor] = None,
    use_conv_dt: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Experimental occupancy loss:
      - Builds soft targets from distance-to-true EDT/DT
      - Applies focal BCE with soft targets
    """
    device = pred_occ_logits.device
    M, P = pred_occ_logits.shape
    p_h, p_w, p_d = patch_shape

    max_d = float(max_distance)
    dg = float(distance_gamma)

    dt_steps = int(math.ceil(max_d))
    true_occ = true_occ_spatial.to(device=device, dtype=torch.float32)

    if dt_from_true is None:
        if use_conv_dt:
            dt_from_true = compute_distance_transform_conv3d(true_occ, patch_shape, max_iterations=dt_steps, normalize=False)
        else:
            dt_from_true = compute_distance_transform_3d(true_occ, patch_shape, max_distance=max_d, normalize=False)

    distance_map = dt_from_true.view(M, P).clamp(0.0, max_d)
    norm_d = (distance_map / max_d).clamp(0.0, 1.0)

    # soft target: 1 at hits, decays with distance
    soft_targets = torch.exp(-(norm_d ** dg))
    true_flat = true_occ.view(M, P)
    soft_targets = torch.where(true_flat > 0.5, torch.ones_like(soft_targets), soft_targets)

    # focal with soft targets
    pred_prob = torch.sigmoid(pred_occ_logits)
    pt = soft_targets * pred_prob + (1.0 - soft_targets) * (1.0 - pred_prob)
    focal_weight = (1.0 - pt).pow(float(gamma))

    bce = -(soft_targets * F.logsigmoid(pred_occ_logits) + (1.0 - soft_targets) * F.logsigmoid(-pred_occ_logits))

    alpha_t = soft_targets * float(alpha) + (1.0 - soft_targets) * (1.0 - float(alpha))
    loss = (alpha_t * focal_weight * bce).mean()

    metrics = {
        "occ_focal_dt/loss": loss.detach(),
        "occ_focal_dt/mean_soft_target": soft_targets.mean().detach(),
        "occ_focal_dt/mean_distance_empty": distance_map[true_flat < 0.5].mean().detach() if (true_flat < 0.5).any() else torch.tensor(0.0, device=device),
    }
    return loss, metrics


# =========================
# Semantic distance-aware loss
# =========================

def distance_aware_semantic_segmentation_loss(
    pred_logits: torch.Tensor,      # [M,P,C]
    idx_targets: torch.Tensor,      # [M,P]
    csr_labels: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (indptr, cls, weights)
    ghost_mask: torch.Tensor,       # [N_hits]
    patch_shape: Tuple[int, int, int],
    *,
    max_distance: float = 5.0,
    gamma_distance: float = 2.0,
    label_smoothing: float = 0.0,
    lambda_cp: float = 1e-3,
    class_threshold: float = 0.01,
    min_weight: float = 0.05,
    exclude_classes_from_dt: Optional[Sequence[int]] = None,
    voxel_keep_prob: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Distance-aware soft CE for semantic segmentation with soft (CSR) labels.

    Reweights per-voxel CE by exp(-d^g / max_d^g) where d is expected distance to
    the spatial support of the target mixture.

    exclude_classes_from_dt can be used to exclude "none" class(es) from DT weighting.
    """
    from utils.losses import build_soft_targets_from_csr, confidence_penalty
    from utils.funcs import csr_keep_rows_torch

    device = pred_logits.device
    ghost_mask = ghost_mask.to(device=device)

    M, P, C = pred_logits.shape
    p_h, p_w, p_d = patch_shape
    max_d = float(max_distance)
    g = float(gamma_distance)
    eps = 1e-6

    # valid voxels = occupied & non-ghost
    is_occ = idx_targets >= 0
    is_ghost = _safe_ghost_mask_lookup(idx_targets, ghost_mask)
    valid_mask = is_occ & ~is_ghost
    if voxel_keep_prob < 1.0:
        keep = torch.rand(valid_mask.shape, device=device) < float(voxel_keep_prob)
        valid_mask = valid_mask & keep
        # Ensure at least 1 supervised voxel per patch row if there was any valid voxel
        row_has_valid = valid_mask.any(dim=1)  # [M]
        row_has_any_before = (is_occ & ~is_ghost).any(dim=1)  # [M] original-valid rows
        need_fix = row_has_any_before & ~row_has_valid  # rows where we dropped all valids
        if need_fix.any():
            # pick 1 random valid voxel from the original valid set for those rows
            noise = torch.rand((M, P), device=device)
            orig_valid = (is_occ & ~is_ghost)
            noise = noise.masked_fill(~orig_valid, float('inf'))
            pick = noise.argmin(dim=1)  # [M]
            valid_mask[need_fix, pick[need_fix]] = True
    valid_flat = valid_mask.view(-1)
    N_valid = int(valid_flat.sum().item())

    if N_valid == 0:
        zero = pred_logits.sum() * 0.0
        return zero, {
            "semantic_dist/loss": zero.detach(),
            "semantic_dist/mean_distance": torch.tensor(0.0, device=device),
            "semantic_dist/mean_weight": torch.tensor(0.0, device=device),
            "semantic_dist/ce_unweighted": torch.tensor(0.0, device=device),
            "semantic_dist/max_distance": torch.tensor(0.0, device=device),
        }

    pred_valid = pred_logits.view(-1, C)[valid_flat]      # [N_valid,C]
    idx_valid = idx_targets.view(-1)[valid_flat]          # [N_valid]

    indptr, cls_ids, weights = csr_labels
    csr_valid = csr_keep_rows_torch(indptr, cls_ids, weights, idx_valid)[:3]

    soft_targets = build_soft_targets_from_csr(*csr_valid, num_classes=C, N=N_valid, ghost_mask=None)
    if label_smoothing > 0:
        eps_ls = float(label_smoothing)
        soft_targets = soft_targets * (1.0 - eps_ls) + eps_ls / float(C)

    # coords for valid voxels
    valid_indices = valid_mask.nonzero(as_tuple=False)  # [N_valid,2] (patch_idx, flat_pos)
    patch_indices = valid_indices[:, 0]
    flat_pos = valid_indices[:, 1]

    z = flat_pos // (p_w * p_d)
    y = (flat_pos % (p_w * p_d)) // p_d
    x = flat_pos % p_d

    excluded = set(exclude_classes_from_dt or ())
    included_classes = [c for c in range(C) if c not in excluded]

    if len(included_classes) == 0:
        log_probs = F.log_softmax(pred_valid, dim=-1)
        ce = -(soft_targets * log_probs).sum(dim=-1).mean()
        if lambda_cp > 0:
            ce = ce + confidence_penalty(pred_valid, lambda_cp)
        return ce, {
            "semantic_dist/loss": ce.detach(),
            "semantic_dist/mean_distance": torch.tensor(0.0, device=device),
            "semantic_dist/mean_weight": torch.tensor(1.0, device=device),
            "semantic_dist/ce_unweighted": ce.detach(),
            "semantic_dist/max_distance": torch.tensor(0.0, device=device),
        }

    # expected distance over included classes
    expected_distance = torch.zeros(N_valid, device=device, dtype=torch.float32)
    included_prob = soft_targets[:, included_classes].sum(dim=-1)
    included_prob_safe = included_prob.clamp_min(eps)

    dt_steps = int(math.ceil(max_d))

    # Compute DT per class support (expensive but correct).
    for c in included_classes:
        class_prob = soft_targets[:, c]
        has_class = class_prob > float(class_threshold)
        if not has_class.any():
            continue

        class_mask_spatial = torch.zeros((M, p_h, p_w, p_d), device=device, dtype=torch.float32)
        class_mask_spatial[
            patch_indices[has_class],
            z[has_class],
            y[has_class],
            x[has_class],
        ] = 1.0

        dt_c = compute_distance_transform_conv3d(class_mask_spatial, patch_shape, max_iterations=dt_steps, normalize=False)
        d_c = dt_c[patch_indices, z, y, x]  # [N_valid]
        expected_distance += d_c * class_prob.to(torch.float32)

    expected_distance = torch.where(
        included_prob > eps,
        expected_distance / included_prob_safe,
        torch.zeros_like(expected_distance),
    )

    dist_weight = torch.exp(-(expected_distance ** g) / (max_d ** g + eps)).clamp(min=float(min_weight))

    log_probs = F.log_softmax(pred_valid, dim=-1)
    ce_vec = -(soft_targets * log_probs).sum(dim=-1)

    loss = (ce_vec * dist_weight).mean()
    if lambda_cp > 0:
        loss = loss + confidence_penalty(pred_valid, lambda_cp)

    metrics = {
        "semantic_dist/loss": loss.detach(),
        "semantic_dist/mean_distance": expected_distance.mean().detach(),
        "semantic_dist/mean_weight": dist_weight.mean().detach(),
        "semantic_dist/ce_unweighted": ce_vec.mean().detach(),
        "semantic_dist/max_distance": expected_distance.max().detach() if expected_distance.numel() > 0 else torch.tensor(0.0, device=device),
        "semantic_dist/included_prob_mean": included_prob.mean().detach(),
    }
    return loss, metrics


# =========================
# Public unified APIs
# =========================

def unified_reconstruction_loss(
    targ_reg: torch.Tensor,         # [N_hits, C]
    pred_occ: torch.Tensor,         # [M, P] logits
    pred_reg: torch.Tensor,         # [M, P*C] or [M,P,C]
    idx_targets: torch.Tensor,      # [M, P]
    ghost_mask: torch.Tensor,       # [N_hits]
    hit_event_id: torch.Tensor,     # [N_hits]
    patch_shape: Tuple[int, int, int],
    dataset,
    preprocessing_input: str,
    *,
    loss_mode: str = "hybrid",      # "standard" | "hybrid" | "distance" | "focal_dt"
    # ----- standard occupancy -----
    occ_label_smoothing: float = 0.0,
    occ_focal_gamma: float = 1.5,
    occ_focal_alpha: Optional[float] = None,
    occ_dilate: int = 2,
    occ_empty_beta: float = 0.5,
    # ----- standard regression -----
    huber_delta: float = 1.0,
    reg_weight_lam: float = 1.0,
    reg_weight_alpha: float = 0.5,
    reg_weight_q0: Optional[float] = None,
    reg_weight_wmax: Optional[float] = None,
    per_event_mean: bool = False,
    # ----- distance-aware (chamfer + distance-weighted reg) -----
    chamfer_weight: float = 0.3,          # in hybrid/distance
    distance_reg_weight: float = 0.3,     # in hybrid/distance
    max_distance: float = 5.0,
    gamma_distance: float = 2.0,
    temperature_chamfer: float = 1.0,
    min_neg_weight: float = 0.0,
    use_conv_dt: bool = True,
    # ----- experimental focal_dt mode (occupancy only) -----
    focal_dt_alpha: float = 0.25,
    focal_dt_gamma: float = 1.5,
    focal_dt_distance_gamma: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Unified reconstruction loss.

    loss_mode:
      - "standard": voxel-wise supervision (your existing masked losses)
      - "distance": only distance-aware terms (Chamfer occ + distance-weighted reg)
      - "hybrid": standard + weighted distance-aware terms
      - "focal_dt": experimental occ loss using DT-soft-target focal, plus standard regression

    Returns:
      (loss_occ, loss_reg, metrics)
    """
    from utils.losses import reconstruction_losses_masked_simple

    if loss_mode not in {"standard", "hybrid", "distance", "focal_dt"}:
        raise ValueError(f"Unknown loss_mode='{loss_mode}'. Use 'standard','hybrid','distance','focal_dt'.")

    device = idx_targets.device
    M, P = idx_targets.shape

    true_occ_spatial = _true_occupancy_from_targets(idx_targets, ghost_mask, patch_shape)

    # Precompute dt(true) once (used by multiple modes)
    dt_true = None
    if loss_mode in {"hybrid", "distance", "focal_dt"}:
        dt_steps = int(math.ceil(float(max_distance)))
        if use_conv_dt:
            dt_true = compute_distance_transform_conv3d(true_occ_spatial, patch_shape, max_iterations=dt_steps, normalize=False)
        else:
            dt_true = compute_distance_transform_3d(true_occ_spatial, patch_shape, max_distance=float(max_distance), normalize=False)

    metrics: Dict[str, torch.Tensor] = {}

    # --------------------
    # STANDARD
    # --------------------
    if loss_mode == "standard":
        loss_occ, loss_reg, m = reconstruction_losses_masked_simple(
            targ_reg=targ_reg,
            pred_occ=pred_occ,
            pred_reg=pred_reg,
            idx_targets=idx_targets,
            ghost_mask=ghost_mask,
            hit_event_id=hit_event_id,
            patch_shape=patch_shape,
            dataset=dataset,
            preprocessing_input=preprocessing_input,
            label_smoothing=occ_label_smoothing,
            focal_gamma=occ_focal_gamma,
            focal_alpha=occ_focal_alpha,
            occ_dilate=occ_dilate,
            huber_delta=huber_delta,
            reg_weight_lam=reg_weight_lam,
            reg_weight_alpha=reg_weight_alpha,
            reg_weight_q0=reg_weight_q0,
            reg_weight_wmax=reg_weight_wmax,
            occ_empty_beta=occ_empty_beta,
            per_event_mean=per_event_mean,
        )
        return loss_occ, loss_reg, m

    # --------------------
    # DISTANCE ONLY
    # --------------------
    if loss_mode == "distance":
        # occupancy: chamfer
        loss_occ, m_occ = soft_chamfer_occupancy_loss(
            pred_occ_logits=pred_occ,
            true_occ_spatial=true_occ_spatial,
            patch_shape=patch_shape,
            temperature=temperature_chamfer,
            max_distance=max_distance,
            gamma_distance=gamma_distance,
            dt_from_true=dt_true,
            use_conv_dt=use_conv_dt,
        )
        # regression: distance weighted
        reg_empty = targ_reg.amin(dim=0)
        loss_reg, m_reg = distance_weighted_regression_loss(
            pred_reg=pred_reg,
            targ_reg=targ_reg,
            idx_targets=idx_targets,
            ghost_mask=ghost_mask,
            distance_map_flat=dt_true.view(M, P),
            max_distance=max_distance,
            gamma_distance=gamma_distance,
            huber_delta=huber_delta,
            reg_empty=reg_empty,
            min_neg_weight=min_neg_weight,
        )

        # apply weights (even in distance mode) so user can scale them
        loss_occ_total = float(chamfer_weight) * loss_occ
        loss_reg_total = float(distance_reg_weight) * loss_reg

        metrics.update(m_occ)
        metrics.update(m_reg)
        metrics["occ/total"] = loss_occ_total.detach()
        metrics["reg/total"] = loss_reg_total.detach()
        metrics["mode/is_distance"] = torch.tensor(1.0, device=device)

        return loss_occ_total, loss_reg_total, metrics

    # --------------------
    # FOCAL_DT (experimental occ) + standard reg
    # --------------------
    if loss_mode == "focal_dt":
        loss_occ, m_occ = focal_distance_transform_occ_loss(
            pred_occ_logits=pred_occ,
            true_occ_spatial=true_occ_spatial,
            patch_shape=patch_shape,
            alpha=focal_dt_alpha,
            gamma=focal_dt_gamma,
            max_distance=max_distance,
            distance_gamma=focal_dt_distance_gamma,
            dt_from_true=dt_true,
            use_conv_dt=use_conv_dt,
        )

        _, loss_reg, m_std = reconstruction_losses_masked_simple(
            targ_reg=targ_reg,
            pred_occ=pred_occ,
            pred_reg=pred_reg,
            idx_targets=idx_targets,
            ghost_mask=ghost_mask,
            hit_event_id=hit_event_id,
            patch_shape=patch_shape,
            dataset=dataset,
            preprocessing_input=preprocessing_input,
            label_smoothing=occ_label_smoothing,
            focal_gamma=occ_focal_gamma,
            focal_alpha=occ_focal_alpha,
            occ_dilate=occ_dilate,
            huber_delta=huber_delta,
            reg_weight_lam=reg_weight_lam,
            reg_weight_alpha=reg_weight_alpha,
            reg_weight_q0=reg_weight_q0,
            reg_weight_wmax=reg_weight_wmax,
            occ_empty_beta=occ_empty_beta,
            per_event_mean=per_event_mean,
        )

        metrics.update(m_std)
        metrics.update(m_occ)
        metrics["occ/total"] = loss_occ.detach()
        metrics["reg/total"] = loss_reg.detach()
        metrics["mode/is_focal_dt"] = torch.tensor(1.0, device=device)
        return loss_occ, loss_reg, metrics

    # --------------------
    # HYBRID
    # --------------------
    loss_occ_std, loss_reg_std, m_std = reconstruction_losses_masked_simple(
        targ_reg=targ_reg,
        pred_occ=pred_occ,
        pred_reg=pred_reg,
        idx_targets=idx_targets,
        ghost_mask=ghost_mask,
        hit_event_id=hit_event_id,
        patch_shape=patch_shape,
        dataset=dataset,
        preprocessing_input=preprocessing_input,
        label_smoothing=occ_label_smoothing,
        focal_gamma=occ_focal_gamma,
        focal_alpha=occ_focal_alpha,
        occ_dilate=occ_dilate,
        huber_delta=huber_delta,
        reg_weight_lam=reg_weight_lam,
        reg_weight_alpha=reg_weight_alpha,
        reg_weight_q0=reg_weight_q0,
        reg_weight_wmax=reg_weight_wmax,
        occ_empty_beta=occ_empty_beta,
        per_event_mean=per_event_mean,
    )
    metrics.update(m_std)

    loss_occ = loss_occ_std
    loss_reg = loss_reg_std

    if chamfer_weight > 0:
        loss_ch, m_ch = soft_chamfer_occupancy_loss(
            pred_occ_logits=pred_occ,
            true_occ_spatial=true_occ_spatial,
            patch_shape=patch_shape,
            temperature=temperature_chamfer,
            max_distance=max_distance,
            gamma_distance=gamma_distance,
            dt_from_true=dt_true,
            use_conv_dt=use_conv_dt,
        )
        loss_occ = loss_occ + float(chamfer_weight) * loss_ch
        metrics.update(m_ch)
        metrics["occ/chamfer_component"] = (float(chamfer_weight) * loss_ch).detach()

    if distance_reg_weight > 0:
        reg_empty = targ_reg.amin(dim=0)
        loss_dr, m_dr = distance_weighted_regression_loss(
            pred_reg=pred_reg,
            targ_reg=targ_reg,
            idx_targets=idx_targets,
            ghost_mask=ghost_mask,
            distance_map_flat=dt_true.view(M, P),
            max_distance=max_distance,
            gamma_distance=gamma_distance,
            huber_delta=huber_delta,
            reg_empty=reg_empty,
            min_neg_weight=min_neg_weight,
        )
        loss_reg = loss_reg + float(distance_reg_weight) * loss_dr
        metrics.update(m_dr)
        metrics["reg/distance_component"] = (float(distance_reg_weight) * loss_dr).detach()

    metrics["occ/total"] = loss_occ.detach()
    metrics["reg/total"] = loss_reg.detach()
    metrics["occ/standard_component"] = loss_occ_std.detach()
    metrics["reg/standard_component"] = loss_reg_std.detach()
    metrics["mode/is_hybrid"] = torch.tensor(1.0, device=device)

    return loss_occ, loss_reg, metrics


def unified_semantic_segmentation_loss(
    pred_logits: torch.Tensor,  # [M,P,C]
    idx_targets: torch.Tensor,  # [M,P]
    csr_labels: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ghost_mask: torch.Tensor,   # [N_hits]
    patch_shape: Tuple[int, int, int],
    *,
    loss_mode: str = "hybrid",  # "standard" | "hybrid" | "distance"
    distance_weight: float = 0.3,      # used in hybrid; also scales distance in distance mode
    max_distance: float = 3.0,
    gamma_distance: float = 2.0,
    class_threshold: float = 0.01,
    min_weight: float = 0.05,
    exclude_classes_from_dt: Optional[Union[int, Sequence[int]]] = None,
    label_smoothing: float = 0.0,
    lambda_cp: float = 1e-3,
    # optional class-weighting for the STANDARD term (typical: downweight "none")
    standard_class_weights: Optional[torch.Tensor] = None,  # [C] or None
    none_index: Optional[int] = None,
    none_row_weight: float = 1.0,
    voxel_keep_prob: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Unified semantic segmentation loss.

    loss_mode:
      - "standard": soft CE (CSR) only
      - "distance": distance-aware reweighted soft CE only
      - "hybrid": standard + distance_weight * distance-aware
    """
    from utils.losses import soft_ce_with_logits_csr
    from utils.funcs import csr_keep_rows_torch

    if loss_mode not in {"standard", "hybrid", "distance"}:
        raise ValueError(f"Unknown loss_mode='{loss_mode}'. Use 'standard','hybrid','distance'.")

    device = pred_logits.device
    M, P, C = pred_logits.shape
    ghost_mask = ghost_mask.to(device=device)

    # valid voxels for standard CSR CE (occupied & non-ghost)
    is_occ = idx_targets >= 0
    is_ghost = _safe_ghost_mask_lookup(idx_targets, ghost_mask)
    valid_mask = is_occ & ~is_ghost
    if voxel_keep_prob < 1.0:
        keep = torch.rand(valid_mask.shape, device=device) < float(voxel_keep_prob)
        valid_mask = valid_mask & keep
        # Ensure at least 1 supervised voxel per patch row if there was any valid voxel
        row_has_valid = valid_mask.any(dim=1)  # [M]
        row_has_any_before = (is_occ & ~is_ghost).any(dim=1)  # [M] original-valid rows
        need_fix = row_has_any_before & ~row_has_valid  # rows where we dropped all valids
        if need_fix.any():
            # pick 1 random valid voxel from the original valid set for those rows
            noise = torch.rand((M, P), device=device)
            orig_valid = (is_occ & ~is_ghost)
            noise = noise.masked_fill(~orig_valid, float('inf'))
            pick = noise.argmin(dim=1)  # [M]
            valid_mask[need_fix, pick[need_fix]] = True
    valid_flat = valid_mask.view(-1)
    N_valid = int(valid_flat.sum().item())

    if N_valid == 0:
        zero = pred_logits.sum() * 0.0
        return zero, {"semantic/total": zero.detach()}

    pred_valid = pred_logits.view(-1, C)[valid_flat]
    idx_valid = idx_targets.view(-1)[valid_flat]

    indptr, cls_ids, weights = csr_labels
    csr_valid = csr_keep_rows_torch(indptr, cls_ids, weights, idx_valid)[:3]

    metrics: Dict[str, torch.Tensor] = {}

    # ---- standard term ----
    loss_std = None
    if loss_mode in {"standard", "hybrid"}:
        loss_std = soft_ce_with_logits_csr(
            pred_valid,
            csr_valid,
            ghost_mask=None,
            label_smoothing=label_smoothing,
            lambda_cp=lambda_cp,
            class_weights=standard_class_weights,
            none_index=none_index,
            none_row_weight=none_row_weight,
        )
        metrics["semantic/standard"] = loss_std.detach()

    # ---- distance term ----
    loss_dist = None
    if loss_mode in {"hybrid", "distance"}:
        loss_dist, m_dist = distance_aware_semantic_segmentation_loss(
            pred_logits=pred_logits,
            idx_targets=idx_targets,
            csr_labels=csr_labels,
            ghost_mask=ghost_mask,
            patch_shape=patch_shape,
            max_distance=max_distance,
            gamma_distance=gamma_distance,
            label_smoothing=label_smoothing,
            lambda_cp=0.0,  # avoid double-counting CP if hybrid
            class_threshold=class_threshold,
            min_weight=min_weight,
            exclude_classes_from_dt=([exclude_classes_from_dt] if isinstance(exclude_classes_from_dt, int) else exclude_classes_from_dt),
            voxel_keep_prob=voxel_keep_prob,
        )
        metrics.update(m_dist)

    if loss_mode == "distance":
        loss_total = float(distance_weight) * loss_dist
        metrics["semantic/distance_component"] = loss_total.detach()
        metrics["semantic/total"] = loss_total.detach()
        return loss_total, metrics

    if loss_mode == "standard":
        metrics["semantic/total"] = loss_std.detach()
        return loss_std, metrics

    # hybrid
    loss_total = loss_std + float(distance_weight) * loss_dist
    metrics["semantic/distance_component"] = (float(distance_weight) * loss_dist).detach()
    metrics["semantic/total"] = loss_total.detach()
    return loss_total, metrics
