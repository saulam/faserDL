"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Distance-aware loss functions for sparse 3D neutrino interactions.
    Addresses the issue of voxel-level losses that heavily penalize 
    spatially close but misaligned predictions.
"""

import torch
from torch.nn import functional as F
from typing import Dict, Tuple, Optional
from scipy.ndimage import distance_transform_edt
import numpy as np


def compute_distance_transform_3d(
    occupancy_mask: torch.Tensor,  # [M, P] or [M, p_h, p_w, p_d]
    patch_shape: Tuple[int, int, int],
    max_distance: float = 10.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute 3D Euclidean Distance Transform for each patch.
    For each voxel, computes distance to nearest occupied voxel (occupied voxels have distance 0).
    
    Args:
        occupancy_mask: Binary mask [M, P] or [M, p_h, p_w, p_d]
        patch_shape: (p_h, p_w, p_d)
        max_distance: Maximum distance to clip to
        normalize: If True, normalize by max_distance
    
    Returns:
        distance_map: [M, P] or [M, p_h, p_w, p_d] with distances
    """
    device = occupancy_mask.device
    
    p_h, p_w, p_d = patch_shape
    P = p_h * p_w * p_d
    
    # Reshape to spatial if needed
    if occupancy_mask.dim() == 2:
        M, _ = occupancy_mask.shape
        occ_spatial = occupancy_mask.view(M, p_h, p_w, p_d)
    else:
        occ_spatial = occupancy_mask
        M = occ_spatial.shape[0]
    
    # Move to CPU for scipy processing
    occ_np = occ_spatial.cpu().numpy()
    
    # Compute distance transform for each patch
    distance_maps = []
    for i in range(M):
        # distance_transform_edt returns distance to nearest zero (False) element.
        # We want distance to nearest occupied voxel => provide input that is 0 at occupied voxels.
        dt = distance_transform_edt(~occ_np[i].astype(bool))
        distance_maps.append(dt)
    
    distance_maps = np.stack(distance_maps, axis=0)
    distance_tensor = torch.from_numpy(distance_maps).to(device=device, dtype=torch.float32)
    
    # Clip and optionally normalize
    distance_tensor = torch.clamp(distance_tensor, max=max_distance)
    if normalize:
        distance_tensor = distance_tensor / max_distance
    
    # Reshape back to flat if input was flat
    if occupancy_mask.dim() == 2:
        distance_tensor = distance_tensor.view(M, P)
    
    return distance_tensor


def compute_distance_transform_conv3d(
    occupancy_mask: torch.Tensor,  # [M, p_h, p_w, p_d]
    patch_shape: Tuple[int, int, int],
    max_iterations: int = 5,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Fast DT-like approximation using iterative neighborhood propagation.
    This approximates a chamfer/Chebyshev-style distance (NOT exact Euclidean EDT).
    
    Args:
        occupancy_mask: Binary mask [M, p_h, p_w, p_d]
        patch_shape: (p_h, p_w, p_d)
        max_iterations: Number of dilation iterations
        normalize: If True, normalize by max_iterations
    
    Returns:
        distance_map: [M, p_h, p_w, p_d] with approximate distances
    """
    device = occupancy_mask.device
    occ = (occupancy_mask > 0.5)

    # float distance field: occupied=0, empty=max_iterations
    dist = torch.full(occ.shape, float(max_iterations), device=device, dtype=torch.float32)
    dist = torch.where(occ, torch.zeros_like(dist), dist)

    dist = dist.unsqueeze(1)  # [M, 1, p_h, p_w, p_d]

    for _ in range(max_iterations):
        padded = F.pad(dist, (1, 1, 1, 1, 1, 1), mode="replicate")
        
        # min over 3x3x3 neighbors via max-pool on negative
        neighbors = -F.max_pool3d(-padded, kernel_size=3, stride=1, padding=0)

        # one-step relaxation (occupied stays 0)
        dist = torch.where(
            occ.unsqueeze(1),
            torch.zeros_like(dist),
            torch.minimum(dist, neighbors + 1.0)
        )

    dist = dist.squeeze(1)
    if normalize:
        dist = dist / float(max_iterations)

    return dist    


def soft_distance_transform_conv3d(
    source_prob: torch.Tensor,  # [M, p_h, p_w, p_d] in [0,1]
    patch_shape: Tuple[int, int, int],
    max_iterations: int = 5,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Differentiable DT-like approximation from a soft occupancy field.
    Uses iterative max-pool "coverage" expansion and accumulates expected distance:
        dist ≈ sum_{t=0..T-1} (1 - coverage_t)
    where coverage_0 = source_prob, coverage_{t+1} = dilate(coverage_t).

    Properties:
    - If source_prob is binary and dilation is ideal, this matches distance in steps (Chebyshev-ish).
    - For soft source_prob, gradients propagate without any hard thresholding.

    Args:
        source_prob: [M, p_h, p_w, p_d] probabilities in [0,1]
        patch_shape: kept for compatibility
        max_iterations: max distance in steps
        normalize: if True, divide by max_iterations

    Returns:
        dist: [M, p_h, p_w, p_d] in [0, max_iterations]
    """

    device = source_prob.device
    cov = source_prob.clamp(0.0, 1.0).to(dtype=torch.float32).unsqueeze(1)  # [M,1,H,W,D]
    dist = torch.zeros_like(cov)

    for _ in range(max_iterations):
        dist = dist + (1.0 - cov)  # adds 0 where coverage=1, adds 1 where coverage=0
        padded = F.pad(cov, (1, 1, 1, 1, 1, 1), mode="replicate")
        cov = F.max_pool3d(padded, kernel_size=3, stride=1, padding=0)

    dist = dist.squeeze(1)  # [M,H,W,D]

    if normalize:
        dist = dist / float(max_iterations)

    return dist


def soft_chamfer_loss_patches(
    pred_occ: torch.Tensor,        # [M, P] logits
    idx_targets: torch.Tensor,     # [M, P] hit indices (-1 for empty)
    ghost_mask: torch.Tensor,      # [N_hits] ghost flags
    patch_shape: Tuple[int, int, int],
    temperature: float = 1.0,
    max_distance: float = 5.0,
    gamma_distance: float = 2.0,
    use_conv_dt: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Trainable symmetric Chamfer-style occupancy loss.

    Fixes vs original:
    - dt_from_pred is now differentiable (no pred_prob > 0.5 threshold).
    - avoids the “true->pred term doesn’t backprop” issue.

    Distance penalty uses:
        w(d) = exp(-(d^g)/(max_d^g))
        penalty(d) = 1 - w(d)
    """
    device = pred_occ.device
    M, P = pred_occ.shape
    p_h, p_w, p_d = patch_shape
    max_d = float(max_distance)
    eps = 1e-6

    ghost_mask = ghost_mask.to(device=device)

    # true occupancy (non-ghost)
    is_occ = (idx_targets >= 0)
    is_ghost = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
    is_ghost[is_occ] = ghost_mask[idx_targets[is_occ]].to(dtype=torch.bool)
    true_occ = (is_occ & ~is_ghost).to(dtype=torch.float32).view(M, p_h, p_w, p_d)

    # predicted occupancy probability
    pred_prob = torch.sigmoid(pred_occ / max(temperature, 1e-6)).view(M, p_h, p_w, p_d)

    # DT from true (constant w.r.t. pred) and DT from pred (differentiable)
    if use_conv_dt:
        dt_from_true = compute_distance_transform_conv3d(
            true_occ, patch_shape,
            max_iterations=int(max_d),
            normalize=False,
        )
        dt_from_pred = soft_distance_transform_conv3d(
            pred_prob, patch_shape,
            max_iterations=int(max_d),
            normalize=False,
        )
    else:
        # exact EDT on CPU (non-differentiable); dt_from_pred will still be non-differentiable here
        dt_from_true = compute_distance_transform_3d(
            true_occ, patch_shape, max_distance=max_d, normalize=False
        )
        dt_from_pred = compute_distance_transform_3d(
            (pred_prob > 0.5).to(true_occ.dtype), patch_shape, max_distance=max_d, normalize=False
        )

    pred_prob_flat = pred_prob.view(M, P)
    true_occ_flat = true_occ.view(M, P)

    # pred -> true
    d_pt = dt_from_true.view(M, P).clamp(0.0, max_d)
    w_pt = torch.exp(-(d_pt ** gamma_distance) / (max_d ** gamma_distance + eps))
    loss_pred = (pred_prob_flat * (1.0 - w_pt)).sum() / (pred_prob_flat.sum() + eps)

    # true -> pred (trainable if dt_from_pred is differentiable)
    d_tp = dt_from_pred.view(M, P).clamp(0.0, max_d)
    w_tp = torch.exp(-(d_tp ** gamma_distance) / (max_d ** gamma_distance + eps))
    loss_true = (true_occ_flat * (1.0 - w_tp)).sum() / (true_occ_flat.sum() + eps)

    loss = 0.5 * (loss_pred + loss_true)

    pred_hard = (pred_prob_flat > 0.5)
    metrics = {
        "chamfer/pred_to_true": loss_pred.detach(),
        "chamfer/true_to_pred": loss_true.detach(),
        "chamfer/mean_pred_dist": d_pt[pred_hard].mean().detach() if pred_hard.any() else torch.tensor(0.0, device=device),
        "chamfer/mean_true_dist": d_tp[true_occ_flat > 0.5].mean().detach() if (true_occ_flat > 0.5).any() else torch.tensor(0.0, device=device),
    }
    return loss, metrics


def distance_weighted_regression_loss(
    pred_reg: torch.Tensor,         # [M, P*C_in] or [M,P,C_in]
    targ_reg: torch.Tensor,         # [N_hits, C_in]
    idx_targets: torch.Tensor,      # [M, P]
    ghost_mask: torch.Tensor,       # [N_hits]
    distance_map: torch.Tensor,     # [M, P] distances to nearest true hit
    patch_shape: Tuple[int, int, int],
    max_distance: float = 5.0,
    gamma: float = 2.0,
    huber_delta: float = 1.0,
    reg_empty: Optional[torch.Tensor] = None,
    min_neg_weight: float = 0.05,    # small >0 if you want *some* pressure near hits
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Distance-weighted regression loss.

    Fix vs original:
    - Now matches the stated intent: EMPTY voxels close to hits get LOWER penalty.
      (occupied voxels still get full weight 1.)

    Weighting:
      w_pos = 1
      w_neg = clamp(d/max_d, 0..1)^gamma  (0 near hits, 1 far away)
    """
    device = pred_reg.device
    ghost_mask = ghost_mask.to(device=device)

    C_in = targ_reg.shape[1]
    M, P = idx_targets.shape
    max_d = float(max_distance)
    eps = 1e-6

    # flatten preds
    if pred_reg.dim() == 2:
        pred_reg_flat = pred_reg.view(-1, C_in)  # [M*P, C_in]
    else:
        pred_reg_flat = pred_reg.reshape(-1, C_in)

    idx_flat = idx_targets.reshape(-1)
    dist_flat = distance_map.reshape(-1).to(device=device, dtype=torch.float32)

    # occupied voxels (non-ghost)
    is_occ = (idx_flat >= 0)
    is_ghost = torch.zeros_like(idx_flat, dtype=torch.bool, device=device)
    is_ghost[is_occ] = ghost_mask[idx_flat[is_occ]].to(dtype=torch.bool)
    pos_mask = is_occ & ~is_ghost

    # reg_empty
    if reg_empty is None:
        reg_empty = targ_reg.amin(dim=0)
    reg_empty = reg_empty.to(device=device, dtype=pred_reg_flat.dtype)

    # targets per voxel
    targ_reg_flat = reg_empty.unsqueeze(0).expand(M * P, -1).clone()
    if pos_mask.any():
        targ_reg_flat[pos_mask] = targ_reg[idx_flat[pos_mask]].to(device=device, dtype=pred_reg_flat.dtype)

    # elementwise huber
    reg_elem = F.smooth_l1_loss(
        pred_reg_flat, targ_reg_flat,
        beta=float(huber_delta),
        reduction="none"
    )  # [M*P, C_in]
    reg_row = reg_elem.sum(dim=1)  # [M*P]

    # weights: pos=1, neg=(d/max_d)^gamma
    d_norm = (dist_flat / max(max_d, eps)).clamp(0.0, 1.0)
    w_neg = d_norm ** float(gamma)
    if min_neg_weight > 0:
        w_neg = w_neg.clamp(min=float(min_neg_weight))

    weights = torch.where(pos_mask, torch.ones_like(w_neg), w_neg)

    weighted = reg_row * weights
    loss = weighted.sum() / (weights.sum() + eps)

    metrics = {
        "reg_dist/total": loss.detach(),
        "reg_dist/pos": weighted[pos_mask].mean().detach() if pos_mask.any() else torch.tensor(0.0, device=device),
        "reg_dist/neg": weighted[~pos_mask].mean().detach() if (~pos_mask).any() else torch.tensor(0.0, device=device),
        "reg_dist/mean_weight": weights.mean().detach(),
    }
    return loss, metrics


def combined_distance_aware_reconstruction_loss(
    targ_reg: torch.Tensor,         # [N_hits, C_in]
    pred_occ: torch.Tensor,         # [M, P]
    pred_reg: torch.Tensor,         # [M, P*C_in]
    idx_targets: torch.Tensor,      # [M, P]
    ghost_mask: torch.Tensor,       # [N_hits]
    hit_event_id: torch.Tensor,     # [N_hits]
    *,
    patch_shape: Tuple[int, int, int],
    dataset,
    preprocessing_input: str,
    # Distance-aware params
    use_chamfer_occ: bool = True,
    use_distance_weighted_reg: bool = True,
    chamfer_weight: float = 0.3,
    distance_reg_weight: float = 0.3,
    max_distance: float = 5.0,
    gamma_distance: float = 2.0,
    temperature_chamfer: float = 1.0,
    # Standard loss params
    label_smoothing: float = 0.0,
    focal_gamma: float = 1.5,
    focal_alpha: Optional[float] = None,
    occ_dilate: int = 2,
    huber_delta: float = 1.0,
    reg_weight_lam: float = 1.0,
    reg_weight_alpha: float = 0.5,
    reg_weight_q0: Optional[float] = None,
    reg_weight_wmax: Optional[float] = None,
    occ_empty_beta: float = 0.5,
    per_event_mean: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Combined reconstruction loss with both standard voxel-level and distance-aware components.
    
    Loss = standard_occ + chamfer_weight * chamfer_occ + 
           standard_reg + distance_reg_weight * distance_weighted_reg
    
    This allows gradual transition and comparison between loss formulations.
    
    Args:
        ... (same as reconstruction_losses_masked_simple)
        use_chamfer_occ: If True, add soft chamfer occupancy loss
        use_distance_weighted_reg: If True, add distance-weighted regression loss
        chamfer_weight: Weight for chamfer occupancy component
        distance_reg_weight: Weight for distance-weighted regression
        max_distance: Maximum distance for spatial weighting
        gamma_distance: Exponent for distance decay
        temperature_chamfer: Temperature for soft chamfer matching
    
    Returns:
        loss_occ: Total occupancy loss (standard + optional chamfer)
        loss_reg: Total regression loss (standard + optional distance-weighted)
        metrics: Combined metrics dictionary
    """
    device = idx_targets.device
    M, P = idx_targets.shape
    C_in = targ_reg.shape[1]
    p_h, p_w, p_d = patch_shape
    
    # Import the standard loss for base computation
    from utils.losses import reconstruction_losses_masked_simple
    
    # Compute standard losses
    loss_occ_standard, loss_reg_standard, metrics_standard = reconstruction_losses_masked_simple(
        targ_reg=targ_reg,
        pred_occ=pred_occ,
        pred_reg=pred_reg,
        idx_targets=idx_targets,
        ghost_mask=ghost_mask,
        hit_event_id=hit_event_id,
        patch_shape=patch_shape,
        dataset=dataset,
        preprocessing_input=preprocessing_input,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        occ_dilate=occ_dilate,
        huber_delta=huber_delta,
        reg_weight_lam=reg_weight_lam,
        reg_weight_alpha=reg_weight_alpha,
        reg_weight_q0=reg_weight_q0,
        reg_weight_wmax=reg_weight_wmax,
        occ_empty_beta=occ_empty_beta,
        per_event_mean=per_event_mean,
    )
    
    metrics = {**metrics_standard}
    loss_occ_total = loss_occ_standard
    loss_reg_total = loss_reg_standard
    
    # Add soft chamfer occupancy loss if requested
    if use_chamfer_occ and chamfer_weight > 0:
        loss_chamfer, metrics_chamfer = soft_chamfer_loss_patches(
            pred_occ=pred_occ,
            idx_targets=idx_targets,
            ghost_mask=ghost_mask,
            patch_shape=patch_shape,
            temperature=temperature_chamfer,
            max_distance=max_distance,
            gamma_distance=gamma_distance,
            use_conv_dt=True,  # Use differentiable version
        )
        loss_occ_total = loss_occ_total + chamfer_weight * loss_chamfer
        metrics.update(metrics_chamfer)
        metrics['occ/chamfer_component'] = (chamfer_weight * loss_chamfer).detach()
    
    # Add distance-weighted regression loss if requested
    if use_distance_weighted_reg and distance_reg_weight > 0:
        # Compute distance transform for regression weighting
        is_occ = (idx_targets >= 0)
        is_ghost = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
        is_ghost[is_occ] = ghost_mask[idx_targets[is_occ]]
        true_occ = (is_occ & ~is_ghost).float().view(M, p_h, p_w, p_d)
        
        distance_map = compute_distance_transform_conv3d(
            true_occ, patch_shape, max_iterations=int(max_distance), normalize=False
        )
        
        reg_empty = targ_reg.amin(dim=0)
        loss_reg_dist, metrics_reg_dist = distance_weighted_regression_loss(
            pred_reg=pred_reg,
            targ_reg=targ_reg,
            idx_targets=idx_targets,
            ghost_mask=ghost_mask,
            distance_map=distance_map.view(M, P),
            patch_shape=patch_shape,
            max_distance=max_distance,
            gamma=gamma_distance,
            huber_delta=huber_delta,
            reg_empty=reg_empty,
        )
        loss_reg_total = loss_reg_total + distance_reg_weight * loss_reg_dist
        metrics.update(metrics_reg_dist)
        metrics['reg/distance_component'] = (distance_reg_weight * loss_reg_dist).detach()
    
    # Add identifiers for standard components
    metrics['occ/standard_component'] = loss_occ_standard.detach()
    metrics['reg/standard_component'] = loss_reg_standard.detach()
    
    return loss_occ_total, loss_reg_total, metrics


def focal_distance_transform_loss(
    pred_occ: torch.Tensor,        # [M, P] logits
    idx_targets: torch.Tensor,     # [M, P]
    ghost_mask: torch.Tensor,      # [N_hits]
    patch_shape: Tuple[int, int, int],
    alpha: float = 0.25,
    gamma: float = 2.0,
    max_distance: float = 5.0,
    distance_gamma: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Focal loss modulated by distance transform.
    
    Instead of binary 0/1 targets, use soft targets based on distance to true voxels.
    This creates smoother gradients for nearby mispredictions.
    
    Args:
        pred_occ: Predicted occupancy logits [M, P]
        idx_targets: Ground truth hit indices [M, P]
        ghost_mask: Ghost particle mask [N_hits]
        patch_shape: (p_h, p_w, p_d)
        alpha: Focal loss alpha (class balancing)
        gamma: Focal loss gamma (focus on hard examples)
        max_distance: Distance normalization
        distance_gamma: Exponent for distance-to-target conversion
    
    Returns:
        loss: Focal DT loss
        metrics: Diagnostics
    """
    device = pred_occ.device
    M, P = pred_occ.shape
    p_h, p_w, p_d = patch_shape
    
    # Get true occupancy
    is_occ = (idx_targets >= 0)
    is_ghost = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
    is_ghost[is_occ] = ghost_mask[idx_targets[is_occ]]
    true_occ = (is_occ & ~is_ghost).float().view(M, p_h, p_w, p_d)
    
    # Compute distance transform (distance from each voxel to nearest occupied voxel)
    distance_map = compute_distance_transform_conv3d(
        true_occ, patch_shape, max_iterations=int(max_distance), normalize=False
    )
    distance_map = distance_map.view(M, P)
    
    # Convert distances to soft targets: occupied=1, far empty=0, nearby empty=soft
    # Use exponential decay: target = exp(-(d/max_d)^distance_gamma)
    normalized_dist = (distance_map / max_distance).clamp(0, 1)
    soft_targets = torch.exp(-normalized_dist ** distance_gamma)
    
    # Clip minimum target for true empty voxels
    soft_targets = torch.where(
        true_occ.view(M, P) > 0.5,
        torch.ones_like(soft_targets),
        soft_targets * 0.9  # Allow some gradient even far away
    )
    
    # Standard focal loss with soft targets
    pred_prob = torch.sigmoid(pred_occ)
    
    # Focal modulation
    pt = soft_targets * pred_prob + (1 - soft_targets) * (1 - pred_prob)
    focal_weight = (1 - pt) ** gamma
    
    # BCE with soft targets
    bce = -(soft_targets * F.logsigmoid(pred_occ) + 
            (1 - soft_targets) * F.logsigmoid(-pred_occ))
    
    # Class balancing (alpha weighting)
    if alpha >= 0:
        alpha_t = soft_targets * alpha + (1 - soft_targets) * (1 - alpha)
        focal_loss = alpha_t * focal_weight * bce
    else:
        focal_loss = focal_weight * bce
    
    loss = focal_loss.mean()
    
    metrics = {
        'focal_dt/loss': loss.detach(),
        'focal_dt/mean_soft_target': soft_targets.mean().detach(),
        'focal_dt/mean_distance': distance_map[true_occ.view(M, P) < 0.5].mean().detach() if (true_occ.view(M, P) < 0.5).any() else torch.tensor(0., device=device),
    }
    
    return loss, metrics


def distance_aware_semantic_segmentation_loss(
    pred_logits: torch.Tensor,      # [M, P, C]
    idx_targets: torch.Tensor,      # [M, P] hit indices
    csr_labels: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (indptr, cls, weights)
    ghost_mask: torch.Tensor,       # [N_hits]
    patch_shape: Tuple[int, int, int],
    max_distance: float = 5.0,
    gamma_distance: float = 2.0,
    label_smoothing: float = 0.0,
    lambda_cp: float = 1e-3,
    class_threshold: float = 0.01,
    min_weight: float = 0.05,
    exclude_classes_from_dt: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Distance-aware soft CE for semantic segmentation using *soft* (energy-fraction) targets in CSR.

    This loss does NOT do spatial matching between prediction and target.
    Instead, it reweights CE on a voxel by how close that voxel is to the spatial support
    of its target class mixture (encourages spatial coherence, downweights ambiguous/boundary voxels).

    Improvements vs your original:
      - configurable lower class_threshold to preserve overlaps (important for energy-fraction labels)
      - optional exclusion of a "none" class from DT weighting (prevents it dominating)
      - handles N_valid == 0 safely
      - clearer naming: expected_distance (not "min")
    """
    from utils.losses import build_soft_targets_from_csr, confidence_penalty
    from utils.funcs import csr_keep_rows_torch

    device = pred_logits.device
    ghost_mask = ghost_mask.to(device=device)

    M, P, num_classes = pred_logits.shape
    p_h, p_w, p_d = patch_shape
    max_d = float(max_distance)
    eps = 1e-6

    def _dt(mask_spatial: torch.Tensor) -> torch.Tensor:
        return compute_distance_transform_conv3d(
            mask_spatial, patch_shape,
            max_iterations=int(max_d), normalize=False,
        )

    # Valid voxels (non-empty and non-ghost)
    is_occ = (idx_targets >= 0)
    is_ghost = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
    is_ghost[is_occ] = ghost_mask[idx_targets[is_occ]].to(dtype=torch.bool)
    valid_mask = is_occ & ~is_ghost  # [M, P]

    valid_flat = valid_mask.view(-1)
    N_valid = int(valid_flat.sum().item())

    if N_valid == 0:
        zero = pred_logits.sum() * 0.0
        metrics = {
            "semantic_dist/loss": zero.detach(),
            "semantic_dist/mean_distance": torch.tensor(0.0, device=device),
            "semantic_dist/mean_weight": torch.tensor(0.0, device=device),
            "semantic_dist/ce_unweighted": torch.tensor(0.0, device=device),
            "semantic_dist/max_distance": torch.tensor(0.0, device=device),
        }
        return zero, metrics

    pred_valid = pred_logits.view(-1, num_classes)[valid_flat]  # [N_valid, C]
    idx_valid = idx_targets.view(-1)[valid_flat]                # [N_valid]

    # Build soft targets from CSR (for these valid voxels)
    indptr, cls_ids, weights = csr_labels
    csr_valid = csr_keep_rows_torch(indptr, cls_ids, weights, idx_valid)[:3]
    soft_targets = build_soft_targets_from_csr(
        *csr_valid, num_classes=num_classes, N=N_valid, ghost_mask=None
    )  # [N_valid, C]

    # Optional label smoothing (still OK with soft labels; use small eps)
    if label_smoothing > 0:
        eps_ls = float(label_smoothing)
        soft_targets = soft_targets * (1.0 - eps_ls) + eps_ls / float(num_classes)

    # Spatial coords for valid voxels
    valid_indices = valid_mask.nonzero(as_tuple=False)  # [N_valid, 2] = (patch_idx, flat_pos)
    patch_indices = valid_indices[:, 0]
    flat_pos = valid_indices[:, 1]

    z = flat_pos // (p_w * p_d)
    y = (flat_pos % (p_w * p_d)) // p_d
    x = flat_pos % p_d

    # Which classes participate in DT weighting?
    excluded = set(exclude_classes_from_dt or ())
    included_classes = [c for c in range(num_classes) if c not in excluded]

    # If everything excluded, fall back to plain soft CE
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

    # Expected distance to the spatial support of the target mixture (over included classes)
    expected_distance = torch.zeros(N_valid, device=device, dtype=torch.float32)

    # Renormalize probabilities over included classes so "none" doesn't dominate the weighting
    included_prob = soft_targets[:, included_classes].sum(dim=-1)  # [N_valid]
    included_prob_safe = included_prob.clamp(min=eps)

    for c in included_classes:
        class_prob = soft_targets[:, c]  # [N_valid]
        has_class = class_prob > float(class_threshold)
        if not has_class.any():
            continue

        # Build binary support for class c in spatial grid
        class_mask_spatial = torch.zeros((M, p_h, p_w, p_d), device=device, dtype=torch.float32)
        class_mask_spatial[
            patch_indices[has_class],
            z[has_class],
            y[has_class],
            x[has_class],
        ] = 1.0

        dt_c = _dt(class_mask_spatial)  # [M, p_h, p_w, p_d]
        d_c = dt_c[patch_indices, z, y, x]  # [N_valid]

        # Accumulate prob-weighted distances
        expected_distance += d_c * class_prob.to(dtype=torch.float32)

    # Normalize over included mass; if included_prob is ~0 (pure "none"), set distance 0
    expected_distance = torch.where(
        included_prob > eps,
        expected_distance / included_prob_safe,
        torch.zeros_like(expected_distance)
    )

    # Weight CE: close to target support -> weight ~1, far -> downweight
    dist_weight = torch.exp(-(expected_distance ** float(gamma_distance)) / (max_d ** float(gamma_distance) + eps))
    dist_weight = dist_weight.clamp(min=float(min_weight))

    # Soft CE
    log_probs = F.log_softmax(pred_valid, dim=-1)
    ce_vec = -(soft_targets * log_probs).sum(dim=-1)  # [N_valid]
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


def combined_distance_aware_segmentation_loss(
    pred_logits: torch.Tensor,  # [M, P, C]
    idx_targets: torch.Tensor,  # [M, P]
    csr_labels: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ghost_mask: torch.Tensor,   # [N_hits]
    patch_shape: Tuple[int, int, int],
    use_distance_weighting: bool = True,
    distance_weight: float = 0.3,
    max_distance: float = 5.0,
    gamma_distance: float = 2.0,
    label_smoothing: float = 0.0,
    lambda_cp: float = 1e-3,
    class_threshold: float = 0.01,
    min_weight: float = 0.05,
    exclude_classes_from_dt: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Hybrid loss = standard soft CE (CSR) + distance_weight * distance-aware reweighted soft CE.
    Suitable for energy-fraction soft labels (multi-particle overlaps).
    """
    from utils.losses import soft_ce_with_logits_csr
    from utils.funcs import csr_keep_rows_torch

    device = pred_logits.device
    ghost_mask = ghost_mask.to(device=device)

    M, P, num_classes = pred_logits.shape

    # Valid voxels
    is_occ = (idx_targets >= 0)
    is_ghost = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
    is_ghost[is_occ] = ghost_mask[idx_targets[is_occ]].to(dtype=torch.bool)
    valid_mask = is_occ & ~is_ghost

    valid_flat = valid_mask.view(-1)
    N_valid = int(valid_flat.sum().item())

    if N_valid == 0:
        zero = pred_logits.sum() * 0.0
        return zero, {"semantic/standard": zero.detach(), "semantic/total": zero.detach()}

    pred_valid = pred_logits.view(-1, num_classes)[valid_flat]
    idx_valid = idx_targets.view(-1)[valid_flat]

    # Filter CSR labels to valid voxels
    indptr, cls_ids, weights = csr_labels
    csr_valid = csr_keep_rows_torch(indptr, cls_ids, weights, idx_valid)[:3]

    # Standard CSR soft CE
    loss_standard = soft_ce_with_logits_csr(
        pred_valid, csr_valid, ghost_mask=None,
        label_smoothing=label_smoothing,
        lambda_cp=lambda_cp,
        class_weights=torch.ones(num_classes, device=pred_logits.device).\
            scatter_(0, torch.tensor(exclude_classes_from_dt, device=pred_logits.device), 0.5)\
            if exclude_classes_from_dt is not None else None,
        none_index=exclude_classes_from_dt,
        none_row_weight=0.3 if exclude_classes_from_dt is not None else None,
    )

    metrics = {"semantic/standard": loss_standard.detach()}
    loss_total = loss_standard

    if use_distance_weighting and distance_weight > 0:
        loss_dist, dist_metrics = distance_aware_semantic_segmentation_loss(
            pred_logits=pred_logits,
            idx_targets=idx_targets,
            csr_labels=csr_labels,
            ghost_mask=ghost_mask,
            patch_shape=patch_shape,
            max_distance=max_distance,
            gamma_distance=gamma_distance,
            label_smoothing=label_smoothing,
            lambda_cp=0.0,  # CP disabled here to avoid double-counting
            class_threshold=class_threshold,
            min_weight=min_weight,
            exclude_classes_from_dt=exclude_classes_from_dt,
        )
        loss_total = loss_total + float(distance_weight) * loss_dist
        metrics.update(dist_metrics)
        metrics["semantic/distance_component"] = (float(distance_weight) * loss_dist).detach()

    metrics["semantic/total"] = loss_total.detach()
    return loss_total, metrics

