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
import torch.nn as nn
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
    For each empty voxel, computes distance to nearest occupied voxel.
    
    Args:
        occupancy_mask: Binary mask [M, P] or [M, p_h, p_w, p_d]
        patch_shape: (p_h, p_w, p_d)
        max_distance: Maximum distance to clip to
        normalize: If True, normalize by max_distance
    
    Returns:
        distance_map: [M, P] or [M, p_h, p_w, p_d] with distances
    """
    device = occupancy_mask.device
    dtype = occupancy_mask.dtype
    
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
        # Distance transform on binary mask (True = occupied)
        # Returns distance to nearest True voxel
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
    Differentiable approximation of distance transform using iterative max pooling.
    This is faster than scipy and maintains gradients for training.
    
    Uses the chamfer-style approach: iteratively propagate distances outward.
    
    Args:
        occupancy_mask: Binary mask [M, p_h, p_w, p_d]
        patch_shape: (p_h, p_w, p_d)
        max_iterations: Number of dilation iterations
        normalize: If True, normalize by max_iterations
    
    Returns:
        distance_map: [M, p_h, p_w, p_d] with approximate distances
    """
    device = occupancy_mask.device
    p_h, p_w, p_d = patch_shape
    M = occupancy_mask.shape[0]
    
    # Initialize: occupied voxels = 0, empty voxels = large value
    dist = torch.where(
        occupancy_mask > 0.5,
        torch.zeros_like(occupancy_mask, dtype=torch.float32),
        torch.full_like(occupancy_mask, float(max_iterations), dtype=torch.float32)
    )
    
    # Iteratively propagate minimum distances
    dist = dist.unsqueeze(1)  # [M, 1, p_h, p_w, p_d]
    
    for iteration in range(max_iterations):
        # 3x3x3 max pool with padding to propagate distances
        # Use negative pooling to get minimum
        padded = F.pad(dist, (1, 1, 1, 1, 1, 1), mode='replicate')
        neighbors = F.max_pool3d(-padded, kernel_size=3, stride=1, padding=0)
        neighbors = -neighbors  # Convert back to min
        
        # Increment distances by 1 for non-occupied voxels
        dist = torch.where(
            occupancy_mask.unsqueeze(1) > 0.5,
            torch.zeros_like(dist),
            torch.minimum(dist, neighbors + 1.0)
        )
    
    dist = dist.squeeze(1)  # [M, p_h, p_w, p_d]
    
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
    Soft Chamfer-style loss for occupancy prediction.
    Instead of hard voxel-to-voxel matching, considers spatial proximity.
    
    For each predicted occupied voxel, find weighted distance to nearest true occupied voxel.
    For each true occupied voxel, find weighted distance to nearest predicted occupied voxel.
    
    Args:
        pred_occ: Predicted occupancy logits [M, P]
        idx_targets: Ground truth hit indices [M, P], -1 for empty sub-voxels
        ghost_mask: Ghost particle mask [N_hits]
        patch_shape: (p_h, p_w, p_d)
        temperature: Softmax temperature for soft matching
        max_distance: Maximum distance penalty
        gamma_distance: Exponent for distance weighting (higher = steeper penalty)
        use_conv_dt: Use conv-based (differentiable) distance transform
    
    Returns:
        loss: Scalar soft chamfer loss
        metrics: Dictionary of diagnostic metrics
    """
    device = pred_occ.device
    M, P = pred_occ.shape
    p_h, p_w, p_d = patch_shape
    
    # Get true occupancy (non-ghost hits)
    is_occ = (idx_targets >= 0)
    is_ghost = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
    is_ghost[is_occ] = ghost_mask[idx_targets[is_occ]]
    true_occ = (is_occ & ~is_ghost).float().view(M, p_h, p_w, p_d)
    
    # Get predicted occupancy probabilities
    pred_prob = torch.sigmoid(pred_occ / temperature).view(M, p_h, p_w, p_d)
    
    # Compute distance transforms
    if use_conv_dt:
        # Differentiable approximation
        dt_from_true = compute_distance_transform_conv3d(
            true_occ, patch_shape, max_iterations=int(max_distance), normalize=False
        )
        dt_from_pred = compute_distance_transform_conv3d(
            pred_prob > 0.5, patch_shape, max_iterations=int(max_distance), normalize=False
        )
    else:
        # Exact but non-differentiable (for evaluation)
        dt_from_true = compute_distance_transform_3d(
            true_occ, patch_shape, max_distance=max_distance, normalize=False
        )
        dt_from_pred = compute_distance_transform_3d(
            pred_prob > 0.5, patch_shape, max_distance=max_distance, normalize=False
        )
    
    # Distance-weighted losses
    # For predicted voxels: weight by distance to nearest true voxel
    pred_to_true_dist = dt_from_true.view(M, P)
    dist_weight_pred = torch.exp(-pred_to_true_dist ** gamma_distance / (max_distance ** gamma_distance))
    loss_pred = (pred_prob.view(M, P) * (1.0 - dist_weight_pred)).sum() / (pred_prob.sum() + 1e-6)
    
    # For true voxels: weight by distance to nearest predicted voxel  
    true_to_pred_dist = dt_from_pred.view(M, P)
    true_occ_flat = true_occ.view(M, P)
    dist_weight_true = torch.exp(-true_to_pred_dist ** gamma_distance / (max_distance ** gamma_distance))
    loss_true = (true_occ_flat * (1.0 - dist_weight_true)).sum() / (true_occ_flat.sum() + 1e-6)
    
    # Symmetric chamfer loss
    loss = 0.5 * (loss_pred + loss_true)
    
    metrics = {
        'chamfer/pred_to_true': loss_pred.detach(),
        'chamfer/true_to_pred': loss_true.detach(),
        'chamfer/mean_pred_dist': pred_to_true_dist[pred_prob.view(M, P) > 0.5].mean().detach() if (pred_prob.view(M, P) > 0.5).any() else torch.tensor(0., device=device),
        'chamfer/mean_true_dist': true_to_pred_dist[true_occ_flat > 0.5].mean().detach() if (true_occ_flat > 0.5).any() else torch.tensor(0., device=device),
    }
    
    return loss, metrics


def distance_weighted_regression_loss(
    pred_reg: torch.Tensor,         # [M, P*C_in]
    targ_reg: torch.Tensor,         # [N_hits, C_in]
    idx_targets: torch.Tensor,      # [M, P] hit indices
    ghost_mask: torch.Tensor,       # [N_hits]
    distance_map: torch.Tensor,     # [M, P] precomputed distances
    patch_shape: Tuple[int, int, int],
    max_distance: float = 5.0,
    gamma: float = 2.0,
    huber_delta: float = 1.0,
    reg_empty: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Distance-weighted regression loss for charge/energy prediction.
    Voxels that are spatially close to true hits get lower penalties.
    
    Args:
        pred_reg: Predicted regression values [M, P*C_in]
        targ_reg: True regression targets [N_hits, C_in]
        idx_targets: Ground truth hit indices [M, P]
        ghost_mask: Ghost particle mask [N_hits]
        distance_map: Precomputed distance transform [M, P]
        patch_shape: (p_h, p_w, p_d)
        max_distance: Maximum distance for weighting
        gamma: Exponent for distance weighting
        huber_delta: Huber loss delta
        reg_empty: Representative "empty" value (if None, use min of targets)
    
    Returns:
        loss: Weighted regression loss
        metrics: Dictionary of diagnostic metrics
    """
    device = pred_reg.device
    C_in = targ_reg.shape[1]
    M, P = idx_targets.shape
    p_h, p_w, p_d = patch_shape
    
    # Handle both [M, P*C_in] and [M, P, C_in] formats
    if pred_reg.dim() == 2:
        pred_reg_flat = pred_reg.view(-1, C_in)  # [M*P, C_in]
    else:  # dim == 3: [M, P, C_in]
        pred_reg_flat = pred_reg.view(-1, C_in)  # [M*P, C_in]
    
    idx_flat = idx_targets.view(-1)          # [M*P]
    dist_flat = distance_map.view(-1)        # [M*P]
    
    # Identify occupied voxels (non-ghost)
    is_occ = (idx_flat >= 0)
    is_ghost = torch.zeros_like(idx_flat, dtype=torch.bool)
    is_ghost[is_occ] = ghost_mask[idx_flat[is_occ]]
    pos_mask = is_occ & ~is_ghost
    
    # Gather targets
    if reg_empty is None:
        reg_empty = targ_reg.amin(dim=0)
    
    targ_reg_flat = reg_empty.unsqueeze(0).expand(M * P, -1).clone()
    targ_reg_flat[pos_mask] = targ_reg[idx_flat[pos_mask]]
    
    # Compute element-wise Huber loss
    reg_elem = F.smooth_l1_loss(
        pred_reg_flat, targ_reg_flat, beta=huber_delta, reduction='none'
    )  # [M*P, C_in]
    reg_row = reg_elem.sum(dim=1)  # [M*P]
    
    # Distance-based weighting: closer predictions get more weight
    # For occupied voxels, use distance=0 (exact match expected)
    # For empty voxels, use distance to nearest true voxel
    effective_dist = torch.where(pos_mask, torch.zeros_like(dist_flat), dist_flat)
    dist_weight = torch.exp(-effective_dist ** gamma / (max_distance ** gamma))
    
    # Apply weights
    weighted_loss = reg_row * dist_weight
    loss = weighted_loss.sum() / (dist_weight.sum() + 1e-6)
    
    metrics = {
        'reg_dist/total': loss.detach(),
        'reg_dist/pos': weighted_loss[pos_mask].mean().detach() if pos_mask.any() else torch.tensor(0., device=device),
        'reg_dist/neg': weighted_loss[~pos_mask].mean().detach() if (~pos_mask).any() else torch.tensor(0., device=device),
        'reg_dist/mean_weight': dist_weight.mean().detach(),
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
    pred_logits: torch.Tensor,      # [M, P, num_classes] predictions
    idx_targets: torch.Tensor,      # [M, P] hit indices
    csr_labels: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (indptr, cls, weights)
    ghost_mask: torch.Tensor,       # [N_hits]
    patch_shape: Tuple[int, int, int],
    max_distance: float = 5.0,
    gamma_distance: float = 2.0,
    label_smoothing: float = 0.0,
    lambda_cp: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Distance-aware soft cross-entropy loss for semantic segmentation.
    
    Uses per-class distance transforms to provide spatial context:
    - For each class, compute distance to nearest voxel with that class
    - Weight cross-entropy by proximity to correct class
    - Encourages spatial coherence and smoothness in predictions
    
    Args:
        pred_logits: Predicted class logits [M, P, num_classes]
        idx_targets: Ground truth hit indices [M, P], -1 for empty
        csr_labels: CSR format labels (indptr, cls_ids, weights)
        ghost_mask: Ghost particle mask [N_hits]
        patch_shape: (p_h, p_w, p_d)
        max_distance: Maximum distance for spatial weighting
        gamma_distance: Distance decay exponent
        label_smoothing: Label smoothing factor
        lambda_cp: Confidence penalty weight
    
    Returns:
        loss: Distance-aware semantic segmentation loss
        metrics: Diagnostics
    """
    from utils.losses import build_soft_targets_from_csr, confidence_penalty
    
    device = pred_logits.device
    M, P, num_classes = pred_logits.shape
    p_h, p_w, p_d = patch_shape
    
    # Get valid (non-empty, non-ghost) voxels
    is_occ = (idx_targets >= 0)
    is_ghost = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
    is_ghost[is_occ] = ghost_mask[idx_targets[is_occ]]
    valid_mask = is_occ & ~is_ghost  # [M, P]
    
    # Extract valid predictions and indices
    valid_flat = valid_mask.view(-1)  # [M*P]
    pred_valid = pred_logits.view(-1, num_classes)[valid_flat]  # [N_valid, num_classes]
    idx_valid = idx_targets.view(-1)[valid_flat]  # [N_valid]
    
    # Build soft targets from CSR
    indptr, cls_ids, weights = csr_labels
    from utils.funcs import csr_keep_rows_torch
    csr_valid = csr_keep_rows_torch(indptr, cls_ids, weights, idx_valid)[:3]
    
    N_valid = pred_valid.shape[0]
    soft_targets = build_soft_targets_from_csr(
        *csr_valid, num_classes=num_classes, N=N_valid, ghost_mask=None
    )  # [N_valid, num_classes]
    
    # Apply label smoothing
    if label_smoothing > 0:
        eps = float(label_smoothing)
        soft_targets = soft_targets * (1.0 - eps) + eps / num_classes
    
    # Compute per-class distance transforms
    # For each class, compute distance from each voxel to nearest voxel with that class
    valid_spatial = valid_mask.view(M, p_h, p_w, p_d)  # [M, p_h, p_w, p_d]
    
    # Get spatial indices of valid voxels for building per-class masks
    valid_indices = valid_mask.nonzero(as_tuple=False)  # [N_valid, 2] = (patch_idx, flat_pos)
    
    # Convert flat positions to spatial coords
    patch_indices = valid_indices[:, 0]  # [N_valid]
    flat_positions = valid_indices[:, 1]  # [N_valid]
    z_coords = flat_positions // (p_w * p_d)
    y_coords = (flat_positions % (p_w * p_d)) // p_d
    x_coords = flat_positions % p_d
    
    # Compute per-class distance transforms for soft labels
    # For each class, build mask weighted by soft label probabilities
    # Then compute minimum weighted distance to target class distribution
    
    min_weighted_distances = torch.zeros(N_valid, device=device)
    class_threshold = 0.05  # Lower threshold for soft labels (was 0.1)
    
    for c in range(num_classes):
        # Get soft probabilities for class c
        class_prob = soft_targets[:, c]  # [N_valid]
        has_class_c = class_prob > class_threshold  # [N_valid]
        
        if not has_class_c.any():
            continue
        
        # Build spatial mask for this class (binarized: present vs absent)
        class_mask_spatial = torch.zeros((M, p_h, p_w, p_d), device=device)
        class_mask_spatial[patch_indices[has_class_c], 
                          z_coords[has_class_c], 
                          y_coords[has_class_c], 
                          x_coords[has_class_c]] = 1.0
        
        # Compute distance transform from class c regions
        dt_class = compute_distance_transform_conv3d(
            class_mask_spatial, patch_shape, 
            max_iterations=int(max_distance), normalize=False
        )  # [M, p_h, p_w, p_d]
        
        # Extract distances for all voxels
        distances_c = dt_class[patch_indices, z_coords, y_coords, x_coords]  # [N_valid]
        
        # Weight distances by target probability for this class
        # If voxel should be class c with prob p, distance matters proportional to p
        weighted_dist_c = distances_c * class_prob
        
        # Accumulate probability-weighted distances
        min_weighted_distances += weighted_dist_c
    
    # Normalize by total probability mass
    total_prob = soft_targets.sum(dim=-1).clamp(min=1e-6)  # [N_valid]
    min_weighted_distances = min_weighted_distances / total_prob
    
    # Distance weighting: voxels close to their target class distribution get higher weight
    dist_weight = torch.exp(-min_weighted_distances ** gamma_distance / (max_distance ** gamma_distance))
    dist_weight = torch.clamp(dist_weight, min=0.05)  # Minimum weight
    
    # Standard cross-entropy loss
    log_probs = F.log_softmax(pred_valid, dim=-1)  # [N_valid, num_classes]
    ce_loss = -(soft_targets * log_probs).sum(dim=-1)  # [N_valid]
    
    # Apply distance weighting
    weighted_ce = ce_loss * dist_weight
    loss = weighted_ce.mean()
    
    # Confidence penalty (optional)
    if lambda_cp > 0:
        cp = confidence_penalty(pred_valid, lambda_cp)
        loss = loss + cp
    
    metrics = {
        'semantic_dist/loss': loss.detach(),
        'semantic_dist/mean_distance': min_weighted_distances.mean().detach(),
        'semantic_dist/mean_weight': dist_weight.mean().detach(),
        'semantic_dist/ce_unweighted': ce_loss.mean().detach(),
        'semantic_dist/max_distance': min_weighted_distances.max().detach() if min_weighted_distances.numel() > 0 else torch.tensor(0., device=device),
    }
    
    return loss, metrics


def combined_distance_aware_segmentation_loss(
    pred_logits: torch.Tensor,
    idx_targets: torch.Tensor,
    csr_labels: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ghost_mask: torch.Tensor,
    patch_shape: Tuple[int, int, int],
    use_distance_weighting: bool = True,
    distance_weight: float = 0.3,
    max_distance: float = 5.0,
    gamma_distance: float = 2.0,
    label_smoothing: float = 0.0,
    lambda_cp: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Hybrid semantic segmentation loss: standard + distance-aware.
    
    Loss = standard_ce + distance_weight × distance_aware_ce
    
    This allows gradual transition from pure voxel-level to spatially-aware.
    
    Args:
        pred_logits: [M, P, num_classes]
        idx_targets: [M, P]
        csr_labels: CSR format (indptr, cls, weights)
        ghost_mask: [N_hits]
        patch_shape: (p_h, p_w, p_d)
        use_distance_weighting: Enable distance component
        distance_weight: Weight for distance-aware component
        max_distance: Spatial weighting range
        gamma_distance: Distance decay exponent
        label_smoothing: Label smoothing factor
        lambda_cp: Confidence penalty
    
    Returns:
        loss: Combined loss
        metrics: Diagnostics
    """
    from utils.losses import soft_ce_with_logits_csr
    
    device = pred_logits.device
    M, P, num_classes = pred_logits.shape
    
    # Reshape predictions for standard loss (expects [N_valid, num_classes])
    is_occ = (idx_targets >= 0)
    is_ghost = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
    is_ghost[is_occ] = ghost_mask[idx_targets[is_occ]]
    valid_mask = is_occ & ~is_ghost
    
    valid_flat = valid_mask.view(-1)
    pred_valid = pred_logits.view(-1, num_classes)[valid_flat]
    idx_valid = idx_targets.view(-1)[valid_flat]
    
    # Filter CSR labels for valid voxels
    indptr, cls_ids, weights = csr_labels
    from utils.funcs import csr_keep_rows_torch
    csr_valid = csr_keep_rows_torch(indptr, cls_ids, weights, idx_valid)[:3]
    
    # Compute standard soft CE loss
    loss_standard = soft_ce_with_logits_csr(
        pred_valid, csr_valid, ghost_mask=None,
        label_smoothing=label_smoothing, lambda_cp=lambda_cp
    )
    
    metrics = {
        'semantic/standard': loss_standard.detach(),
    }
    
    loss_total = loss_standard
    
    # Add distance-aware component if enabled
    if use_distance_weighting and distance_weight > 0:
        loss_distance, dist_metrics = distance_aware_semantic_segmentation_loss(
            pred_logits=pred_logits,
            idx_targets=idx_targets,
            csr_labels=csr_labels,
            ghost_mask=ghost_mask,
            patch_shape=patch_shape,
            max_distance=max_distance,
            gamma_distance=gamma_distance,
            label_smoothing=label_smoothing,
            lambda_cp=lambda_cp,
        )
        
        loss_total = loss_total + distance_weight * loss_distance
        metrics.update(dist_metrics)
        metrics['semantic/distance_component'] = (distance_weight * loss_distance).detach()
    
    metrics['semantic/total'] = loss_total.detach()
    
    return loss_total, metrics
