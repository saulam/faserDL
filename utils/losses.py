"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Custom loss functions.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, Tuple, Optional, Sequence, Union


class KinematicsMultiTaskLoss(nn.Module):
    """
    Predict (p_vis, p_jet). Derive p_lep = p_vis - p_jet.

    forward() inputs:
      p_vis_hat:  (B,3) predicted visible momentum
      p_jet_hat:  (B,3) predicted jet momentum
      p_vis_true: (B,3) true visible momentum
      p_jet_true: (B,3) true jet momentum  (provided)
      is_cc:      (B,)  ground-truth CC mask in {0,1}
      is_cc_hat:  (B,)  predicted CC prob in [0,1] (already sigmoid’d)
      vis_latents, jet_latents: optional latents for tiny priors

    Design:
      - Magnitudes supervised via relative residuals.
      - XY-direction loss added (plus optional 3D direction).
      - Lepton vector supervision is CC-only. Optional NC zero-attractor on raw lep.
    """

    def __init__(
        self,
        *,
        stats,
        # -------- weights / knobs --------
        huber_delta=1.0,
        lam_dir_xy=1.0,               # weight for XY cosine loss
        lam_dir_3d=0.0,               # weight for 3D cosine loss (default off)
        lep_nc_zero_w=0.05,           # small NC zero-attractor on raw lep
        latent_prior_w=0.0,           # tiny N(0,1) prior on provided latents
        enforce_nonneg_truth_pz=True, # clamp truth pz>=0 before deriving p_lep_true
        decouple_radial=False,        # remove radial component from component loss
    ):
        super().__init__()

        # scales
        self.register_buffer("s_vis_xyz", torch.tensor(stats["vis"]["s_xyz"], dtype=torch.float32).view(1,3))
        self.register_buffer("s_jet_xyz", torch.tensor(stats["jet"]["s_xyz"], dtype=torch.float32).view(1,3))
        self.register_buffer("s_lep_xyz", torch.tensor(stats["lep"]["s_xyz"], dtype=torch.float32).view(1,3))

        # per-output floors
        self.tau_pt_vis   = float(stats["vis"]["tau_pt"])
        self.tau_mag_vis  = float(stats["vis"]["tau_mag"])
        self.tau_pt_jet   = float(stats["jet"]["tau_pt"])
        self.tau_mag_jet  = float(stats["jet"]["tau_mag"])
        self.tau_pt_lep   = float(stats["lep"]["tau_pt"])
        self.tau_mag_lep  = float(stats["lep"]["tau_mag"])

        # knobs
        self.huber_delta = float(huber_delta)
        self.lam_dir_xy  = float(lam_dir_xy)
        self.lam_dir_3d  = float(lam_dir_3d)

        self.lep_nc_zero_w = float(lep_nc_zero_w)
        self.latent_prior_w = float(latent_prior_w)
        self.enforce_nonneg_truth_pz = bool(enforce_nonneg_truth_pz)
        self.decouple_radial = bool(decouple_radial)

    # ---------- primitives ----------
    @staticmethod
    def _huber(x, delta):
        ax = x.abs()
        quad = torch.clamp(ax, max=delta)
        lin = ax - quad
        return 0.5 * quad**2 + delta * lin

    @staticmethod
    def _cosine_dir_3d(p_hat, p_true, eps=1e-8):
        num = (p_hat * p_true).sum(-1)
        den = p_hat.norm(dim=-1) * p_true.norm(dim=-1)
        return 1.0 - num / (den + eps)

    @staticmethod
    def _cosine_dir_xy(p_hat, p_true, eps=1e-8):
        v_hat  = p_hat[..., :2]
        v_true = p_true[..., :2]
        num = (v_hat * v_true).sum(-1)
        den = v_hat.norm(dim=-1) * v_true.norm(dim=-1)
        return 1.0 - num / (den + eps)

    def _component_loss(self, p_hat, p_true, s_xyz, eps=1e-8):
        """
        If decouple_radial=True, remove the radial component so this term is
        purely angular in the native Cartesian basis (per-axis scaled).
        """
        e = p_hat - p_true
        if self.decouple_radial:
            tnorm2 = (p_true * p_true).sum(-1, keepdim=True).clamp_min(eps)
            e_rad = ((e * p_true).sum(-1, keepdim=True) / tnorm2) * p_true
            e = e - e_rad
        z = e / s_xyz
        return self._huber(z, self.huber_delta).sum(-1)

    def _relative_residual_loss(self, p_true, p_hat, tau, kind="mag"):
        """
        Relative residual Huber on a scalar derived from vectors p_true/p_hat:
          kind="mag":  uses ||p||
          kind="pt":   uses ||p_xy||
        Norms are computed internally.
        """
        if kind == "mag":
            x_true = p_true.norm(dim=-1)
            x_hat  = p_hat.norm(dim=-1)
        elif kind == "pt":
            x_true = p_true[..., :2].norm(dim=-1)
            x_hat  = p_hat[..., :2].norm(dim=-1)
        else:
            raise ValueError(f"Unknown kind='{kind}'")
        denom = torch.maximum(x_true, torch.as_tensor(tau, dtype=x_true.dtype, device=x_true.device))
        r = (x_true - x_hat) / denom
        return self._huber(r, self.huber_delta)

    # ---------- forward ----------
    def forward(
        self,
        *,
        p_vis_hat, p_jet_hat,
        p_vis_true, p_jet_true,
        is_cc, vis_latents=None, jet_latents=None,
    ):
        device = p_vis_hat.device

        # ----- truths -----
        if self.enforce_nonneg_truth_pz:
            p_vis_true = p_vis_true.clone(); p_vis_true[...,2] = p_vis_true[...,2].clamp_min(0.0)
            p_jet_true = p_jet_true.clone(); p_jet_true[...,2] = p_jet_true[...,2].clamp_min(0.0)
        p_lep_true = p_vis_true - p_jet_true

        # ----- predictions -----
        p_lep_hat = p_vis_hat - p_jet_hat

        m_cc = is_cc.to(p_vis_hat.dtype).view(-1)      # (B,)
        m_nc = 1.0 - m_cc

        # ----- vector losses -----
        # VIS
        L_vis_comp  = self._component_loss(p_vis_hat, p_vis_true, self.s_vis_xyz)
        L_vis_dirxy = self._cosine_dir_xy(p_vis_hat, p_vis_true)
        L_vis_geom = L_vis_comp + self.lam_dir_xy * L_vis_dirxy
        if self.lam_dir_3d != 0.0:
            L_vis_geom = L_vis_geom + self.lam_dir_3d * self._cosine_dir_3d(p_vis_hat, p_vis_true)
        if vis_latents is not None and self.latent_prior_w > 0.0:
            L_vis_geom = L_vis_geom + self.latent_prior_w * (vis_latents.pow(2).sum(-1))
        L_vis_pt   = self._relative_residual_loss(p_vis_true, p_vis_hat, self.tau_pt_vis, kind="pt")
        L_vis_mag  = self._relative_residual_loss(p_vis_true, p_vis_hat, self.tau_mag_vis, kind="mag")

        # JET
        L_jet_comp  = self._component_loss(p_jet_hat, p_jet_true, self.s_jet_xyz)
        L_jet_dirxy = self._cosine_dir_xy(p_jet_hat, p_jet_true)
        L_jet_geom = L_jet_comp + self.lam_dir_xy * L_jet_dirxy
        if self.lam_dir_3d != 0.0:
            L_jet_geom = L_jet_geom + self.lam_dir_3d * self._cosine_dir_3d(p_jet_hat, p_jet_true)
        if jet_latents is not None and self.latent_prior_w > 0.0:
            L_jet_geom = L_jet_geom + self.latent_prior_w * (jet_latents.pow(2).sum(-1))
        L_jet_pt   = self._relative_residual_loss(p_jet_true, p_jet_hat, self.tau_pt_jet, kind="pt")
        L_jet_mag  = self._relative_residual_loss(p_jet_true, p_jet_hat, self.tau_mag_jet, kind="mag")

        # LEPTON (CC-only; use gated prediction).
        L_lep_comp  = self._component_loss(p_lep_hat, p_lep_true, self.s_lep_xyz)
        L_lep_dirxy = self._cosine_dir_xy(p_lep_hat, p_lep_true)
        L_lep_geom = L_lep_comp + self.lam_dir_xy * L_lep_dirxy
        if self.lam_dir_3d != 0.0:
            L_lep_geom = L_lep_geom + (self.lam_dir_3d * self._cosine_dir_3d(p_lep_hat, p_lep_true))
        L_lep_geom = L_lep_geom * m_cc
        L_lep_pt   = self._relative_residual_loss(p_lep_true, p_lep_hat, self.tau_pt_lep, kind="pt") * m_cc
        L_lep_mag  = self._relative_residual_loss(p_lep_true, p_lep_hat, self.tau_mag_lep, kind="mag") * m_cc

        # NC zero-attractor (small) on RAW lepton
        if self.lep_nc_zero_w > 0.0:
            z_nc = (p_lep_hat / self.s_lep_xyz) * m_nc.view(-1,1)
            L_lep_zero_nc = self._huber(z_nc, self.huber_delta).sum(-1)
        else:
            L_lep_zero_nc = torch.zeros_like(m_cc)

        losses = {
            # vis
            'loss_vis/geom': L_vis_geom,
            'loss_vis/pt':   L_vis_pt,
            'loss_vis/mag':  L_vis_mag,
            # jet
            'loss_jet/geom': L_jet_geom,
            'loss_jet/pt':   L_jet_pt,
            'loss_jet/mag':  L_jet_mag,
            # lep (CC-only)
            'loss_lep/geom': L_lep_geom,
            'loss_lep/pt':   L_lep_pt,
            'loss_lep/mag':  L_lep_mag,
            # NC prior
            'loss_lep/zero_nc': L_lep_zero_nc,
        }

        return losses


class MAPE(torch.nn.Module):
    """
    Standard Mean Absolute Percentage Error (MAPE) loss function in PyTorch.

    MAPE is defined as:
        L = (1/n) * sum(|(y_pred - y_true) / y_true|)

    - Handles numerical stability by adding a small epsilon to the denominator.
    - Supports 'mean' or 'sum' reduction.
    
    Args:
        epsilon (float): Small value to prevent division by zero. Default is 1e-8.
        reduction (str): Specifies the reduction method ('mean' or 'sum'). Default is 'mean'.
        preprocessing (str): Type of preprocessing applied ('sqrt', 'log', or None).
        standardize (str): Type of standarisation applied ('z-score', 'unit-var', 'norm').
    """

    def __init__(
            self,
            eps=1e-8,
            reduction='mean',
        ):
        super(MAPE, self).__init__()
        self.eps = eps
        self.reduction = reduction


    def forward(self, pred, target):
        """
        Computes the MAPE loss, but only for target values greater than 0.

        Args:
            pred (torch.Tensor): Predicted values (batch_size, *)
            target (torch.Tensor): Ground truth values (batch_size, *)

        Returns:
            torch.Tensor: Computed MAPE loss, considering only target values > 0.
        """ 
        # Calculate percentage error where the target is greater than 0
        mask = target > 0
        percentage_error = torch.abs((pred - target) / (target + self.eps))
        percentage_error = torch.nan_to_num(percentage_error, nan=0.0) * mask

        if self.reduction == 'mean':
            return percentage_error.sum() / (mask.sum() + self.eps)
        if self.reduction == 'sum':
            return percentage_error.sum()
        
        return percentage_error


class CosineLoss(torch.nn.Module):
    """
    Custom loss function that computes the cosine loss between predicted and target 3D vectors.
    
    The loss is defined as:
        loss = 1 - cosine_similarity(pred, target)
        
    This loss emphasizes directional alignment by normalizing both vectors first.
    
    Args:
        reduction (str): 'mean' (default) to average over the batch, or 'sum' for total loss.
        eps (float): Small value to avoid division by zero in normalization.
    """
    def __init__(self, reduction='mean', eps=1e-6):
        super(CosineLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        # mask out any zero-target rows
        mask = torch.any(target != 0, dim=1).float()

        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        loss = (1.0 - cos_sim) * mask
        
        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + self.eps)
        else:  # 'sum'
            return loss.sum()


class SphericalAngularLoss(torch.nn.Module):
    """
    Custom loss function that combines:
    - Geodesic angular distance for direction optimization.

    This ensures that both the direction (theta, phi) and the magnitude are optimized correctly.

    Args:
        reduction (str): 'mean' (default) to average over batch, or 'sum' for total loss.
    """
    def __init__(self, reduction='mean', eps=1e-6):
        super(SphericalAngularLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target):
        """
        Computes the combined loss given predicted and target 3D direction vectors.

        Args:
            pred (torch.Tensor): Predicted 3D direction vectors (batch_size, 3).
            target (torch.Tensor): True 3D direction vectors (batch_size, 3).

        Returns:
            torch.Tensor: The computed loss.
        """
        mask = torch.any(target != 0, dim=1)
 
        # Compute the dot product (cosine of the angle)
        dot_product = torch.sum(pred * target, dim=1).clamp(-0.9999, 0.9999)

        # Compute the angular distance (geodesic distance on the unit sphere)
        angular_loss = torch.acos(dot_product)  # Returns angles in radians
        angular_loss = torch.nan_to_num(angular_loss, nan=0.0) * mask.float()

        if self.reduction == 'mean':
            valid_loss_count = mask.sum()
            angular_loss = angular_loss.sum() / (valid_loss_count + self.eps)
        elif self.reduction == 'sum':
            angular_loss = angular_loss.sum()

        return angular_loss


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


class StableLogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        return torch.mean(torch.abs(diff) + torch.log1p(torch.exp(-2 * torch.abs(diff))) - torch.log(torch.tensor(2.0)))


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha = None,
    gamma: float = 0.,
    reduction: str = "none",
    sigmoid: bool = True,
) -> torch.Tensor:
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
        alpha: (optional) Weighting factor for each class to balance
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    """
    if sigmoid:
        loss = sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction)
    else:
        loss = softmax_focal_loss(inputs, targets, alpha, gamma, reduction)
    
    return loss


def softmax_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: list = None,
    gamma: float = 0.,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Multi-class focal loss, based on: https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
        alpha: (optional) Weighting factor for each class to balance
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    if inputs.ndim > 2:
        # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
        c = inputs.shape[1]
        inputs = inputs.permute(0, *range(2, inputs.ndim), 1).reshape(-1, c)
        # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)

    if alpha is not None:
        alpha = torch.tensor(alpha, device=inputs.device)
    
    targets = targets.view(-1)

    inputs = inputs.float()
    targets = targets.long()

    # compute weighted cross entropy term: -alpha * log(pt)
    log_p = F.log_softmax(inputs, dim=1)
    ce_loss = F.nll_loss(log_p, targets, weight=alpha, reduction='none')
   
    # get true class column from each row
    all_rows = torch.arange(len(inputs))
    log_pt = log_p[all_rows, targets]

    # compute focal term: (1 - pt)^gamma
    pt = log_pt.exp()
    focal_term = (1 - pt)**gamma

    # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
    loss = focal_term * ce_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


class SigmoidFocalLossWithLogits(nn.Module):
    """
    Focal Loss for binary classification with logits input, matching
    the interface of torch.nn.BCEWithLogitsLoss.

    LOSS = alpha * (1 - p_t)^gamma * BCEWithLogits

    Args:
        alpha (float, optional): weight for the positive class in [0,1].
            Defaults to 0.25.
        gamma (float, optional): focusing parameter >= 0. Defaults to 2.0.
        reduction (str, optional): 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0,1]")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be one of 'none','mean','sum', got {reduction}")

        self.register_buffer("alpha", torch.tensor(alpha))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        # p_t = exp(-bce_loss): stable way to get sigmoid/logits relation
        pt = torch.exp(-bce_loss)

        # alpha_t = alpha if target==1 else (1-alpha)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # compute focal factor (1 - p_t)^gamma ---
        focal_factor = (1 - pt).pow(self.gamma)

        # combine into focal loss per element ---
        loss = alpha_factor * focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
            

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = None,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = None (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def sigmoid_focal_loss_star(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 1,
    reduction: str = "none",
) -> torch.Tensor:
    """
    FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    shifted_inputs = gamma * (inputs * (2 * targets - 1))
    loss = -(F.logsigmoid(shifted_inputs)) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def soft_focal_bce_with_logits(
    logits: torch.Tensor, 
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Focal binary cross-entropy (sigmoid) that supports SOFT targets.
      For each logit x and target t in [0,1], with p = sigmoid(x).
    """
    target = target.to(dtype=logits.dtype)

    # Stable logs: log(sigmoid(x)) and log(1 - sigmoid(x))
    log_p   = F.logsigmoid(logits)      # = -softplus(-x)
    log_1mp = F.logsigmoid(-logits)     # = -softplus(x)
    p       = torch.sigmoid(logits)

    mod_pos = (1.0 - p).clamp_min(eps).pow(gamma)
    mod_neg = p.clamp_min(eps).pow(gamma)

    if alpha is None:
        alpha_pos = alpha_neg = 1.0
    else:
        if not torch.is_tensor(alpha):
            alpha = logits.new_tensor(alpha, dtype=logits.dtype)
        if alpha.numel() == 1:
            alpha_pos = alpha
            alpha_neg = 1.0 - alpha
        elif alpha.numel() == 2:
            alpha_pos, alpha_neg = alpha.reshape(-1)
        else:
            raise ValueError("alpha must be None, a scalar, or a length-2 sequence/tensor")

    loss_pos = -target * mod_pos * log_p * alpha_pos
    loss_neg = -(1.0 - target) * mod_neg * log_1mp * alpha_neg
    loss = loss_pos + loss_neg
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss


def soft_focal_cross_entropy(
    logits: torch.Tensor,              # [N, C]
    target: torch.Tensor,              # [N, C] (soft labels; rows sum≈1)
    gamma: float = 2.0,
    alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Focal cross-entropy that supports SOFT targets.
      loss_i = - sum_c t_ic * (1 - p_ic)^gamma * log p_ic * alpha_c
    """
    logp = F.log_softmax(logits, dim=-1)
    p    = logp.exp()

    mod  = (1.0 - p).clamp_min(eps).pow(gamma)             # [N, C]
    loss = -target * mod * logp                            # [N, C]

    if alpha is not None:
        if not torch.is_tensor(alpha):
            alpha = logits.new_tensor(alpha, dtype=loss.dtype)
        loss = loss * alpha.view(1, -1)                    # broadcast [1, C]

    loss = loss.sum(dim=-1)                                # [N]
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss


@torch.no_grad()
def _build_grouping_from_edges(event_id_rows: torch.Tensor,
                               class_id_edges: torch.Tensor):
    """
    Group edges (edge = one (row i, class c) contribution) by (event, class),
    without bit-packing (supports negative/large class ids).

    Returns:
      inv:  [L]   -> group index in [0..G-1] for each edge
      meta: dict with:
        - ord_g: permutation of groups; when applied yields groups sorted by event
        - rank:  [G] local index of the group within its event segment (0..count_e-1)
        - eidx:  [G] event index in [0..E-1] for each group
        - counts:[E] number of groups per event (pre-threshold)
        - offset:[E] prefix sums for counts (start position of each event segment)
        - events_sorted: [E] unique event ids (sorted) defining segment order
    """
    eid = event_id_rows.to(torch.int64)
    cid = class_id_edges.to(torch.int64)

    pairs = torch.stack([eid, cid], dim=1)                      # [L,2]
    uniq_pairs, inv = torch.unique(pairs, dim=0, return_inverse=True)
    group_eid = uniq_pairs[:, 0]                                # [G]

    # Order groups by event id
    ord_g = torch.argsort(group_eid)
    ge_sorted = group_eid[ord_g]

    # Segment metadata by event
    first = torch.ones_like(ge_sorted, dtype=torch.bool)
    if ge_sorted.numel() > 0:
        first[1:] = ge_sorted[1:] != ge_sorted[:-1]
    starts = torch.nonzero(first, as_tuple=False).squeeze(1)    # [E]
    counts = torch.diff(torch.cat([starts, ge_sorted.new_tensor([ge_sorted.numel()])]))
    E = starts.numel()

    pos = torch.arange(uniq_pairs.size(0), device=cid.device)
    start_for_pos = starts.repeat_interleave(counts)
    rank_sorted = pos - start_for_pos
    rank_in_event = torch.empty_like(rank_sorted)
    rank_in_event[ord_g] = rank_sorted

    eidx_for_pos = torch.arange(E, device=cid.device).repeat_interleave(counts)
    eidx_for_group = torch.empty_like(eidx_for_pos)
    eidx_for_group[ord_g] = eidx_for_pos

    offset = torch.zeros(E, dtype=torch.long, device=cid.device)
    if E > 1:
        offset[1:] = torch.cumsum(counts[:-1], dim=0)

    events_sorted = ge_sorted[starts]
    return inv, {
        "ord_g": ord_g, "rank": rank_in_event, "eidx": eidx_for_group,
        "counts": counts, "offset": offset, "events_sorted": events_sorted
    }


def _sum_indices_deterministic(idx: torch.Tensor, val: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.zeros(size, dtype=val.dtype, device=val.device)
    out.index_add_(0, idx, val)
    return out


def _segment_mean_from_contiguous_idx_det(values: torch.Tensor,
                                          idx: torch.Tensor,
                                          size: int,
                                          weights: torch.Tensor = None) -> torch.Tensor:
    if weights is None:
        sums = torch.zeros(size, dtype=values.dtype, device=values.device)
        sums.index_add_(0, idx, values)
        counts = torch.zeros(size, dtype=values.dtype, device=values.device)
        counts.index_add_(0, idx, torch.ones_like(values, dtype=values.dtype))
        valid = counts > 0
        means = sums[valid] / counts[valid].clamp_min(1)
        return means.mean() if valid.any() else values.new_zeros(())
    else:
        sums = torch.zeros(size, dtype=values.dtype, device=values.device)
        sums.index_add_(0, idx, values * weights)
        den = torch.zeros(size, dtype=values.dtype, device=values.device)
        den.index_add_(0, idx, weights)
        valid = den > 0
        means = sums[valid] / den[valid].clamp_min(1e-12)
        return means.mean() if valid.any() else values.new_zeros(())


def contrastive_with_ghost_shared(
    z: torch.Tensor,
    event_id: torch.Tensor,
    ghost_mask: torch.Tensor,
    label_indptr: torch.Tensor,
    label_class: torch.Tensor,
    label_weight: torch.Tensor,
    num_neg: int = 32,
    pool_mult: int = 1,
    normalize: bool = True,
    min_class_size: float = 10.0,
    temperature: float = 0.07,
    detach_prototypes: bool = True,
):
    if z.numel() == 0 or num_neg <= 0:
        return z.new_zeros(())

    device = z.device
    N, D = z.shape
    if normalize:
        z = F.normalize(z, dim=-1, eps=1e-6)

    MINF32 = torch.finfo(torch.float32).min / 4

    # Expand CSR rows to edges
    lw = label_weight.clamp_min(0)
    deg = (label_indptr[1:] - label_indptr[:-1])  # [N]
    if deg.numel() != N:
        raise ValueError("label_indptr shape mismatch with N")
    if deg.sum().item() == 0:
        return z.new_zeros(())

    rows = torch.arange(N, device=device, dtype=torch.long).repeat_interleave(deg)  # [L]
    eids_edge = event_id[rows].to(torch.int64)  # [L]
    cids_edge = label_class.to(torch.int64)     # [L]
    w_edge    = lw                              # [L]

    # Group by (event, class)
    inv_group_edge, Pidx = _build_grouping_from_edges(eids_edge, cids_edge)
    G = int(Pidx["rank"].numel())

    # Weighted prototypes in fp32
    P_sum = torch.zeros((G, D), device=device, dtype=torch.float32)
    P_sum.index_add_(0, inv_group_edge, z[rows].to(torch.float32) * w_edge.unsqueeze(1).to(torch.float32))
    cnt_vec = torch.zeros(G, device=device, dtype=torch.float32)
    cnt_vec.index_add_(0, inv_group_edge, w_edge.to(torch.float32))
    cnt_vec.clamp_(min=1e-12)

    if detach_prototypes:
        P_sum = P_sum.detach()
        cnt_vec = cnt_vec.detach()

    # Valid groups/events
    def _prepare_valid_groups_soft(Pidx, cnt_vec, min_class_weight: float):
        G = cnt_vec.numel()
        E = int(Pidx["counts"].numel())
        thr = max(2.0, float(min_class_weight))
        eidx_g = Pidx["eidx"]                     # [G]
        ok_cls = (cnt_vec >= thr)                 # [G]
        counts_ok = torch.zeros(E, dtype=torch.long, device=cnt_vec.device)
        if ok_cls.any():
            counts_ok.index_add_(0, eidx_g[ok_cls], torch.ones_like(eidx_g[ok_cls], dtype=torch.long))
        event_ok = (counts_ok >= 2)               # [E]
        group_ok = ok_cls & event_ok[eidx_g]      # [G]

        ord_g = Pidx["ord_g"]
        ord_ok = ord_g[group_ok[ord_g]]

        counts_ok_eff = counts_ok * event_ok.long()
        offset_ok = torch.zeros_like(counts_ok_eff)
        if E > 1:
            offset_ok[1:] = torch.cumsum(counts_ok_eff[:-1], dim=0)

        total_ok = int(counts_ok_eff.sum().item())
        if total_ok == 0:
            gid2compact = cnt_vec.new_full((G,), -1, dtype=torch.long)
            rank_valid = cnt_vec.new_full((G,), -1, dtype=torch.long)
            return group_ok, counts_ok_eff, offset_ok, ord_ok, gid2compact, rank_valid

        pos = torch.arange(total_ok, device=cnt_vec.device)
        eidx_pos = torch.arange(E, device=cnt_vec.device).repeat_interleave(counts_ok_eff)
        rank_sorted = pos - offset_ok[eidx_pos]
        rank_valid = cnt_vec.new_full((G,), -1, dtype=torch.long)
        rank_valid[ord_ok] = rank_sorted

        gid2compact = cnt_vec.new_full((G,), -1, dtype=torch.long)
        gid2compact[ord_ok] = torch.arange(total_ok, device=cnt_vec.device)
        return group_ok, counts_ok_eff, offset_ok, ord_ok, gid2compact, rank_valid

    group_ok, counts_ok, offset_ok, ord_ok, gid2compact, rank_valid = \
        _prepare_valid_groups_soft(Pidx, cnt_vec, min_class_size)

    if int(counts_ok.sum().item()) == 0:
        return z.new_zeros(())

    P_mean = (P_sum[group_ok] / cnt_vec[group_ok].unsqueeze(1))
    if normalize:
        P_mean = F.normalize(P_mean, dim=-1, eps=1e-6)

    g_edge = inv_group_edge
    edge_ok = group_ok[g_edge] & (~ghost_mask.bool()[rows])
    if not torch.any(edge_ok):
        loss = z.new_zeros(())
    else:
        rows_e  = rows[edge_ok]                     # [M]
        gids_e  = g_edge[edge_ok]                   # [M]
        w_e     = w_edge[edge_ok]                   # [M]
        eidx_e  = Pidx["eidx"][gids_e]              # [M]
        local_e = rank_valid[gids_e]                # [M]

        valid_event = (counts_ok[eidx_e] >= 2) & (local_e >= 0)
        rows_e, gids_e, w_e, eidx_e, local_e = \
            rows_e[valid_event], gids_e[valid_event], w_e[valid_event], eidx_e[valid_event], local_e[valid_event]

        if rows_e.numel() == 0:
            loss = z.new_zeros(())
        else:
            cnt_e = cnt_vec[gids_e].to(z.dtype)
            denom = (cnt_e - w_e).clamp_min(1e-12)
            P_pos_raw = (P_sum[gids_e].to(z.dtype) - z[rows_e] * w_e.unsqueeze(1)) / denom.unsqueeze(1)
            if detach_prototypes:
                P_pos_raw = P_pos_raw.detach()
            P_pos = F.normalize(P_pos_raw, dim=-1, eps=1e-6) if normalize else P_pos_raw

            z_e = z[rows_e]
            pos_sim = (z_e.float() * P_pos.float()).sum(-1, keepdim=True)

            # build per-row positive masks over local indices
            urows, inv_rows = torch.unique(rows_e, return_inverse=True)  # urows: [R]
            R = urows.numel()
            lcl_all = rank_valid[g_edge]                                  # [L]
            row_map = torch.full((N,), -1, dtype=torch.long, device=device)
            row_map[urows] = torch.arange(R, device=device)
            r_idx_all = row_map[rows]                                     # [L]
            sel_all = (r_idx_all >= 0) & (lcl_all >= 0)
            r_sel   = r_idx_all[sel_all]
            l_sel   = lcl_all[sel_all]

            euniq = Pidx["events_sorted"]                                 # [E]
            eids_u = event_id[urows].to(torch.int64)                      # [R]
            eidx_rows = torch.searchsorted(euniq, eids_u)                 # [R] in [0..E]
            # clamp and verify to avoid OOB when event not present
            safe_idx = eidx_rows.clamp_max(max(0, euniq.numel() - 1))
            match = (euniq[safe_idx] == eids_u)
            # if an event isn't present (should be rare), set its count to 0
            Ue_rows = counts_ok[safe_idx] * match.to(counts_ok.dtype)     # [R]
            maxU = int(Ue_rows.max().item()) if R > 0 else 0

            ex_mask = torch.zeros(R, maxU, dtype=torch.bool, device=device)
            if r_sel.numel() > 0:
                # only keep positions < maxU
                l_sel_clamped = l_sel.clamp_max(maxU - 1)
                ex_mask[r_sel, l_sel_clamped] = True

            # per-anchor allowed negatives mask
            ar = torch.arange(maxU, device=device)                         # [maxU]
            Ue = counts_ok[eidx_e]                                         # [M]
            valid_by_range = ar.unsqueeze(0) < Ue.unsqueeze(1)             # [M, maxU]
            r_e = inv_rows                                                 # [M]
            pos_mask = ex_mask[r_e]                                        # [M, maxU]
            allowed = valid_by_range & (~pos_mask)                         # [M, maxU]
            allowed_count = allowed.sum(dim=1)
            allowed_count_max = int(allowed_count.max().item())

            if maxU == 0 or allowed_count_max == 0:
                loss = z.new_zeros(())
            elif allowed_count_max <= num_neg:
                # ALL negatives (fast path). Clamp local indices per row to [0, Ue-1]
                max_local = (Ue - 1).clamp_min(0).unsqueeze(1)            # [M,1]
                safe_local = torch.minimum(ar.unsqueeze(0).expand_as(allowed), max_local)  # [M, maxU]
                base_all = offset_ok[eidx_e].unsqueeze(1) + safe_local    # [M, maxU]
                groups_all = ord_ok[base_all]                              # [M, maxU]
                comp_idx = gid2compact[groups_all]                         # [M, maxU]
                P_all = P_mean[comp_idx]                                   # [M, maxU, D]
                sims_all = torch.einsum('md,mkd->mk', z_e.float(), P_all.float())
                sims_all = sims_all.masked_fill(~allowed, MINF32)

                logits32 = torch.cat([pos_sim, sims_all], dim=1) / temperature
                per_edge = F.cross_entropy(
                    logits32, torch.zeros(rows_e.size(0), dtype=torch.long, device=device),
                    reduction='none'
                ) * w_e.to(logits32.dtype)
            else:
                # Masked Gumbel top-K without replacement
                K_cap = max(1, int(num_neg) * int(pool_mult))
                K_pool = min(K_cap, allowed_count_max)

                g = torch.empty_like(allowed, dtype=torch.float32).uniform_(0, 1)
                g = -torch.log(-torch.log(g.clamp_min(1e-12)))
                g = g.masked_fill(~allowed, float('-inf'))

                cand_local = g.topk(k=K_pool, dim=1, largest=True).indices
                selected_valid = allowed.gather(1, cand_local)
 
                max_local = (Ue - 1).clamp_min(0).unsqueeze(1)               # [M,1]
                cand_safe = torch.minimum(cand_local, max_local)             # [M, K_pool]

                max_local = (Ue - 1).clamp_min(0).unsqueeze(1)               # [M,1]
                cand_safe = torch.minimum(cand_local, max_local)             # [M, K_pool]

                base = offset_ok[eidx_e].unsqueeze(1) + cand_safe            # [M, K_pool]  (SAFE)
                neg_group = ord_ok[base]                                     # [M, K_pool]
                P_negs = P_mean[gid2compact[neg_group]]                      # [M, K_pool, D]

                neg_sim_full = torch.einsum('md,mkd->mk', z_e.float(), P_negs.float())
                neg_sim_full = neg_sim_full.masked_fill(~selected_valid, MINF32)

                k = min(num_neg, K_pool)
                neg_sim = torch.topk(neg_sim_full, k=k, dim=1).values

                logits32 = torch.cat([pos_sim, neg_sim], dim=1) / temperature
                per_edge = F.cross_entropy(
                    logits32, torch.zeros(rows_e.size(0), dtype=torch.long, device=device),
                    reduction='none'
                ) * w_e.to(logits32.dtype)

            # Average over anchors that contribute any edge
            per_row_sum = torch.zeros(z.size(0), dtype=per_edge.dtype, device=device)
            per_row_sum.index_add_(0, rows_e, per_edge)
            row_mask = torch.zeros(z.size(0), dtype=torch.bool, device=device)
            row_mask[rows_e] = True
            denom_rows = row_mask.to(per_edge.dtype).sum().clamp_min(1)
            loss = (per_row_sum[row_mask].sum() / denom_rows).to(z.dtype)

    return loss


def occ_supervision_mask(
    idx_targets: torch.Tensor,          # [M, P], -1 => empty sub-voxel, >=0 => raw index of hit
    patch_shape: Tuple[int, int, int],  # (p_h, p_w, p_d)
    ghost_mask: torch.Tensor,
    occ_empty_beta: float = 0.5,
    dilate: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        sup_mask:         [M, P] bool — (positives ∪ border ∪ sampled_negatives)
        sup_targ:         [M, P] float — 1 for positives; 0 otherwise
        pos_mask:         [M, P] bool — non-ghost occupied subvoxels
        border_mask:      [M, P] bool
        sampled_neg_mask: [M, P] bool — negatives actually sampled
    """
    device = idx_targets.device
    M, P   = idx_targets.shape
    p_h, p_w, p_d = patch_shape

    is_occ = (idx_targets >= 0)
    is_ghost = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
    is_ghost[is_occ] = ghost_mask[idx_targets[is_occ]]
    pos_mask = is_occ & ~is_ghost

    # Border via dilation
    occ = pos_mask.float().view(M, 1, p_h, p_w, p_d)
    if dilate > 0:
        ksz = 2 * dilate + 1
        kernel = torch.ones((1, 1, ksz, ksz, ksz), device=device)
        dil = F.conv3d(occ, kernel, padding=dilate) > 0
    else:
        dil = occ.bool()
    border_mask = dil.view(M, P) & (~pos_mask)

    # Eligible negatives & global budget (proportional to #positives)
    eligible_neg = ~(pos_mask | border_mask)
    pos_counts = pos_mask.sum()
    neg_counts_r = eligible_neg.sum(dim=1)  # [M]
    total_neg = int(neg_counts_r.sum().item())

    target_total_negs = int(min(total_neg, round(occ_empty_beta * int(pos_counts.item()))))

    q_r = (neg_counts_r.float() / max(1, total_neg) * target_total_negs).floor().to(torch.long)
    q_r = torch.minimum(q_r, neg_counts_r)
    K_max = int(q_r.max().item())

    sampled_neg_mask = torch.zeros_like(eligible_neg, dtype=torch.bool)
    if K_max > 0:
        rnd = torch.rand(M, P, device=device).masked_fill(~eligible_neg, float("inf"))
        _, idxs = torch.topk(-rnd, k=K_max, dim=1)  # [M, K_max]
        rows = torch.arange(M, device=device).unsqueeze(1).expand(-1, K_max)
        keep = (torch.arange(K_max, device=device).unsqueeze(0) < q_r.unsqueeze(1))
        sampled_neg_mask[rows[keep], idxs[keep]] = True

    sup_mask = pos_mask | border_mask | sampled_neg_mask
    sup_targ = pos_mask.float()

    return sup_mask, sup_targ, pos_mask, border_mask, sampled_neg_mask


def _row_event_ids(idx_targets: torch.Tensor, hit_event_id: torch.Tensor) -> torch.Tensor:
    """
    Map each token row to an event id using the first occupied sub-voxel in that row.
    Rows with no occupied sub-voxels get -1.
    """
    device = idx_targets.device
    M, P = idx_targets.shape
    valid = idx_targets >= 0
    any_valid = valid.any(dim=1)
    first_col = torch.argmax(valid.to(torch.int64), dim=1)  # arbitrary first True
    row_idx = torch.arange(M, device=device)
    raw = idx_targets[row_idx, first_col]
    raw = torch.where(any_valid, raw, torch.zeros_like(raw))
    row_event = hit_event_id[raw]
    row_event = torch.where(any_valid, row_event, torch.full_like(row_event, -1))
    return row_event  # [M]


def eventwise_mean(values: torch.Tensor, event_ids: torch.Tensor) -> torch.Tensor:
    """
    Any (possibly non-contiguous) event ids, -1 = unknown (ignored).
    Unweighted mean per event, then unweighted mean across events.
    """
    valid = (event_ids >= 0)
    if not torch.any(valid):
        return values.mean()
    ev   = event_ids[valid]
    vals = values[valid]
    # map to contiguous bins deterministically via sorting unique ids
    uniq, inv = torch.unique(ev, return_inverse=True)  # sorted -> deterministic
    size = uniq.numel()
    return _segment_mean_from_contiguous_idx_det(vals, inv, size, weights=None)


def eventwise_weighted_mean(values: torch.Tensor, weights: torch.Tensor, event_ids: torch.Tensor) -> torch.Tensor:
    """
    Any (possibly non-contiguous) event ids, -1 = unknown (ignored).
    Weighted mean per event, then unweighted mean across events.
    """
    valid = (event_ids >= 0)
    if not torch.any(valid):
        return (weights * values).sum() / (weights.sum() + 1e-12)
    ev   = event_ids[valid]
    vals = values[valid]
    wts  = weights[valid]
    uniq, inv = torch.unique(ev, return_inverse=True)  # sorted -> deterministic
    size = uniq.numel()
    return _segment_mean_from_contiguous_idx_det(vals, inv, size, weights=wts)


def reconstruction_losses_masked_simple(
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
    # hyper/behaviour
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
    Returns (loss_occ, loss_reg, part_losses_dec).
    If per_event_mean=True, compute mean per event first (where possible) before averaging across events.
    """
    device = idx_targets.device
    M, P = idx_targets.shape
    C_in = targ_reg.shape[1]

    sup_mask, sup_targ, pos_mask, border_mask, sampled_neg_mask = occ_supervision_mask(
        idx_targets, patch_shape, ghost_mask, occ_empty_beta=occ_empty_beta, dilate=occ_dilate
    )

    # ===================== OCC =====================
    if label_smoothing > 0.0:
        eps = label_smoothing
        sup_targ = sup_targ * (1.0 - eps) + 0.5 * eps

    # Flat supervised OCC logits/targets
    occ_logits_sup = pred_occ[sup_mask]
    occ_targ_sup   = sup_targ[sup_mask]

    # Focal BCE
    occ_losses = soft_focal_bce_with_logits(
        occ_logits_sup, occ_targ_sup, gamma=focal_gamma, alpha=focal_alpha, reduction='none'
    )  # [N_sup]

    # Break down for logging
    pos_or_border = (pos_mask | border_mask)[sup_mask]
    neg_only = sampled_neg_mask[sup_mask]
    occ_pos_loss = occ_losses[pos_or_border].mean() if pos_or_border.any() else torch.tensor(0., device=device)
    occ_neg_loss = occ_losses[neg_only].mean() if neg_only.any() else torch.tensor(0., device=device)

    # Per-event mean for OCC (if enabled)
    if per_event_mean:
        row_event = _row_event_ids(idx_targets, hit_event_id)  # [M]
        sup_rows = sup_mask.nonzero(as_tuple=True)[0]          # [N_sup]
        sup_events = row_event[sup_rows]                       # [-1 or event_id]
        loss_occ = eventwise_mean(occ_losses, sup_events)
    else:
        loss_occ = occ_losses.mean()

    # ===================== REG =====================
    flat_idx_targets = idx_targets.view(-1)
    pos_idx = torch.where(pos_mask.view(-1))[0]
    neg_idx = torch.where(sampled_neg_mask.view(-1))[0]
    all_idx = torch.cat([pos_idx, neg_idx], dim=0)
    N_pos = pos_idx.numel()

    # Predictions
    pred_reg_flat = pred_reg.view(-1, C_in)[all_idx]  # [N_all, C_in]

    # Targets (positives from hits; negatives to 'empty' representative)
    reg_empty = targ_reg.amin(dim=0)                  # [C_in]
    targ_reg_flat = reg_empty.unsqueeze(0).expand(all_idx.numel(), -1).clone()
    if N_pos > 0:
        raw_pos = flat_idx_targets[pos_idx]
        targ_reg_flat[:N_pos] = targ_reg[raw_pos]

    # Huber / Smooth L1 row loss
    reg_elem = F.smooth_l1_loss(pred_reg_flat, targ_reg_flat, beta=huber_delta, reduction='none')  # [N_all,C_in]
    reg_row = reg_elem.sum(dim=1)  # [N_all]

    # charge-aware weights (positives only)
    w = torch.ones_like(reg_row)
    if N_pos > 0:
        q_pos_orig = dataset.unpreprocess(
            targ_reg_flat[:N_pos], 'q', preprocessing=preprocessing_input
        ).squeeze(-1)  # [N_pos]
        eps = 1e-6
        if reg_weight_q0 is None:
            q0 = torch.quantile(q_pos_orig.detach(), 0.75).clamp_min(eps)
        else:
            q0 = torch.as_tensor(reg_weight_q0, dtype=q_pos_orig.dtype, device=q_pos_orig.device).clamp_min(eps)

        w_pos = 1.0 + reg_weight_lam * (q_pos_orig / q0).clamp_min(0.).pow(reg_weight_alpha)
        if reg_weight_wmax is not None:
            w_pos = torch.clamp(w_pos, max=float(reg_weight_wmax))
        w[:N_pos] = w_pos

    # Per-event mean for REG (if enabled)
    # Assign each flat index to a row -> event id
    row_event = _row_event_ids(idx_targets, hit_event_id)  # [M]
    all_rows = all_idx // P                                # [N_all]
    all_events = row_event[all_rows]                       # [-1 or event_id]

    if per_event_mean:
        loss_reg = eventwise_weighted_mean(reg_row, w, all_events)
        # For logging: keep pos/neg breakdown (standard means)
        reg_pos_loss = (w[:N_pos] * reg_row[:N_pos]).sum() / (w[:N_pos].sum() + 1e-12) if N_pos > 0 else torch.tensor(0., device=device)
        reg_neg_loss = reg_row[N_pos:].mean() if reg_row.numel() > N_pos else torch.tensor(0., device=device)
    else:
        loss_reg = (w * reg_row).sum() / (w.sum() + 1e-12)
        reg_pos_loss = (w[:N_pos] * reg_row[:N_pos]).sum() / (w[:N_pos].sum() + 1e-12) if N_pos > 0 else torch.tensor(0., device=device)
        reg_neg_loss = reg_row[N_pos:].mean() if reg_row.numel() > N_pos else torch.tensor(0., device=device)

    part_losses_dec = {
        'occ/total': loss_occ.detach(), 'occ/pos': occ_pos_loss.detach(), 'occ/neg': occ_neg_loss.detach(),
        'reg/total': loss_reg.detach(), 'reg/pos': reg_pos_loss.detach(), 'reg/neg': reg_neg_loss.detach(),
    }
    return loss_occ, loss_reg, part_losses_dec


def build_soft_targets_from_csr(indptr, cls, w, ghost_mask, num_classes, N):
    device = cls.device
    rows = torch.arange(N, device=device, dtype=torch.long).repeat_interleave(indptr[1:] - indptr[:-1])
    target = torch.zeros(N, num_classes-1, device=device, dtype=w.dtype)
    target.index_put_((rows, cls.to(torch.long)), w.clamp_min(0), accumulate=True)
    target_sum = target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    probs = target / target_sum  # [N, C]
    ghost_col = ghost_mask.to(device=device, dtype=w.dtype).reshape(N, 1)
    return torch.cat([probs, ghost_col], dim=1)


def soft_ce_with_logits_csr(
    logits: torch.Tensor,
    csr: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (indptr, cls, w)
    ghost_mask: torch.Tensor = None,
):
    N, num_classes = logits.shape
    soft_pid = build_soft_targets_from_csr(*csr, ghost_mask=ghost_mask, num_classes=num_classes, N=N)
    loss = -(soft_pid * torch.log_softmax(logits, dim=-1)).sum(dim=1).mean()
    return loss


def pair_soft_overlap_bce(
    z, event_id, indptr, cls, w,
    ghost_mask=None,
    topk_T=8,
    pairs_per_batch=8192,
    min_group_size=5,
    normalize=True,
    rebalance=True,
    logit_bias=0.0,
    logit_scale=10.0,
):
    device = z.device
    N, D = z.shape
    if N == 0:
        return z.new_zeros(()), {"pairs": 0}

    # fp32 normalize for numeric stability
    z32 = F.normalize(z.float(), dim=-1, eps=1e-6) if normalize else z.float()

    # keep only rows with labels (and non-ghost)
    has_edge = (indptr[1:] > indptr[:-1])
    valid = has_edge.clone()
    if ghost_mask is not None:
        valid &= ~ghost_mask.bool()

    # row-level filter: require at least one class with >= min_group_size in the event
    edge_group_count = None
    if min_group_size > 1 and has_edge.any():
        row_ids = torch.arange(N, device=device)
        nnz_per_row = (indptr[1:] - indptr[:-1])
        edge_row = torch.repeat_interleave(row_ids, nnz_per_row)         # [L]
        edge_event = event_id[edge_row].to(torch.long)                   # [L]
        edge_cls   = cls.to(torch.long)                                  # [L]

        pairs_ec = torch.stack([edge_event, edge_cls], dim=1)            # [L,2]
        _, inverse, counts = torch.unique(
            pairs_ec, dim=0, return_inverse=True, return_counts=True
        )
        edge_group_count = counts[inverse]                               # [L]

        edge_is_large = (edge_group_count >= int(min_group_size)).to(torch.int32)
        row_large_sum = torch.zeros(N, dtype=torch.int32, device=device)
        row_large_sum.index_put_((edge_row,), edge_is_large, accumulate=True)
        valid &= (row_large_sum > 0)

    if valid.sum() < 2:
        return z.new_zeros(()), {"pairs": 0}

    # sort by event and segment
    e = event_id[valid].long()
    order = torch.argsort(e)
    e_sorted = e[order]
    rows_sorted = torch.nonzero(valid, as_tuple=False).squeeze(1)[order]
    Rtot = rows_sorted.numel()

    first = torch.ones_like(e_sorted, dtype=torch.bool)
    first[1:] = e_sorted[1:] != e_sorted[:-1]
    starts = torch.nonzero(first, as_tuple=False).squeeze(1)
    if starts.numel() == 0:
        return z.new_zeros(()), {"pairs": 0}
    lens = torch.diff(torch.cat([starts, e_sorted.new_tensor([Rtot])]))

    # anchors from events with >=2 rows
    seg_id = torch.repeat_interleave(torch.arange(starts.numel(), device=device), lens)
    good_ev = (lens >= 2)
    if not good_ev.any():
        return z.new_zeros(()), {"pairs": 0}
    rows_ok = torch.nonzero(good_ev[seg_id], as_tuple=False).squeeze(1)

    K = int(min(int(pairs_per_batch), rows_ok.numel()))
    if K == 0:
        return z.new_zeros(()), {"pairs": 0}

    anchor_pos = rows_ok[torch.randperm(rows_ok.numel(), device=device)[:K]]
    ev_idx   = seg_id[anchor_pos]
    ev_start = starts[ev_idx]
    ev_len   = lens[ev_idx]  # >=2

    # unbiased mate within same event; avoid self
    offs = torch.floor(torch.rand_like(ev_len.float()) * ev_len.float()).to(torch.long)
    anchor_rel = anchor_pos - ev_start
    same = (offs == anchor_rel)
    offs = (offs + same.long()) % ev_len
    mate_pos = ev_start + offs

    row_i = rows_sorted[anchor_pos]
    row_j = rows_sorted[mate_pos]

    # top-K per used row
    rows_used = torch.unique(torch.cat([row_i, row_j], 0))
    M = rows_used.numel()
    if M == 0:
        return z.new_zeros(()), {"pairs": 0}

    start_r = indptr[rows_used]
    end_r   = indptr[rows_used + 1]
    len_r   = end_r - start_r
    Lmax = int(len_r.max().item())
    if Lmax == 0:
        return z.new_zeros(()), {"pairs": 0}

    ar = torch.arange(Lmax, device=device).unsqueeze(0)
    valid_cols = ar < len_r.unsqueeze(1)
    edge_idx = (start_r.unsqueeze(1) + ar).clamp_max(int(indptr[-1].item()) - 1)

    c_pad = torch.full((M, Lmax), -1, dtype=cls.dtype, device=device)
    w_pad = torch.zeros((M, Lmax), dtype=w.dtype, device=device)
    flat = valid_cols.view(-1)
    if flat.any():
        src = edge_idx.view(-1)[flat]
        c_pad.view(-1)[flat] = cls[src]
        w_pad.view(-1)[flat] = w[src]

    # contribution-level mask: ignore classes with event-count < min_group_size
    if (min_group_size > 1) and (edge_group_count is not None):
        keep_edge = valid_cols & (edge_group_count[edge_idx] >= int(min_group_size))
        w_pad = torch.where(keep_edge, w_pad, torch.zeros_like(w_pad))
        c_pad = torch.where(keep_edge, c_pad, c_pad.new_full(c_pad.shape, -1))

    # top-K and renorm
    T = int(topk_T)
    k = min(T, Lmax)
    topw = torch.zeros((M, T), dtype=w.dtype, device=device)
    topc = torch.full((M, T), -1, dtype=cls.dtype, device=device)
    if k > 0:
        vals, idxs = torch.topk(w_pad, k=k, dim=1)
        topw[:, :k] = vals
        topc[:, :k] = torch.gather(c_pad, 1, idxs)

    # drop pairs where a row lost all mass after masking+Top-K
    denom = topw.sum(dim=1, keepdim=True)
    row_has_mass = (denom.squeeze(1) > 0)
    if not row_has_mass.any():
        return z.new_zeros(()), {"pairs": 0}
    topw = topw / denom.clamp_min(1e-12)

    row2loc = torch.full((N,), -1, dtype=torch.long, device=device)
    row2loc[rows_used] = torch.arange(M, device=device)
    li, lj = row2loc[row_i], row2loc[row_j]

    ci, wi = topc[li], topw[li]    # [K,T]
    cj, wj = topc[lj], topw[lj]    # [K,T]

    # Optional: filter pairs where either side has zero mass (after Top-K)
    keep_pair = (wi.sum(-1) > 0) & (wj.sum(-1) > 0)
    if not keep_pair.any():
        return z.new_zeros(()), {"pairs": 0}
    row_i = row_i[keep_pair]; row_j = row_j[keep_pair]
    ci, wi = ci[keep_pair], wi[keep_pair]
    cj, wj = cj[keep_pair], wj[keep_pair]

    # soft overlap y_ij
    vi, vj = (ci >= 0), (cj >= 0)
    eq = (ci.unsqueeze(2) == cj.unsqueeze(1)) & vi.unsqueeze(2) & vj.unsqueeze(1)
    y = (eq * (wi.unsqueeze(2) * wj.unsqueeze(1))).sum(dim=(1, 2)).float()  # [K] in [0,1]

    # logits from cosine similarity (NO bias)
    sims = (z32[row_i] * z32[row_j]).sum(-1)
    logits = sims * logit_scale + logit_bias

    if rebalance:
        # class-weighting via batch prior
        p = y.detach().mean().clamp(1e-4, 1 - 1e-4)
        pos_w = (1 - p) / p
        w_pair = pos_w * y + (1 - y)
        # keep average weight ~ 1.0
        w_pair = w_pair / (pos_w * p + (1 - p))
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        loss = (bce * w_pair).mean()
    else:
        loss = F.binary_cross_entropy_with_logits(logits, y)

    stats = {
        "pairs": int(y.numel()),
        "target_mean": float(y.mean().detach()),
        "sim_pos_mean": float(sims[y > 0.5].mean().detach()) if (y > 0.5).any() else 0.0,
        "sim_neg_mean": float(sims[y < 0.1].mean().detach()) if (y < 0.1).any() else 0.0,
    }
    return loss.to(z.dtype), stats
