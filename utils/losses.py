"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Custom loss functions.
"""

import math
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
def _build_grouping(event_id: torch.Tensor, class_id: torch.Tensor):
    cid = class_id.to(torch.int64); eid = event_id.to(torch.int64)
    key = (eid << 32) | cid
    uniq_key, inv_group = torch.unique(key, return_inverse=True)   # groups G
    G = uniq_key.numel()
    group_eid = (uniq_key >> 32)
    ord_g = torch.argsort(group_eid)
    ge_sorted = group_eid[ord_g]

    first = torch.ones_like(ge_sorted, dtype=torch.bool)
    first[1:] = ge_sorted[1:] != ge_sorted[:-1]
    starts = torch.nonzero(first, as_tuple=False).squeeze(1)
    counts = torch.diff(torch.cat([starts, ge_sorted.new_tensor([ge_sorted.numel()])]))
    E = starts.numel()

    pos = torch.arange(G, device=class_id.device)
    start_for_pos = starts.repeat_interleave(counts)
    rank_sorted = pos - start_for_pos
    rank_in_event = torch.empty(G, dtype=torch.long, device=class_id.device)
    rank_in_event[ord_g] = rank_sorted

    eidx_for_pos = torch.arange(E, device=class_id.device).repeat_interleave(counts)
    eidx_for_group = torch.empty(G, dtype=torch.long, device=class_id.device)
    eidx_for_group[ord_g] = eidx_for_pos

    offset = torch.zeros(E, dtype=torch.long, device=class_id.device)
    if E > 1: offset[1:] = torch.cumsum(counts[:-1], dim=0)

    events_sorted = ge_sorted[starts]  # unique event ids in segment order

    return inv_group, {
        "ord_g": ord_g, "rank": rank_in_event, "eidx": eidx_for_group,
        "counts": counts, "offset": offset, "events_sorted": events_sorted
    }


def _count_indices_deterministic(idx: torch.Tensor, size: int) -> torch.Tensor:
    # Deterministic integer counts on the same device as idx (instead of bincount).
    out = torch.zeros(size, dtype=torch.long, device=idx.device)
    out.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    return out


def _segment_mean_from_contiguous_idx_det(
    values: torch.Tensor,  # [N]
    idx: torch.Tensor,     # [N] int64 in [0..size-1]
    size: int,
    weights: torch.Tensor = None  # [N] or None
) -> torch.Tensor:
    """
    Deterministic counts via _count_indices_deterministic; sums via index_add_.
    Returns the mean of per-segment means over segments with >0 mass.
    """
    device = values.device
    dtype  = values.dtype

    if weights is None:
        # sums
        sums = torch.zeros(size, dtype=dtype, device=device)
        sums.index_add_(0, idx, values)
        # counts (deterministic)
        counts = _count_indices_deterministic(idx, size).to(dtype)
        valid = counts > 0
        means = sums[valid] / counts[valid].clamp_min(1)
        return means.mean() if valid.any() else values.new_zeros(())
    else:
        # weighted sums and weighted counts
        sums = torch.zeros(size, dtype=dtype, device=device)
        sums.index_add_(0, idx, values * weights)
        den = torch.zeros(size, dtype=dtype, device=device)
        den.index_add_(0, idx, weights)
        valid = den > 0
        means = sums[valid] / den[valid].clamp_min(1e-12)
        return means.mean() if valid.any() else values.new_zeros(())
    

def prototype_contrastive_loss(
    z: torch.Tensor,
    class_id: torch.Tensor,
    event_id: torch.Tensor,
    num_neg: int = 32,
    normalize: bool = True,
    semi_hard: bool = False,
    semi_hard_pool_mult: int = 4,
    semi_hard_margin: float = 0.05,
    min_class_size: int = 4,
    temperature: float = 0.07,
    per_event_mean: bool = False,
) -> torch.Tensor:
    """
    Leave-one-out prototype InfoNCE with negatives from other classes.
    - Positives: class prototype excluding the anchor.
    - Negatives: other class prototypes.
    - Dense fast-path when all event_id are identical (e.g., PID with collapsed events).

    If per_event_mean=True, compute the mean loss per event first, then average across events.
    Returns a scalar tensor on the same dtype/device as z.
    """
    if z.numel() == 0:
        return z.new_zeros(())

    valid = class_id >= 0
    if not torch.any(valid):
        return z.new_zeros(())

    z   = z[valid]
    cid = class_id[valid].long()
    eid = event_id[valid].long()

    if normalize:
        z = F.normalize(z, dim=-1, eps=1e-6)

    # Build grouping over (event, class)
    inv_group, Pidx = _build_grouping(eid, cid)
    G = int(Pidx["rank"].numel())
    D = z.size(1)

    # Per-group sums and counts
    P_sum = z.new_zeros((G, D))
    P_sum.index_add_(0, inv_group, z)
    #cnt_vec = torch.bincount(inv_group, minlength=G).clamp_min_(1)  # [G]
    cnt_vec = _count_indices_deterministic(inv_group, G).clamp_min_(1)

    # Per-hit metadata
    g         = inv_group                         # [M] group id of each hit
    local_pos = Pidx["rank"][g]                   # rank of the class within its event
    eidx      = Pidx["eidx"][g]                   # event index for each hit
    Ue        = Pidx["counts"][eidx]              # #classes in the (hit's) event
    off       = Pidx["offset"][eidx]              # event offset in ord_g
    cls_cnt   = cnt_vec[g]                        # class size for each hit

    # Keep only usable anchors: at least two classes in event and class size >= 2 (or min_class_size)
    keep = (Ue >= 2) & (cls_cnt >= max(2, min_class_size))
    if not torch.any(keep):
        return z.new_zeros(())

    z        = z[keep]
    g        = g[keep]
    local_pos= local_pos[keep]
    Ue       = Ue[keep]
    off      = off[keep]
    cls_cnt  = cls_cnt[keep]
    eidx     = eidx[keep]

    # Positive = leave-one-out prototype
    P_pos = (P_sum[g] - z) / (cls_cnt - 1).unsqueeze(1).to(z.dtype)
    if normalize:
        P_pos = F.normalize(P_pos, dim=-1, eps=1e-6)

    # Global per-group mean (prototype)
    P_mean = P_sum / cnt_vec.unsqueeze(1).to(P_sum.dtype)
    if normalize:
        P_mean = F.normalize(P_mean, dim=-1, eps=1e-6)

    # Positive similarities (fp32 for stability)
    z32, P_pos32 = z.float(), P_pos.float()
    pos_sim = (z32 * P_pos32).sum(-1, keepdim=True)  # [M,1]

    # Fast DENSE path if all events are the same (e.g., PID with collapsed events)
    single_event = (eid.unique().numel() == 1)
    if single_event:
        sim_all = z32 @ P_mean.float().t()           # [M, G]
        sim_all.scatter_(1, g.unsqueeze(1), -1e4)  # mask positive prototype (its own group id)

        # Number of available negatives per row is constant: G-1
        K_avail = max(0, int(P_mean.size(0)) - 1)
        if K_avail == 0:
            return z.new_zeros(())

        K_final = min(num_neg, K_avail)

        if semi_hard:
            # semi-hard band: (pos_sim - margin, pos_sim)
            band_lo = (pos_sim - semi_hard_margin)            # [M,1]
            sh = (sim_all < pos_sim) & (sim_all > band_lo)    # [M,G]
            scores = sim_all.masked_fill(~sh, -1e4)
            idx = torch.topk(scores, k=K_final, dim=1).indices
            # Fallback to hard where band is empty
            got = sh.gather(1, idx).sum(dim=1)
            need_fb = got < K_final
            if need_fb.any():
                fb_idx = torch.topk(sim_all, k=K_final, dim=1).indices
                idx[need_fb] = fb_idx[need_fb]
            neg_sim = sim_all.gather(1, idx)                  # [M,K_final]
        else:
            idx = torch.topk(sim_all, k=K_final, dim=1).indices
            neg_sim = sim_all.gather(1, idx)

        # Logits and CE (fp32), then cast back
        logits32 = torch.cat([pos_sim, neg_sim], dim=1)       # [M,1+K]
        logits32 = logits32 / temperature
        loss = F.cross_entropy(logits32, torch.zeros(z.size(0), dtype=torch.long, device=z.device))
        return loss.to(z.dtype)

    # General (multi-event) path: sampled negatives within each event
    M = z.size(0)
    K_final = num_neg
    K_pool  = (num_neg * semi_hard_pool_mult) if semi_hard else num_neg

    # How many negatives are available per anchor (at most K_pool)
    ftype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
    K_eff = torch.clamp(Ue - 1, max=K_pool)  # [M]

    # Sample negatives (no replacement via index trick around local_pos)
    rnd = torch.rand(M, K_pool, device=z.device, dtype=ftype)
    neg_raw = torch.floor(rnd * K_eff.unsqueeze(1).to(ftype)).to(torch.long)  # [M,K_pool] in [0..(Ue-2)]
    neg_local = neg_raw + (neg_raw >= local_pos.unsqueeze(1)).to(torch.long)  # skip the positive class
    base = off.unsqueeze(1) + neg_local                                       # [M,K_pool]
    neg_group = Pidx["ord_g"][base]                                           # [M,K_pool]
    P_negs = P_mean[neg_group]                                                # [M,K_pool,D]

    # Validity mask for columns where Ue-1 < K_pool
    col = torch.arange(K_pool, device=z.device).unsqueeze(0)
    neg_valid = col < K_eff.unsqueeze(1)                                     # [M,K_pool]

    # Similarities (fp32) and mask invalid with a large negative
    neg_sim_full = torch.einsum('md,mkd->mk', z32, P_negs.float())           # [M,K_pool]
    neg_sim_full = neg_sim_full.masked_fill(~neg_valid, -1e4)

    if semi_hard:
        sh_mask = (neg_sim_full < pos_sim) & (neg_sim_full > (pos_sim - semi_hard_margin))
        sh_scores = neg_sim_full.masked_fill(~sh_mask, -1e4)
        # top-k from semi-hard band; fallback to hard if band is too small
        k = min(K_final, K_pool)
        sh_idx = torch.topk(sh_scores, k=k, dim=1).indices
        got_mask = sh_mask.gather(1, sh_idx)
        enough = got_mask.sum(dim=1) >= k

        fb_idx = torch.topk(neg_sim_full, k=k, dim=1).indices
        idx_sel = torch.where(enough.unsqueeze(1), sh_idx, fb_idx)
        neg_sim = neg_sim_full.gather(1, idx_sel)                            # [M,k]
    else:
        k = min(K_final, K_pool)
        idx_sel = torch.topk(neg_sim_full, k=k, dim=1).indices
        neg_sim = neg_sim_full.gather(1, idx_sel)

    logits32 = torch.cat([pos_sim, neg_sim], dim=1)                          # [M,1+k]
    logits32 = logits32 / temperature

    per_sample = F.cross_entropy(
        logits32, torch.zeros(M, dtype=torch.long, device=z.device), reduction='none'
    )  # [M]

    if per_event_mean:
        E = int(Pidx["counts"].numel())  # number of events in the grouping
        return _segment_mean_from_contiguous_idx_det(per_sample, eidx, E, weights=None).to(z.dtype)
    else:
        return per_sample.mean().to(z.dtype)


def ghost_pushaway_loss(
    z: torch.Tensor,                   # [N, D]
    class_id: torch.Tensor,            # [N] >=0 real, <0 ghost
    event_id: torch.Tensor,            # [N]
    ghost_mask: torch.Tensor,          # [N] True for ghost anchors
    num_neg: int = 32,                 # negatives kept per anchor (top-k). Must be >0
    pool_mult: int = 1,                # oversample factor (K_pool = num_neg * pool_mult)
    normalize: bool = True,
    temperature: float = 0.07,
    per_event_mean: bool = False,
) -> torch.Tensor:
    """
    Push ghost anchors away from real (event,class) prototypes.

    - If all used event_id are identical: single dense mm [M_g,G], keep top num_neg.
    - Else: vectorized per-row sampling (no loops, no [M_g,Umax,D] tensors),
            then keep top num_neg per row.

    If per_event_mean=True, compute the mean loss per event first, then average across events.
    Returns a scalar tensor on z.dtype/device.
    """
    if z.numel() == 0 or num_neg is None or num_neg <= 0:
        return z.new_zeros(())

    gmask = ghost_mask.bool()
    rmask = (class_id >= 0)
    if not torch.any(gmask) or not torch.any(rmask):
        return z.new_zeros(())

    z_g   = z[gmask]
    z_r   = z[rmask]
    cid_r = class_id[rmask].long()
    eid_r = event_id[rmask].long()
    eid_g = event_id[gmask].long()

    if normalize:
        z_g = F.normalize(z_g, dim=-1, eps=1e-6)
        z_r = F.normalize(z_r, dim=-1, eps=1e-6)

    # build real prototypes over (event, class)
    inv_group, Pidx = _build_grouping(eid_r, cid_r)
    G = int(Pidx["rank"].numel())
    D = z_r.size(1)

    P_sum = z_r.new_zeros((G, D))
    P_sum.index_add_(0, inv_group, z_r)
    #cnt_vec = torch.bincount(inv_group, minlength=G).clamp_min_(1)
    cnt_vec = _count_indices_deterministic(inv_group, G).clamp_min_(1)
    P_mean = P_sum / cnt_vec.unsqueeze(1).to(P_sum.dtype)         # [G, D]
    if normalize:
        P_mean = F.normalize(P_mean, dim=-1, eps=1e-6)

    # map ghost events to indices in the grouping’s event order
    euniq = Pidx["events_sorted"]
    eidx_g = torch.searchsorted(euniq, eid_g)
    valid_ev = (eidx_g >= 0) & (eidx_g < euniq.numel()) & (euniq[eidx_g] == eid_g)
    if not torch.any(valid_ev):
        return z.new_zeros(())

    z_g   = z_g[valid_ev]
    eidx_g= eidx_g[valid_ev]

    # fast path: all used events identical -> single dense mm
    single_event = torch.unique(torch.cat([eid_r, eid_g], dim=0)).numel() == 1
    if single_event:
        sims = (z_g @ P_mean.t()).float()                       # [M_g, G]
        k = min(num_neg, sims.size(1))
        if k < sims.size(1):
            sims = torch.topk(sims, k=k, dim=1).values          # keep top-k only
        sims = sims / temperature
        lse = torch.logsumexp(sims, dim=1)
        loss = (lse - math.log(float(k))).mean()
        return loss.to(z.dtype)

    # general multi-event path: vectorized per-row sampling
    counts_e = Pidx["counts"]   # [E] #classes per event
    offset_e = Pidx["offset"]   # [E] start index in ord_g
    ord_g    = Pidx["ord_g"]    # [sum_E counts_e] -> group ids

    M_g   = z_g.size(0)
    Ue    = counts_e[eidx_g]                          # [M_g]
    K_pool= (num_neg * pool_mult)
    # sample per row in [0, Ue_i-1]
    ftype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
    rnd   = torch.rand(M_g, K_pool, device=z.device, dtype=ftype)
    si    = torch.floor(rnd * Ue.unsqueeze(1).to(ftype)).to(torch.long)  # [M_g,K_pool]
    # convert to global group indices
    base  = offset_e[eidx_g].unsqueeze(1) + si                            # [M_g,K_pool]
    grp   = ord_g[base]                                                   # [M_g,K_pool]

    # gather selected prototypes (small 3D tensor: [M_g,K_pool,D])
    P_sel = P_mean[grp]                                                   # [M_g,K_pool,D]

    # similarities and top-k
    sims_full = torch.einsum('md,mkd->mk', z_g.float(), P_sel.float())    # [M_g,K_pool]
    k = min(num_neg, K_pool)
    sims = torch.topk(sims_full, k=k, dim=1).values                       # [M_g,k]
    sims = sims / temperature

    per_sample = (torch.logsumexp(sims, dim=1) - math.log(float(k)))     # [M_g]

    if per_event_mean:
        E = int(Pidx["counts"].numel())
        return _segment_mean_from_contiguous_idx_det(per_sample, eidx_g, E, weights=None).to(z.dtype)
    else:
        return per_sample.mean().to(z.dtype)


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
    Mirrors the 'metric_losses_masked_simple' style: returns (loss_occ, loss_reg, part_losses_dec).
    If per_event_mean=True, compute mean per event first (where possible) before averaging across events.
    """
    device = idx_targets.device
    M, P = idx_targets.shape
    C_in = targ_reg.shape[1]

    sup_mask, sup_targ, pos_mask, border_mask, sampled_neg_mask = occ_supervision_mask(
        idx_targets, patch_shape, ghost_mask, occ_empty_beta=occ_empty_beta, dilate=occ_dilate
    )

    # OCC targets (with optional smoothing)
    if label_smoothing > 0.0:
        eps = label_smoothing
        sup_targ = sup_targ * (1.0 - eps) + 0.5 * eps

    # Flat supervised OCC logits/targets
    occ_logits_sup = pred_occ[sup_mask]
    occ_targ_sup   = sup_targ[sup_mask]

    # Focal BCE (uses the version already in your utils)
    # soft_focal_bce_with_logits is assumed to be available in this module
    occ_losses = soft_focal_bce_with_logits(
        occ_logits_sup, occ_targ_sup, gamma=focal_gamma, alpha=focal_alpha, reduction='none'
    )  # [N_sup]

    # Break down for logging
    pos_or_border = (pos_mask | border_mask)[sup_mask]
    neg_only = sampled_neg_mask[sup_mask]
    occ_pos_loss = occ_losses[pos_or_border].mean() if pos_or_border.any() else torch.tensor(0., device=device)
    occ_neg_loss = occ_losses[neg_only].mean() if neg_only.any() else torch.tensor(0., device=device)

    # ----- Per-event mean for OCC (if enabled) -----
    # Map each supervised (row, col) to a row event, then average per event where known.
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
