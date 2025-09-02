"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Custom loss functions.
"""

from typing import Optional, Sequence, Union
import torch
import torch.nn as nn
from torch.nn import functional as F


class KinematicsMultiTaskLoss(nn.Module):
    """
    Multi-task regression loss for vis/lepton (and optional jet) with analysis-aligned residuals.

    Inputs to forward():
      p_vis_hat:  (B,3) predicted visible momentum (Cartesian)
      p_lep_hat:  (B,3) predicted lepton momentum (Cartesian)
      p_vis_true: (B,3) true visible momentum
      p_lep_true: (B,3) true lepton momentum
      is_cc:      (B,)  1 for CC, 0 for NC (bool or float)
      p_jet_true: (B,3) optional true jet (if using aux jet loss); else None
      p_jet_hat:  (B,3) optional predicted jet; if None, derived as p_vis_hat - p_lep_hat
      vis_latents, lep_latents: optional tensors from Option-A heads (e.g., [zT, zz]);
                                if given, a tiny latent prior is applied.

    Configure with:
      - Scales s_*_* (MADN) for component and magnitude losses (register_buffer’d)
      - tau_ptmiss_cc/nc, tau_evis_cc/nc (per-class denominator floors)
      - Weights: zero_attractor_w, jet_aux_w, latent_prior_w, lam_mag, lam_dir
      - use_uncertainty: learn Kendall–Gal weights over [vis, lep, ptmiss, evis, jet]
      - enforce_nonneg_truth_pz: enforce pz >= 0 for truth vectors (vis and lep)
    """
    def __init__(
        self,
        *,
        # --- robust scales for VIS ---
        s_vis_xyz,                    # (3,) tuple/list: MADN per component for vis
        s_vis_mag,                    # float: MADN of ||p_vis||
        # --- robust scales for LEP ---
        s_lep_xyz,                    # (3,)
        s_lep_mag,                    # float
        # --- optional robust scales for JET (if aux jet loss enabled) ---
        s_jet_xyz=None,               # (3,) or None -> fallback to s_vis_xyz
        s_jet_mag=None,               # float or None -> fallback to s_vis_mag
        # --- residual floors (per-class) ---
        tau_ptmiss_cc=0.05, tau_ptmiss_nc=0.05,
        tau_evis_cc=0.05,   tau_evis_nc=0.05,
        # --- weights / knobs ---
        huber_delta=1.0,
        lam_mag=1.0, lam_dir=1.0,     # weights inside vector losses
        zero_attractor_w=0.1,
        jet_aux_w=0.0,                # set >0 to enable aux jet loss (CC-only)
        latent_prior_w=0.0,           # tiny (e.g., 1e-3) if you pass latents
        enforce_nonneg_truth_pz=True, # enforce pz >= 0 for truth vectors (vis and lep)
    ):
        super().__init__()
        # register robust scales
        self.register_buffer("s_vis_xyz", torch.tensor(s_vis_xyz, dtype=torch.float32).view(1,3))
        self.register_buffer("s_lep_xyz", torch.tensor(s_lep_xyz, dtype=torch.float32).view(1,3))
        self.s_vis_mag = float(s_vis_mag)
        self.s_lep_mag = float(s_lep_mag)

        if s_jet_xyz is None: s_jet_xyz = s_vis_xyz
        if s_jet_mag is None: s_jet_mag = s_vis_mag
        self.register_buffer("s_jet_xyz", torch.tensor(s_jet_xyz, dtype=torch.float32).view(1,3))
        self.s_jet_mag = float(s_jet_mag)

        # residual floors
        self.tau_ptmiss_cc = float(tau_ptmiss_cc)
        self.tau_ptmiss_nc = float(tau_ptmiss_nc)
        self.tau_evis_cc   = float(tau_evis_cc)
        self.tau_evis_nc   = float(tau_evis_nc)

        # knobs
        self.huber_delta = float(huber_delta)
        self.lam_mag = float(lam_mag)
        self.lam_dir = float(lam_dir)
        self.zero_attractor_w = float(zero_attractor_w)
        self.jet_aux_w = float(jet_aux_w)
        self.latent_prior_w = float(latent_prior_w)
        self.enforce_nonneg_truth_pz = bool(enforce_nonneg_truth_pz)

    # ---------- primitives ----------
    @staticmethod
    def _huber(x, delta):
        ax = x.abs()
        quad = torch.clamp(ax, max=delta)
        lin = ax - quad
        return 0.5 * quad**2 + delta * lin

    @staticmethod
    def _cosine_dir_loss(p_hat, p_true, eps=1e-8):
        num = (p_hat * p_true).sum(-1)
        den = p_hat.norm(dim=-1) * p_true.norm(dim=-1)
        return 1.0 - num / (den + eps)

    def _component_loss(self, p_hat, p_true, s_xyz):
        z = (p_hat - p_true) / s_xyz
        return self._huber(z, self.huber_delta).sum(-1)

    def _magnitude_loss(self, p_hat, p_true, s_mag):
        z = (p_hat.norm(dim=-1) - p_true.norm(dim=-1)) / s_mag
        return self._huber(z, self.huber_delta)

    def _residual_scalar_loss(self, x_true, x_hat, tau):
        # tau can be scalar or (B,)
        denom = torch.maximum(x_true, tau)
        r = (x_true - x_hat) / denom
        return self._huber(r, self.huber_delta)

    # ---------- forward ----------
    def forward(
        self,
        *,
        p_vis_hat, p_lep_hat,
        p_vis_true, p_lep_true,
        is_cc,
        p_jet_true=None, p_jet_hat=None,
        vis_latents=None, lep_latents=None,
    ):
        device = p_vis_hat.device
        B = p_vis_hat.shape[0]
        eps = 1e-8

        if self.enforce_nonneg_truth_pz:
            # enforce non-negative pz for truth vectors
            p_vis_true = p_vis_true.clone()
            p_lep_true = p_lep_true.clone()
            p_vis_true[..., 2] = p_vis_true[..., 2].clamp_min(0.0)
            p_lep_true[..., 2] = p_lep_true[..., 2].clamp_min(0.0)

        m_cc = is_cc.to(p_vis_hat.dtype).view(-1)                # (B,)
        m_nc = 1.0 - m_cc

        # ----- Visible vector losses -----
        L_vis_comp = self._component_loss(p_vis_hat, p_vis_true, self.s_vis_xyz)
        L_vis_mag  = self._magnitude_loss(p_vis_hat, p_vis_true, self.s_vis_mag)
        L_vis_dir  = self._cosine_dir_loss(p_vis_hat, p_vis_true)
        L_vis = L_vis_comp + self.lam_mag * L_vis_mag + self.lam_dir * L_vis_dir

        # optional latent prior (tiny)
        if vis_latents is not None and self.latent_prior_w > 0.0:
            L_vis = L_vis + self.latent_prior_w * (vis_latents.pow(2).sum(-1))

        # ----- Lepton vector losses (CC-supervised) -----
        L_lep_comp = self._component_loss(p_lep_hat, p_lep_true, self.s_lep_xyz)
        L_lep_mag  = self._magnitude_loss(p_lep_hat, p_lep_true, self.s_lep_mag)
        L_lep_dir  = self._cosine_dir_loss(p_lep_hat, p_lep_true)
        L_lep_cc   = (L_lep_comp + self.lam_mag * L_lep_mag + self.lam_dir * L_lep_dir) * m_cc

        # zero-attractor on NC
        z_nc = (p_lep_hat / self.s_lep_xyz) * m_nc.view(-1,1)
        L_lep_zero = self._huber(z_nc, self.huber_delta).sum(-1) * self.zero_attractor_w

        if lep_latents is not None and self.latent_prior_w > 0.0:
            L_lep_cc = L_lep_cc + self.latent_prior_w * (lep_latents.pow(2).sum(-1)) * m_cc

        # ----- Residual scalars from visible only -----
        pt_true = torch.sqrt(p_vis_true[...,0]**2 + p_vis_true[...,1]**2 + eps)
        pt_hat  = torch.sqrt(p_vis_hat[...,0]**2  + p_vis_hat[...,1]**2  + eps)
        e_true  = p_vis_true.norm(dim=-1)
        e_hat   = p_vis_hat.norm(dim=-1)

        # per-sample floors
        tau_pt = (self.tau_ptmiss_cc * m_cc + self.tau_ptmiss_nc * m_nc).to(device)
        tau_ev = (self.tau_evis_cc   * m_cc + self.tau_evis_nc   * m_nc).to(device)

        L_ptmiss = self._residual_scalar_loss(pt_true, pt_hat, tau_pt)
        L_evis   = self._residual_scalar_loss(e_true,  e_hat,  tau_ev)

        # ----- Optional jet aux loss (CC-only) -----
        if self.jet_aux_w > 0.0 and (p_jet_true is not None):
            if p_jet_hat is None:
                p_jet_hat = p_vis_hat - p_lep_hat
            L_jet_comp = self._component_loss(p_jet_hat, p_jet_true, self.s_jet_xyz)
            L_jet_mag  = self._magnitude_loss(p_jet_hat, p_jet_true, self.s_jet_mag)
            L_jet_dir  = self._cosine_dir_loss(p_jet_hat, p_jet_true)
            L_jet = (L_jet_comp + self.lam_mag * L_jet_mag + self.lam_dir * L_jet_dir) * m_cc
        else:
            L_jet = torch.zeros(B, device=device)

        out = {
            "L_vis_comp": L_vis_comp,
            "L_vis_mag": L_vis_mag,
            "L_vis_dir": L_vis_dir,
            "L_vis": L_vis,
            "L_lep_comp": L_lep_comp,
            "L_lep_mag": L_lep_mag,
            "L_lep_dir": L_lep_dir,
            "L_lep_cc": L_lep_cc,
            "L_lep_zero": L_lep_zero,
            "L_ptmiss": L_ptmiss,
            "L_evis": L_evis,
            "jet_aux_weight": self.jet_aux_w,
            "L_jet_comp": L_jet_comp,
            "L_jet_mag": L_jet_mag,
            "L_jet_dir": L_jet_dir,
            "L_jet": L_jet,
        }
        return out


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


class MomentumLoss(torch.nn.Module):
    """
    Custom loss function for predicting 3D momentum vectors.
    Combines:
    - Cosine similarity loss for direction.
    - Mean Absolute Error (MAE) for magnitude.

    Args:
        lambda_magnitude (float): Weight for magnitude loss. Default is 0.1.
    """
    def __init__(self, lambda_magnitude=0.1):
        super(MomentumLoss, self).__init__()
        self.lambda_magnitude = lambda_magnitude

    def forward(self, pred, target):
        """
        Compute the loss given predicted and target 3D momentum vectors.

        Args:
            pred (Tensor): Predicted momentum vectors (batch_size, 3).
            target (Tensor): Ground truth momentum vectors (batch_size, 3).

        Returns:
            Tensor: The computed loss value.
        """
        # Normalize predictions and targets to unit vectors
        pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)  # Avoid division by zero
        target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)

        # Cosine similarity loss (1 - cos(theta))
        cos_loss = 1 - torch.sum(pred_norm * target_norm, dim=1).mean()

        # Magnitude loss (Mean Absolute Error on norms)
        mag_loss = torch.abs(torch.norm(pred, dim=1) - torch.norm(target, dim=1)).mean()

        # Combined loss
        loss = cos_loss + self.lambda_magnitude * mag_loss
        return loss


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


def dice_score(inputs: torch.Tensor,
               targets: torch.Tensor,
               smooth_num: float = 0,
               smooth_den: float = 1e-12):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs.
        smooth: A smoothing constant to smooth gradients. 
    Returns:
        dice_score: Dice loss value.
    """
    reduce_axes: list[int] = torch.arange(1, len(inputs.shape)).tolist()
    intersection = torch.sum(targets * inputs, dim=reduce_axes)
    union = torch.sum(targets, dim=reduce_axes) + torch.sum(inputs, dim=reduce_axes)

    dice_score = (2. * intersection + smooth_num) / (union + smooth_den)
    
    return torch.mean(dice_score)


def dice_loss(inputs: torch.Tensor or list[torch.Tensor],
              targets: torch.Tensor or list[torch.Tensor],
              sigmoid: bool = True,
              smooth_num: float = 0,
              smooth_den: float = 1e-12,
              reduction: str = "none",
) -> torch.Tensor:
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs.
        sigmoid: A boolean indicating binary or multi-class.
        smooth: A smoothing constant to avoid division by zero. 
    Returns:
        dice_loss: Dice loss value.

    Note:
        Assumes class in dimension 1.
    """
    # If inputs and targets are not lists, convert them to lists
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(targets, list):
        targets = [targets]

    assert len(inputs) == len(targets), "batch size not the same for inputs and targets"
    batch_size = len(inputs)

    scores = torch.zeros(batch_size, device=inputs[0].device)
    if sigmoid:
        # binary
        for batch_idx, (ipt, tgt) in enumerate(zip(inputs, targets)):
            if ipt.size(-1) == 1:
                ipt = ipt.squeeze(-1)
            if tgt.size(-1) == 1:
                tgt = tgt.squeeze(-1) 
            ipt = torch.sigmoid(ipt)
            scores[batch_idx] = dice_score(ipt, tgt, smooth_num, smooth_den)
    else:
        # multi-class
        for batch_idx, (ipt, tgt) in enumerate(zip(inputs, targets)):
            ipt = torch.softmax(ipt, 1)
            nb_labels = ipt.size(1)
            score = 0.
            # Check target shape: if the target has the same number of dimensions as ipt,
            # assume it is one-hot encoded, otherwise assume it's class IDs.
            if tgt.ndim == ipt.ndim:
                for i in range(nb_labels):
                    ipt_i = ipt[:, i]
                    tgt_i = tgt[:, i].float()
                    score += dice_score(ipt_i, tgt_i, smooth_num, smooth_den)
            else:
                for i in range(nb_labels):
                    ipt_i = ipt[:, i]
                    tgt_i = (tgt == i).float()
                    score += dice_score(ipt_i, tgt_i, smooth_num, smooth_den)
            score /= nb_labels
            scores[batch_idx] = score
        
    loss = 1 - scores

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


def contrastive_loss_class_labels(
    x_i, x_j, labels, temperature=0.1, gather_distributed=False, same_label_weight=0.5
):
    """
    Contrastive loss function from bmdillon/JetCLR

    Uses class labels to define positive pairs.

    Args:
        x_i (torch.Tensor): Input tensor of shape (batch_size, n_features)
        x_j (torch.Tensor): Input tensor of shape after augmentations (batch_size, n_features)
        temperature (float, optional): Temperature parameter. Defaults to 0.1.
    Returns:
        torch.Tensor: Contrastive loss
    """
    if gather_distributed and get_world_size() == 1:
        raise ValueError("gather_distributed=True but number of processes is 1")

    xdevice = x_i.get_device()

    if gather_distributed:
        x_i = torch.cat(GatherLayer.apply(x_i), dim=0)
        x_j = torch.cat(GatherLayer.apply(x_j), dim=0)

    batch_size = x_i.shape[0]
    z_i = F.normalize(x_i, dim=1 )
    z_j = F.normalize(x_j, dim=1 )
    z = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )

    # 0.5 for same class pairs, 1.0 for same image pairs
    labels = torch.cat([labels, labels], dim=0)
    positives_mask = (labels[:, None] == labels[None, :]).float() * same_label_weight
    ids = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
    positives_mask += (ids[:, None] == ids[None, :]).float() * (1.0 - same_label_weight)
    positives_mask *= (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float()
    positives_mask = positives_mask.to(xdevice)
    nominator = positives_mask * torch.exp(similarity_matrix / temperature)

    negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )

    loss_partial = -torch.log( torch.sum(nominator, dim=1) / torch.sum( denominator, dim=1 ) )
    loss = torch.sum( loss_partial )/( 2*batch_size )

    return loss


def label_based_contrastive_loss_random_chunk(projs, labels, temperature=0.1, chunk_size=512, eps=1e-6):
    """
    Computes the contrastive loss for two random chunks to reduce memory usage,
    ensuring that each voxel is compared to all other voxels, and using binary labels.
    
    Args:
        projs: Tensor of shape (N, proj_dim), where N is the number of voxels and proj_dim is the dimensionality of the projections.
        labels: Tensor of shape (N,) containing the binary labels (0 or 1) for each voxel.
        temperature: Scaling factor for logits.
        chunk_size: The size of the chunk of voxels to process at a time.
        eps: Small constant to prevent NaNs.

    Returns:
        loss: The accumulated contrastive loss for the entire input tensor.
    """
    N, device = projs.shape[0], projs.device

    if N < chunk_size * 2:
        chunk_size = N // 2

    indices = torch.randperm(N)
    chunk1_indices = indices[:chunk_size]
    chunk2_indices = indices[chunk_size:2*chunk_size]

    projs1 = projs[chunk1_indices]
    projs2 = projs[chunk2_indices]
    labels1 = labels[chunk1_indices]
    labels2 = labels[chunk2_indices]

    projs1 = F.normalize(projs1)
    projs2 = F.normalize(projs2)

    similarity_matrix = F.cosine_similarity(projs1.unsqueeze(1), projs2.unsqueeze(0), dim=2 )

    positives_mask = (labels1[:, None] == labels2[None, :]).float()
    nominator = positives_mask * torch.exp(similarity_matrix / temperature)
    denominator = torch.exp(similarity_matrix / temperature)    

    loss_partial = -torch.log((torch.sum(nominator, dim=1) + eps) / (torch.sum(denominator, dim=1) + eps))
    loss = torch.sum(loss_partial) / chunk_size

    return loss


def supervised_pixel_contrastive_loss(features_ori_list: torch.Tensor,
                                      features_aug_list: torch.Tensor,
                                      labels_ori_list: torch.Tensor,
                                      labels_aug_list: torch.Tensor,
                                      label_weights: list = None,
                                      temperature: float = 0.07,
                                      chunk_size: int = 512,
                                      ignore_labels: list = [-1],
                                      within_image_loss: bool = False):
    """
    Computes pixel-level supervised contrastive loss for batches with variable-sized inputs.
    
    Inspiration: https://github.com/google-research/google-research/blob/master/supervised_pixel_contrastive_loss/contrastive_loss.py
    Paper: https://arxiv.org/abs/2012.06985
    
    Args:
        features_ori_list: List of tensors for original features
        features_aug_list: List of tensors for augmented features
        labels_ori_list: List of tensors for original labels
        labels_aug_list: List of tensors for augmented labels
        label_weights: Weight for each label in the loss computation
        temperature: Temperature to use in contrastive loss
        chunk_size: Maximum number of voxels per event.
        ignore_labels: A list of labels to ignore.
        within_image_loss: whether to use within_image or cross_image loss.

    Returns:
        Scalar contrastive loss for the batch

    Note:
        Expect feats and labels be (batch_size, -1, num_channels)
    """

    batch_size = len(labels_ori_list)

    features_aug_list = features_aug_list[::-1]
    labels_aug_list = labels_aug_list[::-1]

    total_loss = 0.0
    for i in range(batch_size):
        features_ori = features_ori_list[i]
        features_aug = features_aug_list[i]
        labels_ori = labels_ori_list[i].squeeze()
        labels_aug = labels_aug_list[i].squeeze()

        N_ori, N_aug = features_ori.size(0), features_aug.size(0)
        if N_ori > chunk_size:
            shuffled_idx = torch.randperm(N_ori)
            chunk_idx = shuffled_idx[:chunk_size]
            features_ori = features_ori[chunk_idx]
            labels_ori = labels_ori[chunk_idx]
        if N_aug > chunk_size:
            shuffled_idx = torch.randperm(N_aug)
            chunk_idx = shuffled_idx[:chunk_size]
            features_aug = features_aug[chunk_idx]
            labels_aug = labels_aug[chunk_idx]

        features_ori = F.normalize(features_ori, p=2, dim=-1)
        features_aug = F.normalize(features_aug, p=2, dim=-1)

        if within_image_loss:
            curr_loss = within_image_supervised_pixel_contrastive_loss(features_ori, labels_ori, label_weights, temperature)
        else:
            curr_loss = cross_image_supervised_pixel_contrastive_loss(features_ori, features_aug, labels_ori, labels_aug, label_weights, temperature)

        total_loss += curr_loss

    return total_loss / batch_size


def within_image_supervised_pixel_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, label_weights, temperature):
    """Computes within-image supervised pixel contrastive loss for two individual images."""
    logits = torch.matmul(features, features.T) / temperature
    positive_mask, negative_mask = generate_positive_and_negative_masks(labels, labels, label_weights)
    return compute_contrastive_loss(logits, positive_mask, negative_mask)


def cross_image_supervised_pixel_contrastive_loss(features1, features2, labels1, labels2, label_weights, temperature):
    """Computes cross-image supervised pixel contrastive loss for two individual images."""
    num_pixels1 = features1.size(0)  # Number of pixels in image 1
    num_pixels2 = features2.size(0)  # Number of pixels in image 2

    features2 = torch.cat([features1, features2], dim=0)  # Concatenate pixel features from both images
    labels2 = torch.cat([labels1, labels2], dim=0)        # Concatenate labels

    same_image_mask = generate_same_image_mask([num_pixels1],
                                               [num_pixels1, num_pixels2],
                                               device=features1.device)

    # Compute logits across all pixel pairs from the two images
    logits = torch.matmul(features1, features2.T) / temperature
    #logits = F.cosine_similarity( features.unsqueeze(1), features.unsqueeze(0), dim=2 ) / temperature
    
    positive_mask, negative_mask = generate_positive_and_negative_masks(labels1, labels2, label_weights)
    negative_mask *= same_image_mask  # Only consider negatives within the same image

    return compute_contrastive_loss(logits, positive_mask, negative_mask)


def compute_contrastive_loss(logits, positive_mask, negative_mask, eps=1e-12, original=True):
    """Contrastive loss function."""

    if original:
        exp_logits = torch.exp(logits)
    
        normalized_exp_logits = exp_logits / (exp_logits + torch.sum(exp_logits * negative_mask, dim=1, keepdim=True))
        neg_log_likelihood = -torch.log(normalized_exp_logits)

        positive_mask_sum = torch.sum(positive_mask, dim=1, keepdim=True)
        normalized_weight = positive_mask / torch.clamp(positive_mask_sum, min=1e-6)
        neg_log_likelihood = torch.sum(neg_log_likelihood * normalized_weight, dim=1)

        # Handle the case where there are no positive pairs
        valid_index = 1 - (positive_mask_sum.squeeze() == 0).float()
        normalized_weight = valid_index / torch.clamp(valid_index.sum(), min=1e-6)
    
        loss = torch.mean(neg_log_likelihood * normalized_weight)    
    else:
        validity_mask = 1 - torch.eye(positive_mask.size(0), positive_mask.size(1),
            dtype=bool, device=positive_mask.device).float()
        validity_mask *= (positive_mask + negative_mask)
        positive_mask = positive_mask * validity_mask

        exp_logits = torch.exp(logits)

        nominator = positive_mask * exp_logits
        denominator = validity_mask * exp_logits

        loss_partial = -torch.log((torch.sum(nominator, dim=1) + eps) / (torch.sum(denominator, dim=1)) + eps)

        loss = torch.mean(loss_partial)
    return loss


def generate_same_image_mask(num_pixels1, num_pixels2, device):
    """Generates a mask indicating if two pixels belong to the same image or not."""
    image_ids1, image_ids2 = [], []
    for img_id, pixel_count in enumerate(num_pixels1):
        image_ids1 += [img_id] * pixel_count
    for img_id, pixel_count in enumerate(num_pixels2):
        image_ids2 += [img_id] * pixel_count

    image_ids1 = torch.tensor(image_ids1, device=device).view(-1, 1)
    image_ids2 = torch.tensor(image_ids2, device=device).view(-1, 1)
    same_image_mask = (image_ids1 == image_ids2.T).float()

    return same_image_mask


def generate_positive_and_negative_masks(labels1: torch.Tensor, labels2: torch.Tensor, label_weights):
    """Generates positive and negative masks used by contrastive loss."""
    positive_mask = (labels1[:, None] == labels2[None, :]).float()
    if label_weights is not None:
        weights_tensor = torch.tensor(label_weights, device=labels1.device)
        label_weights = weights_tensor[labels1.long()]
        positive_mask *= label_weights[:, None]
    negative_mask = 1 - positive_mask

    return positive_mask, negative_mask


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
    

def prototype_contrastive_loss_vectorized(
    z: torch.Tensor, class_id: torch.Tensor, event_id: torch.Tensor,
    num_neg: int = 64, temperature: float = 0.07, normalize: bool = True,
    semi_hard: bool = False, semi_hard_pool_mult: int = 4, semi_hard_margin: float = 0.05,
) -> torch.Tensor:
    if z.numel() == 0:
        return z.new_zeros(())

    valid = class_id >= 0
    if not torch.any(valid):
        return z.new_zeros(())

    z   = z[valid]
    cid = class_id[valid].long()
    eid = event_id[valid].long()
    if normalize:
        z = F.normalize(z, dim=-1)

    # Groups & per-group sums / counts
    inv_group, Pidx = _build_grouping(eid, cid)     # inv_group: [M]
    G = int(Pidx["rank"].numel())
    D = z.size(1)

    P_sum = z.new_zeros((G, D))
    P_sum.index_add_(0, inv_group, z)               # sum per (event,class)
    cnt_vec = torch.bincount(inv_group, minlength=G).clamp_min_(1)   # [G] (int)

    # Per-hit metadata
    g        = inv_group
    local_pos= Pidx["rank"][g]
    eidx     = Pidx["eidx"][g]
    Ue       = Pidx["counts"][eidx]                 # classes per event (per hit)
    off      = Pidx["offset"][eidx]
    cls_cnt  = cnt_vec[g]                           # class size (per hit, int)

    # Keep only hits with >=2 classes in event AND class size >=2
    keep = (Ue >= 2) & (cls_cnt >= 2)
    if not torch.any(keep):
        return z.new_zeros(())

    z        = z[keep]
    g        = g[keep]
    local_pos= local_pos[keep]
    Ue       = Ue[keep]
    off      = off[keep]
    cls_cnt  = cls_cnt[keep]

    # Positive = leave-one-out prototype: (sum - z) / (cnt-1)
    P_pos = (P_sum[g] - z) / (cls_cnt - 1).unsqueeze(1).to(z.dtype)
    if normalize:
        P_pos = F.normalize(P_pos, dim=-1)

    # Negatives = means of other classes in same event
    P_mean = P_sum / cnt_vec.unsqueeze(1).to(P_sum.dtype)   # [G,D]
    if normalize:
        P_mean = F.normalize(P_mean, dim=-1)

    # ---- negative sampling (candidate pool) ----
    M = z.size(0)
    # pool size: either num_neg or larger if semi-hard
    K_final = num_neg
    K_pool  = (num_neg * semi_hard_pool_mult) if semi_hard else num_neg

    ftype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
    K_eff = torch.clamp(Ue - 1, max=K_pool)               # per-row #available negatives
    rnd = torch.rand(M, K_pool, device=z.device, dtype=ftype)
    neg_raw = torch.floor(rnd * K_eff.unsqueeze(1).to(ftype)).to(torch.long)  # [M,K_pool] in [0..(Ue-2)]
    # skip the positive class by "gap" after local_pos
    neg_local = neg_raw + (neg_raw >= local_pos.unsqueeze(1)).to(torch.long)
    base = off.unsqueeze(1) + neg_local
    neg_group = Pidx["ord_g"][base]                      # [M,K_pool]
    P_negs = P_mean[neg_group]                           # [M,K_pool,D]

    # mask unused columns when Ue-1 < K_pool
    col = torch.arange(K_pool, device=z.device).unsqueeze(0)
    neg_valid = col < K_eff.unsqueeze(1)                 # [M,K_pool]

    # Similarities
    pos_sim = (z * P_pos).sum(-1, keepdim=True)          # [M,1]
    neg_sim_full = torch.einsum('md,mkd->mk', z, P_negs) # [M,K_pool]
    neg_sim_full = neg_sim_full.masked_fill(~neg_valid, float('-inf'))

    # ---- semi-hard selection (top-K just below positive, within a margin) ----
    if semi_hard:
        # valid semi-hard = below positive but close: pos_sim - margin < neg_sim < pos_sim
        sh_mask = (neg_sim_full < pos_sim) & (neg_sim_full > (pos_sim - semi_hard_margin))
        # rank by similarity (hardest among semi-hard)
        sh_scores = neg_sim_full.masked_fill(~sh_mask, float('-inf'))
        sh_idx = torch.topk(sh_scores, k=K_final, dim=1).indices   # [M,K_final]
        # Check how many valid we actually got; fallback if too few
        got_mask = torch.gather(sh_mask, 1, sh_idx)                # [M,K_final]
        enough = got_mask.sum(dim=1) >= K_final

        # Fallback: purely hard (top by sim) among all valid if not enough semi-hard
        fb_scores = neg_sim_full
        fb_idx = torch.topk(fb_scores, k=K_final, dim=1).indices   # [M,K_final]

        # choose row-wise
        idx_sel = torch.where(enough.unsqueeze(1), sh_idx, fb_idx) # [M,K_final]
        neg_sim = torch.gather(neg_sim_full, 1, idx_sel)           # [M,K_final]
    else:
        # vanilla: just take top-K hard negatives among valid
        idx_sel = torch.topk(neg_sim_full, k=K_final, dim=1).indices
        neg_sim = torch.gather(neg_sim_full, 1, idx_sel)

    # Logits & loss
    logits  = torch.cat([pos_sim, neg_sim], dim=1) / temperature  # [M,1+K_final]
    targets = torch.zeros(M, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, targets, reduction="mean")


def ghost_pushaway_loss(
    z: torch.Tensor,                 # [N, D] masked embeddings (ghosts included)
    class_id: torch.Tensor,          # [N] int64; real >=0, non-real (ghost/ignored) == -1
    event_id: torch.Tensor,          # [N] int64
    ghost_mask: torch.Tensor,        # [N] bool — TRUE = treat as ghost to push away
    num_neg: int = 32,
    temperature: float = 0.07,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Push embeddings with ghost_mask==True away from REAL prototypes (class_id>=0) in the same event.
    Other -1 hits with ghost_mask==False are ignored here.
    """
    # Build real prototypes
    real = class_id >= 0
    if not torch.any(real):
        return z.new_zeros(())
    z_r = z[real]
    eid_r = event_id[real].long()
    cid_r = class_id[real].long()
    if normalize: z_r = F.normalize(z_r, dim=-1)

    inv_group, Pidx = _build_grouping(eid_r, cid_r)
    G = int(Pidx["rank"].numel())
    D = z.size(1)

    P = z_r.new_zeros((G, D))
    P.index_add_(0, inv_group, z_r)
    cnt = torch.bincount(inv_group, minlength=G).clamp_min_(1).unsqueeze(1).to(P.dtype)
    P = P / cnt
    if normalize: P = F.normalize(P, dim=-1)

    # Select ghosts to push
    ghosts = ghost_mask.bool()
    if not torch.any(ghosts):
        return z.new_zeros(())

    z_g = z[ghosts]
    e_g = event_id[ghosts].long()
    if normalize: z_g = F.normalize(z_g, dim=-1)

    # keep only ghosts in events that have real prototypes
    events_sorted = Pidx["events_sorted"]                 # [E_real], ascending
    idx = torch.searchsorted(events_sorted, e_g)
    valid_e = (idx < events_sorted.numel()) & (events_sorted[idx] == e_g)
    if not torch.any(valid_e):
        return z.new_zeros(())

    z_g = z_g[valid_e]
    eidx = idx[valid_e]
    Ue = Pidx["counts"][eidx]                             # #prototypes in that event
    off = Pidx["offset"][eidx]

    # Sample up to num_neg prototypes per ghost within its event (with replacement)
    K = num_neg
    M = z_g.size(0)
    if M == 0:
        return z.new_zeros(())
    ftype = torch.float32 if z.dtype in (torch.float16, torch.bfloat16) else z.dtype
    # cap columns to Ue (mask extras)
    K_eff = torch.clamp(Ue, max=K)
    rnd = torch.rand(M, K, device=z.device, dtype=ftype)
    neg_local = torch.floor(rnd * K_eff.unsqueeze(1).to(ftype)).to(torch.long)  # [M,K] in [0..Ue-1]
    base = off.unsqueeze(1) + neg_local
    neg_group = Pidx["ord_g"][base]                     # [M,K]
    P_negs = P[neg_group]                               # [M,K,D]

    # mask unused columns where col >= K_eff
    col = torch.arange(K, device=z.device).unsqueeze(0)
    neg_valid = col < K_eff.unsqueeze(1)

    # push-away: minimize similarity to prototypes (logsumexp over negatives)
    neg_sim = torch.einsum('md,mkd->mk', z_g, P_negs) / temperature
    neg_sim = neg_sim.masked_fill(~neg_valid, float('-inf'))
    return torch.logsumexp(neg_sim, dim=1).mean()
