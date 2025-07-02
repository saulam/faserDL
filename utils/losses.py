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
    """

    def __init__(self, eps=1e-8, reduction='mean', preprocessing=None):
        super(MAPE, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.preprocessing = preprocessing

    def forward(self, pred, target):
        """
        Computes the MAPE loss, but only for target values greater than 0.

        Args:
            pred (torch.Tensor): Predicted values (batch_size, *)
            target (torch.Tensor): Ground truth values (batch_size, *)

        Returns:
            torch.Tensor: Computed MAPE loss, considering only target values > 0.
        """
        if self.preprocessing == "sqrt":
            pred = pred ** 2
            target = target ** 2
        elif self.preprocessing == "log":
            pred = torch.expm1(pred)
            target = torch.expm1(target)
            
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

