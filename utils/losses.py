import torch
from torch.nn import functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
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
                positive vs negative examples. Default = -1 (no weighting).
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

    if alpha >= 0:
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


def dice_loss(inputs: torch.Tensor,
              targets: torch.Tensor,
              eps: float = 1e-6,
) -> torch.Tensor:
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        eps: A smoothing constant to avoid division by zero. 
    Returns:
        dice_loss: Dice loss value.
    """
    inputs = inputs.float()
    targets = targets.float()
    inputs = torch.sigmoid(inputs)

    reduce_axis: list[int] = torch.arange(1, len(inputs.shape)).tolist()
    intersection = torch.sum(targets * inputs, dim=reduce_axis)
    union = torch.sum(targets, dim=reduce_axis) + torch.sum(inputs, dim=reduce_axis)
     
    dice_score = (2. * intersection + eps) / (union + eps)
    dice_loss = 1 - dice_score

    return torch.mean(dice_loss)


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
                                      temperature: float = 0.07,
                                      chunk_size: int = 512,
                                      within_image_loss: bool = False):
    """
    Computes pixel-level supervised contrastive loss for batches with variable-sized inputs.
    
    Inpiration: https://github.com/google-research/google-research/blob/master/supervised_pixel_contrastive_loss/contrastive_loss.py
    Paper: https://arxiv.org/abs/2012.06985
    
    Args:
        features_ori_list: List of tensors for original features
        features_aug_list: List of tensors for augmented features
        labels_ori_list: List of tensors for original labels
        labels_aug_list: List of tensors for augmented labels
        temperature: Temperature to use in contrastive loss
        chunk_size: Maximum number of voxels per event.
        within_image_loss: whether to use within_image or cross_image loss.

    Returns:
        Scalar contrastive loss for the batch
    """

    # expect feats and labels be (batch_size, -1, num_channels)
    batch_size = len(labels_ori_list)
    shuffled_idx = torch.randperm(batch_size)

    features_aug_list = [features_aug_list[i] for i in shuffled_idx]
    labels_aug_list = [labels_aug_list[i] for i in shuffled_idx]

    total_loss = 0.0
    for i in range(batch_size):
        features_ori = features_ori_list[i]
        features_aug = features_aug_list[i]
        labels_ori = labels_ori_list[i]
        labels_aug = labels_aug_list[i]

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
            loss_ori = within_image_supervised_pixel_contrastive_loss(features_ori, labels_ori, temperature)
            loss_aug = within_image_supervised_pixel_contrastive_loss(features_aug, labels_aug, temperature)
            curr_loss = loss_ori + loss_aug
        else:
            curr_loss = cross_image_supervised_pixel_contrastive_loss(features_ori, features_aug, labels_ori, labels_aug, temperature)                

        total_loss += curr_loss

    return total_loss / batch_size


def within_image_supervised_pixel_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature):
    """Computes within-image supervised pixel contrastive loss for two individual images."""
    logits = torch.matmul(features, features.T) / temperature
    positive_mask, negative_mask = generate_positive_and_negative_masks(labels)
    return compute_contrastive_loss(logits, positive_mask, negative_mask)


def cross_image_supervised_pixel_contrastive_loss(features1, features2, labels1, labels2, temperature):
    """Computes cross-image supervised pixel contrastive loss for two individual images."""
    num_pixels1 = features1.size(0)  # Number of pixels in image 1
    num_pixels2 = features2.size(0)  # Number of pixels in image 2
    
    features = torch.cat([features1, features2], dim=0)  # Concatenate pixel features from both images
    labels = torch.cat([labels1, labels2], dim=0)        # Concatenate labels

    same_image_mask = generate_same_image_mask([num_pixels1, num_pixels2], device=features.device)

    # Compute logits across all pixel pairs from the two images
    logits = torch.matmul(features, features.T) / temperature
    positive_mask, negative_mask = generate_positive_and_negative_masks(labels)
    negative_mask *= same_image_mask  # Only consider negatives within the same image

    return compute_contrastive_loss(logits, positive_mask, negative_mask)


def compute_contrastive_loss(logits, positive_mask, negative_mask):
    """Contrastive loss function."""
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

    return loss


def generate_same_image_mask(num_pixels, device):
    """Generates a mask indicating if two pixels belong to the same image or not."""
    image_ids = []
    for img_id, pixel_count in enumerate(num_pixels):
        image_ids += [img_id] * pixel_count

    image_ids = torch.tensor(image_ids, device=device).view(-1, 1)
    same_image_mask = (image_ids == image_ids.T).float()
    
    return same_image_mask


def generate_positive_and_negative_masks(labels: torch.Tensor):
    """Generates positive and negative masks used by contrastive loss."""
    positive_mask = (labels[:, None] == labels[None, :]).float()
    negative_mask = 1 - positive_mask
    
    return positive_mask, negative_mask

