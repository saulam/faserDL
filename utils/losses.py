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


def label_based_nt_xent_loss(projs, labels, temperature=0.1):
    b, device = projs.shape[0], projs.device
    logits = projs @ projs.t()

    mask = torch.eye(b, device=device).bool()
    logits = logits[~mask].reshape(b, b - 1)
    logits /= temperature

    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    label_matrix = label_matrix.float()
    label_matrix = label_matrix[~mask].reshape(b, b - 1)

    loss = F.binary_cross_entropy_with_logits(logits, label_matrix, reduction='sum')
    loss /= b

    return loss


def label_based_nt_xent_loss_in_chunks(projs, labels, temperature=0.1, chunk_size=64):
    """
    Computes the contrastive loss in smaller chunks to reduce memory usage,
    ensuring that each voxel is compared to all other voxels, and using binary labels.
    
    Args:
        projs: Tensor of shape (N, proj_dim), where N is the number of voxels and proj_dim is the dimensionality of the projections.
        labels: Tensor of shape (N,) containing the binary labels (0 or 1) for each voxel.
        temperature: Scaling factor for logits.
        chunk_size: The size of the chunk of voxels to process at a time.
        
    Returns:
        loss: The accumulated contrastive loss for the entire input tensor.
    """
    N, device = projs.shape[0], projs.device
    total_loss = 0.0

    # Iterate over projections in chunks (mini-batches)
    for i in range(0, N, chunk_size):
        # Determine the end of the chunk
        end = min(i + chunk_size, N)
        projs_chunk = projs[i:end]  # (chunk_size, proj_dim)
        labels_chunk = labels[i:end]  # (chunk_size,)
        logits = projs @ projs_chunk.t()  # (N, chunk_size)
        logits /= temperature

        # Create label matrix based on the entire set of labels
        label_matrix = labels.unsqueeze(1) == labels_chunk.unsqueeze(0)  # (N, chunk_size)
        label_matrix = label_matrix.float()

        # Compute binary cross-entropy loss for this chunk
        loss = F.binary_cross_entropy_with_logits(logits, label_matrix, reduction='sum')
        total_loss += loss

        # Manually free memory for intermediate tensors
        del projs_chunk, labels_chunk, logits, label_matrix
        torch.cuda.empty_cache()  # Optionally clear cache to free memory

    total_loss /= N

    return total_loss


def label_based_contrastive_loss_in_chunks(projs, labels, temperature=0.1, chunk_size=512):
    """
    Computes the contrastive loss in smaller chunks to reduce memory usage,
    ensuring that each voxel is compared to all other voxels, and using binary labels.
    
    Args:
        projs: Tensor of shape (N, proj_dim), where N is the number of voxels and proj_dim is the dimensionality of the projections.
        labels: Tensor of shape (N,) containing the binary labels (0 or 1) for each voxel.
        temperature: Scaling factor for logits.
        chunk_size: The size of the chunk of voxels to process at a time.
        
    Returns:
        loss: The accumulated contrastive loss for the entire input tensor.
    """
    N, device = projs.shape[0], projs.device
    total_loss = 0.0

    projs = F.normalize(projs)

    # Iterate over projections in chunks (mini-batches)
    for i in range(0, N, chunk_size):
        # Determine the end of the chunk
        end = min(i + chunk_size, N)
        projs_chunk = projs[i:end]  # (chunk_size, proj_dim)
        labels_chunk = labels[i:end]  # (chunk_size,)

        similarity_matrix = F.cosine_similarity(projs.unsqueeze(1), projs_chunk.unsqueeze(0), dim=2 )

        positives_mask = labels[:, None] == labels_chunk[None, :]
        nominator = positives_mask.float() * torch.exp(similarity_matrix / temperature)

        negatives_mask = ~positives_mask
        denominator = negatives_mask.float() * torch.exp(similarity_matrix / temperature)
        
        loss_partial = -torch.log(torch.sum(nominator, dim=1) / torch.sum( denominator, dim=1 ))
        total_loss = torch.sum(loss_partial) 

        # Manually free memory for intermediate tensors
        del projs_chunk, labels_chunk, positives_mask, negatives_mask
        torch.cuda.empty_cache()  # Optionally clear cache to free memory

    total_loss /= N

    return total_loss


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

