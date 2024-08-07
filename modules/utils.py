'''
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
'''

import math
import tqdm
import torch
import MinkowskiEngine as ME
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def arrange_sparse_minkowski(data, device):
    """
    Arrange sparse Minkowski tensor from given data.
    
    Args:
        data (dict): Dictionary containing features and coordinates.
        device (torch.device): Device to place the tensor on.

    Returns:
        ME.SparseTensor: Sparse tensor for Minkowski Engine.
    """
    return ME.SparseTensor(
        features=data['f'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )
    

def arrange_truth(data):
    """
    Retrieve ground truth labels from the data.

    Args:
        data (dict): Dictionary containing labels.

    Returns:
        torch.Tensor: Ground truth labels.
    """
    return data['y']


def train(model, loader, bce_loss_fn, dice_loss_fn, optimizer, scheduler, disable=False, epoch=-1, writer=None, device=None):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): Model to train.
        loader (DataLoader): DataLoader for training data.
        bce_loss_fn (callable): Binary cross-entropy loss function.
        dice_loss_fn (callable): Dice loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (_LRScheduler): Learning rate scheduler.
        disable (bool, optional): Disable tqdm progress bar. Default is False.
        epoch (int, optional): Current epoch number. Default is -1.
        writer (SummaryWriter, optional): TensorBoard writer. Default is None.
        device (torch.device, optional): Device to use for training. Default is None.

    Returns:
        float: Average loss over the epoch.
    """
    model.train()
    batch_size = loader.batch_size
    n_batches = int(math.ceil(len(loader.dataset) / batch_size))
    t = tqdm.tqdm(enumerate(loader), total=n_batches, disable=disable)
    sum_loss = 0.0

    optimizer.zero_grad()

    for i, data in t:
        if i % 5 == 0:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

        # Retrieve input, labels, and run model
        batch_input = arrange_sparse_minkowski(data, device)
        batch_target = arrange_truth(data).to(device)
        batch_output = model(batch_input)[0]

        # Calculate binary cross-entropy loss
        bce_loss = bce_loss_fn(batch_output.F, batch_target) if bce_loss_fn else 0.0

        # Calculate dice-score loss
        dice_loss = dice_loss_fn(batch_output.F, batch_target) if dice_loss_fn else 0.0

        # Total loss
        batch_loss = bce_loss + dice_loss
        batch_loss.backward()
        t.set_description(f"train loss = {batch_loss:.5f}, bce_loss = {bce_loss:.5f}, dice_loss = {dice_loss:.5f}")
        sum_loss += batch_loss.item()

        if writer and (i % 100 == 0):
            writer.add_scalar('epoch', epoch, epoch * n_batches + i)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * n_batches + i)
            writer.add_scalar('loss/train', batch_loss.item(), epoch * n_batches + i)
            if bce_loss_fn:
                writer.add_scalar('loss/bce_loss', bce_loss.item(), epoch * n_batches + i)
            if dice_loss_fn:
                writer.add_scalar('loss/dice_loss', dice_loss.item(), epoch * n_batches + i)

        # Optimise
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return sum_loss / n_batches


def test(model, loader, bce_loss_fn, dice_loss_fn, disable=False, epoch=-1, writer=None, device=None):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): Model to evaluate.
        loader (DataLoader): DataLoader for test data.
        bce_loss_fn (callable): Binary cross-entropy loss function.
        dice_loss_fn (callable): Dice loss function.
        disable (bool, optional): Disable tqdm progress bar. Default is False.
        epoch (int, optional): Current epoch number. Default is -1.
        writer (SummaryWriter, optional): TensorBoard writer. Default is None.
        device (torch.device, optional): Device to use for evaluation. Default is None.

    Returns:
        float: Average loss over the test dataset.
    """
    model.eval()
    batch_size = loader.batch_size
    n_batches = int(math.ceil(len(loader.dataset) / batch_size))
    t = tqdm.tqdm(enumerate(loader), total=n_batches, disable=disable)
    sum_loss, sum_bce, sum_dice = 0.0, 0.0, 0.0

    with torch.no_grad():
        for i, data in t:
            if i % 5 == 0:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

            # Retrieve input, labels, and run model
            batch_input = arrange_sparse_minkowski(data, device)
            batch_target = arrange_truth(data).to(device)
            batch_output = model(batch_input)[0]

            # Calculate binary cross-entropy loss
            bce_loss = bce_loss_fn(batch_output.F, batch_target) if bce_loss_fn else 0.0
            sum_bce += bce_loss.item()

            # Calculate dice-score loss
            dice_loss = dice_loss_fn(batch_output.F, batch_target) if dice_loss_fn else 0.0
            sum_dice += dice_loss.item()

            # Total loss
            batch_loss = bce_loss + dice_loss
            t.set_description(f"val loss = {batch_loss:.5f}, bce_loss = {bce_loss:.5f}, dice_loss = {dice_loss:.5f}")
            sum_loss += batch_loss.item()

    # Log losses to TensorBoard
    if writer:
        writer.add_scalar('loss/val', sum_loss / n_batches, epoch * n_batches)
        if bce_loss_fn:
            writer.add_scalar('loss/val_bce', sum_bce / n_batches, epoch * n_batches)
        if dice_loss_fn:
            writer.add_scalar('loss/val_dice', sum_dice / n_batches, epoch * n_batches)

    return sum_loss / n_batches


def testNevents(data):
    """
    Test the model on a single batch of data.

    Args:
        data (dict): Dictionary containing input features and labels.

    Returns:
        tuple: Ground truth labels and model predictions.
    """
    model.eval()

    # Retrieve input, labels, and run model
    batch_input = arrange_sparse_minkowski(data)
    batch_target = arrange_truth(data)
    batch_output = model(batch_input)[0]

    # Free up GPU memory
    del batch_input
    torch.cuda.empty_cache()

    return batch_target, batch_output



class CustomLambdaLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps):
        """
        Initialize a custom LambdaLR learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which the learning rate will be scheduled.
            warmup_steps (int): Number of iterations for warm-up.
        """
        self.warmup_steps = warmup_steps
        super(CustomLambdaLR, self).__init__(optimizer, lr_lambda=self.lr_lambda)

    def lr_lambda(self, step):
        """
        Calculate the learning rate lambda based on the current step and warm-up steps.

        Args:
            step (int): The current step in training.

        Returns:
            float: The learning rate lambda.
        """
        return float(step) / max(1, self.warmup_steps)
        

class CombinedScheduler(_LRScheduler):
    def __init__(self, optimizer, scheduler1, scheduler2, lr_decay=1.0, warmup_steps=100):
        """
        Initialize the CombinedScheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which the learning rate will be scheduled.
            scheduler1 (_LRScheduler): The first scheduler for the warm-up phase.
            scheduler2 (_LRScheduler): The second scheduler for the main phase.
            lr_decay (float): The factor by which the learning rate is decayed after each restart (default: 1.0).
            warmup_steps (int): The number of steps for the warm-up phase (default: 100).
        """
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.lr_decay = lr_decay

    def step(self):
        """
        Update the learning rate based on the current step and the selected scheduler.
        This method alternates between the two provided schedulers based on the current step number.
        After the warm-up phase, it switches to the second scheduler and optionally decays the learning
        rate after each restart.
        """
        if self.step_num < self.warmup_steps:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
            if self.lr_decay < 1.0 and (self.scheduler2.T_cur + 1 == self.scheduler2.T_i):
                # Reduce the learning rate after every restart
                self.scheduler2.base_lrs[0] *= self.lr_decay
        self.step_num += 1


def dice_loss_fn(pred, true, num_classes=3, eps=1e-5):
    """
    Calculate the Dice loss for multi-class predictions.

    Args:
        pred (torch.Tensor): Model predictions.
        true (torch.Tensor): Ground truth labels.
        num_classes (int, optional): Number of classes. Default is 3.
        eps (float, optional): Small value to avoid division by zero. Default is 1e-5.

    Returns:
        float: Dice loss value.
    """
    # Apply softmax to the predictions
    pred = torch.softmax(pred, dim=1)

    # Initialize dice score
    dice = 0.0

    # Iterate over each class
    for c in range(num_classes):
        pred_c = pred[:, c]
        true_c = (true == c).float()
        intersection = torch.sum(true_c * pred_c)

        union = torch.sum(true_c) + torch.sum(pred_c)
        dice_c = (2.0 * intersection + eps) / (union + eps)
        dice += dice_c

    # Average over classes
    dice /= num_classes

    return 1.0 - dice  # minimization problem
