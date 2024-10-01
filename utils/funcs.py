import torch
import MinkowskiEngine as ME
from sklearn.model_selection import KFold
from torch.utils.data import random_split, Subset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def split_dataset(dataset, args, splits=[0.6, 0.1, 0.3], seed=7):
    """
    Splits the dataset into training, validation, and test sets based on the given splits.

    Parameters:
    dataset (torch.utils.data.Dataset): The dataset to split.
    args (Namespace): Arguments containing batch_size and num_workers.
    splits (list): A list of three floats representing the split ratio for train, validation, and test sets.
                   The sum of these ratios must be 1. Default is [0.6, 0.1, 0.3].
    collate_fn (callable, optional): A function to merge a list of samples to form a mini-batch. Default is None.
    seed (int): The seed for random number generation to ensure reproducibility. Default is 7.

    Returns:
    tuple: DataLoader objects for training, validation, and test sets.
    """

    # Ensure the splits sum up to 1
    assert sum(splits) == 1, "The splits should sum up to 1."

    # Calculate the lengths of each split
    fulllen = len(dataset)
    train_len = int(fulllen * splits[0])
    val_len = int(fulllen * splits[1])
    test_len = fulllen - train_len - val_len  # Remaining length for the test set

    # Split the dataset into train, validation, and test sets
    train_set, val_set, test_set = random_split(
        dataset, 
        [train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(seed)
    )

    # Create DataLoader for each split
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_sparse_minkowski
    )
    valid_loader = DataLoader(
        val_set, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_sparse_minkowski
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_sparse_minkowski
    )

    return train_loader, valid_loader, test_loader


def collate_sparse_minkowski(batch):
    coords = [d['coords'] for d in batch]
    feats = torch.cat([d['feats'] for d in batch])
    y = torch.cat([d['labels'] for d in batch])

    ret = {
        'f': feats,
        'c': coords,
        'y': y,
    }

    return ret


def arrange_sparse_minkowski(data, device):
    return ME.SparseTensor(
        features=data['f'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device)


def arrange_truth(data, device):
    y = ME.SparseTensor(
         features=data['y'],
         coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
         device=device)
    return y


def arrange_sparse_minkowski(data, device):
    tensor = ME.SparseTensor(
        features=data['f'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )

    return tensor


def arrange_truth(data, device):
    main_truth = ME.SparseTensor(
        features=data['y'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )

    if 'y_aug' in data and 'c_aug' in data:
        aug_truth = ME.SparseTensor(
            features=data['y_aug'],
            coordinates=ME.utils.batched_coordinates(data['c_aug'], dtype=torch.int),
            device=device
        )
        return main_truth, aug_truth

    return main_truth


def argsort_coords(coordinates):
    # Assume coordinates are integers. Create a large enough multiplier to uniquely represent each dimension.
    # Multiply coordinates by powers of a large number to encode them uniquely into one tensor
    max_val = coordinates.max() + 1
    multipliers = torch.tensor([max_val**i for i in reversed(range(coordinates.shape[1]))], device=coordinates.device)

    # Create a single sortable tensor
    encoded_coords = (coordinates * multipliers).sum(dim=1)

    # Sort based on the encoded coordinates
    sorted_indices = torch.argsort(encoded_coords)

    return sorted_indices

def argsort_sparse_tensor(tensor):
    # Assume coordinates are integers. Create a large enough multiplier to uniquely represent each dimension.
    # Multiply coordinates by powers of a large number to encode them uniquely into one tensor
    max_val = tensor.coordinates.max() + 1
    multipliers = torch.tensor([max_val**i for i in reversed(range(tensor.coordinates.shape[1]))], device=tensor.coordinates.device)
    
    # Create a single sortable tensor
    #encoded_coords = torch.matmul(tensor.coordinates.float(), multipliers.float())
    encoded_coords = (tensor.coordinates * multipliers).sum(dim=1)

    # Sort based on the encoded coordinates
    sorted_indices = torch.argsort(encoded_coords)
    
    return sorted_indices


class CustomLambdaLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps):
        """
        Initialise a custom LambdaLR learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which the learning rate will be scheduled.
            warmup_steps (int): number of iterations for warm-up.
            lr_func (callable): A function to calculate the learning rate lambda.
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
            optimizer (torch.optim.Optimizer): The optimiser for which the learning rate will be scheduled.
            scheduler1 (_LRScheduler): The first scheduler for the warm-up phase.
            scheduler2 (_LRScheduler): The second scheduler for the main phase.
            lr_decay (float): The factor by which the learning rate is decayed after each restart (default: 1.0).
            warmup_steps (int): The number of steps for the warm-up phase (default: 100).
        """
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.warmup_steps = warmup_steps
        self.step_num = 0  # current scheduler step
        self.lr_decay = lr_decay  # decrease of lr after every restart

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
            if self.lr_decay < 1.0 and (self.scheduler2.T_cur+1 == self.scheduler2.T_i):
                # Reduce the learning rate after every restart
                self.scheduler2.base_lrs[0] *= self.lr_decay
        self.step_num += 1



