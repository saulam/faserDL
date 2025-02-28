"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Auxiliary functions definitions.
"""


import copy
import pickle as pkl
import numpy as np
import torch
import MinkowskiEngine as ME
from sklearn.model_selection import KFold
from torch.utils.data import random_split, Subset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def split_dataset(dataset, args, splits=[0.6, 0.1, 0.3], seed=7, test=False):
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

    if args.sets_path is not None:
        # Load saved sets
        with open(args.sets_path, "rb") as fd:
            sets = pkl.load(fd)
        sets["train_files"] = np.char.replace(sets["train_files"], "path", args.dataset_path, count=1)
        sets["valid_files"] = np.char.replace(sets["valid_files"], "path", args.dataset_path, count=1)
        sets["test_files"] = np.char.replace(sets["test_files"], "path", args.dataset_path, count=1)
        train_set = copy.deepcopy(dataset)
        val_set = copy.deepcopy(dataset)
        test_set = copy.deepcopy(dataset)
        train_set.data_files = sets["train_files"]
        val_set.data_files = sets["valid_files"]
        test_set.data_files = sets["test_files"]
        print("Loaded saved splits!")
    else:
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
        collate_fn=collate_test if test else collate_sparse_minkowski
    )
    valid_loader = DataLoader(
        val_set, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_test if test else collate_sparse_minkowski
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_test if test else collate_sparse_minkowski
    )

    return train_loader, valid_loader, test_loader


def collate_test(batch):
    ret = {
        'f': torch.cat([d['feats'] for d in batch]),
        'f_glob': torch.stack([d['feats_global'] for d in batch]),
        'c': [d['coords'] for d in batch],
    }
    
    optional_keys = [
        'run_number', 'event_id', 'primary_vertex', 'is_cc', 'in_neutrino_pdg', 
        'in_neutrino_energy', 'primlepton_labels', 'seg_labels', 'flavour_label',
        'e_vis', 'pt_miss', 'out_lepton_momentum_mag',
        'out_lepton_momentum_dir', 'jet_momentum_mag', 'jet_momentum_dir'
    ]
    
    for key in optional_keys:
        if key in batch[0]:
            ret[key] = ([d[key].numpy() for d in batch] if key in ['primlepton_labels', 'seg_labels', 'flavour_label']
                        else [d[key].item() for d in batch] if key in ['e_vis', 'pt_miss']
                        else [d[key] for d in batch])
    
    return ret


def collate_sparse_minkowski(batch):
    ret = {
        'f': torch.cat([d['feats'] for d in batch]),
        'f_glob': torch.stack([d['feats_global'] for d in batch]),
        'c': [d['coords'] for d in batch],
    }
    
    optional_keys = {
        'primlepton_labels', 'seg_labels', 'flavour_label',
        'e_vis', 'pt_miss', 'out_lepton_momentum_mag', 'out_lepton_momentum_dir',
        'jet_momentum_mag', 'jet_momentum_dir',
    }
    
    for key in optional_keys:
        if key in batch[0]:
            ret[key] = (torch.cat([d[key] for d in batch]) if key in ['primlepton_labels', 'seg_labels', 'flavour_label', 'e_vis', 'pt_miss']
                        else torch.stack([d[key] for d in batch]) if key in ['out_lepton_momentum_mag', 'out_lepton_momentum_dir', 'jet_momentum_mag', 'jet_momentum_dir']
                        else [d[key] for d in batch])
    
    return ret


def arrange_sparse_minkowski(data, device):
    tensor = ME.SparseTensor(
        features=data['f'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )
    tensor_global = data['f_glob']

    return tensor, tensor_global

def arrange_truth(data):
    output = {'coords': [x.detach().cpu().numpy() for x in data['c']]}
    
    optional_keys = [
        'run_number', 'event_id', 'primary_vertex', 'is_cc', 'in_neutrino_pdg',
        'in_neutrino_energy', 'primlepton_labels', 'seg_labels', 'flavour_label',
        'e_vis', 'pt_miss', 'out_lepton_momentum_mag',
        'out_lepton_momentum_dir', 'jet_momentum_mag', 'jet_momentum_dir'
    ]
    
    for key in optional_keys:
        if key in data:
            output[key] = data[key]
    
    return output

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
    def __init__(self, optimizer, scheduler1, scheduler2, lr_decay=1.0, warmup_steps=100, start_cosine_step=100):
        """
        Initialize the CombinedScheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimiser for which the learning rate will be scheduled.
            scheduler1 (_LRScheduler): The first scheduler for the warm-up phase.
            scheduler2 (_LRScheduler): The second scheduler for the main phase.
            lr_decay (float): The factor by which the learning rate is decayed after each restart (default: 1.0).
            warmup_steps (int): The number of steps for the warm-up phase (default: 100).
            start_cosine_step (int): The step to start cosine annealing scheduling.
        """
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.warmup_steps = warmup_steps
        self.start_cosine_step = start_cosine_step
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
        elif self.step_num >= self.start_cosine_step:
            self.scheduler2.step()
            if self.lr_decay < 1.0 and (self.scheduler2.T_cur+1 == self.scheduler2.T_i):
                # Reduce the learning rate after every restart
                self.scheduler2.base_lrs[0] *= self.lr_decay
        self.step_num += 1

    def state_dict(self):
        """Return the state of the scheduler."""
        return {
            'warmup_steps': self.warmup_steps,
            'start_cosine_step': self.start_cosine_step,
            'step_num': self.step_num,
            'lr_decay': self.lr_decay,
            'scheduler1': self.scheduler1.state_dict() if self.scheduler1 else None,
            'scheduler2': self.scheduler2.state_dict() if self.scheduler2 else None,
        }

    def load_state_dict(self, state_dict):
        """Load the scheduler state."""
        self.warmup_steps = state_dict['warmup_steps']
        self.start_cosine_step = state_dict['start_cosine_step']
        self.step_num = state_dict['step_num']
        self.lr_decay = state_dict['lr_decay']
        if self.scheduler1:
            self.scheduler1.load_state_dict(state_dict['scheduler1'])
        if self.scheduler2:
            self.scheduler2.load_state_dict(state_dict['scheduler2'])

