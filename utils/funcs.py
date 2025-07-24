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
        args (Namespace): Arguments containing batch_size, num_workers.
        splits (list): A list of three floats representing the split ratios [train, validation, test]. Must sum to 1.
        seed (int): Seed for reproducibility. Default is 7.
        test (bool): Whether to use collate_test or collate_sparse_minkowski. Default is False.

    Returns:
        tuple: DataLoader objects for training, validation, and test sets.
    """
    assert sum(splits) == 1, "The splits should sum up to 1."

    fulllen = len(dataset)
    train_len = int(fulllen * splits[0])
    val_len = int(fulllen * splits[1])
    test_len = fulllen - train_len - val_len  # Remaining length for the test set

    # Split the dataset
    train_split, val_split, test_split = random_split(
        dataset, 
        [train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(seed)
    )

    def extract_files(indices):
        return [dataset.data_files[i] for i in indices]

    train_set, val_set, test_set = (copy.deepcopy(dataset) for _ in range(3))
    train_set.data_files = extract_files(train_split.indices)
    val_set.data_files = extract_files(val_split.indices)
    test_set.data_files = extract_files(test_split.indices)
    
    train_set.augmentations_enabled = args.augmentations_enabled
    collate_fn = collate_test if test else collate_sparse_minkowski
    persistent = args.num_workers > 0

    def create_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=persistent,
            collate_fn=collate_fn
        )

    return (
        create_loader(train_set, shuffle=True),
        create_loader(val_set, shuffle=False),
        create_loader(test_set, shuffle=False),
    )


def collate_test(batch):
    coords_list, feats_list = [], []
    module_to_event, module_pos = [], []

    for ev_idx, sample in enumerate(batch):
        coords, mods, feats = sample['coords'], sample['modules'], sample['feats']
        '''
        for m in torch.unique(mods):
            mask = (mods == m)
            coords_list.append(coords[mask])
            feats_list.append(feats[mask])
            module_to_event.append(ev_idx)
            module_pos.append(int(m))
        '''
        coords_list.append(coords)
        feats_list.append(feats)
        module_to_event.append(ev_idx)
        module_pos.append(0)

    ret = {
        'f': torch.cat(feats_list, dim=0),
        'f_glob': torch.stack([d['feats_global'] for d in batch]),
        'module_hits': torch.stack([d['module_hits'] for d in batch]),
        'faser_cal_modules': torch.stack([d['faser_cal_modules'] for d in batch]),
        'rear_cal_modules': torch.stack([d['rear_cal_modules'] for d in batch]),
        'rear_hcal_modules': torch.stack([d['rear_hcal_modules'] for d in batch]),
        'c': coords_list,
        'module_to_event': torch.tensor(module_to_event, dtype=torch.long),
        'module_pos':      torch.tensor(module_pos,      dtype=torch.long),
    }
    
    optional_keys = [
        'run_number', 'event_id', 'primary_vertex', 'is_cc', 'in_neutrino_pdg', 
        'in_neutrino_energy', 'primlepton_labels', 'seg_labels', 'flavour_label',
        'charm', 'e_vis', 'pt_miss', 'out_lepton_momentum_mag',
        'out_lepton_momentum_dir', 'jet_momentum_mag', 'jet_momentum_dir'
    ]
    
    for key in optional_keys:
        if key in batch[0]:
            ret[key] = ([d[key].numpy() for d in batch] if key in ['primlepton_labels', 'seg_labels', 'is_cc', 'flavour_label', 'charm']
                        else [d[key].item() for d in batch] if key in ['e_vis', 'pt_miss', 'out_lepton_momentum_mag', 'jet_momentum_mag']
                        else [d[key] for d in batch])
    
    return ret


def collate_sparse_minkowski(batch):
    coords_list, feats_list = [], []
    module_to_event, module_pos = [], []

    for ev_idx, sample in enumerate(batch):
        coords, mods, feats = sample['coords'], sample['modules'], sample['feats']
        '''
        for m in torch.unique(mods):
            mask = (mods == m)
            coords_list.append(coords[mask])
            feats_list.append(feats[mask])
            module_to_event.append(ev_idx)
            module_pos.append(int(m))
        '''
        coords_list.append(coords)
        feats_list.append(feats)
        module_to_event.append(ev_idx)
        module_pos.append(0)

    ret = {
        'f': torch.cat(feats_list, dim=0),
        'f_glob': torch.stack([d['feats_global'] for d in batch]),
        'module_hits': torch.stack([d['module_hits'] for d in batch]),
        'faser_cal_modules': torch.stack([d['faser_cal_modules'] for d in batch]),
        'rear_cal_modules': torch.stack([d['rear_cal_modules'] for d in batch]),
        'rear_hcal_modules': torch.stack([d['rear_hcal_modules'] for d in batch]),
        'c': coords_list,
        'module_to_event': torch.tensor(module_to_event, dtype=torch.long),
        'module_pos':      torch.tensor(module_pos,      dtype=torch.long),
    }
    
    optional_keys = {
        'primlepton_labels', 'seg_labels', 'is_cc', 'flavour_label', 'charm',
        'e_vis', 'pt_miss', 'out_lepton_momentum_mag', 'out_lepton_momentum_dir',
        'jet_momentum_mag', 'jet_momentum_dir',
    }
    
    for key in optional_keys:
        if key in batch[0]:
            ret[key] = (torch.cat([d[key] for d in batch]) if key in ['primlepton_labels', 'seg_labels', 'is_cc', 'flavour_label', 'charm', 'e_vis', 'pt_miss', 'out_lepton_momentum_mag', 'jet_momentum_mag']
                        else torch.stack([d[key] for d in batch]) if key in ['out_lepton_momentum_dir', 'jet_momentum_dir']
                        else [d[key] for d in batch])
    
    return ret


def arrange_sparse_minkowski(data, device):
    tensor = ME.SparseTensor(
        features=data['f'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )
    module_hits = data['module_hits']
    faser_cal = data['faser_cal_modules']
    rear_cal = data['rear_cal_modules']
    rear_hcal = data['rear_hcal_modules']
    tensor_global = data['f_glob']

    return tensor, faser_cal, rear_cal, rear_hcal, tensor_global

def arrange_truth(data):
    output = {'coords': [x.detach().cpu().numpy() for x in data['c']]}
    
    optional_keys = [
        'run_number', 'event_id', 'primary_vertex', 'is_cc', 'in_neutrino_pdg',
        'in_neutrino_energy', 'primlepton_labels', 'seg_labels', 'flavour_label',
        'charm', 'e_vis', 'pt_miss', 'out_lepton_momentum_mag',
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


def transfer_weights(model_seg, model_enc):
    """
    Load weights from segmentation model to encoder model
    """
    model_enc.stem.load_state_dict(model_seg.stem.state_dict())
    model_enc.global_feats_encoder.load_state_dict(model_seg.global_feats_encoder.state_dict())
    N_bb = len(model_enc.shared_encoders)
    for i in range(N_bb):
        model_enc.shared_encoders[i].load_state_dict(model_seg.encoder_layers[i].state_dict())
        model_enc.shared_se_layers[i].load_state_dict(model_seg.se_layers[i].state_dict())
        if i < N_bb - 1:
            model_enc.shared_downsamples[i].load_state_dict(model_seg.downsample_layers[i].state_dict())

    last = N_bb  # index in model_seg: encoder_layers[last], se_layers[last], downsample_layers[last]
    for name, branch in model_enc.branches.items():
        # branch is an nn.ModuleDict, so indexing by string
        branch["downsample"].load_state_dict(model_seg.downsample_layers[last - 1].state_dict())
        branch["encoder"].load_state_dict(model_seg.encoder_layers[last].state_dict())
        branch["se"].load_state_dict(model_seg.se_layers[last].state_dict())

