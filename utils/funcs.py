"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Auxiliary functions definitions.
"""


import copy
import torch
import MinkowskiEngine as ME
from functools import partial
from sklearn.model_selection import KFold
from torch.utils.data import random_split, ConcatDataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def split_dataset(dataset, args, splits=[0.6, 0.1, 0.3], seed=7, test=False, extra_dataset=None):
    """
    Splits the dataset into training, validation, and test sets based on the given splits.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset to split.
        args (Namespace): Arguments containing batch_size, num_workers.
        splits (list): A list of three floats representing the split ratios [train, validation, test]. Must sum to 1.
        seed (int): Seed for reproducibility. Default is 7.
        test (bool): Whether to use collate_test or collate_sparse_minkowski. Default is False.
        extra_dataset (torch.utils.data.Dataset or None): Extra dataset to append to the training loader. Default is None.

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

    if args.train and args.augmentations_enabled and not args.stage1 and args.mixup_alpha > 0:
        train_set.calc_primary_vertices()

    train_set.augmentations_enabled = args.augmentations_enabled   
    collate_fn = partial(collate, test=test)

    return (
        create_loader(train_set, shuffle=True, drop_last=True, collate_fn=collate_fn, args=args),
        create_loader(val_set, shuffle=False, drop_last=True, collate_fn=collate_fn, args=args),
        create_loader(test_set, shuffle=False, drop_last=False, collate_fn=collate_fn, args=args),
    )


def create_loader(ds, shuffle, drop_last, collate_fn=None, args=None):
    if collate_fn is None:
        collate_fn = partial(collate, test=not args.train)
    persistent = args.num_workers > 0
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        pin_memory=True,
        persistent_workers=persistent,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )


def collate(batch, test: bool = True):
    """
    Unified collate for 'train' and 'test'.
    - mode='train': matches collate_train behavior (tensors via cat/stack).
    - mode='test' : matches collate_test behavior (lists; some .numpy()/.item()).
    """
    mode = 'test' if test else 'train'
    batch = [d for d in batch if len(d["coords"]) > 0]

    coords_list = [d["coords"] for d in batch]
    feats_list  = [d["feats"] for d in batch]
    num_hits = torch.tensor([len(x) for x in feats_list], dtype=torch.long)
    hit_event_id = torch.arange(len(feats_list), dtype=torch.long).repeat_interleave(num_hits)

    ret = {
        "f": torch.cat(feats_list, dim=0),
        "c": coords_list,
        "hit_event_id": hit_event_id,
        "f_glob": torch.stack([d["feats_global"] for d in batch]),
        "faser_cal_modules": torch.stack([d["faser_cal_modules"] for d in batch]),
        "rear_cal_modules": torch.stack([d["rear_cal_modules"] for d in batch]),
        "rear_hcal_modules": torch.stack([d["rear_hcal_modules"] for d in batch]),
    }

    if mode == "test":
        optional_keys = [
            "run_number", "event_id", "primary_vertex", "is_cc", "in_neutrino_pdg",
            "in_neutrino_energy", "primlepton_labels", "seg_labels", "flavour_label",
            "charm", "e_vis", "pt_miss",
            "vis_sp_momentum", "vis_sp_momentum_mag", "vis_sp_momentum_dir",
            "out_lepton_momentum", "out_lepton_momentum_mag", "out_lepton_momentum_dir",
            "jet_momentum", "jet_momentum_mag", "jet_momentum_dir",
        ]
        to_numpy = {"primlepton_labels", "seg_labels", "flavour_label", "charm"}
        to_item  = {
            "e_vis", "pt_miss", "vis_sp_momentum_mag",
            "out_lepton_momentum_mag", "jet_momentum_mag",
        }

        for key in optional_keys:
            if key in batch[0]:
                if key in to_numpy:
                    ret[key] = [d[key].numpy() for d in batch]
                elif key in to_item:
                    ret[key] = [d[key].item() for d in batch]
                else:
                    ret[key] = [d[key] for d in batch]
        return ret

    # mode == "train"
    opt_all = {
        "hit_track_id", "hit_primary_id", "hit_pdg", "ghost_mask",
        "primlepton_labels", "seg_labels", "is_cc", "flavour_label", "charm",
        "e_vis", "pt_miss", "vis_sp_momentum", "out_lepton_momentum",
        "vis_sp_momentum_mag", "vis_sp_momentum_dir",
        "out_lepton_momentum_mag", "out_lepton_momentum_dir",
        "jet_momentum", "jet_momentum_mag", "jet_momentum_dir",
    }
    cat_keys = {
        "hit_track_id", "hit_primary_id", "hit_pdg", "ghost_mask",
        "primlepton_labels", "seg_labels", "is_cc", "flavour_label", "charm",
        "e_vis", "pt_miss", "vis_sp_momentum_mag",
        "out_lepton_momentum_mag", "jet_momentum_mag",
    }
    stack_keys = {
        "vis_sp_momentum", "out_lepton_momentum", "jet_momentum",
        "vis_sp_momentum_dir", "out_lepton_momentum_dir", "jet_momentum_dir",
    }

    for key in opt_all:
        if key in batch[0]:
            if key in cat_keys:
                ret[key] = torch.cat([d[key] for d in batch])
            elif key in stack_keys:
                ret[key] = torch.stack([d[key] for d in batch])
            else:
                ret[key] = [d[key] for d in batch]

    return ret


def arrange_sparse_minkowski(data, device):
    tensor = ME.SparseTensor(
        features=data['f'],
        coordinates=ME.utils.batched_coordinates(data['c'], dtype=torch.int),
        device=device
    )
    faser_cal = data['faser_cal_modules']
    rear_cal = data['rear_cal_modules']
    rear_hcal = data['rear_hcal_modules']
    tensor_global = data['f_glob']

    return tensor, faser_cal, rear_cal, rear_hcal, tensor_global

def arrange_truth(data):
    output = {'coords': [x.detach().cpu().numpy() for x in data['c']]}
    
    optional_keys = [
        'hit_track_id', 'hit_primary_id', 'hit_pdg', 'ghost_mask', 'hit_event_id',
        'run_number', 'event_id', 'primary_vertex', 'is_cc', 'in_neutrino_pdg',
        'in_neutrino_energy', 'primlepton_labels', 'seg_labels', 'flavour_label',
        'charm', 'e_vis', 'pt_miss', 
        'vis_sp_momentum', 'vis_sp_momentum_mag', 'vis_sp_momentum_dir',
        'out_lepton_momentum', 'out_lepton_momentum_mag', 'out_lepton_momentum_dir',
        'jet_momentum', 'jet_momentum_mag', 'jet_momentum_dir'
    ]
    
    for key in optional_keys:
        if key in data:
            output[key] = data[key]
    
    return output


def weighted_loss(L, s):
    """
    Calculates the uncertainty-weighted loss for any task.
    L: The raw loss for the task (e.g., MSE, BCE, CrossEntropy).
    s: The learnable log-variance parameter for the task.

    https://arxiv.org/pdf/1705.07115
    """
    return torch.exp(-s) * L + 0.5 * s


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


def load_mae_encoder(model_vit, mae_ckpt):
    sd = {key.replace("model.", ""): value for key, value in mae_ckpt['state_dict'].items()}
    # Keep only encoder keys that exist in the fine-tune model
    vit_sd = model_vit.state_dict()
    #keep = {k: v for k, v in sd.items() if k in vit_sd and v.shape == vit_sd[k].shape}
    #missing = [k for k in vit_sd.keys() if k not in keep]
    #dropped = [k for k in sd.keys() if k not in keep]
    msg = model_vit.load_state_dict(sd, strict=False)
    #print("Loaded:", len(keep))
    #print("Missing in ckpt:", len(missing))
    #print("Dropped from ckpt:", len(dropped))
    print("Load msg:", msg)

