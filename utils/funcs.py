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
    coords = [d['coords'] for d in batch]
    feats = torch.cat([d['feats'] for d in batch])
    feats_global = torch.stack([d['feats_global'] for d in batch])

    ret = {
        'f': feats,
        'f_glob': feats_global,
        'c': coords,
    }

    if 'prim_vertex' in batch[0]:
        prim_vertex = [d['prim_vertex'] for d in batch]
        ret['prim_vertex'] = prim_vertex

    if 'in_neutrino_pdg' in batch[0]:
        in_neutrino_pdg = [d['in_neutrino_pdg'] for d in batch]
        ret['in_neutrino_pdg'] = in_neutrino_pdg

    if 'in_neutrino_energy' in batch[0]:
        in_neutrino_energy = [d['in_neutrino_energy'] for d in batch]
        ret['in_neutrino_energy'] = in_neutrino_energy

    if 'primlepton_labels' in batch[0]:
        primlepton_labels = [d['primlepton_labels'].numpy() for d in batch]
        ret['primlepton_labels'] = primlepton_labels

    if 'seg_labels' in batch[0]:
        seg_labels = [d['seg_labels'].numpy() for d in batch]
        ret['seg_labels'] = seg_labels

    if 'flavour_label' in batch[0]:
        flavour_label = [d['flavour_label'].unsqueeze(0).numpy() for d in batch]
        ret['flavour_label'] = flavour_label

    if 'evis' in batch[0]:
        evis = [d['evis'].item() for d in batch]
        ret['evis'] = evis

    if 'ptmiss' in batch[0]:
        ptmiss = [d['ptmiss'].item() for d in batch]
        ret['ptmiss'] = ptmiss

    return ret


def collate_sparse_minkowski(batch):
    coords = [d['coords'] for d in batch]
    feats = torch.cat([d['feats'] for d in batch])
    feats_global = torch.stack([d['feats_global'] for d in batch])

    ret = {
        'f': feats,
        'f_glob': feats_global,
        'c': coords,
    }

    if 'prim_vertex' in batch[0]:
        prim_vertex = [d['prim_vertex'] for d in batch]
        ret['prim_vertex'] = prim_vertex

    if 'in_neutrino_pdg' in batch[0]:
        in_neutrino_pdg = [d['in_neutrino_pdg'] for d in batch]
        ret['in_neutrino_pdg'] = in_neutrino_pdg

    if 'in_neutrino_energy' in batch[0]:
        in_neutrino_energy = [d['in_neutrino_energy'] for d in batch]
        ret['in_neutrino_energy'] = in_neutrino_energy

    if 'y' in batch[0]:
        y = torch.cat([d['labels'] for d in batch])
        ret['y'] = y

    if 'primlepton_labels' in batch[0]:
        primlepton_labels = torch.cat([d['primlepton_labels'] for d in batch])
        ret['primlepton_labels'] = primlepton_labels

    if 'seg_labels' in batch[0]:
        seg_labels = torch.cat([d['seg_labels'] for d in batch])
        ret['seg_labels'] = seg_labels

    if 'flavour_label' in batch[0]:
        flavour_label = torch.cat([d['flavour_label'] for d in batch])
        ret['flavour_label'] = flavour_label

    if 'evis' in batch[0]:
        evis = torch.cat([d['evis'] for d in batch])
        ret['evis'] = evis

    if 'ptmiss' in batch[0]:
        ptmiss = torch.cat([d['ptmiss'] for d in batch])
        ret['ptmiss'] = ptmiss

    if 'out_lepton_momentum' in batch[0]:
        out_lepton_momentum = torch.stack([d['out_lepton_momentum'] for d in batch])
        ret['out_lepton_momentum'] = out_lepton_momentum

    if 'jet_momentum' in batch[0]:
        jet_momentum = torch.stack([d['jet_momentum'] for d in batch])
        ret['jet_momentum'] = jet_momentum

    if 'global_feats' in batch[0]:
        global_labels = torch.stack([d['global_feats'] for d in batch])
        ret['global_feats'] = global_labels

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
    output = {'coords': [x.detach().cpu().numpy() for x in data['c']],
              'prim_vertex': data['prim_vertex'],
              'in_neutrino_pdg': data['in_neutrino_pdg'],
              'in_neutrino_energy': data['in_neutrino_energy'],
              'primlepton_labels': data['primlepton_labels'] if 'primlepton_labels' in data else None,
              'seg_labels': data['seg_labels'] if 'seg_labels' in data else None,
              'flavour_label': data['flavour_label'],
              'evis': data['evis'],
              'ptmiss': data['ptmiss'],
              'out_lepton_momentum': data['out_lepton_momentum'] if 'out_lepton_momentum' in data else None,
              'jet_momentum': data['jet_momentum'] if 'jet_momentum' in data else None,
             }
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
            'scheduler1': self.scheduler1.state_dict(),
            'scheduler2': self.scheduler2.state_dict()
        }

    def load_state_dict(self, state_dict):
        """Load the scheduler state."""
        self.warmup_steps = state_dict['warmup_steps']
        self.start_cosine_step = state_dict['start_cosine_step']
        self.step_num = state_dict['step_num']
        self.lr_decay = state_dict['lr_decay']
        self.scheduler1.load_state_dict(state_dict['scheduler1'])
        self.scheduler2.load_state_dict(state_dict['scheduler2'])

