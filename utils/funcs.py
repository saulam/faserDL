"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Auxiliary functions definitions.
"""


import copy
import torch
from spconv.pytorch import SparseConvTensor
from functools import partial
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def split_dataset(dataset, args, splits=[0.6, 0.1, 0.3], seed=7, test=False, extra_dataset=None):
    """
    Splits the dataset into training, validation, and test sets based on the given splits.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset to split.
        args (Namespace): Arguments containing batch_size, num_workers.
        splits (list): A list of three floats representing the split ratios [train, validation, test]. Must sum to 1.
        seed (int): Seed for reproducibility. Default is 7.
        test (bool): Default is False.
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


def _make_spconv_tensor(
    feats,                 # [N, C]
    coords_list,           # length B, each [Ni, D] (no batch col), ints
    device,
    axis_order,            # "XYZ" (keep ME-style) or "ZYX"
    spatial_shape = None,  # predefine if you want fixed grid
):
    assert len(coords_list) > 0, "Empty batch."
    D = coords_list[0].shape[1]
    assert all(c.shape[1] == D for c in coords_list), "All coords must have same D."

    if device is None:
        device = feats.device

    # Build batched indices in XYZ first: [N, 1+D] = [b, x, y, (z…)]
    bcols, ccols = [], []
    for b, c in enumerate(coords_list):
        n = c.size(0)
        if n == 0:  # skip empty items (we filtered earlier, but be safe)
            continue
        bcols.append(torch.full((n, 1), b, dtype=torch.int32, device=device))
        ccols.append(c.to(device=device, dtype=torch.int32))
    indices_xyz = torch.cat([torch.cat(bcols, 0), torch.cat(ccols, 0)], dim=1)  # [N, 1+D]
    assert indices_xyz.numel() > 0, "No coordinates after batching."

    # Choose axis order for spconv
    if axis_order.upper() == "XYZ":
        indices_spatial = indices_xyz[:, 1:]
        if spatial_shape is None:
            spatial_shape = tuple((indices_spatial.max(0).values + 1).tolist())  # (X, Y, Z…)
        indices_spconv = indices_xyz  # [b, x, y, z…]
    elif axis_order.upper() == "ZYX":
        xyz = indices_xyz[:, 1:]
        zyx = xyz.flip(dims=[1])
        if spatial_shape is None:
            spatial_shape = tuple((zyx.max(0).values + 1).tolist())             # (Z, Y, X…)
        indices_spconv = torch.cat([indices_xyz[:, :1], zyx], dim=1)            # [b, z, y, x…]
    else:
        raise ValueError("axis_order must be 'XYZ' or 'ZYX'.")

    x_sp = SparseConvTensor(
        features=feats.to(device),
        indices=indices_spconv.to(torch.int32),
        spatial_shape=spatial_shape,
        batch_size=len(coords_list),
    )
    return x_sp, spatial_shape


def collate(
    batch,
    test: bool = True,
    *,
    axis_order: str = "XYZ",
    device = None,
    spatial_shape = (48, 48, 200),  # set (X, Y, Z) if fixed grid
):
    """
    Collate that returns a spconv-ready dict.
    Expects each item 'd' to have:
      - d["coords"]: [Ni, D] integer coords (no batch column)
      - d["feats"] : [Ni, C] features
      - optional: same fields you already propagate
    """
    mode = 'test' if test else 'train'
    batch = [d for d in batch if len(d["coords"]) > 0]
    if len(batch) == 0:
        # Return an empty-shaped stub to avoid downstream crashes
        return {
            "x_sp": None, "spatial_shape": spatial_shape, "batch_size": 0,
            "hit_event_id": torch.empty(0, dtype=torch.long),
        }

    coords_list = [d["coords"] for d in batch]
    feats_list  = [d["feats"]  for d in batch]

    # Build hit_event_id exactly like your current collate
    num_hits = torch.tensor([len(x) for x in feats_list], dtype=torch.long)
    hit_event_id = torch.arange(len(feats_list), dtype=torch.long).repeat_interleave(num_hits)

    feats_cat = torch.cat(feats_list, dim=0)
    x_sp, spatial_shape_out = _make_spconv_tensor(
        feats_cat, coords_list, device=device, axis_order=axis_order, spatial_shape=spatial_shape
    )

    ret = {
        "x_sp": x_sp,
        "spatial_shape": spatial_shape_out,
        "batch_size": len(coords_list),
        "hit_event_id": hit_event_id,
    }

    # Keep your existing extras 1:1
    ret["f_glob"] = torch.stack([d["feats_global"] for d in batch])
    ret["faser_cal_modules"] = torch.stack([d["faser_cal_modules"] for d in batch])
    ret["rear_cal_modules"] = torch.stack([d["rear_cal_modules"] for d in batch])
    ret["rear_hcal_modules"] = torch.stack([d["rear_hcal_modules"] for d in batch])

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
        to_item  = {"e_vis", "pt_miss", "vis_sp_momentum_mag",
                    "out_lepton_momentum_mag", "jet_momentum_mag"}

        for key in optional_keys:
            if key in batch[0]:
                if key in to_numpy:
                    ret[key] = [d[key].numpy() for d in batch]
                elif key in to_item:
                    ret[key] = [d[key].item() for d in batch]
                else:
                    ret[key] = [d[key] for d in batch]
        return ret
    
    # CSRs: stack row-wise
    for key in ["trk", "pri", "pdg"]:
        ret[f"csr_{key}_indptr"], ret[f"csr_{key}_ids"], ret[f"csr_{key}_weights"] = csr_stack_rows_torch(
            [d[f"csr_{key}_indptr"] for d in batch],
            [d[f"csr_{key}_ids"] for d in batch],
            [d[f"csr_{key}_weights"] for d in batch]
        )

    # mode == "train"
    opt_all = {
        "ghost_mask", "primlepton_labels", "seg_labels", "is_cc", "flavour_label", "charm",
        "e_vis", "pt_miss", "vis_sp_momentum", "out_lepton_momentum",
        "vis_sp_momentum_mag", "vis_sp_momentum_dir",
        "out_lepton_momentum_mag", "out_lepton_momentum_dir",
        "jet_momentum", "jet_momentum_mag", "jet_momentum_dir",
    }
    cat_keys = {
        "ghost_mask", "primlepton_labels", "seg_labels", "is_cc", 
        "flavour_label", "charm", "e_vis", "pt_miss", "vis_sp_momentum_mag",
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


def csr_stack_rows_torch(indptr_list, class_list, weight_list):
    """
    Concatenate K per-event CSRs row-wise into one CSR.

    Inputs (lists length K):
      indptr_list[k]:  [N_k+1] long
      class_list[k]:   [L_k]   long
      weight_list[k]:  [L_k]   float

    Returns:
      indptr: [N_total+1] long
      cls:    [L_total]   long
      w:      [L_total]   float
    """
    assert len(indptr_list) == len(class_list) == len(weight_list)
    device = indptr_list[0].device
    # Ensure everything is on the same device/dtype
    indptr_list = [t.to(device=device, dtype=torch.long) for t in indptr_list]
    class_list  = [t.to(device=device, dtype=torch.long) for t in class_list]
    weight_list = [t.to(device=device, dtype=weight_list[0].dtype) for t in weight_list]

    # Row counts per event, then one big cumulative sum
    counts_each = [ip[1:] - ip[:-1] for ip in indptr_list]         # list of [N_k]
    counts_all  = torch.cat(counts_each, dim=0) if counts_each else torch.zeros(0, dtype=torch.long, device=device)
    indptr = torch.empty(counts_all.numel() + 1, dtype=torch.long, device=device)
    indptr[0] = 0
    if counts_all.numel():
        indptr[1:] = counts_all.cumsum(0)

    # Concatenate columns
    cls = torch.cat(class_list, dim=0) if class_list else torch.zeros(0, dtype=torch.long, device=device)
    w   = torch.cat(weight_list, dim=0) if weight_list else torch.zeros(0, dtype=weight_list[0].dtype, device=device)

    # Sanity
    assert indptr[-1].item() == cls.numel() == w.numel()
    return indptr, cls, w


def arrange_input(data):
    x_sp = data['x_sp']
    faser_cal = data['faser_cal_modules']
    rear_cal = data['rear_cal_modules']
    rear_hcal = data['rear_hcal_modules']
    tensor_global = data['f_glob']

    return x_sp, faser_cal, rear_cal, rear_hcal, tensor_global


def arrange_truth(data):
    output = {}
    
    optional_keys = [
        'csr_trk_indptr', 'csr_trk_ids', 'csr_trk_weight',
        'csr_pri_indptr', 'csr_pri_ids', 'csr_pri_weight',
        'csr_pdg_indptr', 'csr_pdg_ids', 'csr_pdg_weight',
        'ghost_mask', 'hit_event_id',
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


def csr_keep_rows_torch(label_indptr, label_ids, label_weight, raw_idx):
    """
    Keep only the rows at integer indices `row_idx` from a CSR (label_indptr, label_ids, label_weight).

    Args:
      label_indptr : [N+1] long, row pointers
      label_ids    : [L]    long, column ids per nonzero
      label_weight : [L]    float/half, values per nonzero
      raw_idx      : [M]    long/int, row indices to keep (can be any order; duplicates allowed)

    Returns:
      new_indptr : [M+1] long
      new_ids    : [L_kept] long
      new_weight : [L_kept] float/half
      kept_rows  : [M] long  (same as `row_idx`, returned for convenience)
    """
    device = label_indptr.device
    if not (label_ids.device == device and label_weight.device == device):
        raise ValueError("All CSR tensors must be on the same device.")
    label_indptr = label_indptr.to(torch.long)
    label_ids = label_ids.to(torch.long)
    raw_idx = raw_idx.to(device=device, dtype=torch.long)

    N = label_indptr.numel() - 1
    if raw_idx.numel() == 0:
        return label_indptr.new_zeros(1), label_ids.new_empty(0), label_weight.new_empty(0), raw_idx

    if (raw_idx.min() < 0) or (raw_idx.max() >= N):
        raise IndexError("raw_idx contains out-of-bounds indices.")

    starts = label_indptr[raw_idx]          # [M]
    ends   = label_indptr[raw_idx + 1]      # [M]
    counts = ends - starts                  # [M]

    new_indptr = label_indptr.new_zeros(raw_idx.numel() + 1)
    if counts.numel():
        new_indptr[1:] = counts.cumsum(0)

    L_kept = int(new_indptr[-1].item())
    if L_kept == 0:
        return new_indptr, label_ids.new_empty(0), label_weight.new_empty(0), raw_idx

    start_rep   = torch.repeat_interleave(starts, counts)              # [L_kept]
    row_offsets = torch.repeat_interleave(new_indptr[:-1], counts)     # [L_kept]
    in_row_pos  = torch.arange(L_kept, device=device) - row_offsets    # [L_kept]
    edge_sel    = start_rep + in_row_pos                               # [L_kept] in [0..L-1]

    new_ids    = label_ids[edge_sel]
    new_weight = label_weight[edge_sel]

    return new_indptr, new_ids, new_weight, raw_idx


def weighted_loss(L, s, kind):
    """
    Calculates Kendall et al. uncertainty-weighted loss for any task.
    L: The raw loss for the task (e.g., MSE, BCE, CrossEntropy).
    s: The learnable log-variance parameter for the task.

    https://arxiv.org/pdf/1705.07115
    """
    #if kind in ("ce","bce","nce","focal"):    # classification-ish (InfoNCE/focal/BCE/CE)
    #    return torch.exp(-s) * L + 0.5 * s
    #return 0.5 * torch.exp(-s) * L + 0.5 * s  # regression-ish
    return torch.exp(-s) * L + 0.5 * s  # works best for me in practice


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
    def __init__(self, optimizer, scheduler1, scheduler2, warmup_steps=100, start_cosine_step=100):
        """
        Initialize the CombinedScheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimiser for which the learning rate will be scheduled.
            scheduler1 (_LRScheduler): The first scheduler for the warm-up phase.
            scheduler2 (_LRScheduler): The second scheduler for the main phase.
            warmup_steps (int): The number of steps for the warm-up phase (default: 100).
            start_cosine_step (int): The step to start cosine annealing scheduling.
        """
        self.optimizer = optimizer
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.warmup_steps = warmup_steps
        self.start_cosine_step = start_cosine_step
        self.step_num = 0  # current scheduler step

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
        self.step_num += 1

    def state_dict(self):
        """Return the state of the scheduler."""
        return {
            'warmup_steps': self.warmup_steps,
            'start_cosine_step': self.start_cosine_step,
            'step_num': self.step_num,
            'scheduler1': self.scheduler1.state_dict() if self.scheduler1 else None,
            'scheduler2': self.scheduler2.state_dict() if self.scheduler2 else None,
        }

    def load_state_dict(self, state_dict):
        """Load the scheduler state."""
        self.warmup_steps = state_dict['warmup_steps']
        self.start_cosine_step = state_dict['start_cosine_step']
        self.step_num = state_dict['step_num']
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


def move_obj(o, device):
    if isinstance(o, SparseConvTensor):
        ind = o.indices.to(device)
        if ind.dtype != torch.int32:
            ind = ind.int()
        return SparseConvTensor(o.features.to(device), ind, o.spatial_shape, o.batch_size)
    if isinstance(o, torch.Tensor):
        return o.to(device)
    if isinstance(o, dict):
        return {k: move_obj(v, device) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(move_obj(v, device) for v in o)
    return o
