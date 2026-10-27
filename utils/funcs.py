"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Auxiliary functions definitions.
"""

import math
import copy
import hashlib
import numpy as np
import torch
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
try:
    from spconv.pytorch import SparseConvTensor
except ModuleNotFoundError:  # spconv-free environments use model.flash_vit.collate_tokens
    SparseConvTensor = None
from functools import partial
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def resolve_checkpoint_path(path, argument_name="checkpoint"):
    """Resolve an optional checkpoint path and fail if an explicit path is absent.

    Training launchers must not silently turn a requested resume/load operation
    into a fresh run merely because a path is misspelled or unavailable.
    """
    if path is None:
        return None
    checkpoint = Path(path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"{argument_name} checkpoint does not exist: {checkpoint}"
        )
    return str(checkpoint)


# The test-set evaluation (evaluation/inference_common.TASK_MONITORS) reads only
# the best checkpoint of these five task losses.
EVALUATED_MONITORS = [
    "loss_cls/flavour/val",
    "loss_cls/charm/val",
    "loss_vis/geom/val",
    "loss_jet/geom/val",
    "loss_vertex/val",
]
ALL_MONITORS = [
    "loss_total/val",
    "loss_cls/flavour/val",
    "loss_cls/charm/val",
    "loss_vis/geom/val",
    "loss_jet/geom/val",
    "loss_lep/geom/val",
    "loss_vertex/val",
]


def checkpoint_monitors(selection):
    """Fine-tuning validation losses to checkpoint: all, or only the evaluated ones."""
    return list(EVALUATED_MONITORS if selection == "evaluated" else ALL_MONITORS)


def split_fingerprint(*file_lists):
    """sha256 of the ordered event identities ("directory/file") of each list.

    Independent of where the events are stored, so equal fingerprints of
    (train, val, test) mean identical splits on every machine.
    """
    digest = hashlib.sha256()
    for files in file_lists:
        for path in files:
            path = Path(path)
            digest.update(f"{path.parent.name}/{path.name}\n".encode())
        digest.update(b"--\n")
    return digest.hexdigest()


def require_split_fingerprint(expected, train_files, val_files, test_files):
    """Print the split fingerprint and fail if it differs from ``expected``."""
    actual = split_fingerprint(train_files, val_files, test_files)
    counts = f"{len(train_files)} train / {len(val_files)} val / {len(test_files)} test"
    print(f"- Split fingerprint: {actual} ({counts})")
    if expected and actual != expected:
        raise RuntimeError(
            f"The train/val/test split differs from the expected one ({counts}; fingerprint "
            f"{actual} != {expected}).  Is the dataset copy complete and identical?")
    return actual


def share_file_lists(*loaders):
    """Store the loaders' event paths in numpy arrays before workers start.

    Forked DataLoader workers then index the paths without touching Python
    object reference counts, so they do not copy the list's pages
    (copy-on-read).  With 64 KB pages (e.g. Clariden's GH200 nodes) that copy
    reached ~1 GB per worker for the ~1M training paths.
    """
    for loader in loaders:
        loader.dataset.data_files = np.array(loader.dataset.data_files)


def split_dataset(dataset, args, splits=[0.6, 0.1, 0.3], seed=7, test=False, collate_fn=None):
    """
    Splits the dataset into training, validation, and test sets based on the given splits.

    Parameters:
        dataset (torch.utils.data.Dataset): The dataset to split.
        args (Namespace): Arguments containing batch_size, num_workers.
        splits (list): A list of three floats representing the split ratios [train, validation, test]. Must sum to 1.
        seed (int): Seed for reproducibility. Default is 7.
        test (bool): Default is False.
        collate_fn (callable): collate taking (batch, test=...); default spconv collate.

    Returns:
        tuple: DataLoader objects for training, validation, and test sets.
    """
    assert len(splits) == 3 and math.isclose(sum(splits), 1.0, rel_tol=0, abs_tol=1e-6), "Splits must be a list of three floats that sum to 1."

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
    val_set.data_files   = extract_files(val_split.indices)
    test_set.data_files  = extract_files(test_split.indices)

    if args.train and args.augmentations_enabled and not args.stage1 and args.mixup_alpha > 0:
        train_set.calc_primary_vertices()

    train_set.augmentations_enabled = args.augmentations_enabled   
    collate_fn = partial(collate if collate_fn is None else collate_fn, test=test)

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
    if SparseConvTensor is None:
        raise ModuleNotFoundError("spconv is required for the spconv collate")
    D = coords_list[0].shape[1]
    assert all(c.shape[1] == D for c in coords_list), "All coords must have same D."

    if device is None:
        device = feats.device

    # Build batched indices in XYZ first: [N, 1+D] = [b, x, y, (z…)]
    bcols, ccols = [], []
    for b, c in enumerate(coords_list):
        n = c.size(0)
        if n == 0:
            # Add a dummy coordinate at origin to maintain batch consistency
            # The attention mask in the model will ignore this token
            dummy_coord = torch.zeros((1, D), dtype=torch.int32, device=device)
            bcols.append(torch.full((1, 1), b, dtype=torch.int32, device=device))
            ccols.append(dummy_coord)
        else:
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


def pack_muspec(nb_muspec_tracks, muspec_info):
    """
    nb_muspec_tracks: tensor (B, 1)
    muspec_info: list of length B
        each element is a tensor of shape (K_i, 5)
    returns:
        feats: (B, K_max, 5)
        attn_mask: (B, K_max)  # True = keep, False = pad
        counts: (B, 1)  # number of tracks per batch element
    """
    # pad_sequence expects (seq_len, *) so we tell it batch_first=True
    feats = pad_sequence(muspec_info, batch_first=True, padding_value=0.0)  # (B, K_max, 5)
    B, K_max, D = feats.shape
    if K_max == 0:
        # The encoder deliberately creates one safe dummy key for events with
        # no tracks.  A whole batch (notably a batch-size-one smoke test) can
        # otherwise have sequence length zero and no index 0 to mark as safe.
        feats = feats.new_zeros(B, 1, D)
        K_max = 1

    lengths = torch.tensor([t.size(0) for t in muspec_info], device=feats.device)
    idxs = torch.arange(K_max, device=feats.device).unsqueeze(0).expand(B, -1)
    attn_mask = idxs < lengths.unsqueeze(1)

    # Return feats (B, K_max, 5), mask (B, K_max), and the counts (B, 1)
    return feats, attn_mask, nb_muspec_tracks


def collate(
    batch,
    test: bool = True,
    *,
    axis_order: str = "XYZ",
    device = None,
    spatial_shape = (48, 48, 200),  # set (X, Y, Z) if fixed grid
    ahcal_spatial_shape = (18, 18, 40),
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
    coords_list = [d["coords"] for d in batch]
    feats_list  = [d["feats"]  for d in batch]
    ahcal_coords_list = [d["ahcal_hits_coords"] for d in batch]
    ahcal_feats_list  = [d["ahcal_hits_feats"]  for d in batch]

    # Ensure empty ahcal events have at least one zero-feature entry
    # This prevents batch size mismatch when creating sparse tensors
    for i, (coords, feats) in enumerate(zip(ahcal_coords_list, ahcal_feats_list)):
        if len(coords) == 0:
            # Add dummy coordinate at origin with zero feature
            ahcal_coords_list[i] = torch.zeros((1, 3), dtype=coords.dtype, device=coords.device)
            ahcal_feats_list[i] = torch.zeros((1, feats.shape[1] if feats.ndim > 1 else 1), 
                                             dtype=feats.dtype, device=feats.device)

    # Build hit_event_id exactly like your current collate
    num_hits = torch.tensor([len(x) for x in feats_list], dtype=torch.long)
    hit_event_id = torch.arange(len(feats_list), dtype=torch.long).repeat_interleave(num_hits)

    feats_cat = torch.cat(feats_list, dim=0)
    x_sp, spatial_shape_out = _make_spconv_tensor(
        feats_cat, coords_list, device=device, axis_order=axis_order, spatial_shape=spatial_shape
    )
    # AHCAL hits spconv tensor
    ahcal_feats_cat = torch.cat(ahcal_feats_list, dim=0)
    ahcal_x_sp, ahcal_spatial_shape = _make_spconv_tensor(
        ahcal_feats_cat, ahcal_coords_list, device=device, axis_order=axis_order, spatial_shape=ahcal_spatial_shape
    )

    # Build hit_event_id_ahcal exactly like your current collate
    num_ahcal_hits = torch.tensor([len(x) for x in ahcal_feats_list], dtype=torch.long)
    hit_event_id_ahcal = torch.arange(len(ahcal_feats_list), dtype=torch.long).repeat_interleave(num_ahcal_hits)

    ret = {
        "x_sp": x_sp,
        "spatial_shape": spatial_shape_out,
        "batch_size": len(coords_list),
        "hit_event_id": hit_event_id,
        "ahcal_x_sp": ahcal_x_sp,
        "ahcal_spatial_shape": ahcal_spatial_shape,
        "hit_event_id_ahcal": hit_event_id_ahcal,
    }

    ret["ecal_hits"] = torch.stack([d["ecal_hits"] for d in batch])
    nb_muspec_tracks = torch.stack([d["nb_muspec_tracks"] for d in batch])
    muspec_info = [d["muspec_info"] for d in batch]    

    muspec_feats, muspec_attn_mask, muspec_counts = pack_muspec(nb_muspec_tracks, muspec_info)
    ret["muspec_feats"] = muspec_feats
    ret["muspec_attn_mask"] = muspec_attn_mask
    ret["muspec_counts"] = muspec_counts

    return collate_targets(batch, ret, mode)


def collate_targets(batch, ret, mode):
    """Add the per-event targets (and test-time identifiers) of ``batch`` to ``ret``.

    Shared by the spconv collate and the flash-attention token collate.
    """
    if mode == "test":
        optional_keys = [
            "run_number", "event_id", "primary_vertex", "is_cc", "in_neutrino_pdg",
            "in_neutrino_energy", "primlepton_labels", "seg_labels", "flavour_label",
            "charm_label", "e_vis", "pt_miss",
            "vis_sp_momentum", "vis_sp_momentum_mag", "vis_sp_momentum_dir",
            "out_lepton_momentum", "out_lepton_momentum_mag", "out_lepton_momentum_dir",
            "jet_momentum", "jet_momentum_mag", "jet_momentum_dir",
        ]
        to_numpy = {"primlepton_labels", "seg_labels", "flavour_label", "charm_label"}
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
    for key in ["hie", "dec", "pid"]:
        if f"csr_{key}_indptr" in batch[0]:
            ret[f"csr_{key}_indptr"], ret[f"csr_{key}_ids"], ret[f"csr_{key}_weights"] = csr_stack_rows_torch(
                [d[f"csr_{key}_indptr"] for d in batch],
                [d[f"csr_{key}_ids"] for d in batch],
                [d[f"csr_{key}_weights"] for d in batch]
            )

    # mode == "train"
    opt_all = {
        "ghost_mask", "primlepton_labels", "seg_labels", "is_cc", "flavour_label", "charm_label",
        "e_vis", "pt_miss", "vis_sp_momentum", "out_lepton_momentum",
        "vis_sp_momentum_mag", "vis_sp_momentum_dir",
        "out_lepton_momentum_mag", "out_lepton_momentum_dir",
        "jet_momentum", "jet_momentum_mag", "jet_momentum_dir",
        "primary_vertex",
    }
    cat_keys = {
        "ghost_mask", "primlepton_labels", "seg_labels", "is_cc", 
        "flavour_label", "charm_label", "e_vis", "pt_miss", "vis_sp_momentum_mag",
        "out_lepton_momentum_mag", "jet_momentum_mag",
    }
    stack_keys = {
        "vis_sp_momentum", "out_lepton_momentum", "jet_momentum",
        "vis_sp_momentum_dir", "out_lepton_momentum_dir", "jet_momentum_dir",
        "primary_vertex",
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
    ahcal_x_sp = data['ahcal_x_sp']
    ecal_hits = data['ecal_hits']
    muspec_feats = data['muspec_feats']
    muspec_attn_mask = data['muspec_attn_mask']
    muspec_counts = data['muspec_counts']

    return x_sp, ahcal_x_sp, ecal_hits, muspec_feats, muspec_attn_mask, muspec_counts


def arrange_truth(data):
    output = {}
    
    optional_keys = [
        'csr_hie_indptr', 'csr_hie_ids', 'csr_hie_weights',
        'csr_dec_indptr', 'csr_dec_ids', 'csr_dec_weights',
        'csr_pid_indptr', 'csr_pid_ids', 'csr_pid_weights',
        'ghost_mask', 'hit_event_id', 'hit_event_id_ahcal',
        'run_number', 'event_id', 'primary_vertex', 'is_cc', 'in_neutrino_pdg',
        'in_neutrino_energy', 'primlepton_labels', 'seg_labels', 'flavour_label',
        'charm_label', 'e_vis', 'pt_miss', 
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
      label_indptr : [N+1]  long, row pointers
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


def weighted_loss(
    L: torch.Tensor,
    u: torch.Tensor,
    w_min: float = 0.3,
    w_max: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Kendall uncertainty weighting with a *bounded* weight:
      w = w_min + (w_max - w_min) * sigmoid(u)   in (w_min, w_max)

    Uses the usual regularizer term with s = -log(w):
      loss = w * L + 0.5 * s

    Returns:
      loss_weighted, w, s

    https://arxiv.org/pdf/1705.07115
    """
    w = w_min + (w_max - w_min) * torch.sigmoid(u)
    s = -torch.log(w.clamp_min(1e-12))
    loss = w * L + 0.5 * s
    return loss, w, s


class CustomLambdaLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps):
        """
        Initialise a custom LambdaLR learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which the learning rate will be scheduled.
            warmup_steps (int): number of iterations for warm-up.
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
        return min(1.0, (step + 1) / self.warmup_steps)


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


def extract_pretrained_encoder_state(checkpoint, encoder_source="auto"):
    """Extract a bare encoder state dict from MAE or JEPA checkpoints.

    ``auto`` preserves the historical MAE behaviour and prefers the EMA target
    encoder when a JEPA checkpoint contains both branches.
    """
    if encoder_source not in {"auto", "context", "target"}:
        raise ValueError(f"Unknown encoder source: {encoder_source}")

    state = checkpoint.get("state_dict", checkpoint)
    target_prefixes = ("model.target_encoder.", "target_encoder.")
    context_prefixes = ("model.context_encoder.", "context_encoder.")

    def _has_prefix(prefixes):
        return any(any(key.startswith(prefix) for prefix in prefixes) for key in state)

    has_target = _has_prefix(target_prefixes)
    has_context = _has_prefix(context_prefixes)
    if encoder_source == "auto":
        selected = "target" if has_target else ("context" if has_context else "mae")
    else:
        selected = encoder_source

    if selected == "target":
        if not has_target:
            raise ValueError("The checkpoint does not contain a JEPA target encoder.")
        prefixes = target_prefixes
    elif selected == "context" and has_context:
        prefixes = context_prefixes
    elif selected == "context":
        # An MAE checkpoint has one encoder, which is equivalent to the context
        # branch for downstream initialisation.
        selected = "mae"
        prefixes = ()
    else:
        prefixes = ()

    if prefixes:
        extracted = {}
        for key, value in state.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    extracted[key[len(prefix):]] = value
                    break
    else:
        extracted = {
            (key[len("model."):] if key.startswith("model.") else key): value
            for key, value in state.items()
            if not key.startswith(("model.context_encoder.", "model.target_encoder."))
        }

    return extracted, selected


def load_mae_encoder(
    model_vit,
    mae_ckpt,
    encoder_source="auto",
    require_full_encoder=False,
):
    """Load compatible pretrained encoder weights into the fine-tuning model.

    Loading remains shape/key compatible for cross-domain transfer utilities,
    while the main in-domain fine-tuning entry point sets
    ``require_full_encoder=True`` to enforce an identical backbone.
    """
    state, selected = extract_pretrained_encoder_state(mae_ckpt, encoder_source)
    target_state = model_vit.state_dict()
    shape_mismatches = {
        key: (tuple(value.shape), tuple(target_state[key].shape))
        for key, value in state.items()
        if key in target_state and value.shape != target_state[key].shape
    }
    compatible = {
        key: value
        for key, value in state.items()
        if key in target_state and value.shape == target_state[key].shape
    }
    if not compatible:
        raise RuntimeError("No compatible encoder weights were found in the checkpoint.")

    # SparseViT's supervised-only parameters are all under heads.* when using
    # the global-pooling configuration.  Excluding those gives a meaningful
    # target-side encoder coverage even when tiny is transferred into base.
    target_encoder_keys = [
        key for key in target_state if not key.startswith("heads.")
    ]
    loaded_encoder_keys = [key for key in compatible if key in target_encoder_keys]
    target_encoder_numel = sum(target_state[key].numel() for key in target_encoder_keys)
    loaded_encoder_numel = sum(target_state[key].numel() for key in loaded_encoder_keys)
    encoder_coverage = loaded_encoder_numel / max(1, target_encoder_numel)
    missing_encoder = [key for key in target_encoder_keys if key not in compatible]
    unused_source = [key for key in state if key not in compatible]
    if require_full_encoder and missing_encoder:
        preview = ", ".join(missing_encoder[:8])
        if len(missing_encoder) > 8:
            preview += ", ..."
        raise RuntimeError(
            "The pretraining checkpoint does not fully cover the downstream "
            f"encoder ({encoder_coverage:.1%} by parameters; missing "
            f"{len(missing_encoder)} tensors: {preview}). Use the same model "
            "variant for pretraining and fine-tuning."
        )
    msg = model_vit.load_state_dict(compatible, strict=False)
    print(
        f"Loaded {len(compatible)} encoder tensors from the {selected} branch; "
        f"target encoder coverage={encoder_coverage:.1%} by parameters; "
        f"missing encoder tensors={len(missing_encoder)}, "
        f"missing total={len(msg.missing_keys)}, unused source={len(unused_source)}"
    )
    return {
        "source": selected,
        "loaded": sorted(compatible),
        "missing": list(msg.missing_keys),
        "unexpected": list(msg.unexpected_keys),
        "missing_encoder": sorted(missing_encoder),
        "unused_source": sorted(unused_source),
        "shape_mismatches": shape_mismatches,
        "encoder_coverage": encoder_coverage,
        "loaded_encoder_numel": loaded_encoder_numel,
        "target_encoder_numel": target_encoder_numel,
    }


def move_obj(o, device):
    if SparseConvTensor is not None and isinstance(o, SparseConvTensor):
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
