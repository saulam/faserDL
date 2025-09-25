"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Auxiliary functions for data augmentations.
"""

import numpy as np
from typing import Dict


ROTATIONS = {
    'x': {0: np.eye(3),
         90: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
         180: np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
         270: np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])},
    'y': {0: np.eye(3),
          90: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
          180: np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
          270: np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])},
    'z': {0: np.eye(3),
          90: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
          180: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
          270: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])}
}


def augment(
    coords, 
    modules, 
    feats, 
    labels, 
    momenta, 
    global_feats, 
    primary_vertex,
    metadata,
    stage1 = False,
    aug_prob=1.0
):
    """
    Performs augmentations.
    """        
    # Mirror
    if np.random.random() < aug_prob:
        coords, modules, momenta, global_feats['rear_cal_modules'], primary_vertex, _ = mirror(
            coords, modules, momenta, global_feats['rear_cal_modules'], 
            primary_vertex, metadata, selected_axes=['x', 'y', 'z'] if stage1 else ['x', 'y'],
        )              

    # Rotation
    if np.random.random() < aug_prob:
        coords, momenta, global_feats['rear_cal_modules'], primary_vertex, _ = rotate_90(
            coords, momenta, global_feats['rear_cal_modules'], 
            primary_vertex, metadata, selected_axes=['z'],
        )

    # Translation
    if np.random.random() < aug_prob:
        coords, modules, global_feats['rear_cal_modules'], primary_vertex, _ = translate(
            coords, modules, global_feats['rear_cal_modules'], 
            primary_vertex, metadata, selected_axes=['x', 'y'],
        )

    # Global features multiplicative jitter
    if np.random.random() < aug_prob:
        global_feats = module_multiplicative_jitter(
            global_feats, log_sigma=dict(faser=0.1, rear_cal=0.1, rear_hcal=0.1, rear_mucal=0.1),
        )

    # Scaling
    if np.random.random() < aug_prob:
        feats, global_feats, _, _ = scale_all_by_global_shift_lognormal(
            feats, global_feats, log_sigma=0.1
        )

    # Jitter per-hit multiplicative
    if np.random.random() < aug_prob:
        feats = jitter_energy_multiplicative(
            feats, log_sigma=0.12, clamp_min=0.0
        )

    # Jitter sqrt-law additive
    if np.random.random() < aug_prob:
        feats = jitter_energy_sqrtlaw(
            feats, a=0.06, b=0.18, clamp_min=0.0
        )

    # Voxel dropping
    if np.random.random() < aug_prob:
        coords, modules, feats, labels = drop_hits(
            coords, modules, feats, labels, max_drop=0.05, min_hits=5,
        )

    return coords, modules, feats, labels, momenta, global_feats, primary_vertex


def mirror(
    coords,
    modules,
    dirs,
    rear_cal_modules,
    primary_vertex,
    metadata,
    selected_axes=None
):
    """
    Randomly mirror coords, module indices, dir‐vectors, and a 2D rear_cal_modules image 
    along each of the X/Y/Z axes (independently) if that axis is in selected_axes.
    Returns all updated arrays, plus a list of axes actually flipped.
    """
    if selected_axes is None:
        selected_axes = ['x', 'y', 'z']
    axes = ['x', 'y', 'z']
    flipped = []
    coords = coords.copy()
    dirs = [d.copy() for d in dirs]
    primary_vertex = primary_vertex.copy()
    for axis_idx, ax in enumerate(axes):
        if ax in selected_axes and np.random.rand() < 0.5:
            # flip coords and primary vertex
            L = metadata[ax].shape[0]
            coords[:, axis_idx] = (L - 1) - coords[:, axis_idx]
            primary_vertex[axis_idx] = (L - 1) - primary_vertex[axis_idx]

            # flip direction vectors
            for d in dirs:
                d[axis_idx] *= -1

            if axis_idx == 2:
                # assume modules is a 1D array of module‐IDs in [0..n_mod-1]
                n_mod = metadata['z'][:, 1].max() + 1
                modules = (n_mod - 1) - modules
            else:
                # flip the 2D rear_cal_modules in XY plane
                flip_axis = 1 - axis_idx  # x => 1 (cols), y => 0 (rows)
                rear_cal_modules = np.flip(rear_cal_modules, axis=flip_axis)

            flipped.append(ax)

    return coords, modules, dirs, rear_cal_modules, primary_vertex, flipped


def rotate_90(
    coords,
    dirs,
    rear_cal_modules,
    primary_vertex,
    metadata,
    selected_axes=None
):
    """
    Randomly rotates (multiple of 90 degress) coords, dir‐vectors, and a 2D rear_cal_modules image 
    along each of the X/Y/Z axes (independently) if that axis is in selected_axes.
    Returns all updated arrays, plus a list of axes actually rotated and the angles.
    """
    if selected_axes is None:
        selected_axes = ['x','y','z']

    # build the composite rotation
    R_final = np.eye(3)
    chosen_angles: Dict[str,int] = {}
    for ax in ['x', 'y', 'z']:
        if ax in selected_axes:
            choices = [0, 90, 180, 270] if ax == 'z' else [0, 180]
            angle = int(np.random.choice(choices))
            chosen_angles[ax] = angle
            R_final = R_final @ ROTATIONS[ax][angle]

    # rotate the RearCal image only if z‑axis turned
    rear_rot = rear_cal_modules
    if chosen_angles.get('z', 0) != 0:
        k = chosen_angles['z'] // 90
        rear_rot = np.rot90(rear_rot, k)

    # apply to coords and vertex about the true centre
    center = np.array([
        (metadata['x'].shape[0]-1)/2.,
        (metadata['y'].shape[0]-1)/2.,
        (metadata['z'].shape[0]-1)/2.,
    ])
    pts = (coords - center) @ R_final + center
    vert = (primary_vertex - center) @ R_final + center

    # rotate the direction vectors
    dirs_rot = [(d @ R_final) for d in dirs]

    return pts, dirs_rot, rear_rot, vert, chosen_angles


def translate(
    coords,
    modules,
    rear_cal_modules,
    primary_vertex,
    metadata,
    selected_axes=None,
):
    """
    Translate a point‐cloud + module indices + primary‐vertex within the grid.

    coords are voxel indices in X/Y/Z (0…W-1 / 0…H-1, 0..D-1).
    modules are module‐indices (0…n_mod-1).

    X/Y: pick a single integer shift so that after shifting, all coords[:,axis]
         lie in [0, grid_len-1].  If any hit already touches 0 or grid_len-1,
         we consider that “escaping” and do not shift along that axis.

    Z: ±K‐module shifts.
    """
    if selected_axes is None:
        selected_axes = ['x','y','z']

    coords = coords.copy()
    modules = modules.copy()
    primary_vertex = primary_vertex.copy()
    rear = rear_cal_modules.copy()
    shifts: Dict[str,int] = {}

    # helpers
    def _is_touching_border(axis):
        L = metadata[axis].shape[0]
        c = coords[:, {'x':0, 'y':1}[axis]]
        return np.any((c <= 0) | (c >= L-1))

    def _shift_image(img, p, axis):
        """ Shift 2D img by p pixels along axis (0=rows, 1=cols). 
            Vacated entries get zero. """
        out = np.zeros_like(img)
        if p > 0:
            if axis == 1:
                out[:, p:] = img[:, :-p]
            else:  # axis==0
                out[p:, :] = img[:-p, :]
        elif p < 0:
            if axis == 1:
                out[:, :p] = img[:, -p:]
            else:
                out[:p, :] = img[-p:, :]
        else:
            out = img.copy()
        return out

    # X and Y
    for ax, idx in (('x', 0), ('y', 1)):
        if ax in selected_axes and not _is_touching_border(ax):
            L = metadata[ax].shape[0]
            c = coords[:, idx]
            s = np.random.randint(-c.min(), (L-1) - c.max() + 1)
            coords[:, idx]      += s
            primary_vertex[idx] += s
            shifts[ax] = s

            # compute 5×5 pixel shift
            ps = int(round(s * 5 / L))
            rear = _shift_image(rear, ps, axis=1-idx)
                
    # Z
    if 'z' in selected_axes:
        module_size = (metadata['z'][:, 1] == 0).sum()
        n_mod = int(metadata['z'][:,1].max() + 1)

        # shift modules by whole modules
        cur_min, cur_max = int(modules.min()), int(modules.max())
        valid_s = list(range(-cur_min, n_mod - cur_max))
        if valid_s:
            s_mod = np.random.choice(valid_s)
            modules += s_mod
            coords[:, 2]       += s_mod * module_size
            primary_vertex[2]  += s_mod * module_size
            shifts['z'] = s_mod * module_size

    return coords, modules, rear, primary_vertex, shifts


def drop_hits(
    coords,
    modules,
    feats,
    labels,
    max_drop=0.05,
    min_hits=5,
):
    """
    Randomly drop up to max_drop fraction of hits, but never below min_hits.
    """
    N = len(coords)
    p = np.random.rand() * max_drop
    mask = np.random.rand(N) > p

    # don’t drop if under min_hits
    if mask.sum() < min_hits:
        return coords, modules, feats, labels
    
    labels_masked = []
    for label in labels:
        if label is not None:
            if isinstance(label, np.ndarray):
                labels_masked.append(label[mask])
            else:
                csr = csr_keep_rows_numpy(*label, mask)
                labels_masked.append(csr)

    return coords[mask], modules[mask], feats[mask], labels_masked


def scale_all_by_global_shift(feats, momentums, global_feats, std_dev=0.1):
    """
    Apply a global multiplicative energy/momentum scale shift to all relevant features.
    
    Parameters:
    - feats: np.ndarray, typically (N_hits,) or (N_hits, D), representing hit charges or features in MeV.
    - momentums: list of np.ndarray, each of shape (3,) representing a 3D momentum vector (in MeV/c)
    - global_feats: dict[str, np.ndarray], global calorimeter features, etc., values in MeV or GeV.
    - std_dev: float, standard deviation for Gaussian noise on shift factor (centered at 1.0)

    Returns:
    - scaled_feats: np.ndarray
    - scaled_momentums: list of np.ndarray
    - scaled_global_feats: dict[str, np.ndarray]
    - shift: float, the applied scalar shift factor
    """
    shift = 1 - np.random.randn() * std_dev

    # Scale hits
    scaled_feats = feats * shift

    # Scale each 3D momentum vector
    scaled_momentums = [p * shift for p in momentums]

    # Scale each global feature (array or scalar)
    scaled_global_feats = {
        k: v * shift for k, v in global_feats.items()
    }

    return scaled_feats, scaled_momentums, scaled_global_feats, shift


def scale_all_by_global_shift_lognormal(feats, global_feats, log_sigma=0.1, momenta=None):
    # shift ~ LogNormal(mean=0, sigma=log_sigma) so E[shift]≈exp(0.5*log_sigma^2)
    # If you want mean≈1 exactly, divide by that factor.
    shift = np.exp(np.random.randn() * log_sigma)
    shift /= np.exp(0.5 * log_sigma**2)   # center around 1.0
    momenta = [p * shift for p in momenta] if momenta is not None else None
    return feats * shift, {k: v * shift for k, v in global_feats.items()}, momenta, shift


def jitter_energy_additive(feats, sigma=0.3, clamp_min=0.0):
    """
    Additive Gaussian jitter to voxel energies.
    
    Args:
        feats (Tensor): voxel energies, shape [N] or [B, ...]
        sigma (float): std dev of Gaussian noise (in same units as feats)
        clamp_min (float): minimum value after jitter (default=0 for energies)
    
    Returns:
        Tensor of same shape as feats, jittered.
    """
    noise = np.random.randn(*feats.shape) * sigma
    return (feats + noise).clip(min=clamp_min)


def jitter_energy_multiplicative(feats, log_sigma=0.15, clamp_min=0.0):
    mult = np.exp(np.random.randn(*feats.shape) * log_sigma)
    mult /= np.exp(0.5 * log_sigma**2)   # mean≈1
    return np.maximum(feats * mult, clamp_min)


def jitter_energy_sqrtlaw(feats, a=0.1, b=0.2, clamp_min=0.0, eps=1e-6):
    # sigma(feat) = sqrt(a^2 + b^2 * max(feat,0))
    sigma = np.sqrt(a*a + b*b * np.clip(feats, 0, None))
    return np.maximum(feats + np.random.randn(*feats.shape) * sigma, clamp_min)


def smooth_labels(targets, smoothing: float, num_classes: int = None):
    """
    Apply label smoothing.

    Parameters
    ----------
    targets : array-like, shape (N,) or (N, C)
        - If num_classes is provided: 1-D array of integer class labels in [0, num_classes-1].
        - Else:
          - 1-D (N,) or 2-D (N,1): binary labels or probabilities for the positive class.
          - 2-D (N, C) with C>1: one-hot or soft multi-class labels.
    smoothing : float in [0, 1)
        amount of smoothing to apply.
    num_classes : int, optional
        number of classes to one‑hot encode 1-D `targets` into before smoothing.
    
    Returns
    -------
    smoothed : ndarray of shape (N,) or (N, C)
        smoothed probabilities.
    """
    if smoothing <= 0:
        return targets
    targets = np.asarray(targets, dtype=np.float32)

    # If supplied num_classes, force multi-class path
    if num_classes is not None:
        if targets.ndim != 1:
            raise ValueError("With num_classes set, targets must be 1-D class indices.")
        N = targets.shape[0]
        C = num_classes
        # one-hot encode
        one_hot = np.zeros((N, C), dtype=np.float32)
        one_hot[np.arange(N), targets.astype(int)] = 1.0
        # smooth
        return one_hot * (1.0 - smoothing) + smoothing / float(C)

    # --- binary classification case ---
    if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
        probs = targets.reshape(-1)
        smooth_pos = probs * (1.0 - smoothing) + 0.5 * smoothing
        return smooth_pos.reshape(targets.shape)

    # --- multi‑class classification case ---
    elif targets.ndim == 2:
        N, C = targets.shape
        return targets * (1.0 - smoothing) + smoothing / float(C)

    else:
        raise ValueError(f"Unsupported target shape {targets.shape}, must be 1-D or 2-D.")


def _sync_totals_from_modules(global_feats, eps=1e-9):
    """Make totals equal the sum of their module arrays (when present)."""
    if 'faser_cal_modules' in global_feats:
        s = float(np.clip(np.sum(global_feats['faser_cal_modules']), 0.0, None))
        global_feats['faser_cal_energy'] = max(s, eps)
    if 'rear_cal_modules' in global_feats:
        s = float(np.clip(np.sum(global_feats['rear_cal_modules'] / 1000.), 0.0, None))   # correction to GeV
        global_feats['rear_cal_energy'] = max(s, eps)
    if 'rear_hcal_modules' in global_feats:
        s = float(np.clip(np.sum(global_feats['rear_hcal_modules']) / 1000., 0.0, None))  # correction to GeV
        global_feats['rear_hcal_energy'] = max(s, eps)
    # mucal has no module map; ensure strictly positive:
    if 'rear_mucal_energy' in global_feats:
        global_feats['rear_mucal_energy'] = max(float(global_feats['rear_mucal_energy']), eps)
    return global_feats


def module_multiplicative_jitter(global_feats, log_sigma=dict(faser=0.1, rear_cal=0.1, rear_hcal=0.1, rear_mucal=0.1)):
    """
    Independent per-cell multiplicative jitter (mean≈1), then sync totals.
    """
    gf = global_feats
    def _per_cell(a, s):
        mult = np.exp(np.random.randn(*a.shape) * s) / np.exp(0.5 * s * s)
        return np.maximum(a * mult, 0.0)

    if 'faser_cal_modules' in gf:
        gf['faser_cal_modules'] = _per_cell(np.asarray(gf['faser_cal_modules'], dtype=float),
                                            log_sigma.get('faser', 0.1))
    if 'rear_cal_modules' in gf:
        gf['rear_cal_modules'] = _per_cell(np.asarray(gf['rear_cal_modules'], dtype=float),
                                           log_sigma.get('rear_cal', 0.1))
    if 'rear_hcal_modules' in gf:
        gf['rear_hcal_modules'] = _per_cell(np.asarray(gf['rear_hcal_modules'], dtype=float),
                                            log_sigma.get('rear_hcal', 0.1))
    if 'rear_mucal_energy' in gf:
        gf['rear_mucal_energy'] = _per_cell(np.asarray(gf['rear_mucal_energy'], dtype=float),
                                            log_sigma.get('rear_mucal', 0.1))
    return _sync_totals_from_modules(gf)


def csr_keep_rows_numpy(label_indptr, label_ids, label_weight, mask):
    """
    NumPy version of row filtering for CSR.
    """
    N = label_indptr.size - 1
    assert mask.size == N and mask.dtype == bool

    kept_rows = np.flatnonzero(mask)                 # [M]
    M = kept_rows.size
    starts = label_indptr[:-1][kept_rows]
    ends   = label_indptr[1:][kept_rows]
    counts = ends - starts

    L = int(label_indptr[-1])
    if L == 0 or M == 0:
        return (np.zeros(M + 1, dtype=label_indptr.dtype),
                np.empty(0, dtype=label_ids.dtype),
                np.empty(0, dtype=label_weight.dtype),
                kept_rows)

    diff = np.zeros(L + 1, dtype=np.int64)
    np.add.at(diff, starts,  1)
    np.add.at(diff, ends,   -1)
    edge_mask = np.cumsum(diff[:-1]) > 0

    sel = np.flatnonzero(edge_mask)
    new_ids    = label_ids[sel]
    new_weight = label_weight[sel]

    new_indptr = np.zeros(M + 1, dtype=label_indptr.dtype)
    if M > 0:
        new_indptr[1:] = np.cumsum(counts)

    return new_indptr, new_ids, new_weight, kept_rows

