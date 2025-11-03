"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Auxiliary functions for data augmentations.
"""

import numpy as np
from typing import Dict


AHCAL_SHAPE = np.array([18, 18, 40], dtype=np.int32)
AHCAL_VOXEL_FACTOR = 4

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

    ecal_hits  = global_feats.get("ecal_hits", None)
    ahcal_hits = global_feats.get("ahcal_hits", None)
    muspec_p   = global_feats.get("muspec_p", None)

    # Mirror
    if np.random.random() < aug_prob:
        coords, modules, momenta, ecal_hits, ahcal_hits, muspec_p, primary_vertex, _ = mirror(
            coords, modules, momenta, ecal_hits, ahcal_hits, muspec_p,
            primary_vertex, metadata, selected_axes=['x', 'y', 'z'] if stage1 else ['x', 'y'],
        )   

    # Rotation
    if np.random.random() < aug_prob:
        coords, momenta, ecal_hits, ahcal_hits, muspec_p, primary_vertex, _ = rotate_90(
            coords, momenta, ecal_hits, ahcal_hits, muspec_p,
            primary_vertex, metadata, selected_axes=['x', 'y', 'z'] if stage1 else ['z'],
        )
    # Translation
    if np.random.random() < aug_prob:
        coords, modules, ecal_hits, ahcal_hits, primary_vertex, _ = translate(
            coords, modules, ecal_hits, ahcal_hits,
            primary_vertex, metadata, selected_axes=['x', 'y'],
        )

    # Re-store possibly modified extras
    if ecal_hits is not None:
        global_feats["ecal_hits"] = ecal_hits
    if ahcal_hits is not None:
        global_feats["ahcal_hits"] = ahcal_hits
    if muspec_p is not None:
        global_feats["muspec_p"] = muspec_p

    # Global features multiplicative jitter
    if np.random.random() < aug_prob:
        global_feats = module_multiplicative_jitter(
            global_feats, log_sigma=0.1,
        )

    # Scaling
    if np.random.random() < aug_prob:
        feats, global_feats, _, _ = scale_all_by_global_shift_lognormal(
            feats, global_feats, log_sigma=0.1
        )

    # After jitter + scaling, refresh locals from global_feats so we use
    ecal_hits  = global_feats.get("ecal_hits", ecal_hits)
    ahcal_hits = global_feats.get("ahcal_hits", ahcal_hits)
    muspec_p   = global_feats.get("muspec_p", muspec_p)

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
        coords, modules, feats, labels, ahcal_hits = drop_hits(
            coords, modules, feats, labels, ahcal_hits, max_drop=0.05, min_hits=5,
        )
        global_feats["ahcal_hits"] = ahcal_hits

    return coords, modules, feats, labels, momenta, global_feats, primary_vertex


def mirror(
    coords,
    modules,
    dirs,
    ecal_hits,
    ahcal_hits,
    muspec_p,
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
            muspec_p[:, axis_idx] *= -1

            if axis_idx == 2:
                # assume modules is a 1D array of module‐IDs in [0..n_mod-1]
                n_mod = metadata['z'][:, 1].max() + 1
                modules = (n_mod - 1) - modules
            else:
                # flip the 2D ecal_hits in XY plane
                flip_axis = 1 - axis_idx  # x => 1 (cols), y => 0 (rows)
                ecal_hits = np.flip(ecal_hits, axis=flip_axis).copy()

            # flip the AHCAL cloud in its own grid
            ah_L = AHCAL_SHAPE[axis_idx]
            ahcal_hits[:, axis_idx] = (ah_L - 1) - ahcal_hits[:, axis_idx]

            flipped.append(ax)

    return coords, modules, dirs, ecal_hits, ahcal_hits, muspec_p, primary_vertex, flipped


def rotate_90(
    coords,
    dirs,
    ecal_hits,
    ahcal_hits,
    muspec_p,
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
    rear_rot = ecal_hits
    if chosen_angles.get('x', 0) == 180:
        rear_rot = np.flip(rear_rot, axis=0).copy()
    if chosen_angles.get('y', 0) == 180:
        rear_rot = np.flip(rear_rot, axis=1).copy()
    if chosen_angles.get('z', 0) != 0:
        k = chosen_angles['z'] // 90
        rear_rot = np.rot90(rear_rot, k).copy()

    # rotate AHCAL cloud around the centre of its own grid
    if ahcal_hits.size > 0:
        ah_center = (AHCAL_SHAPE - 1) / 2.0  # (3,)
        ah_xyz = ahcal_hits[:, :3]
        ah_xyz_rot = (ah_xyz - ah_center) @ R_final + ah_center
        ahcal_hits = ahcal_hits.copy()
        ahcal_hits[:, :3] = ah_xyz_rot

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
    muspec_p = (muspec_p @ R_final)

    return pts, dirs_rot, rear_rot, ahcal_hits, muspec_p, vert, chosen_angles


def translate(
    coords,
    modules,
    ecal_hits,
    ahcal_hits,
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
    rear = ecal_hits.copy()
    ahcal = ahcal_hits.copy()
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

            # compute pixel shift
            ps = int(round(s * rear.shape[1-idx] / L))
            rear = _shift_image(rear, ps, axis=1-idx)

            ah_shift = int(np.round(s / float(AHCAL_VOXEL_FACTOR)))
            if ah_shift != 0 and ahcal.size > 0:
                ah_ax_len = AHCAL_SHAPE[idx]
                ahcal[:, idx] += ah_shift
                ahcal[:, idx] = np.clip(ahcal[:, idx], 0, ah_ax_len - 1)
                
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
            dz = s_mod * module_size
            coords[:, 2]       += dz
            primary_vertex[2]  += dz
            shifts['z'] = dz

            if ahcal.size > 0:
                ah_shift_z = int(np.round(dz / float(AHCAL_VOXEL_FACTOR)))
                if ah_shift_z != 0:
                    ah_z_len = AHCAL_SHAPE[2]
                    ahcal[:, 2] += ah_shift_z
                    ahcal[:, 2] = np.clip(ahcal[:, 2], 0, ah_z_len - 1)

    return coords, modules, rear, ahcal, primary_vertex, shifts


def drop_hits(
    coords,
    modules,
    feats,
    labels,
    ahcal_hits,
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
        ahcal_hits = _drop_ahcal_hits(ahcal_hits, max_drop, min_hits=min_hits)
        return coords, modules, feats, labels, ahcal_hits

    labels_masked = []
    for label in labels:
        if label is None:
            labels_masked.append(None)
        elif isinstance(label, np.ndarray):
            labels_masked.append(label[mask])
        else:
            csr = csr_keep_rows_numpy(*label, mask)
            labels_masked.append(csr)

    coords = coords[mask]
    modules = modules[mask]
    feats = feats[mask]
    labels = labels_masked
    ahcal_hits = _drop_ahcal_hits(ahcal_hits, max_drop, min_hits=min_hits)

    return coords, modules, feats, labels, ahcal_hits


def _drop_ahcal_hits(ahcal_hits, max_drop, min_hits=2):
    """Helper: randomly drop some AHCAL hits as well."""
    M = len(ahcal_hits)
    if M == 0:
        return ahcal_hits
    p_ah = np.random.rand() * max_drop
    mask_ah = np.random.rand(M) > p_ah
    if mask_ah.sum() < min_hits:
        return ahcal_hits
    return ahcal_hits[mask_ah]


def scale_all_by_global_shift_lognormal(
    feats,
    global_feats,
    momenta=None,
    log_sigma=0.1,
):
    """
    Global multiplicative scale (lognormal, mean≈1).
    """
    # draw shift
    shift = np.exp(np.random.randn() * log_sigma)
    shift /= np.exp(0.5 * log_sigma**2)   # center around 1.0

    # scale hit feats
    feats = feats * shift

    # scale momenta list if provided
    if momenta is not None:
        momenta = [p * shift for p in momenta]

    # scale existing scalar/array globals (simple heuristic: just multiply)
    scaled_global_feats = {}
    for k, v in global_feats.items():
        # we will handle the special ones below
        if k in ("ahcal_hits", "muspec_p", "muspec_q", "muspec_chi2"):
            scaled_global_feats[k] = v
        else:
            try:
                scaled_global_feats[k] = v * shift
            except Exception:
                # non-numeric, keep as is
                scaled_global_feats[k] = v

    # AHCAL charges
    if "ahcal_hits" in global_feats and global_feats["ahcal_hits"] is not None:
        ah = np.asarray(global_feats["ahcal_hits"], dtype=float).copy()
        if ah.size > 0:
            ah[:, 3] = ah[:, 3] * shift
        scaled_global_feats["ahcal_hits"] = ah

    # mu-spec momenta
    if "muspec_p" in global_feats and global_feats["muspec_p"] is not None:
        mp = np.asarray(global_feats["muspec_p"], dtype=float).copy()
        if mp.size > 0:
            mp = mp * shift
        scaled_global_feats["muspec_p"] = mp

    # mu-spec q
    if "muspec_q" in global_feats and global_feats["muspec_q"] is not None:
        mq = np.asarray(global_feats["muspec_q"], dtype=float).copy()
        scaled_global_feats["muspec_q"] = mq * shift

    # mu-spec chi2
    if "muspec_chi2" in global_feats and global_feats["muspec_chi2"] is not None:
        mc = np.asarray(global_feats["muspec_chi2"], dtype=float).copy()
        scaled_global_feats["muspec_chi2"] = mc * shift

    return feats, scaled_global_feats, momenta, shift


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


def module_multiplicative_jitter(global_feats, log_sigma=0.1):
    """
    Apply independent multiplicative lognormal jitter (mean≈1) to the physics-y
    pieces of global_feats:
    """
    def _lognormal(shape, s):
        mult = np.exp(np.random.randn(*shape) * s)
        mult /= np.exp(0.5 * s * s)
        return mult

    # AHCAL charges
    if "ahcal_hits" in global_feats and global_feats["ahcal_hits"] is not None:
        ah = np.asarray(global_feats["ahcal_hits"], dtype=float).copy()
        if ah.size > 0:
            # jitter only the charge channel
            charge = ah[:, 3]
            mult = _lognormal(charge.shape, log_sigma)
            ah[:, 3] = np.maximum(charge * mult, 0.0)
        global_feats["ahcal_hits"] = ah

    # mu-spec momenta (vectors)
    if "muspec_p" in global_feats and global_feats["muspec_p"] is not None:
        mp = np.asarray(global_feats["muspec_p"], dtype=float).copy()
        if mp.size > 0:
            # one multiplier per vector, then broadcast to 3 components
            nvec = mp.shape[0]
            mult = _lognormal((nvec, 1), log_sigma)
            mp = mp * mult
        global_feats["muspec_p"] = mp

    # mu-spec charges
    if "muspec_q" in global_feats and global_feats["muspec_q"] is not None:
        mq = np.asarray(global_feats["muspec_q"], dtype=float).copy()
        mult = _lognormal(mq.shape, log_sigma)
        global_feats["muspec_q"] = mq * mult

    # mu-spec chi2
    if "muspec_chi2" in global_feats and global_feats["muspec_chi2"] is not None:
        mc = np.asarray(global_feats["muspec_chi2"], dtype=float).copy()
        mult = _lognormal(mc.shape, log_sigma)
        global_feats["muspec_chi2"] = mc * mult

    return global_feats


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

