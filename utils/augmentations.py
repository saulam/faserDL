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
    '''
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
    '''
    # FASERCAL ±1 voxel translation in x/y
    if np.random.random() < aug_prob:
        coords, modules, feats, labels, primary_vertex, (dx, dy) = translate_fasercal_xy_pm1(
            coords, modules, feats, labels, primary_vertex, metadata,
            prob_shift_x=0.5, prob_shift_y=0.5, min_keep=5,
        )

    # calo-only gain jitter
    if np.random.random() < aug_prob:
        global_feats = module_multiplicative_jitter(global_feats, log_sigma=0.1)

    # calo-only global gain drift
    if np.random.random() < aug_prob:
        feats, global_feats, _, _ = scale_all_by_global_shift_lognormal(
            feats, global_feats, log_sigma=0.1
        )

    # ECAL extra noise/dropout
    if np.random.random() < aug_prob and global_feats.get("ecal_hits", None) is not None:
        global_feats["ecal_hits"] = ecal_additive_noise_and_dropout(
            global_feats["ecal_hits"],
            a=0.02, b=0.10, max_drop=0.10
        )

    # AHCAL extra per-hit smearing
    if np.random.random() < aug_prob and global_feats.get("ahcal_hits", None) is not None:
        global_feats["ahcal_hits"] = ahcal_charge_smear(
            global_feats["ahcal_hits"],
            log_sigma=0.12, a=0.03, b=0.12
        )

    # muon spectrometer augmentation (independent of calo gain)
    if np.random.random() < aug_prob:
        n = global_feats.get("nb_muspec_tracks", 0)
        q = global_feats.get("muspec_q", np.zeros((0,), dtype=np.float32))
        p = global_feats.get("muspec_p", np.zeros((0, 3), dtype=np.float32))
        c = global_feats.get("muspec_chi2", np.zeros((0,), dtype=np.float32))

        n2, q2, p2, c2 = augment_muspec(
            n, q, p, c,
            permute=True,
            drop_prob=0.10,
            rel_p_logsigma=0.02,
            chi2_logsigma=0.10,
            flip_q_prob=0.0,   # keep 0 unless you explicitly want charge mis-ID
        )
        global_feats["nb_muspec_tracks"] = n2
        global_feats["muspec_q"] = q2
        global_feats["muspec_p"] = p2
        global_feats["muspec_chi2"] = c2

    # FASERCAL hit jitters
    if np.random.random() < aug_prob:
        feats = jitter_energy_multiplicative(feats, log_sigma=0.12, clamp_min=0.0)
    if np.random.random() < aug_prob:
        feats = jitter_energy_sqrtlaw(feats, a=0.06, b=0.18, clamp_min=0.0)

    # voxel dropping (already drops AHCAL hits too)
    if np.random.random() < aug_prob:
        coords, modules, feats, labels, ahcal_hits = drop_hits(
            coords, modules, feats, labels, global_feats.get("ahcal_hits", None),
            max_drop=0.05, min_hits=5,
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


def translate_fasercal_xy_pm1(
    coords,
    modules,
    feats,
    labels,
    primary_vertex,
    metadata,
    *,
    prob_shift_x=0.5,
    prob_shift_y=0.5,
    min_keep=5,
):
    """
    Randomly translate FASERCal hits by +/- 1 voxel in x and/or y.
    - If a hit goes out of bounds, it is removed.
    - If too few hits remain (min_keep), the augmentation is skipped.
    - CSR labels are kept consistent by filtering rows with the same mask.

    coords are assumed to be in voxel index space (after voxelise()).
    """
    # Decide shift in x and y independently
    dx = 0
    dy = 0
    if np.random.rand() < prob_shift_x:
        dx = int(np.random.choice([-1, 1]))
    if np.random.rand() < prob_shift_y:
        dy = int(np.random.choice([-1, 1]))

    # If neither axis selected, do nothing
    if dx == 0 and dy == 0:
        return coords, modules, feats, labels, primary_vertex, (0, 0)

    x_max = metadata["x"].shape[0] - 1
    y_max = metadata["y"].shape[0] - 1

    c2 = coords.copy()
    c2[:, 0] = c2[:, 0] + dx
    c2[:, 1] = c2[:, 1] + dy

    # Keep only hits that remain inside the volume
    keep = (
        (c2[:, 0] >= 0) & (c2[:, 0] <= x_max) &
        (c2[:, 1] >= 0) & (c2[:, 1] <= y_max)
    )

    # Must keep at least some hits; otherwise skip augmentation
    if int(keep.sum()) < min_keep:
        return coords, modules, feats, labels, primary_vertex, (0, 0)

    # Apply mask to coords/modules/feats
    c2 = c2[keep]
    m2 = modules[keep]
    f2 = feats[keep]

    # Apply mask to labels (CSR + arrays + None)
    labels_out = []
    for lab in labels:
        if lab is None:
            labels_out.append(None)
        elif isinstance(lab, np.ndarray):
            labels_out.append(lab[keep])
        else:
            # CSR tuple: (indptr, ids, weights)
            new_indptr, new_ids, new_w, _ = csr_keep_rows_numpy(*lab, keep)
            labels_out.append((new_indptr, new_ids, new_w))

    # update primary vertex consistently
    pv2 = primary_vertex
    if primary_vertex is not None:
        pv2 = np.array(primary_vertex, copy=True)

        # shift x/y (assumes pv[0]=x, pv[1]=y)
        pv2[0] = pv2[0] + dx * 10  # 10 = FASERCal voxel size in mm
        pv2[1] = pv2[1] + dy * 10  # 10 = FASERCal voxel size in mm

    return c2, m2, f2, tuple(labels_out), pv2, (dx, dy)


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
            new_indptr, new_ids, new_w, _ = csr_keep_rows_numpy(*label, mask)
            labels_masked.append((new_indptr, new_ids, new_w))

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


def ecal_additive_noise_and_dropout(ecal_hits, a=0.02, b=0.10, max_drop=0.10, clamp_min=0.0):
    """
    sqrt-law additive noise + random dead cells.
    Works for any dense shape (5x5, 25, etc).
    """
    ec = np.asarray(ecal_hits, dtype=np.float32).copy()
    if ec.size == 0:
        return ec

    # sqrt-law noise
    sigma = np.sqrt(a*a + b*b * np.clip(ec, 0, None))
    ec = ec + np.random.randn(*ec.shape).astype(np.float32) * sigma
    ec = np.maximum(ec, clamp_min)

    # dead cells
    p = np.random.rand() * max_drop
    flat = ec.reshape(-1)
    mask = (np.random.rand(flat.size) > p)
    if mask.sum() >= 1:
        flat[~mask] = 0.0
    return flat.reshape(ec.shape)


def ahcal_charge_smear(ahcal_hits, log_sigma=0.12, a=0.03, b=0.12, clamp_min=0.0):
    """
    Per-hit multiplicative + sqrt-law additive smearing on AHCAL charge column.
    """
    ah = np.asarray(ahcal_hits, dtype=np.float32).copy()
    if ah.size == 0:
        return ah

    q = ah[:, 3]

    # multiplicative (lognormal, mean≈1)
    mult = np.exp(np.random.randn(*q.shape).astype(np.float32) * log_sigma)
    mult /= np.exp(0.5 * log_sigma**2)
    q = q * mult

    # sqrt-law additive
    sigma = np.sqrt(a*a + b*b * np.clip(q, 0, None))
    q = q + np.random.randn(*q.shape).astype(np.float32) * sigma

    ah[:, 3] = np.maximum(q, clamp_min)
    return ah


def augment_muspec(
    nb_tracks,
    muspec_q,       # shape [T]
    muspec_p,       # shape [T,3]
    muspec_chi2,    # shape [T]
    *,
    permute=True,
    drop_prob=0.10,
    rel_p_logsigma=0.02,     # per-track multiplicative on p-vector (keeps direction)
    abs_p_sigma=0.0,         # optional additive component noise
    chi2_logsigma=0.10,      # jitter log(chi2)
    flip_q_prob=0.0,         # set small (e.g. 1e-3) if want mis-ID simulation
    min_keep=0,
):
    p = np.asarray(muspec_p, dtype=np.float32)
    q = np.asarray(muspec_q, dtype=np.float32)
    c = np.asarray(muspec_chi2, dtype=np.float32)

    if p.size == 0:
        return np.float32(0), q, p, c

    T = p.shape[0]
    idx = np.arange(T)

    if permute:
        np.random.shuffle(idx)
        p = p[idx]
        if q.size: q = q[idx]
        if c.size: c = c[idx]

    # drop tracks
    if drop_prob > 0:
        keep = (np.random.rand(T) > drop_prob)
        if min_keep > 0 and keep.sum() < min_keep:
            keep[np.random.choice(T, size=min_keep, replace=False)] = True
        p = p[keep]
        if q.size: q = q[keep]
        if c.size: c = c[keep]

    # smear momentum (one multiplier per track)
    if p.size:
        mult = np.exp(np.random.randn(p.shape[0], 1).astype(np.float32) * rel_p_logsigma)
        mult /= np.exp(0.5 * rel_p_logsigma**2)
        p = p * mult
        if abs_p_sigma > 0:
            p = p + np.random.randn(*p.shape).astype(np.float32) * abs_p_sigma

    # jitter chi2 in log space (keeps it positive, doesn’t correlate with calo gain)
    if c.size and chi2_logsigma > 0:
        l = np.log(c + 1e-3)
        l = l + np.random.randn(*l.shape).astype(np.float32) * chi2_logsigma
        c = np.exp(l).astype(np.float32)

    # optional charge flips (rare)
    if q.size and flip_q_prob > 0:
        flip = (np.random.rand(q.shape[0]) < flip_q_prob)
        q = q.copy()
        q[flip] *= -1.0
        q = np.sign(q)  # force back to ±1

    return np.float32(p.shape[0]), q, p, c


def scale_all_by_global_shift_lognormal(
    feats,
    global_feats,
    momenta=None,
    log_sigma=0.1,
):
    """
    Global gain drift (lognormal, mean≈1) applied to calorimeters.
    """
    # draw shift
    shift = np.exp(np.random.randn() * log_sigma)
    shift /= np.exp(0.5 * log_sigma**2)   # center around 1.0

    # scale hit feats
    feats = feats * shift

    out = dict(global_feats)

    # scale ECAL
    if "ecal_hits" in out and out["ecal_hits"] is not None:
        ec = np.asarray(out["ecal_hits"], dtype=np.float32).copy()
        if ec.size > 0:
            ec *= shift
            ec = np.maximum(ec, 0.0)
        out["ecal_hits"] = ec

    # scale AHCAL charge column
    if "ahcal_hits" in out and out["ahcal_hits"] is not None:
        ah = np.asarray(out["ahcal_hits"], dtype=np.float32).copy()
        if ah.size > 0:
            ah[:, 3] *= shift
            ah[:, 3] = np.maximum(ah[:, 3], 0.0)
        out["ahcal_hits"] = ah

    # leave muspec_p, muspec_q, muspec_chi2, nb_muspec_tracks unchanged
    return feats, out, momenta, shift


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
    Per-subdetector gain jitter (mean≈1).
    muspec_* and nb_muspec_tracks unchanged here
    """
    def _lognormal(shape, s):
        mult = np.exp(np.random.randn(*shape) * s)
        mult /= np.exp(0.5 * s * s)
        return mult

    out = dict(global_feats)

    # ECAL (dense 5x5)
    if "ecal_hits" in out and out["ecal_hits"] is not None:
        ec = np.asarray(out["ecal_hits"], dtype=np.float32).copy()
        if ec.size > 0:
            ec *= _lognormal(ec.shape, log_sigma)
            ec = np.maximum(ec, 0.0)
        out["ecal_hits"] = ec

    # AHCAL (sparse hits: [x,y,z,q])
    if "ahcal_hits" in out and out["ahcal_hits"] is not None:
        ah = np.asarray(out["ahcal_hits"], dtype=np.float32).copy()
        if ah.size > 0:
            q = ah[:, 3]
            q *= _lognormal(q.shape, log_sigma)
            ah[:, 3] = np.maximum(q, 0.0)
        out["ahcal_hits"] = ah

    return out


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

