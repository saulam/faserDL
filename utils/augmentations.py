"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Auxiliary functions for data augmentations.
"""


import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import ini_argparse, random_rotation_saul


# for indexing in add_noise_global_params()
REAR_CAL_ENERGY_IDX = 0
REAR_HCAL_ENERGY_IDX = 1
REAR_MUCAL_ENERGY_IDX = 2
FASER_CAL_ENERGY_IDX = 3

REAR_HCAL_START = 4
REAR_HCAL_END = 13  # 9 values (4:13)

FASER_CAL_START = 13
FASER_CAL_END = 28  # 15 values (13:28)


def is_escaping(coords, metadata):
    """
    Check whether hits are on the edge(s) of the volume
    (potential escaping particles)
    """
    on_boundary = (
        (coords[:, 0] <= 0) | (coords[:, 0] >= metadata['x'].shape[0] - 1),
        (coords[:, 1] <= 0) | (coords[:, 1] >= metadata['y'].shape[0] - 1),
        (coords[:, 2] <= 0) | (coords[:, 2] >= metadata['z'].shape[0] - 1)
    )

    return np.any(on_boundary, axis=1)


def mirror(coords, modules, dirs, rear_cal_modules, primary_vertex, metadata, selected_axes=['x', 'y', 'z']):
    axes = ['x', 'y', 'z']
    for axis in range(3):
        if axes[axis] in selected_axes and np.random.choice([True, False]):
            # flip coords
            if axis<2:
                coords[:, axis] = metadata[axes[axis]].shape[0] - coords[:, axis] - 1
            else:
                module_size = int((metadata['z'][:,1]==0).sum())
                n_mod       = metadata['z'][:,1].max() + 1
                coords[:, axis] = module_size - coords[:, axis] - 1
                modules = n_mod - modules - 1

            # flip primary vertex
            primary_vertex[axis] = metadata[axes[axis]].shape[0] - primary_vertex[axis] - 1

            # flip direction vectors
            for x in dirs:
                x[axis] *= -1

            # flip the rearCal modules
            if axis < 2:
                # axis==0 (x): flip columns  => axis=1 of the 2D array
                # axis==1 (y): flip rows     => axis=0 of the 2D array
                flip_axis = 1 - axis
                rear_cal_modules = np.flip(rear_cal_modules, axis=flip_axis)

    return coords, modules, dirs, rear_cal_modules, primary_vertex


def rotate_90(coords, dirs, rear_cal_modules, primary_vertex, metadata, selected_axes=['x', 'y', 'z']):
    # Rotation matrices for 90, 180, and 270 degrees on each axis
    rotations = {
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

    final_rotation_matrix = np.eye(3)
    reference_point = np.array([
        (metadata['x'].shape[0] - 1) / 2.,
        (metadata['y'].shape[0] - 1) / 2.,
        (metadata['z'][:, 1] == 0).sum() / 2., #(metadata['z'].shape[0] - 1) / 2.
    ])  

    for axis in ['x', 'y', 'z']:
        if axis in selected_axes:
            angle = np.random.choice([0, 90, 180, 270])
            final_rotation_matrix = final_rotation_matrix @ rotations[axis][angle]

            if axis == 'z' and angle != 0:
                k = angle // 90  # number of 90° CCW turns
                rear_cal_modules = np.rot90(rear_cal_modules, k)

    translated_points = coords - reference_point
    translated_vertex = primary_vertex - reference_point
    rotated_points = translated_points @ final_rotation_matrix
    rotated_vertex = translated_vertex @ final_rotation_matrix
    final_points = rotated_points + reference_point
    final_vertex = rotated_vertex + reference_point
    rotated_dirs = [x @ final_rotation_matrix for x in dirs]

    return final_points, rotated_dirs, rear_cal_modules, final_vertex


def shear_rotation_2d(points_2d, theta):
    """
    Applies the shear-based rotation in 2D (for points in one plane).
    
    This implements the three-shear algorithm for 2D rotation:
      1. x = x - tan(theta/2) * y
      2. y = y + sin(theta) * x
      3. x = x - tan(theta/2) * y
    
    Parameters:
      points_2d (np.ndarray): Array of shape (N, 2) containing points in 2D.
      theta (float): Rotation angle in radians.
    
    Returns:
      np.ndarray: Rotated points (rounded after each shear).
    """
    pts = points_2d.copy()
    pts[:, 0] = np.round(pts[:, 0] - np.tan(theta / 2) * pts[:, 1])
    pts[:, 1] = np.round(pts[:, 1] + np.sin(theta) * pts[:, 0])
    pts[:, 0] = np.round(pts[:, 0] - np.tan(theta / 2) * pts[:, 1])
    
    return pts


def shear_rotation_axis(points, axis, theta):
    """
    Applies a shear-based rotation about a single axis in 3D.
    
    The function extracts the appropriate 2D plane, applies the shear rotation,
    and then puts the points back together.
    
    Parameters:
      points (np.ndarray): The input point cloud of shape (N, 3).
      axis (str): The axis to rotate about ("x", "y", or "z").
      theta (float): The rotation angle (in radians) to apply.
    
    Returns:
      np.ndarray: The transformed point cloud.
    """
    pts = points.copy()
    
    if axis == "z":
        pts_xy = pts[:, [0, 1]]  # get x, y
        pts_rot = shear_rotation_2d(pts_xy, theta)
        pts[:, 0] = pts_rot[:, 0]
        pts[:, 1] = pts_rot[:, 1]
        
    elif axis == "x":
        pts_yz = pts[:, [1, 2]]
        pts_rot = shear_rotation_2d(pts_yz, theta)
        pts[:, 1] = pts_rot[:, 0]
        pts[:, 2] = pts_rot[:, 1]
        
    elif axis == "y":
        pts_zx = pts[:, [2, 0]]
        pts_rot = shear_rotation_2d(pts_zx, theta)
        pts[:, 2] = pts_rot[:, 0]
        pts[:, 0] = pts_rot[:, 1]
        
    else:
        raise ValueError("Invalid axis: choose 'x', 'y', or 'z'.")
    
    return pts


def shear_rotation_random(coords, dirs, primary_vertex, metadata, selected_axes=['x', 'y', 'z']):
    """
    Applies shear-based rotations about each specified axis sequentially.
       
    For each axis specified in selected_axes, a random rotation angle is chosen
    (from fixed values based on: https://graphicsinterface.org/wp-content/uploads/gi1986-15.pdf)
    and the shear rotation (decomposed into 3 shear steps) is applied. Rounding is performed 
    after each shear to ensure that the mapping is exactly grid-preserving (assuming the input points, 
    or their rounded version, are on an integer grid).
    
    Parameters:
      coords (np.ndarray): Input point cloud, shape (N, 3).
      dirs (np.ndarray): Direction vectors to be rotated. Can be None.
      primary_vertex (np.ndarray): Primary vertex that is rotated.
      metadata: Dataset metadata. Expects 'x', 'y', 'z' keys with shape information.
      selected_axes (list, optional): List of axes (e.g., ["x", "y", "z"]) to rotate about.
      
    Returns:
      np.ndarray: Transformed point cloud after all shear rotations.
    """
    pts = coords.copy()
    primary_vertex = primary_vertex.copy()
    if dirs is not None:
        dirs = dirs.copy() 

    reference_point = np.array([
        (metadata['x'].shape[0] - 1) / 2.,
        (metadata['y'].shape[0] - 1) / 2.,
        (metadata['z'].shape[0] - 1) / 2.
    ]).astype(int)

    pts -= reference_point
    primary_vertex -= reference_point

    for axis in selected_axes:
        # Why no more than 45 degrees: https://graphicsinterface.org/wp-content/uploads/gi1986-15.pdf
        #angle_deg = np.random.uniform(-45, 45)
        #angle_deg = np.random.choice([0, 22.62, 28.07, 36.87, 53.13, 67.38, 73.74])  # values from link above
        angle_deg = np.random.uniform(5, 5)
        angle = np.deg2rad(angle_deg)
        if angle == 0:
            continue
        pts = shear_rotation_axis(pts, axis, angle)
        rotation = R.from_euler(axis, angle_deg, degrees=True)
        primary_vertex = rotation.apply(primary_vertex)
        if dirs is not None:
            dirs = rotation.apply(dirs) 

    pts += reference_point
    primary_vertex += reference_point
    return pts, dirs, primary_vertex


def rotate(coords, dirs, primary_vertex, metadata, selected_axes=['x', 'y', 'z']):
    escaping = is_escaping(coords, metadata)
    #if np.all(escaping):
    #    return coords, dirs, primary_vertex

    # Compute center of volume as rotation reference point
    reference_point = np.array([
        (metadata['x'].shape[0] - 1) / 2.,
        (metadata['y'].shape[0] - 1) / 2.,
        (metadata['z'].shape[0] - 1) / 2.
    ])

    # Shift coords and primary vertex relative to the center
    shifted_coords = coords - reference_point
    shifted_primary = primary_vertex - reference_point

    if dirs is not None:
        dirs = np.asarray(dirs)
        rotated_dirs = dirs.copy()
    else:
        rotated_dirs = None

    axes_indices = {'x': 0, 'y': 1, 'z': 2}
    affected_axes = {
        'x': ['y', 'z'],
        'y': ['x', 'z'],
        'z': ['x', 'y'],
    }

    for axis in selected_axes:
        affected = affected_axes[axis]
        angle_deg = np.random.uniform(-2, 2)  # You can adjust this range
        rotation = R.from_euler(axis, angle_deg, degrees=True)

        shifted_coords = rotation.apply(shifted_coords)
        shifted_primary = rotation.apply(shifted_primary[np.newaxis, :])[0]

        if rotated_dirs is not None:
            rotated_dirs = rotation.apply(rotated_dirs)

    # Recenter coordinates back
    rotated_coords = (shifted_coords + reference_point).round()
    rotated_vertex = shifted_primary + reference_point

    if len(np.unique(rotated_coords, axis=0)) < 2:
        return coords, dirs, primary_vertex

    return rotated_coords, rotated_dirs, rotated_vertex


def translate(coords, modules, primary_vertex, metadata, shift=8, selected_axes=['x', 'y', 'z'], shift_modules=False):
    """
    coords[:,2] is assumed to be the *in-module* z (0..module_size-1).
    modules[:] is the integer module index (0..n_mod-1).
    primary_vertex[2] is also a module index.

    We allow X/Y shifts ±5 voxels,
    and Z shifts by ±k modules (no partial modules).
    """

    if 'x' in selected_axes:
        shift_x = np.random.randint(0, shift)
        coords[:, 0]       += shift_x
        primary_vertex[0]  += shift_x

    if 'y' in selected_axes:
        shift_y = np.random.randint(0, shift)
        coords[:, 1]       += shift_y
        primary_vertex[1]  += shift_y

    if 'z' in selected_axes:
        module_size = (metadata['z'][:, 1] == 0).sum()
        n_mod = int(metadata['z'][:,1].max() + 1)

        # pick shifts (in module units) so that all (modules + s) stay in [0, n_mod-1]
        cur_min, cur_max = int(modules.min()), int(modules.max())
        if shift_modules:
            valid_shifts = [
                s for s in range(-cur_min, n_mod - cur_max)
            ]
            if valid_shifts:
                s = np.random.choice(valid_shifts)
                modules += s         # shift every hit’s module

        shift_z = np.random.randint(0, shift//2)
        coords[:, 2]       += shift_z
        primary_vertex[2]  += shift_z

    return coords, modules, primary_vertex


def drop(coords, modules, feats, labels, std_dev=0.1):
    if np.random.rand() > 0.5:
        return coords, modules, feats, labels
    p = abs(np.random.randn(1) * std_dev)
    mask = np.random.rand(coords.shape[0]) > p
    if mask.sum() < 2 or len(np.unique(coords[mask], axis=0)) < 2:
        #don't drop all coordinates
        return coords, modules, feats, labels
    return coords[mask], modules[mask], feats[mask], [x[mask] for x in labels]


def shift_q_uniform(feats, max_scale_factor=0.1):
    shift = 1 - np.random.rand(*feats.shape) * max_scale_factor
    return feats * shift


def shift_q_gaussian(feats, std_dev=0.1):
    shift = 1 - np.random.randn(*feats.shape) * std_dev
    return feats * shift


def add_noise_global_params(feats_global):
    # Compute initial sums before applying noise
    rear_hcal_sum = feats_global[REAR_HCAL_START:REAR_HCAL_END].sum()
    faser_cal_sum = feats_global[FASER_CAL_START:FASER_CAL_END].sum()

    # Apply Gaussian noise
    feats_global[REAR_CAL_ENERGY_IDX] = shift_q_gaussian(feats_global[REAR_CAL_ENERGY_IDX])
    feats_global[REAR_MUCAL_ENERGY_IDX] = shift_q_gaussian(feats_global[REAR_MUCAL_ENERGY_IDX])
    feats_global[REAR_HCAL_START:REAR_HCAL_END] = shift_q_gaussian(feats_global[REAR_HCAL_START:REAR_HCAL_END])
    feats_global[FASER_CAL_START:FASER_CAL_END] = shift_q_gaussian(feats_global[FASER_CAL_START:FASER_CAL_END])

    # Adjust the energy values proportionally after adding noise
    new_rear_hcal_sum = feats_global[REAR_HCAL_START:REAR_HCAL_END].sum()
    new_faser_cal_sum = feats_global[FASER_CAL_START:FASER_CAL_END].sum()
    if rear_hcal_sum > 0:
        feats_global[REAR_HCAL_ENERGY_IDX] *= new_rear_hcal_sum / rear_hcal_sum
    if faser_cal_sum > 0:
        feats_global[FASER_CAL_ENERGY_IDX] *= new_faser_cal_sum / faser_cal_sum

    return feats_global


def add_gaussian_noise(probs, std=0.05, shuffle_prob=0.01):
    """
    Adds Gaussian noise to a probability distribution while ensuring the sum remains 1.
    Additionally, in 1% of cases, it either shuffles the probabilities (if num_classes > 1)
    or applies (1 - current_value) when num_classes = 1.

    :param probs: NumPy array of shape (batch_size, num_classes) with probability distributions.
    :param std: Standard deviation of Gaussian noise.
    :param shuffle_prob: Probability of applying shuffling (or 1 - value if num_classes = 1).
    :return: Noisy probability distributions of the same shape as probs.
    """
    batch_size, num_classes = probs.shape

    # Add Gaussian noise
    noise = np.random.normal(0, std, size=(batch_size, num_classes))
    noisy_probs = probs + noise
    noisy_probs = np.clip(noisy_probs, 0, 1)  # Ensure values are in the range (0,1)
    if num_classes > 1:
        noisy_probs /= noisy_probs.sum(axis=1, keepdims=True)  # Normalize each row

    # Create mask for selected rows that need shuffling (num_classes > 1) or inversion (num_classes = 1)
    shuffle_mask = np.random.rand(batch_size) < shuffle_prob

    if num_classes > 1:
        # Shuffle probabilities within selected rows
        shuffled_probs = np.apply_along_axis(np.random.permutation, 1, probs)
        noisy_probs[shuffle_mask] = shuffled_probs[shuffle_mask]
    else:
        # Apply (1 - current_value) for single-class cases
        noisy_probs[shuffle_mask] = 1 - noisy_probs[shuffle_mask]

    return noisy_probs

