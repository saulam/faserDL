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
import scipy.ndimage

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


def mirror(coords, dirs, primary_vertex, metadata, selected_axes=['x', 'y', 'z']):
    axes = ['x', 'y', 'z']
    for axis in range(3):
        if axes[axis] in selected_axes and np.random.choice([True, False]):
            coords[:, axis] = metadata[axes[axis]].shape[0] - coords[:, axis] - 1
            primary_vertex[axis] = metadata[axes[axis]].shape[0] - primary_vertex[axis] - 1
            for x in dirs:
                x[axis] *= -1

    return coords, dirs, primary_vertex


def rotate_90(coords, dirs, primary_vertex, metadata, selected_axes=['x', 'y', 'z']):
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
        (metadata['z'].shape[0] - 1) / 2.
    ])  

    for axis in ['x', 'y', 'z']:
        if axis in selected_axes:
            angle = np.random.choice([0, 90, 180, 270])
            final_rotation_matrix = final_rotation_matrix @ rotations[axis][angle]

    translated_points = coords - reference_point
    translated_vertex = primary_vertex - reference_point
    rotated_points = translated_points @ final_rotation_matrix
    rotated_vertex = translated_vertex @ final_rotation_matrix
    final_points = rotated_points + reference_point
    final_vertex = rotated_vertex + reference_point
    rotated_dirs = [x @ final_rotation_matrix for x in dirs]

    return final_points, rotated_dirs, final_vertex


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


def shear_rotation_random(coords, dirs, primary_vertex, metadata, selected_axes=['x', 'y', 'z'], range_angle=45):
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
        angle_deg = np.random.uniform(-range_angle, range_angle)
        # angle_deg = np.random.choice([0, 22.62, 28.07, 36.87, 53.13, 67.38, 73.74])  # values from link above
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


def rotate(coords, dirs, primary_vertex, metadata, selected_axes=['x', 'y', 'z'], max_attempts=10):
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
        #if any(escaping[axes_indices[a]] for a in affected):
        #    continue

        for attempt in range(max_attempts):
            angle_deg = np.random.uniform(0, 360)  # You can adjust this range
            rotation = R.from_euler(axis, angle_deg, degrees=True)

            rotated_coords = rotation.apply(shifted_coords) + reference_point

            # Check bounds
            #if (
            #    np.all(rotated_coords[:, 0] >= 0) and np.all(rotated_coords[:, 0] < metadata['x'].shape[0]) and
            #    np.all(rotated_coords[:, 1] >= 0) and np.all(rotated_coords[:, 1] < metadata['y'].shape[0]) and
            #    np.all(rotated_coords[:, 2] >= 0) and np.all(rotated_coords[:, 2] < metadata['z'].shape[0])
            #):
            if True:
                # Valid rotation found → apply it
                shifted_coords = rotation.apply(shifted_coords)
                shifted_primary = rotation.apply(shifted_primary[np.newaxis, :])[0]

                if rotated_dirs is not None:
                    rotated_dirs = rotation.apply(rotated_dirs)

                break  # Stop attempting after success

    # Recenter coordinates back
    coords = shifted_coords + reference_point
    primary_vertex = shifted_primary + reference_point

    return coords, rotated_dirs, primary_vertex


def translate(coords, primary_vertex, metadata, selected_axes=['x', 'y', 'z']):
    escaping = is_escaping(coords, metadata)
    if np.all(escaping):
        return coords, primary_vertex

    if 'x' in selected_axes and not escaping[0]:
        x_axis_len = metadata['x'].shape[0]
        valid_shift_x = [
            shift for shift in range(-5, 6)
            if 0 <= coords[:, 0].min() + shift and coords[:, 0].max() + shift < x_axis_len
        ]
        if valid_shift_x:
            shift_x = np.random.choice(valid_shift_x)
            coords[:, 0] += shift_x
            primary_vertex[0] += shift_x

    if 'y' in selected_axes and not escaping[1]:
        y_axis_len = metadata['y'].shape[0]
        valid_shift_y = [
            shift for shift in range(-5, 6)
            if 0 <= coords[:, 1].min() + shift and coords[:, 1].max() + shift < y_axis_len
        ]
        if valid_shift_y:
            shift_y = np.random.choice(valid_shift_y)
            coords[:, 1] += shift_y
            primary_vertex[1] += shift_y

    # For Z, shift by multiple of module size
    if 'z' in selected_axes and not escaping[2]:
        z_axis_len = metadata['z'].shape[0]

        module_size = (metadata['z'][:, 1] == 0).sum()
        max_module_shift = (z_axis_len // module_size) - 1

        possible_shifts = np.array([
            i * module_size for i in range(-max_module_shift, max_module_shift + 1)
        ])

        z_coords = coords[:, 2]
        valid_shifts = [
            shift for shift in possible_shifts
            if 0 <= z_coords.min() + shift and z_coords.max() + shift < z_axis_len
        ]

        if valid_shifts:
            shift_z = np.random.choice(valid_shifts)
            coords[:, 2] += shift_z
            primary_vertex[2] += shift_z

    return coords, primary_vertex


def drop(coords, feats, labels, std_dev=0.1):
    if np.random.rand() > 0.5:
        return coords, feats, labels
    p = abs(np.random.randn(1) * std_dev)
    mask = np.random.rand(coords.shape[0]) > p
    if mask.sum() < 2:
        #don't drop all coordinates
        return coords, feats, labels
    return coords[mask], feats[mask], [x[mask] for x in labels]


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

def modif_vox_energy(coords, feats, labels, 
                     res_smearing_std=0.3, 
                     z_fluct_strength=0.8, 
                     subcluster_fluct_range=(0.4, 1.6), 
                     apply_global_bias=True,
                     global_bias_std=0.2,
                     primleton_aug_prob=0.5,
                     primleton_shift_std=5.0):
    """
    Focused voxel augmentation with extra distortion for Primleton voxels.

    Parameters:
    - coords: (N, 3) voxel coordinates
    - feats: (N, 1) voxel energy
    - labels: list of arrays matching feats
    - primleton_voxels: bool mask or int indices of Primleton voxels
    - primleton_aug_prob: probability of applying Primleton augmentations
    - primleton_shift_std: std dev of xy-coordinate shift for Primletons

    Returns:
    - coords, feats, labels
    """

    # 1. Resolution-based smearing
    if res_smearing_std > 0:
        random_noise = np.random.normal(0, res_smearing_std * feats[:, 0])
        feats[:, 0] += random_noise

    # 2. Z-profile energy modulation
    if z_fluct_strength > 0:
        z = coords[:, 2]
        z_mean = z.mean()
        z_std = z.std() + 1e-6
        attenuation = 1.0 - z_fluct_strength * ((z - z_mean) / z_std)
        feats[:, 0] *= attenuation

    # 3. Sub-cluster fluctuation
    if subcluster_fluct_range:
        idx = np.random.choice(len(feats), size=len(feats)//2, replace=False)
        scale = np.random.uniform(*subcluster_fluct_range, size=len(idx))
        feats[idx, 0] *= scale

    # 4. Global energy bias
    if apply_global_bias and global_bias_std > 0:
        bias = np.random.normal(0, global_bias_std, size=len(feats))
        feats[:, 0] *= (1 + bias)
    
    #add random high energetic voxels 
    if np.random.rand() < 0.4:
        high_energy_indices = np.random.choice(len(feats), size=len(feats)//500, replace=False)
        high_energy_values = np.random.uniform(10,500,size=len(high_energy_indices))
        feats[high_energy_indices, 0] += high_energy_values

    # # 5. Primleton-specific energy and xy-coordinate distortion
    if labels[0] is not None:
        # Convert primleton_voxels from ndarray of 0s and 1s to a boolean mask
        primleton_mask = labels[0].flatten().astype(bool)

        # Randomly select voxels for augmentation based on the given probability
        selected = np.random.rand(len(feats)) < primleton_aug_prob
        apply_mask = primleton_mask & selected

        # Apply energy distortion for the selected voxels
        feats[apply_mask, 0] *= np.random.uniform(0.3, 1, size=apply_mask.sum())

        primleton_indices = np.where(apply_mask)[0]

        for idx in primleton_indices:
            x0, y0, z0 = coords[idx, 0], coords[idx, 1], coords[idx, 2]

            # Generate random shift in range [-std, std]
            dx = np.random.randint(-primleton_shift_std, primleton_shift_std + 1)
            dy = np.random.randint(-primleton_shift_std, primleton_shift_std + 1)
            new_x, new_y = x0 + dx, y0 + dy
            # Check if a voxel already exists at the new (x, y, z)
            match_mask = (
                (coords[:, 0] == new_x) &
                (coords[:, 1] == new_y) &
                (coords[:, 2] == z0)
            )
            existing_indices = np.where(match_mask)[0]

            if len(existing_indices) > 0:
                # If voxel exists there, swap labels
                swap_idx = existing_indices[0]  # pick the first match
                labels[0][idx], labels[0][swap_idx] = labels[0][swap_idx], labels[0][idx]
            else:
                # Otherwise, move the voxel's coordinate x,y
                coords[idx, 0] = new_x
                coords[idx, 1] = new_y

    return coords, feats, labels




def translate_z_regions(coords, metadata, max_translation=(5, 5), N_reg=10):
    """
    Safely translates voxel coordinates in different z-regions without exceeding metadata bounds.

    Args:
    - coords: (N, 3) array of voxel coordinates.
    - metadata: Dict with keys 'x', 'y', 'z' providing axis metadata.
    - regions_z: List of tuples defining z-axis regions [(z_min, z_max), ...].
    - max_translation: Max allowed shift in x and y directions (±int).

    Returns:
    - translated_coords: Transformed coordinates (same shape).
    """

    #divide in 10 different regions and get regions_z
    z_min = coords[:,2].min()
    z_max = coords[:,2].max()
    z_step = int((z_max - z_min)/ N_reg)
    regions_z = [(z_min + i * z_step, z_min + (i + 1) * z_step) for i in range(10)]

    translated_coords = coords.copy()

    x_min, x_max = 0, metadata['x'].shape[0]
    y_min, y_max = 0, metadata['y'].shape[0]

    for z_min, z_max in regions_z:
        region_mask = (coords[:, 2] >= z_min) & (coords[:, 2] < z_max)

        if not np.any(region_mask):
            continue

        region_coords = translated_coords[region_mask]

        # X shift
        valid_shift_x = [
            dx for dx in range(-max_translation[0], max_translation[0] + 1)
            if x_min <= region_coords[:, 0].min() + dx and region_coords[:, 0].max() + dx < x_max
        ]
        shift_x = np.random.choice(valid_shift_x) if valid_shift_x else 0

        # Y shift
        valid_shift_y = [
            dy for dy in range(-max_translation[1], max_translation[1] + 1)
            if y_min <= region_coords[:, 1].min() + dy and region_coords[:, 1].max() + dy < y_max
        ]
        shift_y = np.random.choice(valid_shift_y) if valid_shift_y else 0

        # Apply shift
        translated_coords[region_mask, 0] += shift_x
        translated_coords[region_mask, 1] += shift_y

    return translated_coords