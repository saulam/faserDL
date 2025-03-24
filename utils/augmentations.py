"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Auxiliary functions for data augmentations.
"""


import torch
import numpy as np
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


def mirror(point_cloud, dirs, primary_vertex, metadata, selected_axes=['x', 'y', 'z']):
    axes = ['x', 'y', 'z']
    for axis in range(3):
        if axes[axis] in selected_axes and np.random.choice([True, False]):
            point_cloud[:, axis] = metadata[axes[axis]].shape[0] - point_cloud[:, axis]
            primary_vertex[axis] = metadata[axes[axis]].shape[0] - primary_vertex[axis]
            for x in dirs:
                x[axis] *= -1

    return point_cloud, dirs, primary_vertex


def rotate_90(point_cloud, dirs, primary_vertex, metadata, selected_axes=['x', 'y', 'z']):
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

    for axis in ['x', 'y', 'z']:
        if axis in selected_axes:
            angle = np.random.choice([0, 90, 180, 270])
            final_rotation_matrix = final_rotation_matrix @ rotations[axis][angle]

    translated_points = point_cloud - primary_vertex
    rotated_points = translated_points @ final_rotation_matrix
    final_points = rotated_points + primary_vertex
    rotated_dirs = [x @ final_rotation_matrix for x in dirs]

    return final_points, rotated_dirs


def rotate(coords, dirs, primary_vertex):
    """Random rotation along"""
    angle_limits = torch.tensor([
        [-torch.pi/180, -torch.pi/180, -torch.pi],
        [torch.pi/180, torch.pi/180,  torch.pi]
    ])

    if (angle_limits==0).all():
        # no rotation at all
        return coords, dirs

    return random_rotation_saul(coords=coords,
                                dirs=dirs,
                                angle_limits=angle_limits,
                                origin=primary_vertex)


def translate(coords, primary_vertex, selected_axes=['x', 'y', 'z']):
    shift_x, shift_y = np.random.randint(low=-5, high=5+1, size=(2,))
    shift_z = np.random.randint(low=-15, high=15+1)
    if 'x' in selected_axes:
        coords[:, 0] += shift_x
        primary_vertex[0] += shift_x
    if 'y' in selected_axes: 
        coords[:, 1] += shift_y
        primary_vertex[1] += shift_y
    if 'z' in selected_axes:
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

