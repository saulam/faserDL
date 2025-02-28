"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: Dataset file.
"""

import pickle as pk
import numpy as np
import torch
import os
import sys
from glob import glob
from torch.utils.data import Dataset
from utils import ini_argparse
from utils.augmentations import *
import MinkowskiEngine as ME


class SparseFASERCALDataset(Dataset):
    def __init__(self, args):
        """
        Initializes the SparseFASERCALDataset class.

        Args:
        root (str): Root directory containing the data files.
        shuffle (bool): Whether to shuffle the dataset (default: False).
        """
        self.root = args.dataset_path
        self.data_files = self.processed_file_names
        self.load_seg = args.load_seg
        self.stage1 = args.stage1
        self.train = args.train
        self.augmentations = False
        self.total_events = self.__len__
        with open(self.root + "/metadata.pkl", "rb") as fd:
            self.metadata = pk.load(fd)
            self.metadata['x'] = np.array(self.metadata['x'])
            self.metadata['y'] = np.array(self.metadata['y'])
            self.metadata['z'] = np.array(self.metadata['z'])

            
    def set_augmentations_on(self):
        """Sets augmentations on dinamically."""
        print("Setting augmentations: ON.")
        self.augmentations = True
    
    
    def set_augmentations_off(self):
        """Sets augmentations off dinamically."""
        print("Setting augmentations: OFF.")
        self.augmentations = False
 

    @property
    def processed_dir(self):
        """
        Returns the processed directory path.

        Returns:
        str: Path to the processed directory.
        """
        return f'{self.root}'
    
    
    @property
    def processed_file_names(self):
        """
        Returns a list of processed file names.

        Returns:
        list: List of file names.
        """
        return glob(f'{self.processed_dir}/*.npz')
    
    
    def __len__(self):
        """
        Returns the total number of data files.

        Returns:
        int: Number of data files.
        """
        return len(self.data_files)

    
    def _augment(self, coords_ori, feats_ori, labels_ori, dirs_ori, primary_vertex):
        coords, feats = coords_ori.copy(), feats_ori.copy()
        labels = [x.copy() for x in labels_ori]
        dirs = [x.copy() for x in dirs_ori]
 
        # rotate
        coords, dirs = rotate(coords, dirs, primary_vertex)
        #coords, dirs = rotate_90(coords, dirs, primary_vertex, self.metadata)
        
        # translate
        coords, primary_vertex = translate(coords, primary_vertex)
        # drop voxels
        coords, feats, labels = drop(coords, feats, labels, std_dev=0.1)
        # shift feature values
        feats = shift_q_gaussian(feats, std_dev=0.01)
        # keep within limits
        coords, feats, labels = self.within_limits(coords, feats, labels, voxelised=False, mask_axes=[2])
        
        if coords.shape[0] < 2:
            return coords_ori, feats_ori, labels_ori, dirs_ori

        return coords, feats, labels, dirs
    
    
    def within_limits(self, coords, feats=None, labels=None, voxelised=False, mask_axes=(0, 1, 2)):
        """
        Filters coordinates, features, and labels based on given axis limits.
        Allows selecting which axes to apply the mask using mask_axes parameter.
        """
        if voxelised:
            range_x = 0, self.metadata['x'].shape[0]
            range_y = 0, self.metadata['y'].shape[0]
            range_z = 0, self.metadata['z'].shape[0]
        else:
            range_x = self.metadata['x'].min(), self.metadata['x'].max()
            range_y = self.metadata['y'].min(), self.metadata['y'].max()
            range_z = self.metadata['z'][:, 0].min(), self.metadata['z'][:, 0].max()

        ranges = [range_x, range_y, range_z]
        mask = np.ones(coords.shape[0], dtype=bool)

        for axis in mask_axes:
            mask &= (coords[:, axis] >= ranges[axis][0]) & (coords[:, axis] < ranges[axis][1])

        coords_filtered = coords[mask]
        feats_filtered = feats[mask] if feats is not None else None
        labels_filtered = [x[mask] for x in labels] if labels is not None else None
        '''
        mask = (coords[:, 0] >= range_x[0]) & (coords[:, 0] < range_x[1]) & \
           (coords[:, 1] >= range_y[0]) & (coords[:, 1] < range_y[1]) & \
           (coords[:, 2] >= range_z[0]) & (coords[:, 2] < range_z[1])
        coords_filtered = coords[mask]
        feats_filtered = feats[mask] if feats is not None else None
        labels_filtered = [x[mask] for x in labels] if labels is not None else None
        '''
        
        return coords_filtered, feats_filtered, labels_filtered
 

    def voxelise(self, coords, reverse=False):
        """
        Voxelises or unvoxelises given coordinates.
        
        Note: Voxelisation different for Z since module positions are not uniform.
        """
        min_x, max_x = self.metadata['x'].min(), self.metadata['x'].max()
        min_y, max_y = self.metadata['y'].min(), self.metadata['y'].max()
        range_x = self.metadata['x'].shape[0] - 1
        range_y = self.metadata['y'].shape[0] - 1

        def forward_transform(values, min_val, max_val, range_max):
            return range_max * (values - min_val) / (max_val - min_val)

        def inverse_transform(values, min_val, max_val, range_max):
            return min_val + (values / range_max) * (max_val - min_val)

        transform = inverse_transform if reverse else forward_transform

        mapped = np.empty_like(coords, dtype=np.float32)
        mapped[..., 0] = transform(coords[..., 0], min_x, max_x, range_x)
        mapped[..., 1] = transform(coords[..., 1], min_y, max_y, range_y)
        if reverse:
            mapped[..., 2] = self.metadata['z'][coords[..., 2].astype(int), 0]
        else:
            mapped[..., 2] = np.searchsorted(self.metadata['z'][:, 0], coords[..., 2])

        return mapped.round()  # round needed only for augmentations
        
        
    def aggregate_duplicate_coords(self, coords, feats, primlepton_labels, seg_labels):
        """
        Aggregate duplicate coordinates by:
        - Keeping unique 3D coordinates.
        - Prioritising primary leptons when handling collisions, as primlepton_labels indicate primary lepton presence.
        - Handling seg_labels as follows:
            - The first column corresponds to ghost identification, meaning ghosts disappear if they collide with non-ghosts.
            - The second and third columns represent total energy deposits from electromagnetic and hadronic particles, respectively,
            which should be summed in case of collisions.
        """
        coords_tuple = [tuple(row) for row in coords]
        unique_coords, indices = np.unique(coords_tuple, axis=0, return_inverse=True)

        unique_feats = np.zeros((len(unique_coords), 1))
        unique_primlepton_labels = np.zeros((len(unique_coords), 1))
        unique_seg_labels = np.zeros((len(unique_coords), 3))

        for i, idx in enumerate(indices):
            unique_feats[idx, 0] += feats[i, 0]
            unique_primlepton_labels[idx] = max(unique_primlepton_labels[idx], primlepton_labels[i])
            unique_seg_labels[idx, 0] = min(unique_seg_labels[idx, 0], seg_labels[i, 0]) if unique_seg_labels[idx, 0] != 0 else seg_labels[i, 0]
            unique_seg_labels[idx, 1] += seg_labels[i, 1]
            unique_seg_labels[idx, 2] += seg_labels[i, 2]

        return unique_coords, unique_feats, unique_primlepton_labels, unique_seg_labels
    
    
    def normalise_seg_labels(self, seg_labels, eps=1e-8):
        """
        Normalise seg_labels by:
        - Keeping the first column unchanged as it represents ghost identification.
        - Normalizing the second and third columns (energy depositions from electromagnetic and hadronic components) so that their sum equals (1 - first column value) per row.
        """
        
        sum_values = np.sum(seg_labels[:, 1:], axis=1, keepdims=True)
        mask = sum_values.squeeze() > 0
        seg_labels[mask, 1:] = (1 - seg_labels[mask, 0:1]) * (seg_labels[mask, 1:] / sum_values[mask, :])
    
        return seg_labels

        
    def pdg2label(self, pdg, iscc, name=False):
        """
        PDG to label.
        """
        if iscc:
            if pdg in [-12, 12]:
                label = "CC nue" if name else 0
            elif pdg in [-14, 14]:
                label = "CC numu" if name else 1
            elif pdg in [-16, 16]:
                label = "CC nutau" if name else 2
        else:
            label = "NC" if name else 3

        return label
    
    
    def decompose_momentum(self, momentum):
        """
        Given a 3D momentum vector or an array of N 3D momentum vectors, 
        return the magnitudes and directions separately.
        
        Parameters:
            momentum (np.ndarray): A (3,) or (N, 3) NumPy array representing momentum vector(s).
            
        Returns:
            magnitude (np.ndarray): The magnitude(s) of the momentum vector(s).
            direction (np.ndarray): A unit vector (or unit vectors) representing the direction(s) of the momentum.
        """
        momentum = np.atleast_2d(momentum)  # Converts (3,) -> (1, 3), leaves (N,3) unchanged
        magnitudes = np.linalg.norm(momentum, axis=1, keepdims=True)
        directions = np.divide(momentum, magnitudes, where=magnitudes != 0)
    
        if magnitudes.shape[0] == 1:
            return magnitudes[0], directions[0]
    
        return magnitudes.flatten(), directions

    
    def reconstruct_momentum(self, magnitude, direction):
        """
        Given magnitude and direction, reconstruct the original momentum vector.
        
        Parameters:
            magnitude (float or np.ndarray): A scalar or an array of shape (N,) representing momentum magnitudes.
            direction (np.ndarray): A (3,) or (N, 3) NumPy array representing unit direction vectors.
            
        Returns:
            momentum (np.ndarray): The reconstructed momentum vector(s).
        """
        magnitude = np.atleast_1d(magnitude)
        direction = np.atleast_2d(direction)  # Converts (3,) -> (1, 3), leaves (N,3) unchanged
    
        momentum = magnitude[:, np.newaxis] * direction
    
        return momentum[0] if magnitude.shape[0] == 1 else momentum

    
    def standardize(self, x, param_name):
        return (x - self.metadata[param_name]['mean']) / self.metadata[param_name]['std']    
    
    
    def divide_by_std(self, x, param_name):
        return x / self.metadata[param_name]['std']
    
    
    def get_param(self, data, param_name, preprocess=False):
        if param_name not in data:
            return None

        param = data[param_name]
        if param.ndim == 0:
            param = param.reshape(1,) if preprocess else param.item()
        param = self.divide_by_std(param, param_name) if preprocess else param
        
        return param

    
    def __getitem__(self, idx):
        """
        Retrieves a data sample by index.

        Args:
        idx (int): Index of the data sample.

        Returns:
        dict: Data sample with filename, coordinates, features, and labels.
        """
        data = np.load(self.data_files[idx], allow_pickle=True)
        
        run_number = self.get_param(data, 'run_number')
        event_id = self.get_param(data, 'event_id')
        true_hits = self.get_param(data, 'true_hits')
        reco_hits = self.get_param(data, 'reco_hits')
        reco_hits_true = self.get_param(data, 'reco_hits_true')
        in_neutrino_pdg = self.get_param(data, 'in_neutrino_pdg')
        in_neutrino_energy = self.get_param(data, 'in_neutrino_energy')
        out_lepton_pdg = self.get_param(data, 'out_lepton_pdg')
        out_lepton_momentum = self.get_param(data, 'out_lepton_momentum', preprocess=False)
        vis_sp_momentum = self.get_param(data, 'vis_sp_momentum', preprocess=False)
        jet_momentum = self.get_param(data, 'jet_momentum', preprocess=False)
        is_cc = self.get_param(data, 'is_cc')
        e_vis = self.get_param(data, 'e_vis', preprocess=True)
        pt_miss = self.get_param(data, 'pt_miss', preprocess=False)
        rear_cal_energy = self.get_param(data, 'rear_cal_energy', preprocess=True)
        rear_hcal_energy = self.get_param(data, 'rear_hcal_energy', preprocess=True)
        rear_mucal_energy = self.get_param(data, 'rear_mucal_energy', preprocess=True)
        faser_cal_energy = self.get_param(data, 'faser_cal_energy', preprocess=True)
        rear_hcal_modules = self.get_param(data, 'rear_hcal_modules', preprocess=True)
        faser_cal_modules = self.get_param(data, 'faser_cal_modules', preprocess=True)
        primary_vertex = data['primary_vertex']
        
        if not is_cc:
            # Fix jet momentum and no outgoing lepton momentum for NC events
            jet_momentum = jet_momentum + out_lepton_momentum
            out_lepton_momentum.fill(0)
        
        # momentum -> direction + magnitude
        out_lepton_momentum_mag, out_lepton_momentum_dir = self.decompose_momentum(out_lepton_momentum)
        out_lepton_momentum_mag = self.divide_by_std(out_lepton_momentum_mag, 'out_lepton_momentum_magnitude')
        jet_momentum_mag, jet_momentum_dir = self.decompose_momentum(jet_momentum)
        jet_momentum_mag = self.divide_by_std(jet_momentum_mag, 'jet_momentum_magnitude')
        
        # retrieve coordiantes and features (energy deposited)
        coords = reco_hits[:, :3]
        q = self.divide_by_std(reco_hits[:, 4].reshape(-1, 1), 'q')
        
        # process labels
        primlepton_labels = self.get_param(data, 'primlepton_labels')
        seg_labels = self.get_param(data, 'seg_labels')
        if self.load_seg:
            # load labels from pretrained model predictions
            file_name = self.data_files[idx].replace("events_v3.5_sample", "events_v3.5_seg_results")
            predictions = np.load(file_name)
            primlepton_labels_pred, seg_labels_pred = predictions['primlepton_labels'], predictions['seg_labels']
            
            # convert electromagnetic and hadronic probabilities to energy desposits (using truth info)
            seg_labels_pred[:, 1:] *= (seg_labels[:, 1:].sum(axis=1, keepdims=True) + 1e-8)            
            
            # predictions become labels
            primlepton_labels = primlepton_labels_pred
            seg_labels = seg_labels_pred
            
        flavour_label = self.pdg2label(in_neutrino_pdg, is_cc)
        primlepton_labels = primlepton_labels.reshape(-1, 1)
        seg_labels = seg_labels.reshape(-1, 3)

        augmented, feats = False, q
        if self.augmentations and np.random.rand() > 0.01:           
            # augmented event
            (
                coords, feats, (primlepton_labels, seg_labels),
                (out_lepton_momentum_dir, jet_momentum_dir, vis_sp_momentum)
            ) = self._augment(
                coords, feats, (primlepton_labels, seg_labels),
                (out_lepton_momentum_dir, jet_momentum_dir, vis_sp_momentum),
                primary_vertex,
            )
            
            augmented = True
        else:
            seg_labels = self.normalise_seg_labels(seg_labels)
            
        # voxelise coordinates and prepare global features
        coords = self.voxelise(coords)
        feats_global = np.concatenate([rear_cal_energy, rear_hcal_energy, rear_mucal_energy, 
                                       faser_cal_energy, rear_hcal_modules, faser_cal_modules])
            
        if augmented:
            # merge duplicated coordinates and finalise with augmentations
            coords, feats, primlepton_labels, seg_labels = self.aggregate_duplicate_coords(coords, feats, primlepton_labels, seg_labels)
            seg_labels = self.normalise_seg_labels(seg_labels)
            primlepton_labels = add_gaussian_noise(primlepton_labels)
            seg_labels = add_gaussian_noise(seg_labels)
            feats_global = add_noise_global_params(feats_global)
        
        # ptmiss
        pt_miss = np.sqrt(np.array([vis_sp_momentum[0]**2 + vis_sp_momentum[1]**2]))
        pt_miss = self.divide_by_std(pt_miss, 'pt_miss')

        # output
        output = {}
        if not self.train:
            output['run_number'] = run_number
            output['event_id'] = event_id
            output['primary_vertex'] = primary_vertex
            output['is_cc'] = is_cc
            output['in_neutrino_pdg'] = in_neutrino_pdg
            output['in_neutrino_energy'] = in_neutrino_energy
            output['out_lepton_momentum_dir'] = torch.from_numpy(out_lepton_momentum_dir).float()
        if self.stage1:
            output['primlepton_labels'] = torch.from_numpy(primlepton_labels).float()
            output['seg_labels'] = torch.from_numpy(seg_labels).float()
        else:
            feats = np.concatenate((feats, primlepton_labels, seg_labels), axis=1)
            output['flavour_label'] = torch.tensor([flavour_label]).long()
            output['e_vis'] = torch.from_numpy(e_vis).float()
            output['pt_miss'] = torch.from_numpy(pt_miss).float()
            output['out_lepton_momentum_mag'] = torch.from_numpy(out_lepton_momentum_mag).float()
            output['out_lepton_momentum_dir'] = torch.from_numpy(out_lepton_momentum_dir).float()
            output['jet_momentum_mag'] = torch.from_numpy(jet_momentum_mag).float()
            output['jet_momentum_dir'] = torch.from_numpy(jet_momentum_dir).float()
        output['coords'] = torch.from_numpy(coords.reshape(-1, 3)).float()
        output['feats'] = torch.from_numpy(feats).float()
        output['feats_global'] = torch.from_numpy(feats_global).float()

        return output
