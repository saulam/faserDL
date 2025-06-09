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
        self.augmentations_enabled = False
        self.is_v5 = True if 'v5' in args.dataset_path else False 
        self.total_events = self.__len__
        with open(self.root + "/metadata.pkl", "rb") as fd:
            self.metadata = pk.load(fd)
            self.metadata['x'] = np.array(self.metadata['x'])
            self.metadata['y'] = np.array(self.metadata['y'])
            self.metadata['z'] = np.array(self.metadata['z'])
        self.module_size = int((self.metadata['z'][:,1]==0).sum())
        self.standardize_input = args.standardize_input
        self.standardize_output = args.standardize_output
        self.preprocessing_input = args.preprocessing_input
        self.preprocessing_output = args.preprocessing_output

            
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
        return sorted(glob(f'{self.processed_dir}/*.npz'))
    
    
    def __len__(self):
        """
        Returns the total number of data files.

        Returns:
        int: Number of data files.
        """
        return len(self.data_files)

    
    def _augment(self, coords_ori, modules_ori, feats_ori, labels_ori, dirs_ori, primary_vertex_ori):
        coords, modules, feats = coords_ori.copy(), modules_ori.copy(), feats_ori.copy()
        primary_vertex = primary_vertex_ori.copy()
        labels = [x.copy() for x in labels_ori]
        dirs = [x.copy() for x in dirs_ori]

        # mirror
        coords, modules, dirs, primary_vertex = mirror(coords, modules, dirs, primary_vertex, self.metadata, selected_axes=['x', 'y'])
        # rotate
        coords, dirs, primary_vertex = rotate_90(coords, dirs, primary_vertex, self.metadata, selected_axes=['z'])
        # translate
        coords, modules, primary_vertex = translate(coords, modules, primary_vertex, self.metadata, selected_axes=['x', 'y', 'z'])
        # drop voxels
        coords, modules, feats, labels = drop(coords, modules, feats, labels, std_dev=0.1)
        # shift feature values
        feats = shift_q_gaussian(feats, std_dev=0.05)
        # keep within limits
        #coords, feats, labels = self.within_limits(coords, feats, labels, voxelised=True, mask_axes=[2])
        if coords.shape[0] < 2:
            return coords_ori, modules_ori, feats_ori, labels_ori, dirs_ori, primary_vertex_ori

        return coords, modules, feats, labels, dirs, primary_vertex
    
    
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

        return mapped#.round()  # round needed only for augmentations
        
        
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


    def preprocess(self, x, param_name, preprocessing=None, standardize=False):
        if np.ndim(x) == 0:
            x = np.atleast_1d(x)

        internal_name = param_name
        if preprocessing == "sqrt":
            if np.any(x < 0):
                raise ValueError(f"{param_name}: negative values cannot take sqrt")
            x = np.sqrt(x)
            internal_name = param_name + "_sqrt"
        elif preprocessing == "log":
            if np.any(x <= -1):
                raise ValueError(f"{param_name}: values <= -1 cannot take log1p")
            x = np.log1p(x)
            internal_name = param_name + "_log1p"

        if standardize:
            stats = self.metadata[internal_name]
            if stats["std"] == 0:
                raise ValueError(f"{internal_name}: std is zero in metadata")
            x = (x - stats["mean"]) / stats["std"]

        return x


    def unpreprocess(self, x, param_name, preprocessing=None, standardize=False):
        if np.ndim(x) == 0 or (isinstance(x, np.ndarray) and x.shape == ()):
            x = np.atleast_1d(x)

        internal_name = param_name
        if preprocessing == "sqrt":
            internal_name = param_name + "_sqrt"
        elif preprocessing == "log":
            internal_name = param_name + "_log1p"

        if standardize:
            stats = self.metadata[internal_name]
            if stats["std"] == 0:
                raise ValueError(f"{internal_name}: std is zero in metadata")
            x = x * stats["std"] + stats["mean"]

        if preprocessing == "sqrt":
            x = x ** 2
        elif preprocessing == "log":
            x = np.expm1(x)

        return x
        

    def get_param(self, data, param_name, preprocessing=None, standardize=False, suffix=""):
        if param_name not in data:
            return None

        param = data[param_name]
        param = self.preprocess(param, param_name + suffix, preprocessing, standardize)

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
        out_lepton_momentum = self.get_param(data, 'out_lepton_momentum')
        vis_sp_momentum = self.get_param(data, 'vis_sp_momentum')
        jet_momentum = self.get_param(data, 'jet_momentum')
        is_cc = self.get_param(data, 'is_cc')
        charm = self.get_param(data, 'charm')
        e_vis = self.get_param(data, 'e_vis', preprocessing=self.preprocessing_output, standardize=self.standardize_output, 
                               suffix='_cc' if is_cc else '_nc')
        pt_miss = self.get_param(data, 'pt_miss', preprocessing=self.preprocessing_output, standardize=self.standardize_output)
        faser_cal_energy = self.get_param(data, 'faser_cal_energy', preprocessing=self.preprocessing_input, standardize=self.standardize_input)
        faser_cal_modules = self.get_param(data, 'faser_cal_modules', preprocessing=self.preprocessing_input, standardize=self.standardize_input)
        rear_cal_energy = self.get_param(data, 'rear_cal_energy', preprocessing=self.preprocessing_input, standardize=self.standardize_input)
        rear_cal_modules = self.get_param(data, 'rear_cal_modules', preprocessing=self.preprocessing_input, standardize=self.standardize_input)
        rear_hcal_energy = self.get_param(data, 'rear_hcal_energy', preprocessing=self.preprocessing_input, standardize=self.standardize_input)
        rear_hcal_modules = self.get_param(data, 'rear_hcal_modules', preprocessing=self.preprocessing_input, standardize=self.standardize_input)
        rear_mucal_energy = self.get_param(data, 'rear_mucal_energy', preprocessing=self.preprocessing_input, standardize=self.standardize_input)
        
        primary_vertex = data['primary_vertex']
        if not is_cc:
            out_lepton_momentum.fill(0)
        
        # momentum -> direction + magnitude
        out_lepton_momentum_mag, out_lepton_momentum_dir = self.decompose_momentum(out_lepton_momentum)
        out_lepton_momentum_mag = self.preprocess(out_lepton_momentum_mag, 'out_lepton_momentum_magnitude', 
                                                  preprocessing=self.preprocessing_output, standardize=self.standardize_output)
        jet_momentum_mag, jet_momentum_dir = self.decompose_momentum(jet_momentum)
        jet_momentum_mag = self.preprocess(jet_momentum_mag, 'jet_momentum_magnitude', 
                                           preprocessing=self.preprocessing_output, standardize=self.standardize_output)

        # retrieve coordiantes and features (energy deposited)
        coords = reco_hits[:, :3]
        q = self.preprocess(reco_hits[:, 4].reshape(-1, 1), 'q', preprocessing=self.preprocessing_input, standardize=self.standardize_input)

        # look-up hit modules
        module_idx = np.searchsorted(self.metadata['z'][:, 0], coords[:, 2])
        modules = self.metadata['z'][module_idx, 1]
        
        # process labels
        primlepton_labels = self.get_param(data, 'primlepton_labels')
        seg_labels = self.get_param(data, 'seg_labels')
        
        # voxelise coordinates and prepare global features and labels
        coords = self.voxelise(coords)
        primary_vertex = self.voxelise(primary_vertex)
        
        feats_global = np.concatenate([faser_cal_energy, rear_cal_energy, rear_hcal_energy, rear_mucal_energy])   
        flavour_label = self.pdg2label(in_neutrino_pdg, is_cc)
        primlepton_labels = primlepton_labels.reshape(-1, 1)
        seg_labels = seg_labels.reshape(-1, 3)

        if self.load_seg:
            # load labels from pretrained model predictions
            version = "v5.1" if self.is_v5 else "v3.5"
            file_name = self.data_files[idx].replace(f"events_{version}", f"events_{version}_seg_results")
            predictions = np.load(file_name)
            primlepton_labels_pred, seg_labels_pred = predictions['out_primlepton'], predictions['out_seg']
            
            # convert electromagnetic and hadronic probabilities to energy desposits (using truth info)
            seg_labels_pred[:, 1:] *= (seg_labels[:, 1:].sum(axis=1, keepdims=True) + 1e-8)            
            
            # predictions become labels
            primlepton_labels = primlepton_labels_pred
            seg_labels = seg_labels_pred

        # relative coords to each module
        coords[:, 2] = coords[:, 2] % self.module_size

        augmented, feats = False, q
        if self.augmentations_enabled: 
            # augmented event
            (
                coords, modules, feats, (primlepton_labels, seg_labels),
                (out_lepton_momentum_dir, jet_momentum_dir, vis_sp_momentum), primary_vertex
            ) = self._augment(
                coords, modules, feats, (primlepton_labels, seg_labels),
                (out_lepton_momentum_dir, jet_momentum_dir, vis_sp_momentum),
                primary_vertex
            )
            augmented = True
        else:
            seg_labels = self.normalise_seg_labels(seg_labels)
          
        if augmented:
            # merge duplicated coordinates and finalise with augmentations
            #coords, feats, primlepton_labels, seg_labels = self.aggregate_duplicate_coords(coords, feats, primlepton_labels, seg_labels)
            seg_labels = self.normalise_seg_labels(seg_labels)
            feats_global = shift_q_gaussian(feats_global, std_dev=0.05)
            faser_cal_modules = shift_q_gaussian(faser_cal_modules, std_dev=0.05)
            rear_cal_modules = shift_q_gaussian(rear_cal_modules, std_dev=0.05)
            rear_hcal_modules = shift_q_gaussian(rear_hcal_modules, std_dev=0.05)
        
        # ptmiss
        pt_miss = np.sqrt(np.array([vis_sp_momentum[0]**2 + vis_sp_momentum[1]**2]))
        pt_miss = self.preprocess(pt_miss, 'pt_miss', preprocessing=self.preprocessing_output, standardize=self.standardize_output)

        # output
        output = {}
        if not self.train:
            output['run_number'] = run_number
            output['event_id'] = event_id
            output['primary_vertex'] = primary_vertex
            output['in_neutrino_pdg'] = in_neutrino_pdg
            output['in_neutrino_energy'] = in_neutrino_energy
            output['vis_sp_momentum'] = vis_sp_momentum
        if not self.train or not self.stage1:
            output['charm'] = torch.tensor(charm).float()
            output['e_vis'] = torch.from_numpy(e_vis).float()
            output['pt_miss'] = torch.from_numpy(pt_miss).float()
            output['out_lepton_momentum_mag'] = torch.from_numpy(out_lepton_momentum_mag).float()
            output['out_lepton_momentum_dir'] = torch.from_numpy(out_lepton_momentum_dir).float()
            output['jet_momentum_mag'] = torch.from_numpy(jet_momentum_mag).float()
            output['jet_momentum_dir'] = torch.from_numpy(jet_momentum_dir).float()
        if self.stage1:
            output['primlepton_labels'] = torch.from_numpy(primlepton_labels).float()
            output['seg_labels'] = torch.from_numpy(seg_labels).float()
        elif self.load_seg:
            feats = np.concatenate((feats, primlepton_labels, seg_labels), axis=1)
        output['flavour_label'] = torch.tensor([flavour_label]).long()
        output['coords'] = torch.from_numpy(coords.reshape(-1, 3)).float()
        output['modules'] = torch.from_numpy(modules).long()
        output['feats'] = torch.from_numpy(feats).float()
        output['feats_global'] = torch.from_numpy(feats_global).float()
        output['faser_cal_modules'] = torch.from_numpy(faser_cal_modules).float()
        output['rear_cal_modules'] = torch.from_numpy(rear_cal_modules).float()
        output['rear_hcal_modules'] = torch.from_numpy(rear_hcal_modules).float()
        output['is_cc'] = torch.from_numpy(is_cc).float()

        return output

