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
from glob import glob
from torch.utils.data import Dataset
from utils import ini_argparse
from utils.augmentations import *


class SparseFASERCALDataset(Dataset):
    """
    A PyTorch Dataset for handling sparse FASERCal data.
    """
    def __init__(self, args):
        """
        Initialises the dataset.
        """
        self.root = args.dataset_path
        self.data_files = sorted(glob(f'{self.root}/*.npz'))

        # Configuration from args
        self.train = args.train
        self.stage1 = args.stage1
        self.augmentations_enabled = False
        self.standardize_input = args.standardize_input
        self.standardize_output = args.standardize_output
        self.preprocessing_input = args.preprocessing_input
        self.preprocessing_output = args.preprocessing_output
        self.mixup = args.mixup

        # Load metadata
        with open(os.path.join(self.root, "metadata.pkl"), "rb") as fd:
            self.metadata = pk.load(fd)
            for key in ['x', 'y', 'z']:
                self.metadata[key] = np.array(self.metadata[key])

        self.module_size = int((self.metadata['z'][:, 1] == 0).sum())
        self.num_modules = int(self.metadata['z'][:, 1].max() + 1)
        self.primary_vertices = None

    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_files)

        
    def calc_primary_vertices(self):
        self.primary_vertices = np.array([np.load(f)["primary_vertex"] for f in self.data_files])
        print("Primary vertices pre-loaded.")

    
    def _augment(self, coords, modules, feats, labels, momentums, global_feats, primary_vertex, transformations=None):
        if transformations is None:
            transformations = {}
            
        # Mirror
        if 'mirror' in transformations:
            flipped = transformations['mirror']
            coords, modules, momentums, global_feats['rear_cal_modules'], primary_vertex = apply_mirror(
                coords, modules, momentums, global_feats['rear_cal_modules'], 
                primary_vertex, self.metadata, flipped,
            )
        else:
            coords, modules, momentums, global_feats['rear_cal_modules'], primary_vertex, flipped = mirror(
                coords, modules, momentums, global_feats['rear_cal_modules'], 
                primary_vertex, self.metadata, selected_axes=['x', 'y'],
            )
            transformations['mirror'] = flipped

        # Rotation
        if 'rotate' in transformations:
            chosen_angles = transformations['rotate']
            coords, momentums, global_feats['rear_cal_modules'], primary_vertex = apply_rotate_90(
                coords, momentums, global_feats['rear_cal_modules'], 
                primary_vertex, self.metadata, chosen_angles,
            )
        else:
            coords, momentums, global_feats['rear_cal_modules'], primary_vertex, chosen_angles = rotate_90(
                coords, momentums, global_feats['rear_cal_modules'], 
                primary_vertex, self.metadata, selected_axes=['z'],
            )
            transformations['rotate'] = chosen_angles

        # Translation
        if 'translate' in transformations:
            shifts = transformations['translate']
            coords, modules, global_feats['rear_cal_modules'], primary_vertex = apply_translate(
                coords, modules, global_feats['rear_cal_modules'], 
                primary_vertex, self.metadata, shifts,
            )
        else:
            coords, modules, global_feats['rear_cal_modules'], primary_vertex, shifts = translate(
                coords, modules, global_feats['rear_cal_modules'], 
                primary_vertex, self.metadata, selected_axes=['x', 'y'],
            )
            transformations['translate'] = shifts

        # Scaling
        feats, momentums, global_feats, _ = scale_all_by_global_shift(
            feats, momentums, global_feats, std_dev=0.1,
        )

        return coords, modules, feats, labels, momentums, global_feats, primary_vertex, transformations

    
    def voxelise(self, coords, reverse=False):
        """
        Voxelises or de-voxelises coordinates based on detector geometry.
        Z-axis is handled differently due to non-uniform module spacing.
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
            
        return mapped


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

    
    def normalise_seg_labels(self, seg_labels, primlepton_labels, smoothing=0.0, eps=1e-8):
        """
        Normalizes segmentation labels and combines them into a final format.
        [ghost, em, had] -> [ghost, em_norm, had_norm, primlepton]
        """
        labels = seg_labels.copy().astype(np.float32)
        sum_vals = np.sum(labels[:, 1:], axis=1, keepdims=True)
        mask = sum_vals.squeeze() > eps
        
        # Normalize EM and Hadronic components to sum to (1 - ghost_fraction)
        labels[mask, 1:] = (1 - labels[mask, :1]) * (labels[mask, 1:] / sum_vals[mask])
        
        prim = primlepton_labels.reshape(-1, 1).astype(labels.dtype)
        result = np.concatenate([labels, prim], axis=1)
        
        # Zero out other components if it's a primary lepton
        prim_mask = prim.squeeze() == 1
        result[prim_mask, :3] = 0.0

        if smoothing > 0:
            return smooth_labels(result, smoothing=smoothing)
        
        return result

    
    def pdg2label(self, pdg, iscc):
        """Converts PDG ID to a classification label (0-3)."""
        if iscc:
            if pdg in [-12, 12]: return 0  # CC nue
            if pdg in [-14, 14]: return 1  # CC numu
            if pdg in [-16, 16]: return 2  # CC nutau
        return 3  # NC

    
    def decompose_momentum(self, momentum):
        """Splits 3D momentum vectors into magnitude and direction."""
        momentum = np.atleast_2d(momentum)
        magnitudes = np.linalg.norm(momentum, axis=1, keepdims=True)
        directions = np.divide(momentum, magnitudes, where=magnitudes != 0)
        
        is_single_vector = magnitudes.shape[0] == 1
        return magnitudes[0] if is_single_vector else magnitudes.flatten(), \
               directions[0] if is_single_vector else directions

        
    def preprocess(self, x, param_name, preprocessing=None, standardize=None):
        """Applies a sequence of preprocessing steps (e.g., log, z-score)."""
        x = np.atleast_1d(x)
        internal_name = param_name
        
        # Apply non-linear transformations first
        if preprocessing == "sqrt":
            if np.any(x < 0): raise ValueError(f"{param_name}: negative values cannot take sqrt")
            x = np.sqrt(x)
            internal_name += "_sqrt"
        elif preprocessing == "log":
            if np.any(x <= -1): raise ValueError(f"{param_name}: values <= -1 cannot take log1p")
            x = np.log1p(x)
            internal_name += "_log1p"

        # Apply standardization
        if standardize:
            stats = self.metadata[internal_name]
            std = stats.get("std", 1.0)
            if std == 0: raise ValueError(f"{internal_name}: std is zero in metadata")

            if standardize == "z-score":
                x = (x - stats["mean"]) / std
            elif standardize == "unit-var":
                x = x / std
            else: # Min-max scaling
                rng = stats["max"] - stats["min"]
                if rng == 0: raise ValueError(f"{param_name}: max and min are equal")
                x = (x - stats["min"]) / rng
        return x


    def _prepare_event(self, idx, transformations=None):
        """
        Loads and processes a single event up through augmentations.
        Returns a dict of intermediate arrays.
        """
        data = np.load(self.data_files[idx], allow_pickle=True)
        reco_hits = data['reco_hits']
        primary_vertex = data['primary_vertex']
        is_cc = data['is_cc']
        in_neutrino_pdg = data['in_neutrino_pdg']
        vis_sp_momentum = data['vis_sp_momentum']
        seg_labels_raw = data['seg_labels']
        primlepton_labels = data['primlepton_labels']
        out_lepton_momentum = data['out_lepton_momentum']
        jet_momentum = data['jet_momentum']
        charm = data['charm']
        global_feats = {
            'e_vis':             data['e_vis'],
            'faser_cal_energy':  data['faser_cal_energy'],
            'rear_cal_energy':   data['rear_cal_energy'],
            'rear_hcal_energy':  data['rear_hcal_energy'],
            'rear_mucal_energy': data['rear_mucal_energy'],
            'faser_cal_modules': data['faser_cal_modules'],
            'rear_cal_modules':  data['rear_cal_modules'],
            'rear_hcal_modules': data['rear_hcal_modules'],
        }
        # initial transformations
        coords = reco_hits[:, :3]
        q = reco_hits[:, 4].reshape(-1, 1)
        orig_coords = coords.copy()
        module_idx = np.searchsorted(self.metadata['z'][:, 0], coords[:, 2])
        modules = self.metadata['z'][module_idx, 1]
        coords = self.voxelise(coords)
        primary_vertex = self.voxelise(primary_vertex)

        # augmentations (if applicable)
        if self.train and self.augmentations_enabled:
            coords, modules, q, (primlepton_labels, seg_labels_raw), \
            (out_lepton_momentum, jet_momentum, vis_sp_momentum), \
            global_feats, _, transformations = self._augment(
                coords, modules, q, (primlepton_labels, seg_labels_raw),
                (out_lepton_momentum, jet_momentum, vis_sp_momentum),
                global_feats, primary_vertex, transformations
            )
            charm = smooth_labels(np.array([charm]), smoothing=0.1)
            flavour_label = smooth_labels(
                np.array([self.pdg2label(in_neutrino_pdg, is_cc)]),
                smoothing=0.1,
                num_classes=4
            )
            seg_labels = self.normalise_seg_labels(seg_labels_raw, primlepton_labels, smoothing=0.1)
        else:
            flavour_label = np.array([self.pdg2label(in_neutrino_pdg, is_cc)])
            seg_labels = self.normalise_seg_labels(seg_labels_raw, primlepton_labels)
            if not is_cc:
                out_lepton_momentum.fill(0)

        # decompose momentum
        out_lepton_magnitude, out_lepton_dir = self.decompose_momentum(out_lepton_momentum)
        jet_magnitude, jet_dir = self.decompose_momentum(jet_momentum)

        return {
            'coords': coords,
            'modules': modules,
            'q': q,
            'seg_labels': seg_labels,
            'primlepton_labels': primlepton_labels,
            'out_lepton_momentum_mag': out_lepton_magnitude,
            'out_lepton_momentum_dir': out_lepton_dir,
            'jet_momentum_mag': jet_magnitude,
            'jet_momentum_dir': jet_dir,
            'global_feats': global_feats,
            'flavour_label': flavour_label,
            'charm': charm,
            'vis_sp_momentum': vis_sp_momentum,
            'e_vis': global_feats['e_vis'],
            'primary_vertex': primary_vertex,
            'transformations': transformations,
        }

    
    def _finalise_event(self, event):
        """
        Applies preprocessing and converts arrays into the final torch tensors output.
        """
        # Preprocess features
        feats = self.preprocess(event['q'], 'q', self.preprocessing_input, self.standardize_input)
        event_hits = self.preprocess(
            len(event['q']), 'event_hits', self.preprocessing_input, self.standardize_input
        )
        module_hits = np.bincount(event['modules'].astype(int), minlength=self.num_modules)
        module_hits = self.preprocess(
            module_hits, 'module_hits', self.preprocessing_input, self.standardize_input
        )
        feats_global = np.concatenate([
            event_hits,
            self.preprocess(event['global_feats']['faser_cal_energy'], 'faser_cal_energy', self.preprocessing_input, self.standardize_input),
            self.preprocess(event['global_feats']['rear_cal_energy'], 'rear_cal_energy', self.preprocessing_input, self.standardize_input),
            self.preprocess(event['global_feats']['rear_hcal_energy'], 'rear_hcal_energy', self.preprocessing_input, self.standardize_input),
            self.preprocess(event['global_feats']['rear_mucal_energy'], 'rear_mucal_energy', self.preprocessing_input, self.standardize_input)
        ])
        faser_mod = self.preprocess(event['global_feats']['faser_cal_modules'], 'faser_cal_modules', self.preprocessing_input, self.standardize_input)
        rear_cal_mod = self.preprocess(event['global_feats']['rear_cal_modules'], 'rear_cal_modules', self.preprocessing_input, self.standardize_input)
        rear_hcal_mod = self.preprocess(event['global_feats']['rear_hcal_modules'], 'rear_hcal_modules', self.preprocessing_input, self.standardize_input)

        # Preprocess outputs
        pt_miss = np.sqrt(event['vis_sp_momentum'][0]**2 + event['vis_sp_momentum'][1]**2)
        pt_miss = self.preprocess(pt_miss, 'pt_miss', self.preprocessing_output, self.standardize_output)
        e_vis = self.preprocess(event['e_vis'], 'e_vis', self.preprocessing_output, self.standardize_output)
        out_mag = self.preprocess(event['out_lepton_momentum_mag'], 'out_lepton_momentum_magnitude', self.preprocessing_output, self.standardize_output)
        jet_mag = self.preprocess(event['jet_momentum_mag'], 'jet_momentum_magnitude', self.preprocessing_output, self.standardize_output)

        # Assemble output
        output = {
            'coords': torch.from_numpy(event['coords']).float(),
            'modules': torch.from_numpy(event['modules']).long(),
            'feats': torch.from_numpy(feats).float(),
            'feats_global': torch.from_numpy(feats_global).float(),
            'faser_cal_modules': torch.from_numpy(faser_mod).float(),
            'rear_cal_modules': torch.from_numpy(rear_cal_mod).float(),
            'rear_hcal_modules': torch.from_numpy(rear_hcal_mod).float(),
            'flavour_label': torch.from_numpy(event['flavour_label']),
        }
        if self.stage1:
            output['seg_labels'] = torch.from_numpy(event['seg_labels']).float()
        if not self.train or not self.stage1:
            output.update({
                'charm': torch.from_numpy(np.atleast_1d(event['charm'])).float(),
                'e_vis': torch.from_numpy(np.atleast_1d(e_vis)).float(),
                'pt_miss': torch.from_numpy(np.atleast_1d(pt_miss)).float(),
                'out_lepton_momentum_mag': torch.from_numpy(np.atleast_1d(out_mag)).float(),
                'out_lepton_momentum_dir': torch.from_numpy(event['out_lepton_momentum_dir']).float(),
                'jet_momentum_mag': torch.from_numpy(np.atleast_1d(jet_mag)).float(),
                'jet_momentum_dir': torch.from_numpy(event['jet_momentum_dir']).float(),
            })
        return output


    def _mixup(self, event1, event2, alpha=0.8):
        """
        Performs mixup of two prepared events with weight alpha, handling overlapping voxels.
        Coordinates and module indices are concatenated; features and labels are mixed.
        """
        return event1
        

    def __getitem__(self, idx):
        event = self._prepare_event(idx)
        '''
        if self.train and self.augmentations_enabled and not self.stage1:
            # find candidate with similar vertex for mixup
            primary_vertex = self.primary_vertices[idx]
            candidate_indices = np.random.randint(len(self), size=5000)
            candidate_vertices = self.primary_vertices[candidate_indices]
            dists = np.linalg.norm(candidate_vertices - primary_vertex, axis=1)
            within_threshold = np.where(dists < 50)[0]
            if within_threshold.size > 0:
                selected_idx = candidate_indices[within_threshold[0]]
            else:
                selected_idx = candidate_indices[np.argmin(dists)]
    
            event2 = self._prepare_event(selected_idx, event['transformations'])
            event = self._mixup(event, event2, alpha=self.mixup)
        '''
        return self._finalise_event(event)
