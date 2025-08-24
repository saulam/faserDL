"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 07.25

Description: Dataset file.
"""

import io
import pickle as pk
import numpy as np
import torch
import webdataset as wds
from glob import glob
from torch.utils.data import Dataset, IterableDataset
from utils.augmentations import augment, smooth_labels


class SparseFASERCALDataset(Dataset):
    """
    A PyTorch Dataset for handling sparse FASERCal data.
    """
    def __init__(self, args):
        """
        Initialises the dataset.
        """
        # Configuration from args
        self.train = args.train
        self.stage1 = args.stage1
        self.augmentations_enabled = False
        self.preprocessing_input = args.preprocessing_input
        self.preprocessing_output = args.preprocessing_output
        self.label_smoothing = args.label_smoothing
        self.mixup_alpha = args.mixup_alpha
        self.crop_around_vertex = args.crop_around_vertex
        self.epoch = 0

        # Load metadata
        with open(args.metadata_path, "rb") as fd:
            self.metadata = pk.load(fd)
            for key in ['x', 'y', 'z']:
                self.metadata[key] = np.array(self.metadata[key])

        self.module_size = int((self.metadata['z'][:, 1] == 0).sum())
        self.num_modules = int(self.metadata['z'][:, 1].max() + 1)

    
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
        if isinstance(labels, (list, tuple)):
            labels_filtered = [x[mask] for x in labels]
        else:
            labels_filtered = labels[mask] if labels is not None else None
        
        return coords_filtered, feats_filtered, labels_filtered

    
    def normalise_seg_labels(self, seg_labels, smoothing=0.0, eps=1e-8):
        """
        Normalises segmentation labels and combines them into a final format.
        """
        labels = seg_labels.copy().astype(np.float32)
        sum_vals = np.sum(labels[:, 1:], axis=1, keepdims=True)
        mask = sum_vals.squeeze() > eps
        
        # Normalize EM and Hadronic components to sum to (1 - ghost_fraction)
        labels[mask, 1:] = (1 - labels[mask, :1]) * (labels[mask, 1:] / sum_vals[mask])

        if smoothing > 0:
            return smooth_labels(labels, smoothing=smoothing)
        
        return labels

    
    def pdg2label(self, pdg, iscc):
        """Converts PDG ID to a classification label (0-3)."""
        if iscc:
            if pdg in [-12, 12]: return 0  # CC nue
            if pdg in [-14, 14]: return 1  # CC numu
            if pdg in [-16, 16]: return 2  # CC nutau
        return 0  # NC (not used)


    def decompose_momentum(self, momentum):
        """Splits 3D momentum vectors into magnitude and direction."""
        if not isinstance(momentum, torch.Tensor):
            momentum = torch.as_tensor(momentum)
        momentum = torch.atleast_2d(momentum)
        magnitudes = torch.linalg.norm(momentum, dim=1, keepdim=True)
        directions = torch.where(magnitudes != 0, momentum / magnitudes, torch.zeros_like(momentum))
        
        is_single_vector = magnitudes.shape[0] == 1
        return (magnitudes[0] if is_single_vector else magnitudes.flatten(),
                directions[0] if is_single_vector else directions)


    def reconstruct_momentum(self, magnitude, direction):
        """Given magnitude and direction, reconstruct the original momentum vector."""
        if not isinstance(magnitude, torch.Tensor):
            magnitude = torch.as_tensor(magnitude)
        if not isinstance(direction, torch.Tensor):
            direction = torch.as_tensor(direction)
        direction = torch.atleast_2d(direction)
        if magnitude.ndim == 0:
            magnitude = torch.atleast_1d(magnitude)
        if magnitude.device != direction.device:
            magnitude = magnitude.to(direction.device)
        if magnitude.dtype != direction.dtype:
            magnitude = magnitude.to(direction.dtype)
        momentum = magnitude[:, None] * direction
        return momentum[0] if magnitude.shape[0] == 1 else momentum


    def robust_standardize(self, x, params):
        k   = params["k"]
        mu  = params["mu"]
        sig = params["sigma"]
        tname = params["transform"]

        lookup = {
            "identity": lambda u: u,
            "log1p":    lambda u: torch.log1p(u),
            "sqrt":     lambda u: torch.sqrt(u),
        }
        if tname not in lookup:
            raise ValueError(f"Unknown transform: {tname}")

        f = lookup[tname]
        u = f(x / k)
        z = (u - mu) / sig
        return z


    def robust_unstandardize(self, z, params):
        k   = params["k"]
        mu  = params["mu"]
        sig = params["sigma"]
        tname = params["transform"]

        inv_lookup = {
            "identity": lambda u: u,
            "log1p":    lambda u: torch.expm1(u),
            "sqrt":     lambda u: torch.square(u),
        }
        if tname not in inv_lookup:
            raise ValueError(f"Unknown transform: {tname}")

        u = z * sig + mu
        x_over_k = inv_lookup[tname](u)
        return x_over_k * k


    def preprocess(self, x, param_name, preprocessing=None):
        """Applies a sequence of preprocessing steps (e.g., log, z-score)."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = torch.atleast_1d(x)
        internal_name = param_name
        
        if preprocessing is not None:
            if preprocessing == "sqrt":
                internal_name += "_sqrt"
            elif preprocessing == "log":
                internal_name += "_log1p"

            stats = self.metadata[internal_name]
            x = self.robust_standardize(x, params=stats)
            
        return x


    def unpreprocess(self, x, param_name, preprocessing=None):
        """Reverses the preprocessing."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.ndim == 0:
            x = torch.atleast_1d(x)

        internal_name = param_name
        if preprocessing is not None:
            if preprocessing == "sqrt":
                internal_name = param_name + "_sqrt"
            elif preprocessing == "log":
                internal_name = param_name + "_log1p"

            stats = self.metadata[internal_name]
            x = self.robust_unstandardize(x, params=stats)

        return x


    def _prepare_event(self, data, transformations=None):
        """
        Loads and processes a single event up through augmentations.
        Returns a dict of intermediate arrays.
        """
        run_number = data['run_number'].item()
        event_id = data['event_id'].item()
        reco_hits = data['reco_hits']
        primary_vertex = data['primary_vertex']
        is_cc = data['is_cc']
        in_neutrino_pdg = data['in_neutrino_pdg']
        in_neutrino_energy = data['in_neutrino_energy']
        vis_sp_momentum = data['vis_sp_momentum']
        seg_labels_raw = data['seg_labels']
        out_lepton_momentum = data['out_lepton_momentum']
        jet_momentum = data['jet_momentum']
        tauvis_momentum = data['tauvis_momentum']
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

        if is_cc and in_neutrino_pdg in [-16, 16]:  # nutau:
            out_lepton_momentum = tauvis_momentum

        # augmentations (if applicable)
        if self.train and self.augmentations_enabled:
            coords, modules, q, (seg_labels_raw,), \
            (out_lepton_momentum, jet_momentum, vis_sp_momentum), \
            global_feats, _, transformations = augment(
                coords, modules, q, (seg_labels_raw,),
                (out_lepton_momentum, jet_momentum, vis_sp_momentum),
                global_feats, primary_vertex, self.metadata, transformations, self.stage1
            )
            charm = smooth_labels(np.array([charm]), smoothing=self.label_smoothing)
            flavour_label = smooth_labels(
                np.array([self.pdg2label(in_neutrino_pdg, is_cc)]),
                smoothing=self.label_smoothing,
                num_classes=3
            )
            is_cc = smooth_labels(np.array([float(is_cc)]), smoothing=self.label_smoothing)
            seg_labels = self.normalise_seg_labels(seg_labels_raw, smoothing=self.label_smoothing)
        else:
            flavour_label = np.array([self.pdg2label(in_neutrino_pdg, is_cc)])
            seg_labels = self.normalise_seg_labels(seg_labels_raw)
            if not is_cc:
                out_lepton_momentum.fill(0)

        return {
            'run_number': run_number,
            'event_id': event_id,
            'coords': coords,
            'modules': modules,
            'q': q,
            'seg_labels': seg_labels,
            'vis_sp_momentum': vis_sp_momentum,
            'out_lepton_momentum': out_lepton_momentum,
            'jet_momentum': jet_momentum,
            'global_feats': global_feats,
            'flavour_label': flavour_label,
            'charm': charm,
            'vis_sp_momentum': vis_sp_momentum,
            'e_vis': global_feats['e_vis'],
            'primary_vertex': primary_vertex,
            'transformations': transformations,
            'in_neutrino_pdg': in_neutrino_pdg,
            'in_neutrino_energy': in_neutrino_energy,
            'is_cc': is_cc,
        }


    def _mixup(self, event1, event2, alpha=0.8):
        """
        Performs mixup of two prepared events with weight alpha, handling overlapping voxels.
        Coordinates and module indices are concatenated; features and labels are mixed.
        """
        lam_mix = np.random.beta(alpha, alpha)
        
        # event1
        coords1 = event1['coords']
        modules1 = event1['modules']
        q1_raw = event1['q']
        seg1_raw = event1['seg_labels']
        
        # event2
        coords2 = event2['coords']
        modules2 = event2['modules']
        q2_raw = event2['q']
        seg2_raw = event2['seg_labels']

        # weight features and labels: event1=lam_mix, event2=1-lam_mix
        q1 = q1_raw * lam_mix
        q2 = q2_raw * (1.0 - lam_mix)
        seg1 = seg1_raw * lam_mix
        seg2 = seg2_raw * (1.0 - lam_mix)

        # concatenate all voxel-level arrays
        coords = np.concatenate([coords1, coords2], axis=0)
        modules = np.concatenate([modules1, modules2], axis=0)
        q = np.concatenate([q1, q2], axis=0)
        seg = np.concatenate([seg1, seg2], axis=0)
        
        # deduplicate overlapping voxels: aggregate q and seg
        keys = np.concatenate([coords, modules.reshape(-1,1)], axis=1)
        unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
        new_coords = unique_keys[:, :3]
        new_modules = unique_keys[:, 3].astype(int)
        new_q = np.zeros((unique_keys.shape[0], q.shape[1]), dtype=q.dtype)
        new_seg = np.zeros((unique_keys.shape[0], seg.shape[1]), dtype=seg.dtype)
        for i in range(q.shape[1]):
            new_q[:, i] = np.bincount(inv, weights=q[:, i], minlength=unique_keys.shape[0])
        for j in range(seg.shape[1]):
            new_seg[:, j] = np.bincount(inv, weights=seg[:, j], minlength=unique_keys.shape[0])
        coords, modules, q, seg = new_coords, new_modules, new_q, new_seg

        # mix global features
        gf1, gf2 = event1['global_feats'], event2['global_feats']
        mixed_gf = {}
        for key in ['faser_cal_energy','rear_cal_energy','rear_hcal_energy','rear_mucal_energy']:
            mixed_gf[key] = gf1[key]*lam_mix + gf2[key]*(1-lam_mix)
        for key in ['faser_cal_modules','rear_cal_modules','rear_hcal_modules']:
            mixed_gf[key] = gf1[key]*lam_mix + gf2[key]*(1-lam_mix)
        
        # mix scalars
        mixed_vis_sp     = event1['vis_sp_momentum']     * lam_mix + event2['vis_sp_momentum']     * (1-lam_mix)
        mixed_e_vis      = event1['e_vis']               * lam_mix + event2['e_vis']               * (1-lam_mix)
        mixed_flavour    = event1['flavour_label']       * lam_mix + event2['flavour_label']       * (1-lam_mix)
        mixed_charm      = event1['charm']               * lam_mix + event2['charm']               * (1-lam_mix)
        mixed_vis_sp_mom = event1['vis_sp_momentum']     * lam_mix + event2['vis_sp_momentum']     * (1-lam_mix)
        mixed_lep_mom    = event1['out_lepton_momentum'] * lam_mix + event2['out_lepton_momentum'] * (1-lam_mix)
        mixed_jet        = event1['jet_momentum']        * lam_mix + event2['jet_momentum']        * (1-lam_mix)

        # Filter out voxels outside the detector
        coords, q, seg = self.within_limits(coords, feats=q, labels=seg, voxelised=True)
        
        mixed_event = {
            'coords': coords,
            'modules': modules,
            'q': q,
            'seg_labels': seg,
            'global_feats': mixed_gf,
            'flavour_label': mixed_flavour,
            'vis_sp_momentum': mixed_vis_sp_mom,
            'out_lepton_momentum': mixed_lep_mom,
            'jet_momentum': mixed_jet,
            'vis_sp_momentum': mixed_vis_sp,
            'charm': mixed_charm,
            'e_vis': mixed_e_vis,
        }
        return mixed_event

    
    def _finalise_event(self, event):
        """
        Applies preprocessing and converts arrays into the final torch tensors output.
        """
        # Preprocess features
        feats = self.preprocess(event['q'], 'q', self.preprocessing_input)
        event_hits = self.preprocess(len(event['q']), 'event_hits', self.preprocessing_input)
        feats_global = torch.cat([
            event_hits,
            self.preprocess(event['global_feats']['faser_cal_energy'], 'faser_cal_energy', self.preprocessing_input),
            self.preprocess(event['global_feats']['rear_cal_energy'], 'rear_cal_energy', self.preprocessing_input),
            self.preprocess(event['global_feats']['rear_hcal_energy'], 'rear_hcal_energy', self.preprocessing_input),
            self.preprocess(event['global_feats']['rear_mucal_energy'], 'rear_mucal_energy', self.preprocessing_input)
        ])
        faser_mod = self.preprocess(event['global_feats']['faser_cal_modules'], 'faser_cal_modules', self.preprocessing_input)
        rear_cal_mod = self.preprocess(event['global_feats']['rear_cal_modules'], 'rear_cal_modules', self.preprocessing_input)
        rear_hcal_mod = self.preprocess(event['global_feats']['rear_hcal_modules'], 'rear_hcal_modules', self.preprocessing_input)

        vis_sp_momentum = event['vis_sp_momentum']
        out_lepton_momentum = event['out_lepton_momentum']
        jet_momentum = event['jet_momentum']

        # Decompose momentum
        vis_magnitude, vis_dir = self.decompose_momentum(vis_sp_momentum)
        out_lepton_magnitude, out_lepton_dir = self.decompose_momentum(out_lepton_momentum)
        jet_magnitude, jet_dir = self.decompose_momentum(jet_momentum)

        # Preprocess outputs
        #pt_miss = torch.tensor([np.sqrt(event['vis_sp_momentum'][0]**2 + event['vis_sp_momentum'][1]**2)])
        #pt_miss = self.preprocess(pt_miss, 'pt_miss', self.preprocessing_output)
        #e_vis = self.preprocess(event['e_vis'], 'e_vis', self.preprocessing_output)
        vis_sp_momentum = self.preprocess(vis_sp_momentum, 'vis_sp_momentum')
        out_lepton_momentum = self.preprocess(out_lepton_momentum, 'out_lepton_momentum')
        jet_momentum = self.preprocess(jet_momentum, 'jet_momentum')
        #vis_mag = self.preprocess(vis_magnitude, 'vis_sp_momentum_magnitude', self.preprocessing_output)
        #out_mag = self.preprocess(out_lepton_magnitude, 'out_lepton_momentum_magnitude', self.preprocessing_output)
        #jet_mag = self.preprocess(jet_magnitude, 'jet_momentum_magnitude', self.preprocessing_output)

        if self.crop_around_vertex:
            # Keep coordinates around the primary vertex module
            primary_vertex_module = event['primary_vertex'][2] // self.num_modules
            coords_module = event['coords'][:, 2] // self.num_modules
            vertex_mask = (coords_module >= primary_vertex_module - 1) & (coords_module <= primary_vertex_module + 4)
        else:
            vertex_mask = np.ones(event['coords'].shape[0], dtype=bool)

        # Assemble output
        output = {
            'coords': torch.from_numpy(event['coords'][vertex_mask]).float(),
            'modules': torch.from_numpy(event['modules'][vertex_mask]).long(),
            'feats': feats[vertex_mask].float(),
            'feats_global': feats_global.float(),
            'faser_cal_modules': faser_mod.float(),
            'rear_cal_modules': rear_cal_mod.float(),
            'rear_hcal_modules': rear_hcal_mod.float(),
            'flavour_label': torch.from_numpy(event['flavour_label']),
        }
        if self.stage1:
            output['seg_labels'] = torch.from_numpy(event['seg_labels'][vertex_mask]).float()
        if not self.train or not self.stage1:
            output.update({
                'charm': torch.from_numpy(np.atleast_1d(event['charm'])).float(),
                #'e_vis': e_vis.float(),
                #'pt_miss': pt_miss.float(),
                'vis_sp_momentum': vis_sp_momentum.float(),
                #'vis_sp_momentum_mag': vis_mag.float(),
                #'vis_sp_momentum_dir': vis_dir.float(),
                'out_lepton_momentum': out_lepton_momentum.float(),
                #'out_lepton_momentum_mag': out_mag.float(),
                #'out_lepton_momentum_dir': out_lepton_dir.float(),
                'jet_momentum': jet_momentum.float(),
                #'jet_momentum_mag': jet_mag.float(),
                #'jet_momentum_dir': jet_dir.float(),
                'is_cc': torch.tensor(event['is_cc']).reshape(1,).float(),
            })
        if not self.train:
            output.update({
                'run_number': event['run_number'],
                'event_id': event['event_id'],
                'in_neutrino_pdg': event['in_neutrino_pdg'],
                'in_neutrino_energy': event['in_neutrino_energy'],
            })
        return output
        

    def _process_sample(self, data):
        event = self._prepare_event(data)
        return self._finalise_event(event)


class SparseFASERCALMapDataset(SparseFASERCALDataset, Dataset):
    def __init__(self, args):
        super().__init__(args)
        self.root = args.dataset_path
        self.data_files = sorted(glob(f'{self.root}/*.npz'))

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx], allow_pickle=True)
        return self._process_sample(data)



class SparseFASERCALIterableDataset(SparseFASERCALDataset, IterableDataset):
    def __init__(self, args, split, meta, shard_pattern, shardshuffle=False, shuffle=0):
        super().__init__(args)
        self.shuffle = shuffle
        self.shardshuffle = shardshuffle
        self.shard_pattern = shard_pattern
        self._len = meta["splits"][split]["num_samples"]

    def __len__(self):
        return self._len
    
    def _decode_npz_only(self, sample):
        """Decode .npz files from bytes into numpy objects."""
        for k in list(sample.keys()):
            if k.endswith(".npz") and not hasattr(sample[k], "files"):
                sample[k] = np.load(io.BytesIO(sample[k]), allow_pickle=True)
        return sample
    
    def __iter__(self):
        dataset = (
            wds.WebDataset(
                self.shard_pattern,
                shardshuffle=self.shardshuffle,
                empty_check=True,
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
            )
            .with_epoch(self.epoch)
            .shuffle(self.shuffle)
            .map(self._decode_npz_only)
            .to_tuple("data.npz")
        )

        for data, in dataset:
            yield self._process_sample(data)
