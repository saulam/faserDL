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
    

    def build_per_hit_ids_from_csr(
        self,
        true_track_id: np.ndarray,      # [T] int64
        true_primary_id: np.ndarray,    # [T] int64
        indptr: np.ndarray,             # [N+1] int64
        true_index: np.ndarray,         # [E] int64
        link_weight: np.ndarray,        # [E] float
        ghost_mask: np.ndarray | None = None,  # [N] bool
        train: bool = False,            # False: argmax; True: weighted sampling
        weight_threshold: float = 0.0,
    ):
        """
        Fully vectorized per-hit ID selection from CSR rows.
        Returns:
        hit_track_id   [N] int64  (=-1 if none)
        hit_primary_id [N] int64  (=-1 if none)
        active         [N] bool   (True if some edge selected)
        chosen_true    [N] int64  chosen true index per hit (=-1 if none)
        """

        indptr = indptr.astype(np.int64, copy=False)
        true_index = true_index.astype(np.int64, copy=False)
        w = link_weight.astype(np.float64, copy=False)  # float64 for stable cumsum
        t_tr = true_track_id.astype(np.int64, copy=False)
        t_pr = true_primary_id.astype(np.int64, copy=False)

        N = indptr.size - 1
        deg = np.diff(indptr)                 # [N]
        rows = np.arange(N, dtype=np.int64)
        has_edges = deg > 0

        # Map each edge -> its row (length E)
        row_of_edge = np.repeat(rows, deg)

        # Apply threshold (<= thr treated as zero for sampling / invalid for argmax)
        if weight_threshold > 0.0:
            w_valid = np.where(w > weight_threshold, w, 0.0)
        else:
            # assume non-negative weights; keep zeros
            w_valid = np.where(w >= 0.0, w, 0.0)

        # Zero-out all edges belonging to ghost rows (they will be ignored)
        if ghost_mask is not None:
            w_valid[ghost_mask[row_of_edge]] = 0.0

        # Outputs (defaults)
        hit_trk = np.full(N, -1, dtype=np.int64)
        hit_pri = np.full(N, -1, dtype=np.int64)
        active  = np.zeros(N, dtype=bool)
        chosen_true = np.full(N, -1, dtype=np.int64)

        if train:
            # -------- Weighted sampling per row (O(E)) --------
            # Global cumsum; sample target in [base, base+rowsum) for each row
            cs = np.cumsum(w_valid)                             # [E]
            cs_prev = np.concatenate(([0.0], cs[:-1]))          # [E]

            s = indptr[:-1]                                     # starts
            e = indptr[1:]                                      # ends
            base = np.zeros(N, dtype=np.float64)
            rowsum = np.zeros(N, dtype=np.float64)

            # Only rows with edges contribute to base/rowsum
            base[has_edges] = cs_prev[s[has_edges]]
            rowsum[has_edges] = cs[e[has_edges] - 1] - base[has_edges]

            # Active rows are those with positive total weight
            positive = has_edges & (rowsum > 0)
            if positive.any():
                u = np.random.random(positive.sum()) * rowsum[positive]
                targets = base[positive] + u                    # [R+]
                # Pick first index where cs > target (rightmost cdf inverse)
                edge_idx = np.searchsorted(cs, targets, side='right')  # [R+], in [0..E-1]

                # Fill outputs for those rows
                pos_rows = np.flatnonzero(positive)
                t_idx = true_index[edge_idx]
                hit_trk[pos_rows] = t_tr[t_idx]
                hit_pri[pos_rows] = t_pr[t_idx]
                active[pos_rows] = True
                chosen_true[pos_rows] = t_idx

            # Rows with rowsum==0 (all zeroed weights) remain -1/inactive

        else:
            # -------- Argmax per row (O(E log E) via single sort) --------
            # Make invalid edges -inf so they lose the argmax
            w_arg = w.copy()
            # threshold
            if weight_threshold > 0.0:
                w_arg[w_arg <= weight_threshold] = -np.inf
            # ghosts
            if ghost_mask is not None:
                w_arg[ghost_mask[row_of_edge]] = -np.inf

            # Sort by (row, weight), take the last edge within each row
            order = np.lexsort((w_arg, row_of_edge))            # by row, then weight asc
            ends_in_order = np.cumsum(deg) - 1                  # position of last edge of each row in 'order'
            rows_nonempty = np.flatnonzero(has_edges)
            if rows_nonempty.size:
                last_pos = ends_in_order[rows_nonempty]         # indices into 'order'
                edge_max = order[last_pos]                      # edge indices chosen for those rows

                # Exclude rows where all edges were invalid (-inf)
                valid_rows_mask = ~np.isneginf(w_arg[edge_max])
                if valid_rows_mask.any():
                    sel_rows = rows_nonempty[valid_rows_mask]
                    sel_edges = edge_max[valid_rows_mask]
                    t_idx = true_index[sel_edges]
                    hit_trk[sel_rows] = t_tr[t_idx]
                    hit_pri[sel_rows] = t_pr[t_idx]
                    active[sel_rows] = True
                    chosen_true[sel_rows] = t_idx

        return hit_trk, hit_pri, active, chosen_true



    def _prepare_event(self, data):
        """
        Loads and processes a single event up through augmentations.
        Returns a dict of intermediate arrays.
        """
        run_number = data['run_number'].item()
        event_id = data['event_id'].item()
        true_hits = data['true_hits']                   # [T]
        true_track_id = true_hits[:, 0]                 # [T]
        true_primary_id = true_hits[:, 2]               # [T]
        reco_hits = data['reco_hits']                   # [N]
        indptr = data['indptr']                         # [N+1] CSR row pointers for reco->true contributions
        true_index = data['true_index']                 # [E] concatenated true-hit indices for each reco hit
        link_weight = data['link_weight']               # [E] Per-edge weights
        ghost_mask = data['ghost_mask']                 # [N] Ghost mask for reco hits
        primary_vertex = data['primary_vertex']
        is_cc = data['is_cc']
        is_tau = data['is_tau']
        in_neutrino_pdg = data['in_neutrino_pdg']
        in_neutrino_energy = data['in_neutrino_energy']
        vis_sp_momentum = data['vis_sp_momentum']
        seg_labels_raw = data['seg_labels']
        out_lepton_momentum = data['out_lepton_momentum']
        jet_momentum = data['jet_momentum']
        tau_vis_momentum = data['tau_vis_momentum']
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
        if is_tau:
            assert in_neutrino_pdg in [-16, 16], "Tau events must have PDG ID of ±16"

        hit_track_id, hit_primary_id = None, None
        if self.stage1:
            hit_track_id, hit_primary_id, _, _ = self.build_per_hit_ids_from_csr(
                true_track_id=true_track_id,
                true_primary_id=true_primary_id,
                indptr=indptr,
                true_index=true_index,
                link_weight=link_weight,
                ghost_mask=ghost_mask,
                train=self.train,            # argmax if False, sampling if True
                weight_threshold=0.05,
            )

        # initial transformations
        coords = reco_hits[:, :3]
        q = reco_hits[:, 4].reshape(-1, 1)
        module_idx = np.searchsorted(self.metadata['z'][:, 0], coords[:, 2])
        modules = self.metadata['z'][module_idx, 1]
        coords = self.voxelise(coords)
        primary_vertex = self.voxelise(primary_vertex)

        if is_cc and is_tau:  # nutau
            out_lepton_momentum = tau_vis_momentum

        # augmentations (if applicable)
        if self.train and self.augmentations_enabled:
            coords, modules, q, (hit_track_id, hit_primary_id, ghost_mask), \
            (out_lepton_momentum, jet_momentum, vis_sp_momentum), \
            global_feats, _ = augment(
                coords, modules, q, (hit_track_id, hit_primary_id, ghost_mask), 
                (out_lepton_momentum, jet_momentum, vis_sp_momentum),
                global_feats, primary_vertex, self.metadata, self.stage1
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
            'hit_track_id': hit_track_id,
            'hit_primary_id': hit_primary_id,
            'ghost_mask': ghost_mask,
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
            'in_neutrino_pdg': in_neutrino_pdg,
            'in_neutrino_energy': in_neutrino_energy,
            'is_cc': is_cc,
        }

    
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

        # Assemble output
        output = {
            'coords': torch.from_numpy(event['coords']).float(),
            'modules': torch.from_numpy(event['modules']).long(),
            'feats': feats.float(),
            'feats_global': feats_global.float(),
            'faser_cal_modules': faser_mod.float(),
            'rear_cal_modules': rear_cal_mod.float(),
            'rear_hcal_modules': rear_hcal_mod.float(),
            'flavour_label': torch.from_numpy(event['flavour_label']),
        }
        if self.stage1:
            output.update({
                'hit_track_id': torch.from_numpy(event['hit_track_id']).long(),
                'hit_primary_id': torch.from_numpy(event['hit_primary_id']).long(),
                'ghost_mask': torch.from_numpy(event['ghost_mask']).bool(),
            })
            
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
