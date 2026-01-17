"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 07.25

Description: Dataset file.
"""

import os
import io
import pickle as pk
import numpy as np
import torch
import webdataset as wds
from glob import glob
from itertools import chain
from torch.utils.data import Dataset, IterableDataset
from utils.augmentations import augment, smooth_labels
from utils.pdg import cluster_labels_from_pdgs


AHCAL_SHAPE = np.array([18, 18, 40], dtype=np.int32)

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
        if args.metadata_path is not None:
            with open(args.metadata_path, "rb") as fd:
                self.metadata = pk.load(fd)
                for key in ['x', 'y', 'z']:
                    self.metadata[key] = np.array(self.metadata[key])

                self.module_size = int((self.metadata['z'][:, 1] == 0).sum())
                self.num_modules = int(self.metadata['z'][:, 1].max() + 1)
        else:
            print("Warning: No metadata path provided; some functionalities may be limited.")

    
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

    
    def standardize_vertex(self, vertex, reverse=False):
        """
        Standardises or unstandardizes primary vertex coordinates using z-score normalization.
        
        Args:
            vertex: array-like of shape (..., 3) with (x, y, z) coordinates
            reverse: if True, unstandardizes back to physical coords
        
        Returns:
            Standardized/unstandardized vertex coordinates
        """
        # Statistics from training data
        mean = np.array([174.26, 109.02, -66.38], dtype=np.float32)
        std = np.array([135.14, 125.34, 652.16], dtype=np.float32)
        
        if not isinstance(vertex, np.ndarray):
            vertex = np.array(vertex, dtype=np.float32)
        
        if reverse:
            return vertex * std + mean
        else:
            return (vertex - mean) / std

    
    def pdg2label(self, pdg, is_cc, tau_decay_mode, split_tau=False):
        """Converts PDG ID to a classification label (0-5).

        Returns:
        - int: 
            0 - CC nue
            1 - CC numu
            2 - CC nutau (tau -> e)
            3 - CC nutau (tau -> mu)
            4 - CC nutau (tau -> hadrons)
            5 - NC
        """
        if is_cc:
            if pdg in {-12, 12}: return 0           # CC nue
            if pdg in {-14, 14}: return 1           # CC numu
            if pdg in {-16, 16}:                    # CC nutau
                if split_tau:
                    assert tau_decay_mode > 0, "Tau events must have a valid decay mode"
                    if tau_decay_mode == 1:   return 2  # tau -> e
                    elif tau_decay_mode == 2: return 3  # tau -> mu
                    else:                     return 4  # tau -> hadrons
                else:
                    return 2  # all CC nutau together
        return 5 if split_tau else 3  # NC


    def charmdecay2label(self, is_charmed, charm_decay, split_charm=False):
        """Classifies charm decays based on decay products.
        
        Returns:
        - int: 
            0 - no charm
            1 - charm -> e
            2 - charm -> mu
            3 - charm -> hadron (everything else)
        """
        if not is_charmed:
            return 0  # no charm
        if split_charm:
            if any(pid in {13, -13} for pid in charm_decay):
                return 2  # charm -> mu
            elif any(pid in {11, -11} for pid in charm_decay):
                return 1  # charm -> e
            return 3      # charm -> hadron
        return 1  # all charm decays together


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
    

    def process_muspec(self, muspec):
        ntracks = 0
        tracks = []
        if len(muspec) == 0:
            muspec = np.zeros((11, 0))
        for i in range(muspec.shape[1]):
            info = muspec[:, i]
            charge, npoints, px, py, pz, p, chi2, ndof, pval, fperr, fiperr = info
            if npoints >= 3 and chi2 > 0:
                ntracks += 1
                tracks.append([charge, px, py, pz, chi2])
                
        return ntracks, np.array(tracks).reshape(ntracks, 5)
    
    
    def build_per_hit_labels_from_csr(
        self,
        true_hie: np.ndarray,           # [T] int64
        true_dec: np.ndarray,           # [T] int64
        true_pid: np.ndarray,           # [T] int64
        indptr: np.ndarray,             # [N+1] int64  CSR row ptr per hit
        true_index: np.ndarray,         # [E]   int64  edge -> true index
        link_weight: np.ndarray,        # [E]   float  contribution weight per edge (>=0)
        ghost_mask: np.ndarray,         # [N]   bool   hits to be treated as ghosts
        weight_threshold: float = 0.0,  # drop edges with w <= thr BEFORE renorm
        normalize_rows: bool = True,
    ):
        """
        Build compact per-hit soft-label CSRs for three label spaces: HIE, DEC, PRIMARY.

        Returns a dict with three entries: 'hie', 'dec', 'primary'.
        Each entry is a tuple:
            (label_indptr: [N+1] int64,
            label_ids:    [L]   int64,   deduped per row,
            label_weight: [L]   float32, row-normalized if normalize_rows=True,
            row_has_labels: [N] bool)
        """
        indptr = indptr.astype(np.int64, copy=False)
        true_index = true_index.astype(np.int64, copy=False)
        w = link_weight.astype(np.float64, copy=False)    # float64 for stable sums
        t_hie = true_hie.astype(np.int64, copy=False)
        t_dec = true_dec.astype(np.int64, copy=False)
        t_pid = true_pid.astype(np.int64, copy=False)

        N = indptr.size - 1
        deg = np.diff(indptr)                 # [N]
        rows = np.arange(N, dtype=np.int64)
        row_of_edge = np.repeat(rows, deg)    # [E]

        if weight_threshold > 0.0:
            w_eff = np.where(w > weight_threshold, w, 0.0)
        else:
            w_eff = np.where(w >= 0.0, w, 0.0)
        if ghost_mask is not None:
            w_eff[ghost_mask[row_of_edge]] = 0.0

        # Edge -> per-field ids
        t_idx = true_index
        edge_hie = t_hie[t_idx].astype(np.int64, copy=False)
        edge_dec = t_dec[t_idx].astype(np.int64, copy=False)
        edge_pid = t_pid[t_idx].astype(np.int64, copy=False)

        def _build_single(edge_ids: np.ndarray):
            """
            Build one CSR over hits for a given edge-level id array.
            """
            # Keep only edges with positive effective weight
            pos = w_eff > 0.0
            if not np.any(pos):
                # Empty CSR
                return (np.zeros(N + 1, dtype=np.int64),
                        np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.float32),
                        np.zeros(N, dtype=bool))

            r = row_of_edge[pos]
            c = edge_ids[pos]
            ww = w_eff[pos]

            # Sort by (row, id) and segment-sum duplicates
            order = np.lexsort((c, r))
            r = r[order]; c = c[order]; ww = ww[order]

            seg_start = np.ones_like(r, dtype=bool)
            seg_start[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
            seg_id = np.cumsum(seg_start) - 1
            nseg = int(seg_id[-1]) + 1

            # Sum weights per (row, id) segment
            w_acc = np.zeros(nseg, dtype=np.float64)
            np.add.at(w_acc, seg_id, ww)

            # Representative (row,id) per segment
            r_rep = r[seg_start]
            c_rep = c[seg_start]

            keep = (w_acc > 0.0)
            if not np.any(keep):
                return (np.zeros(N + 1, dtype=np.int64),
                        np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.float32),
                        np.zeros(N, dtype=bool))

            r_keep = r_rep[keep]
            c_keep = c_rep[keep]
            w_keep = w_acc[keep]

            # Build indptr via bincount over kept rows
            indptr_out = np.zeros(N + 1, dtype=np.int64)
            counts = np.bincount(r_keep, minlength=N).astype(np.int64)
            indptr_out[1:] = np.cumsum(counts)

            # Optional per-row normalization
            if normalize_rows:
                row_sums = np.zeros(N, dtype=np.float64)
                np.add.at(row_sums, r_keep, w_keep)
                w_keep = (w_keep / row_sums[r_keep]).astype(np.float32, copy=False)
            else:
                w_keep = w_keep.astype(np.float32, copy=False)

            row_has = np.zeros(N, dtype=bool)
            row_has[r_keep] = True

            return indptr_out, c_keep.astype(np.int64, copy=False), w_keep, row_has

        # Build all three CSRs
        hie_csr = _build_single(edge_hie)
        dec_csr = _build_single(edge_dec)
        pid_csr = _build_single(edge_pid)

        return {
            "hie": hie_csr,
            "dec": dec_csr,
            "pid": pid_csr,
        }


    def _prepare_event(self, data):
        """
        Loads and processes a single event up through augmentations.
        Returns a dict of intermediate arrays.
        """
        run_number = data['run_number'].item()
        event_id = data['event_id'].item()
        true_hits = data['true_hits']                   # [T]
        true_pdg = true_hits[:, 3]                      # [T]
        true_primary = true_hits[:, 9]                  # [T]
        true_secondary = true_hits[:, 10]               # [T]
        true_tau_decay = true_hits[:, 11]               # [T]
        true_charm_decay = true_hits[:, 12]             # [T]
        reco_hits = data['reco_hits']                   # [N]
        indptr = data['indptr']                         # [N+1] CSR row pointers for reco->true contributions
        true_index = data['true_index']                 # [E] concatenated true-hit indices for each reco hit
        link_weight = data['link_weight']               # [E] Per-edge weights
        ghost_mask = data['ghost_mask']                 # [N] Ghost mask for reco hits
        primary_vertex = data['primary_vertex']
        is_cc = data['is_cc']
        is_tau = data['is_tau']
        in_neutrino_pdg = data['in_neutrino_pdg'].item()
        in_neutrino_energy = data['in_neutrino_energy'].item()
        vis_sp_momentum = data['vis_sp_momentum']
        out_lepton_momentum = data['out_lepton_momentum']
        jet_momentum = data['jet_momentum']
        tau_vis_momentum = data['tau_vis_momentum']
        tau_decay_mode = data['tau_decay_mode'].item()
        is_charmed = data['is_charmed']
        charm_decay = data['charm_decay']
        muspec_info = data['muspec_info']
        nb_muspec_tracks, muspec_tracks = self.process_muspec(muspec_info)
        global_feats = {
            'ecal_hits':        data['ecal_hits'],
            'ahcal_hits':       data['ahcal_hits'],
            'nb_muspec_tracks': nb_muspec_tracks,
            'muspec_q':         muspec_tracks[:, 0],
            'muspec_p':         muspec_tracks[:, 1:4],
            'muspec_chi2':      muspec_tracks[:, 4],
        }
        # ensure we have at least 1 AHCAL hit
        if global_feats["ahcal_hits"].size == 0:
            # random xyz in the 18x18x40 box, charge = 0
            xyz = np.array([
                np.random.randint(0, AHCAL_SHAPE[0]),
                np.random.randint(0, AHCAL_SHAPE[1]),
                np.random.randint(0, AHCAL_SHAPE[2]),
            ], dtype=float)
            ahcal_hits = np.array([[xyz[0], xyz[1], xyz[2], 0.0]], dtype=float)
            global_feats["ahcal_hits"] = ahcal_hits

        if is_tau:
            assert in_neutrino_pdg in [-16, 16], "Tau events must have PDG ID of ±16"

        csr_hie, csr_dec, csr_pid = None, None, None
        if self.stage1:
            true_hierarchy = true_primary.copy()
            true_hierarchy[true_secondary > 0] = 2
            true_decay = true_tau_decay.copy()
            true_decay[true_charm_decay > 0] = 2
            true_pid = cluster_labels_from_pdgs(true_pdg, tau_decay_mode)
            csr = self.build_per_hit_labels_from_csr(
                true_hie=true_hierarchy,
                true_dec=true_decay,
                true_pid=true_pid,
                indptr=indptr,
                true_index=true_index,
                link_weight=link_weight,
                ghost_mask=ghost_mask,
                weight_threshold=0.0,
            )
            csr_hie = csr["hie"][:3]
            csr_dec = csr["dec"][:3]
            csr_pid = csr["pid"][:3]

        # initial transformations
        coords = reco_hits[:, :3]
        q = reco_hits[:, 4].reshape(-1, 1)
        module_idx = np.searchsorted(self.metadata['z'][:, 0], coords[:, 2])
        modules = self.metadata['z'][module_idx, 1]
        coords = self.voxelise(coords)
        if is_cc and is_tau:  
            # for CC tau events, replace the outgoing lepton momentum with the visible tau decay products
            out_lepton_momentum = tau_vis_momentum
        if not is_cc:
            # NC events don't have visible outgoing leptons
            out_lepton_momentum.fill(0)
        split_tau, split_charm = True, True
        flavour_label = np.array([self.pdg2label(in_neutrino_pdg, is_cc, tau_decay_mode, split_tau=split_tau)])
        charm_label = np.array([self.charmdecay2label(is_charmed, charm_decay['pdg'], split_charm=split_charm)])

        # augmentations (if applicable)
        if self.train and self.augmentations_enabled:
            coords, modules, q, (csr_hie, csr_dec, csr_pid, ghost_mask), \
            (out_lepton_momentum, jet_momentum, vis_sp_momentum), \
            global_feats, _ = augment(
                coords, modules, q, (csr_hie, csr_dec, csr_pid, ghost_mask), 
                (out_lepton_momentum, jet_momentum, vis_sp_momentum),
                global_feats, primary_vertex, self.metadata, self.stage1
            )
            charm_label = smooth_labels(
                charm_label, 
                smoothing=self.label_smoothing,
                num_classes=4 if split_charm else 2
            )
            flavour_label = smooth_labels(
                flavour_label,
                smoothing=self.label_smoothing,
                num_classes=6 if split_tau else 4
            )

        return {
            'run_number': run_number,
            'event_id': event_id,
            'csr_hie': csr_hie,
            'csr_dec': csr_dec,
            'csr_pid': csr_pid,
            'ghost_mask': ghost_mask,
            'coords': coords,
            'modules': modules,
            'q': q,
            'vis_sp_momentum': vis_sp_momentum,
            'out_lepton_momentum': out_lepton_momentum,
            'jet_momentum': jet_momentum,
            'global_feats': global_feats,
            'flavour_label': flavour_label,
            'charm_label': charm_label,
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
        feats = self.preprocess(event['q'] * 10, 'q', self.preprocessing_input)
        event_hits = self.preprocess(len(event['q']), 'event_hits', self.preprocessing_input)

        ahcal_hits_coords = event['global_feats']['ahcal_hits'][:, :3]
        ahcal_hits_feats = self.preprocess(event['global_feats']['ahcal_hits'][:, 3] * 10, 'ahcal_hits', self.preprocessing_input)
        ecal_hits = self.preprocess(event['global_feats']['ecal_hits'], 'ecal_hits', self.preprocessing_input)
        nb_muspec_tracks = self.preprocess(event['global_feats']['nb_muspec_tracks'], 'nb_muspec_tracks')
        muspec_p = event['global_feats']['muspec_p']
        muspec_p[:, 0] = self.preprocess(muspec_p[:, 0], 'muspec_px', "identity")
        muspec_p[:, 1] = self.preprocess(muspec_p[:, 1], 'muspec_py', "identity")
        muspec_p[:, 2] = self.preprocess(muspec_p[:, 2], 'muspec_pz', "identity")
        muspec_info = torch.cat([
            self.preprocess(event['global_feats']['muspec_q'], 'muspec_q').reshape(-1, 1),
            torch.from_numpy(muspec_p).reshape(-1, 3),
            self.preprocess(event['global_feats']['muspec_chi2'], 'muspec_chi2', self.preprocessing_input).reshape(-1, 1),
        ], dim=1)
        
        vis_sp_momentum = event['vis_sp_momentum']
        out_lepton_momentum = event['out_lepton_momentum']
        jet_momentum = event['jet_momentum']

        # Preprocess outputs
        vis_sp_momentum = self.preprocess(vis_sp_momentum, 'vis_sp_momentum')
        out_lepton_momentum = self.preprocess(out_lepton_momentum, 'out_lepton_momentum')
        jet_momentum = self.preprocess(jet_momentum, 'jet_momentum')

        # Standardise primary vertex
        primary_vertex_standardised = self.standardize_vertex(event['primary_vertex'], reverse=False)

        # Assemble output
        output = {
            'coords': torch.from_numpy(event['coords']).float(),
            'modules': torch.from_numpy(event['modules']).long(),
            'feats': feats.float(),
            'ahcal_hits_coords': torch.from_numpy(ahcal_hits_coords).float(),
            'ahcal_hits_feats': ahcal_hits_feats.float().reshape(-1, 1),
            'ecal_hits': ecal_hits.float(),
            'nb_muspec_tracks': nb_muspec_tracks.float(),
            'muspec_info': muspec_info.float(),
            'flavour_label': torch.from_numpy(event['flavour_label']),
            'charm_label': torch.from_numpy(event['charm_label']),
            'vis_sp_momentum': vis_sp_momentum.float(),
            'out_lepton_momentum': out_lepton_momentum.float(),
            'jet_momentum': jet_momentum.float(),
            'is_cc': torch.tensor(event['is_cc']).reshape(1,).float(),
            'primary_vertex': torch.from_numpy(primary_vertex_standardised).float(),
        }
        if self.stage1:
            output.update({
                'csr_hie_indptr': torch.from_numpy(event['csr_hie'][0]).long(),
                'csr_hie_ids': torch.from_numpy(event['csr_hie'][1]).long(),
                'csr_hie_weights': torch.from_numpy(event['csr_hie'][2]).float(),
                'csr_dec_indptr': torch.from_numpy(event['csr_dec'][0]).long(),
                'csr_dec_ids': torch.from_numpy(event['csr_dec'][1]).long(),
                'csr_dec_weights': torch.from_numpy(event['csr_dec'][2]).float(),
                'csr_pid_indptr': torch.from_numpy(event['csr_pid'][0]).long(),
                'csr_pid_ids': torch.from_numpy(event['csr_pid'][1]).long(),
                'csr_pid_weights': torch.from_numpy(event['csr_pid'][2]).float(),
                'ghost_mask': torch.from_numpy(event['ghost_mask']).bool(),
            })
        if not self.train:
            output.update({
                'run_number': event['run_number'],
                'event_id': event['event_id'],
                'in_neutrino_pdg': event['in_neutrino_pdg'],
                'in_neutrino_energy': event['in_neutrino_energy'],
                'primary_vertex_raw': event['primary_vertex'],  # Keep raw for reference
            })
        return output
        

    def _process_sample(self, data):
        event = self._prepare_event(data)
        return self._finalise_event(event)


class SparseFASERCALMapDataset(SparseFASERCALDataset, Dataset):
    def __init__(self, args):
        super().__init__(args)
        self.root = args.dataset_path
        self.data_files = sorted(
            chain(
                glob(os.path.join(self.root, "*.npz")),
                glob(os.path.join(self.root, "*", "*.npz")),
            ),
            key=str.lower,
        )

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
