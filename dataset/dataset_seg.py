import pickle as pk
import numpy as np
import torch
import numpy as np
import os
import sys
from glob import glob
from torch.utils.data import Dataset
from utils import ini_argparse, random_rotation_saul
import MinkowskiEngine as ME

       
class SparseFASERCALDatasetSeg(Dataset):
    def __init__(self, args):
        """
        Initializes the SparseFASERCALDataset class.   

        Args:
        root (str): Root directory containing the data files.
        shuffle (bool): Whether to shuffle the dataset (default: False).
        """
        self.root = args.dataset_path
        self.contrastive = args.contrastive
        self.data_files = self.processed_file_names
        self.glob = args.glob
        self.training = False
        self.total_events = self.__len__
        with open(self.root + "/metadata.pkl", "rb") as fd:
            self.metadata = pk.load(fd)
            self.metadata['x'] = np.array(self.metadata['x'])
            self.metadata['y'] = np.array(self.metadata['y'])
            self.metadata['z'] = np.array(self.metadata['z'])

    def set_training_mode(self, training=True):
        """Sets the split type dynamically."""
        print("Setting training mode to {}.".format(training))
        self.training = training


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

    def _augment(self, coords_ori, feats_ori, labels1_ori, labels2_ori, prim_vertex, round_coords=True):
        coords, feats, labels1, labels2 = coords_ori.copy(), feats_ori.copy(), labels1_ori.copy(), labels2_ori.copy()

        # rotate
        coords = self._rotate(coords, prim_vertex)
        #coords = self._rotate_90(coords)
        # translate
        coords = self._translate(coords)
        # drop voxels
        coords, feats, labels1, labels2 = self._drop(coords, feats, labels1, labels2, std_dev=0.1)
        # shift feature values
        feats = self._shift_q_gaussian(feats, std_dev=0.01)
        # keep within limits
        coords, feats, labels1, labels2 = self._within_limits(coords, feats, labels1, labels2)
        
        if coords.shape[0] == 0:
            return coords_ori, feats_ori, labels1_ori, labels2_ori
        
        if round_coords:
            coords = coords.round()

        # Quantize (voxelise and detect duplicates)
        _, indices = ME.utils.sparse_quantize(
            coordinates=coords,
            return_index=True,
            quantization_size=1.0
        )
        coords = coords[indices]
        feats = feats[indices]
        labels1 = labels1[indices]
        labels2 = labels2[indices]

        return coords, feats, labels1, labels2

    def _rotate_90(self, point_cloud):
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

        reference_point = np.array([
            (self.metadata['x'].shape[0]-1) / 2.,
            (self.metadata['y'].shape[0]-1) / 2.,
            (self.metadata['z'].shape[0]-1) / 2.]
        )
        final_rotation_matrix = np.eye(3)

        for axis in ['x', 'y', 'z']:  # Loop over each axis
            if np.random.choice([False, True]):
                angle = np.random.choice([0, 90, 180, 270])
                final_rotation_matrix = final_rotation_matrix @ rotations[axis][angle]

        translated_points = point_cloud - reference_point
        rotated_points = translated_points @ final_rotation_matrix.T
        final_points = rotated_points + reference_point
        return final_points


    def _rotate(self, coords, prim_vertex):
        """Random rotation along"""
        angle_limits = torch.tensor([
            [-torch.pi/8, -torch.pi/8, -torch.pi],  # Min angles for X, Y, Z
            [torch.pi/8, torch.pi/8,  torch.pi]   # Max angles for X, Y, Z
        ])
        if (angle_limits==0).all():
            # no rotation at all
            return coords
        return random_rotation_saul(coords=coords,
                                    angle_limits=angle_limits,
                                    origin=prim_vertex)

    def _translate(self, coords):
        shift_x, shift_y = np.random.randint(low=-5, high=5, size=(2,))
        shift_z = np.random.randint(low=-15, high=15)
        coords[:, 0] += shift_x
        coords[:, 1] += shift_y
        coords[:, 2] += shift_z
        return coords

    def _drop(self, coords, feats, labels1, labels2, std_dev=0.1):
        p = abs(np.random.randn(1) * std_dev)
        mask = np.random.rand(coords.shape[0]) > p
        #don't drop all coordinates
        if mask.sum() == 0:
            return coords, feats, labels1, labels2
        return coords[mask], feats[mask], labels1[mask], labels2[mask]

    def _shift_q_uniform(self, feats, max_scale_factor=0.1):
        shift = 1 - np.random.rand(*feats.shape) * max_scale_factor
        return feats * shift

    def _shift_q_gaussian(self, feats, std_dev=0.1):   
        shift = 1 - np.random.randn(*feats.shape) * std_dev
        return feats * shift

    def _within_limits(self, coords, feats, labels1, labels2):  
        lim_x = self.metadata['x'].shape[0]
        lim_y = self.metadata['y'].shape[0]
        lim_z = self.metadata['z'].shape[0]
        mask = (coords[:, 0] >= 0) & (coords[:, 0] < lim_x) & \
           (coords[:, 1] >= 0) & (coords[:, 1] < lim_y) & \
           (coords[:, 2] >= 0) & (coords[:, 2] < lim_z)  
        return coords[mask], feats[mask], labels1[mask], labels2[mask]

    def voxelise(self, coords):
        """
        Convert physical coordinates to voxel coordinates by mapping them to
        their closest indices within the x, y, and z metadata ranges.
        """
        coords[..., 0] = np.searchsorted(self.metadata['x'], coords[..., 0])
        coords[..., 1] = np.searchsorted(self.metadata['y'], coords[..., 1])
        coords[..., 2] = np.searchsorted(self.metadata['z'][:, 0], coords[..., 2])

    def unvoxelise(self, coords):
        """
        Convert voxel coordinates back to physical coordinates using the
        corresponding indices in the metadata.
        """
        coords[..., 0] = self.metadata['x'][coords[..., 0].astype(int)]
        coords[..., 1] = self.metadata['y'][coords[..., 1].astype(int)]
        coords[..., 2] = self.metadata['z'][coords[..., 2].astype(int), 0]
        
    def contains_primary_lepton(self, hits, lepton_pdg, iscc):
        """
        Checks if any row in the numpy array satisfies the following conditions:
        - ftrackID equals fprimaryID.
        - fparentID equals 0.
        - fPDG is leptonic.

        Returns:
        bool: True if any row satisfies the conditions, False otherwise.
        """
        if iscc == False:
            return False

        # Check the conditions across all rows
        condition = (
            (hits[:, 0] == hits[:, 2]) &
            (hits[:, 1] == 0) &
            (np.isin(hits[:, 3], lepton_pdg))
        )

        return np.any(condition)

    def process_labels(self, reco_hits_true, true_hits, pdg, iscc):
        """
        Process a list of labels into binary classification arrays:
        - seg_labels: (non-ghost/ghost label, muonic, electromagnetic, hadronic)
        - primlepton_labels: whether voxel belongs to primary lepton
        """
        num_hits = len(reco_hits_true)
        seg_labels = np.zeros((num_hits, 3))  # ghost, muonic+electromagnetic, hadronic
        primlepton_labels = np.zeros((num_hits, 1))

        ghost_pdg = list(self.metadata['ghost_pdg'])   
        muonic_pdg = list(self.metadata['muonic_pdg']) 
        electromagnetic_pdg = list(self.metadata['electromagnetic_pdg'])
        hadronic_pdg = list(self.metadata['hadronic_pdg'])
        for i, reco_hit_true in enumerate(reco_hits_true):
            reco_hit_true = np.array(reco_hit_true).astype(int)
            try:
                matched_hits = true_hits[reco_hit_true]
            except Exception as e:
                raise ValueError(f"Error accessing true_hits with reco_hit_true: {reco_hit_true}, type: {type(reco_hit_true)}") from e
            m_mask = np.isin(matched_hits[:, 3], muonic_pdg)
            e_mask = np.isin(matched_hits[:, 3], electromagnetic_pdg)
            h_mask = np.isin(matched_hits[:, 3], hadronic_pdg)

            m_edepo = matched_hits[m_mask, -1].sum()   
            e_edepo = matched_hits[e_mask, -1].sum()   
            h_edepo = matched_hits[h_mask, -1].sum()   
            total_edepo = matched_hits[:, -1].sum()

            primlepton_labels[i, 0] = self.contains_primary_lepton(matched_hits, pdg, iscc)
            seg_labels[i, 0] = 1 if reco_hit_true[0] == -1 else 0
            seg_labels[i, 1] = 0 if reco_hit_true[0] == -1 else (m_edepo + e_edepo) / total_edepo
            seg_labels[i, 2] = 0 if reco_hit_true[0] == -1 else h_edepo / total_edepo

        return primlepton_labels, seg_labels
    
    def pdg2label(self, pdg, iscc):
        """
        PDG to label.
        """
        if iscc:
            if pdg in [-12, 12]:  # CC nue
                label = 0
            elif pdg in [-14, 14]:  # CC numu
                label = 1
            elif pdg in [-16, 16]:  # CC nutau
                label = 2
        else:
            label = 3  # NC

        return label

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
        
        # retrieve fields
        run_number = self.get_param(data, 'run_number')
        event_id = self.get_param(data, 'event_id')
        true_hits = self.get_param(data, 'true_hits')
        reco_hits = self.get_param(data, 'reco_hits')
        reco_hits_true = self.get_param(data, 'reco_hits_true')
        in_neutrino_pdg = self.get_param(data, 'in_neutrino_pdg')
        in_neutrino_energy = self.get_param(data, 'in_neutrino_energy')
        out_lepton_pdg = self.get_param(data, 'out_lepton_pdg')
        iscc = self.get_param(data, 'iscc')
        evis = self.get_param(data, 'evis', preprocess=True)
        ptmiss = self.get_param(data, 'ptmiss', preprocess=True)
        rearcal_energydeposit = self.get_param(data, 'rearcal_energydeposit', preprocess=True)
        rearhcal_energydeposit = self.get_param(data, 'rearhcal_energydeposit', preprocess=True)
        rearmucal_energydeposit = self.get_param(data, 'rearmucal_energydeposit', preprocess=True)
        fasercal_energydeposit = self.get_param(data, 'fasercal_energydeposit', preprocess=True)
        rearhcalmodules = self.get_param(data, 'rearhcalmodules', preprocess=True)
        fasercalmodules = self.get_param(data, 'fasercalmodules', preprocess=True)
        prim_vertex = self.get_param(data, 'prim_vertex')
        
        # retrieve coordiantes and features (energy deposited)
        coords = reco_hits[:, :3]
        q = self.divide_by_std(reco_hits[:, 4].reshape(-1, 1), 'q')
        
        # voxelise coordinates and vertex
        self.voxelise(coords)
        self.voxelise(prim_vertex)
        
        # process labels
        primlepton_labels = self.get_param(data, 'primlepton_labels')
        seg_labels = self.get_param(data, 'seg_labels')
        if 'primlepton_labels' not in data or 'seg_labels' not in data:
            primlepton_labels, seg_labels = self.process_labels(reco_hits_true, true_hits, out_lepton_pdg, iscc)
        flavour_label = self.pdg2label(in_neutrino_pdg, iscc)

        # output
        output = {'run_number': run_number,
                  'event_id': event_id}
        
        feats = q
        if self.training and np.random.rand() > 0.01:
            # augmented event
            coords, feats, seg_labels, primlepton_labels = self._augment(coords, q, seg_labels, primlepton_labels, prim_vertex, round_coords=True)

        # global features
        feats_global = np.concatenate([rearcal_energydeposit, rearhcal_energydeposit, rearmucal_energydeposit, fasercal_energydeposit, rearhcalmodules, fasercalmodules])
    
        output['prim_vertex'] = prim_vertex
        output['in_neutrino_pdg'] = in_neutrino_pdg
        output['in_neutrino_energy'] = in_neutrino_energy
        output['coords'] = torch.from_numpy(coords.reshape(-1, 3)).float()
        output['feats'] = torch.from_numpy(feats.reshape(-1, 1)).float()
        output['feats_global'] = torch.from_numpy(feats_global).float()
        output['primlepton_labels'] = torch.from_numpy(primlepton_labels.reshape(-1, 1)).float()
        output['seg_labels'] = torch.from_numpy(seg_labels.reshape(-1, 3)).float()
        output['flavour_label'] = torch.tensor([flavour_label]).long()
        output['evis'] = torch.from_numpy(evis).float()
        output['ptmiss'] = torch.tensor(ptmiss).float()

        return output

