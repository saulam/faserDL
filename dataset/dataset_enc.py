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

class SparseFASERCALDatasetEnc(Dataset):
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
        self.load_seg = args.load_seg
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

    def _augment(self, coords_ori, feats_ori, labels1_ori, labels2_ori, momentum1_ori, momentum2_ori, prim_vertex, round_coords=True):
        coords, feats, labels1 = coords_ori.copy(), feats_ori.copy(), labels1_ori.copy()
        labels2, momentum1, momentum2 = labels2_ori.copy(), momentum1_ori.copy(), momentum2_ori.copy()
 
        # rotate
        #coords, momentum1, momentum2 = self._rotate(coords, momentum1, momentum2, prim_vertex)
        # translate
        coords = self._translate(coords)
        # drop voxels
        coords, feats, labels1, labels2 = self._drop(coords, feats, labels1, labels2, std_dev=0.1)
        # shift feature values
        feats = self._shift_q_gaussian(feats, std_dev=0.01)
        # keep within limits
        coords, feats, labels1, labels2 = self._within_limits(coords, feats, labels1, labels2)
        
        if coords.shape[0] == 0:
            return coords_ori, feats_ori, labels1_ori, labels2_ori, momentum1_ori, momentum2_ori

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
        labels1 = labels1[indices].reshape(-1, 1)
        labels2 = labels2[indices].reshape(-1, 3)

        return coords, feats, labels1, labels2, momentum1, momentum2


    def _rotate(self, coords, momentum1, momentum2, prim_vertex):
        """Random rotation along"""
        angle_limits = torch.tensor([
            [-torch.pi/8, -torch.pi/8, -torch.pi],  # Min angles for X, Y, Z
            [torch.pi/8, torch.pi/8,  torch.pi]   # Max angles for X, Y, Z
        ])
        if (angle_limits==0).all():
            # no rotation at all
            return coords, momentum1, momentum2
        return random_rotation_saul(coords=coords,
                                    momentum1=momentum1,
                                    momentum2=momentum2,
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
        if mask.sum() < 2:
            return coords, feats, labels1, labels2
        return coords[mask], feats[mask], labels1[mask], labels2[mask]


    def _shift_q_uniform(self, feats, max_scale_factor=0.1):
        shift = 1 - np.random.rand(*feats.shape) * max_scale_factor
        return feats * shift


    def _shift_q_gaussian(self, feats, std_dev=0.1):
        shift = 1 - np.random.randn(*feats.shape) * std_dev
        return feats * shift


    def _within_limits(self, coords, feats, labels1, labels2):
        mask = (coords[:, 0] >= 0) & (coords[:, 0] < 48) & \
           (coords[:, 1] >= 0) & (coords[:, 1] < 48) & \
           (coords[:, 2] >= 0) & (coords[:, 2] < 300)
        return coords[mask], feats[mask], labels1[mask], labels2[mask]

 
    def voxelise(self, coords):
        """
        Convert physical coordinates to voxel coordinates by mapping them to 
        their closest indices within the x, y, and z metadata ranges.
        """
        coords[:, 0] = np.searchsorted(self.metadata['x'], coords[:, 0])
        coords[:, 1] = np.searchsorted(self.metadata['y'], coords[:, 1])
        coords[:, 2] = np.searchsorted(self.metadata['z'][:, 0], coords[:, 2])

    def unvoxelise(self, coords):
        """
        Convert voxel coordinates back to physical coordinates using the 
        corresponding indices in the metadata.
        """
        coords[:, 0] = self.metadata['x'][coords[:, 0].astype(int)]
        coords[:, 1] = self.metadata['y'][coords[:, 1].astype(int)]
        coords[:, 2] = self.metadata['z'][coords[:, 2].astype(int), 0]

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
            except:
                print("hey", reco_hit_true, type(reco_hit_true))
                assert False
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

    def standardize(self, x, mean, std):
        return (x-mean)/std


    def add_gaussian_noise(self, probs, std=0.05, shuffle_prob=0.01):
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


    def to_one_hot(self, indices, length):
        """
        Converts a vector of indices into a batch of one-hot encoded vectors.

        :param indices: List or NumPy array of indices (shape: (batch_size,))
        :param length: Length of the one-hot encoded vectors
        :return: NumPy array of shape (batch_size, length) with one-hot encodings
        """
        if np.any((indices < 0) | (indices >= length)):
            raise ValueError("Some indices are out of range for one-hot encoding")

        batch_size = len(indices)
        one_hot = np.zeros((batch_size, length), dtype=np.float32)
        one_hot[np.arange(batch_size), indices.astype(int)] = 1.0
        return one_hot


    def __getitem__(self, idx):
        """
        Retrieves a data sample by index.

        Args:
        idx (int): Index of the data sample.

        Returns:
        dict: Data sample with filename, coordinates, features, and labels.
        """
        data = np.load(self.data_files[idx], allow_pickle=True)
        run_number = data['run_number'].item()
        event_id = data['event_id'].item()
        true_hits = data['true_hits']
        reco_hits = data['reco_hits']
        reco_hits_true = data['reco_hits_true']
        in_neutrino_pdg = data['in_neutrino_pdg'].item()
        in_neutrino_energy = data['in_neutrino_energy'].item()
        out_lepton_pdg = data['out_lepton_pdg'].item()
        iscc = data['iscc'].item()
        evis = self.standardize(data['evis'].reshape(1,), self.metadata['evis_mean'], self.metadata['evis_std'])
        ptmiss = self.standardize(data['ptmiss'].reshape(1,), self.metadata['ptmiss_mean'], self.metadata['ptmiss_std']) 
        rearcal_energydeposit = self.standardize(data['rearcal_energydeposit'].reshape(1,), self.metadata['rearcal_energydeposit_mean'], self.metadata['rearcal_energydeposit_std'])
        rearhcal_energydeposit = self.standardize(data['rearhcal_energydeposit'].reshape(1,), self.metadata['rearhcal_energydeposit_mean'], self.metadata['rearhcal_energydeposit_std'])
        rearmucal_energydeposit = self.standardize(data['rearmucal_energydeposit'].reshape(1,), self.metadata['rearmucal_energydeposit_mean'], self.metadata['rearmucal_energydeposit_std'])
        fasercal_energydeposit = self.standardize(data['fasercal_energydeposit'].reshape(1,), self.metadata['fasercal_energydeposit_mean'], self.metadata['fasercal_energydeposit_std'])
        rearhcalmodules = self.standardize(data['rearhcalmodules'], self.metadata['rearhcalmodules_mean'], self.metadata['rearhcalmodules_std'])
        fasercalmodules = self.standardize(data['fasercalmodules'], self.metadata['fasercalmodules_mean'], self.metadata['fasercalmodules_std'])
        out_lepton_momentum = data['out_lepton_momentum'] / self.metadata['out_lepton_momentum_std']
        out_lepton_energy = self.standardize(data['out_lepton_energy'].reshape(1,), self.metadata['out_lepton_energy_mean'], self.metadata['out_lepton_energy_std'])
        jet_momentum = data['jet_momentum'] / self.metadata['jet_momentum_std']
        prim_vertex = data['prim_vertex'].reshape(1, 3)
        
        # retrieve coordiantes and features (energy deposited)
        coords = reco_hits[:, :3]
        feats = self.standardize(reco_hits[:, 4].reshape(-1, 1), self.metadata['q_mean'], self.metadata['q_std'])

        # voxelise
        self.voxelise(coords)
        
        # process labels
        if self.load_seg:
            # load labels from pretrained model predictions
            file_name = self.data_files[idx].replace("events_v3.5", "events_v3.5_seg_results")
            segs = np.load(file_name)
            primlepton_labels, seg_labels = segs['out_primlepton'], segs['out_seg']
        else:
            primlepton_labels, seg_labels = self.process_labels(reco_hits_true, true_hits, out_lepton_pdg, iscc)
        flavour_label = self.pdg2label(in_neutrino_pdg, iscc)
        primlepton_labels = primlepton_labels.reshape(-1, 1)
        seg_labels = seg_labels.reshape(-1, 3)

        # output
        output = {'run_number': run_number,
                  'event_id': event_id}

        if self.training and np.random.rand() > 0.01:
            self.voxelise(prim_vertex)
            prim_vertex = prim_vertex.reshape(3)
           
            # augmented event
            coords, feats, primlepton_labels, seg_labels, out_lepton_momentum, jet_momentum = self._augment(coords, feats, primlepton_labels, seg_labels, out_lepton_momentum, jet_momentum, prim_vertex, round_coords=False)
            primlepton_labels = self.add_gaussian_noise(primlepton_labels)
            seg_labels = self.add_gaussian_noise(seg_labels)

        # log features
        #feats = np.log(feats).reshape(-1, 1)
        feats = feats.reshape(-1, 1)
        feats = np.concatenate((feats, primlepton_labels, seg_labels), axis=1)
        feats_global = np.concatenate([rearcal_energydeposit, rearhcal_energydeposit, rearmucal_energydeposit, fasercal_energydeposit, rearhcalmodules, fasercalmodules])

        output['prim_vertex'] = prim_vertex
        output['in_neutrino_pdg'] = in_neutrino_pdg
        output['in_neutrino_energy'] = in_neutrino_energy
        output['coords'] = torch.from_numpy(coords.reshape(-1, 3)).float()
        output['feats'] = torch.from_numpy(feats).float()
        output['feats_global'] = torch.from_numpy(feats_global).float()
        output['flavour_label'] = torch.tensor([flavour_label]).long()
        output['evis'] = torch.from_numpy(evis).float()
        output['ptmiss'] = torch.tensor(ptmiss).float()
        output['out_lepton_momentum'] = torch.from_numpy(out_lepton_momentum)
        output['jet_momentum'] = torch.from_numpy(jet_momentum)

        return output

