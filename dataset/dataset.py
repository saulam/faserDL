import pickle as pk
import numpy as np
import torch
import numpy as np
import MinkowskiEngine as ME
from glob import glob
from torch.utils.data import Dataset
from utils import random_rotation_saul

    
class SparseFASERCALDataset(Dataset):
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
        self.training = False
        self.total_events = self.__len__
        with open(self.root + "/metadata.pkl", "rb") as fd:
            self.metadata = pk.load(fd)


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

    def _augment(self, coords, feats, labels, prim_vertex, round_coords=True):
        # rotate
        coords = self._rotate(coords, prim_vertex)
        # translate
        coords = self._translate(coords)
        # drop voxels
        coords, feats, labels = self._drop(coords, feats, labels, std_dev=0.1)
        # shift feature values
        feats = self._shift_q_gaussian(feats, std_dev=0.05)
        # keep within limits
        #coords, feats, labels = self._within_limits(coords, feats, labels)
        if round_coords:
            coords = coords.round()

        return coords, feats, labels

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
        shift_x, shift_y = np.random.randint(low=-10, high=10, size=(2,))
        coords[:, 0] += shift_x
        coords[:, 1] += shift_y
        return coords


    def _drop(self, coords, feats, labels, std_dev=0.1):
        p = abs(np.random.randn(1) * std_dev)
        mask = np.random.rand(coords.shape[0]) > p
        #don't drop all coordinates
        if mask.sum() == 0:
            return coords, feats
        return coords[mask], feats[mask], labels[mask]


    def _shift_q_uniform(self, feats, max_scale_factor=0.1):
        shift = 1 - np.random.rand(*feats.shape) * max_scale_factor
        return feats * shift


    def _shift_q_gaussian(self, feats, std_dev=0.1):
        shift = 1 - np.random.randn(*feats.shape) * std_dev
        return feats * shift


    def _within_limits(self, coords, feats, labels):
        mask = (coords[:, 0] >= 0) & (coords[:, 0] < 48) & \
           (coords[:, 1] >= 0) & (coords[:, 1] < 48) & \
           (coords[:, 2] >= 0) & (coords[:, 2] < 400)
        return coords[mask], feats[mask], labels[mask]

 
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

    def process_labels(self, labels):
        """
        Process a list of labels into binary classification arrays:
        - y: (non-ghost/ghost label, muonic, electromagnetic, hadronic)
        """
        num_labels = len(labels)
        y = np.zeros((num_labels, 4))  # ghost, muonic, electromagnetic, hadronic

        ghost_pdg = self.metadata['ghost_pdg']
        muonic_pdg = self.metadata['muonic_pdg']
        electromagnetic_pdg = self.metadata['electromagnetic_pdg']
        hadronic_pdg = self.metadata['hadronic_pdg']

        for i, label_set in enumerate(labels):
            label_set = set(label_set)
            # Assign real hit types based on the presence of PDG codes
            y[i, 0] = 1 if label_set & ghost_pdg else 0
            y[i, 1] = 1 if label_set & muonic_pdg else 0
            y[i, 2] = 1 if label_set & electromagnetic_pdg else y[i, 1]
            y[i, 3] = 1 if label_set & hadronic_pdg else 0

        return y

    def __getitem__(self, idx):
        """
        Retrieves a data sample by index.

        Args:
        idx (int): Index of the data sample.

        Returns:
        dict: Data sample with filename, coordinates, features, and labels.
        """
        data = np.load(self.data_files[idx], allow_pickle=True)
        run_number = data['run_number']
        event_id = data['event_id']
        reco_hits = data['reco_hits']
        reco_hit_pdgs = data['reco_hit_pdgs']

        # retrieve coordiantes and features (energy deposited)
        coords = reco_hits[:, :3]
        feats = reco_hits[:, 4].reshape(-1, 1) 

        # voxelise
        self.voxelise(coords)
        
        # PDGs to labels
        labels = self.process_labels(reco_hit_pdgs)
       
        # output
        output = {'run_number': run_number,
                  'event_id': event_id}

        if self.contrastive or (self.training and np.random.rand() > 0.05):
            coords_true = coords.copy()
            feats_true = feats.copy()
            labels_true = labels.copy() 

            prim_vertex = data['prim_vertex'].reshape(1, 3)
            self.voxelise(prim_vertex)
            prim_vertex = prim_vertex.reshape(3)
            coords_aug, feats_aug, labels_aug = self._augment(coords, feats, labels, prim_vertex, round_coords=True)
            
            # Quantize (voxelise and detect duplicates)
            _, indices = ME.utils.sparse_quantize(
                coordinates=coords_aug, 
                return_index=True,
                quantization_size=1.0
            )
            coords_aug = coords_aug[indices]
            feats_aug = feats_aug[indices]
            labels_aug = labels_aug[indices]

            if self.contrastive:
                # restore to original values
                coords, feats, labels = coords_true, feats_true, labels_true
                if (self.training and np.random.rand() < 0.2):
                    # don't augment 20% of the times during training
                    coords_aug, feats_aug, labels_aug = coords, feats, labels
                feats_aug = np.log(feats_aug)
                output['coords_aug'] = torch.from_numpy(coords_aug).float()
                output['feats_aug'] = torch.from_numpy(feats_aug).float()
                output['labels_aug'] = torch.from_numpy(labels_aug).float()
            else:
                coords, feats, labels = coords_aug, feats_aug, labels_aug

        # log features
        feats = np.log(feats)

        output['coords'] = torch.from_numpy(coords).float()
        output['feats'] = torch.from_numpy(feats).float()
        output['labels'] = torch.from_numpy(labels).float()

        return output

