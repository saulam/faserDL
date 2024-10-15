import pickle as pk
import numpy as np
import torch
import numpy as np
import MinkowskiEngine as ME
from glob import glob
from torch.utils.data import Dataset
from utils import random_rotation_saul

    
class SparseFASERCALDatasetCls(Dataset):
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

    def _augment(self, coords, feats, prim_vertex, round_coords=True):
        coords, feats = coords.copy(), feats.copy()

        # rotate
        #coords = self._rotate(coords, prim_vertex)
        coords = self._rotate_90(coords)
        # translate
        coords = self._translate(coords)
        # drop voxels
        coords, feats = self._drop(coords, feats, std_dev=0.1)
        # shift feature values
        feats = self._shift_q_gaussian(feats, std_dev=0.05)
        # keep within limits
        #coords, feats, labels = self._within_limits(coords, feats, labels)
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

        return coords, feats


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

         final_rotation_matrix = np.eye(3) 

         for axis in ['x', 'y', 'z']:  # Loop over each axis
             if np.random.choice([True, False]):  # Randomly decide if we rotate
                 angle = np.random.choice([0, 90, 180, 270])
                 final_rotation_matrix = final_rotation_matrix @ rotations[axis][angle]

         return point_cloud @ final_rotation_matrix.T

 
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


    def _drop(self, coords, feats, std_dev=0.1):
        p = abs(np.random.randn(1) * std_dev)
        mask = np.random.rand(coords.shape[0]) > p
        #don't drop all coordinates
        if mask.sum() == 0:
            return coords, feats, labels
        return coords[mask], feats[mask]


    def _shift_q_uniform(self, feats, max_scale_factor=0.1):
        shift = 1 - np.random.rand(*feats.shape) * max_scale_factor
        return feats * shift


    def _shift_q_gaussian(self, feats, std_dev=0.1):
        shift = 1 - np.random.randn(*feats.shape) * std_dev
        return feats * shift


    def _within_limits(self, coords, feats):
        mask = (coords[:, 0] >= 0) & (coords[:, 0] < 48) & \
           (coords[:, 1] >= 0) & (coords[:, 1] < 48) & \
           (coords[:, 2] >= 0) & (coords[:, 2] < 400)
        return coords[mask], feats[mask]

 
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

    def pdg2label(self, pdg, iscc):
        """
        PDG to label.
        """
        if iscc:
            if pdg in [-12, 12]:  # ICC nue
                label = 0
            elif pdg in [-14, 14]:  # CC numu
                label = 1
            elif pdg in [-16, 16]:  # CC nutau
                label = 2
        else:
            label = 3  # NC

        return label

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
        true_hits = data['true_hits']
        reco_hits = data['reco_hits']
        reco_hits_true = data['reco_hits_true']
        pdg = data['in_neutrino_pdg']
        iscc = data['iscc']

        # retrieve coordiantes and features (energy deposited)
        coords = reco_hits[:, :3]
        feats = reco_hits[:, 4].reshape(-1, 1) 

        # voxelise
        self.voxelise(coords)
        
        # PDGs to labels
        label = self.pdg2label(pdg, iscc)

        # output
        output = {'run_number': run_number,
                  'event_id': event_id}

        if self.training and np.random.rand() > 0.05:
            prim_vertex = data['prim_vertex'].reshape(1, 3)
            self.voxelise(prim_vertex)
            prim_vertex = prim_vertex.reshape(3)
           
            # augmented event
            coords, feats = self._augment(coords, feats, prim_vertex, round_coords=False)
           
        # log features
        feats = np.log(feats)

        output['coords'] = torch.from_numpy(coords).float()
        output['feats'] = torch.from_numpy(feats).float()
        output['labels'] = torch.tensor([label]).long()

        return output

