import pickle as pk
import numpy as np
import torch
import numpy as np
import MinkowskiEngine as ME
from glob import glob
from torch.utils.data import Dataset
    
    
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

    def collate_sparse_minkowski(self, batch):
        """
        Collates a batch of data into a format suitable for MinkowskiEngine.

        Args:
        batch (list): List of data dictionaries.

        Returns:
        dict: Collated data with coordinates, features, and labels.
        """
        coords = [d['coords'].int() for d in batch if d['coords'] is not None]
        feats = torch.cat([d['feats'] for d in batch if d['coords'] is not None])
        labels = torch.cat([d['labels'] for d in batch if d['coords'] is not None])
        
        return {'f': feats, 'c': coords, 'y': labels}
    
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
            y[i, 2] = 1 if label_set & electromagnetic_pdg else 0
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
        feats = np.log(reco_hits[:, 4]).reshape(-1, 1) 

        # voxelise
        self.voxelise(coords)
        
        # PDGs to labels
        labels = self.process_labels(reco_hit_pdgs)
        
        # torch tensors
        coords = torch.from_numpy(coords).float()
        feats = torch.from_numpy(feats).float()
        labels = torch.from_numpy(labels).float()

        '''
        # Quantize (voxelise and detect duplicates)
        coords, feats, labels = ME.utils.sparse_quantize(
            coordinates=coords, features=feats, labels=labels, quantization_size=1.0
        )
        '''

        return {'run_number': run_number,
                'event_id': event_id,
                'coords': coords, 
                'feats': feats, 
                'labels': labels,
               }
    
