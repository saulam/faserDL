import torch
import numpy as np
import MinkowskiEngine as ME
from glob import glob
from torch.utils.data import Dataset

# Define the PDG codes for hadronic and electronic particles
def pdg2label(pdg_codes):
    """
    Classifies PDG codes into three categories:
    0: Muonic
    1: Electromagnetic
    2: Hadronic

    Args:
    pdg_codes (np.array): Array of PDG codes to be classified.

    Returns:
    None (The input array is modified in place)
    """
    # Create masks for muonic and electromagnetic particles
    muonic_mask = np.isin(np.abs(pdg_codes), [13, -13, 14, -14])
    electromagnetic_mask = np.isin(np.abs(pdg_codes), [11, -11, 22])

    # Assign labels based on the masks
    pdg_codes[muonic_mask] = 0  # muonic
    pdg_codes[electromagnetic_mask] = 1  # electromagnetic
    pdg_codes[~(electromagnetic_mask | muonic_mask)] = 2  # hadronic
    

def normalize_log_scale(data, data_min=0.5, data_max=389.8885154724121, epsilon=1e-12):    
    """
    Normalizes the input data using log scale transformation and rescales to [-1, 1].

    Args:
    data (np.array): The data to be normalized.
    data_min (float): Minimum value for normalization.
    data_max (float): Maximum value for normalization.
    epsilon (float): Small value to avoid division by zero.

    Returns:
    np.array: Normalized data.
    """
    # Step 1: Apply log transformation
    data_log = np.log(data + epsilon)
    
    # Step 3: Rescale to [-1, 1]
    data_log_min = np.log(data_min + epsilon)
    data_log_max = np.log(data_max + epsilon)
    
    data_final = 2 * (data_log - data_log_min) / (data_log_max - data_log_min) - 1
    
    return data_final
    
    
class SparseFASERCALDataset(Dataset):
    def __init__(self, root, shuffle=False, **kwargs):
        """
        Initializes the SparseFASERCALDataset class.

        Args:
        root (str): Root directory containing the data files.
        shuffle (bool): Whether to shuffle the dataset (default: False).
        """
        self.root = root
        self.data_files = self.processed_file_names
        self.train = False
        self.total_events = self.__len__
        self.charge_range = (0.5, 389.8885154724121)

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
    
    def remove_empty_events(self, idx):
        """
        Removes empty events based on energy deposition threshold.

        Args:
        idx (int): Index of the data file.

        Returns:
        float: Maximum energy deposition in the filtered hits.
        """
        data = np.load(self.data_files[idx])
        hits = data['hits']
    
        # Filter hits where energy (hits[:, 7]) is >= 0.5
        filtered_hits = hits[hits[:, 7] >= 0.5]
    
        if filtered_hits.shape[0] > 0:
            # Save the filtered hits back to the file if there are any hits left
            np.savez(self.data_files[idx], filename=data['filename'], hits=filtered_hits)
            return filtered_hits[:, 7].max()
        else:
            # Remove the file if no hits meet the energy threshold
            os.remove(self.data_files[idx])
            return 0
        
    def __getitem__(self, idx):
        """
        Retrieves a data sample by index.

        Args:
        idx (int): Index of the data sample.

        Returns:
        dict: Data sample with filename, coordinates, features, and labels.
        """
        data = np.load(self.data_files[idx])
        hits = data['hits']

        coords = hits[:, 4:7]
        feats = hits[:, 7:8]
        labels = hits[:, 3]

        # Remove empty hits
        mask = feats[:, 0] > self.charge_range[0]

        # Filter empty events
        if coords.shape[0] == 0 or mask.sum() == 0:
            return {'filename': data['filename'], 'coords': None}
        
        coords = coords[mask]
        feats = feats[mask]
        labels = labels[mask]

        # Normalize features and convert labels
        feats = normalize_log_scale(feats, self.charge_range[0], self.charge_range[1])
        pdg2label(labels)

        coords = torch.FloatTensor(coords)
        feats = torch.FloatTensor(feats)
        labels = torch.IntTensor(labels)

        # Quantize (voxelise and detect duplicates)
        coords, feats, labels = ME.utils.sparse_quantize(
            coordinates=coords, features=feats, labels=labels, quantization_size=1.0
        )

        # Remove points with labels -100 (duplicates)
        valid_indices = labels != -100
        coords = coords[valid_indices]
        feats = feats[valid_indices]
        labels = labels[valid_indices]
        
        return {'filename': data['filename'], 'coords': coords, 'feats': feats, 'labels': labels}