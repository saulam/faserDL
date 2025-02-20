#!pip install matplotlib
from tqdm.notebook import tqdm
import torch
import os
import numpy as np
#import MinkowskiEngine as ME
from glob import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
        data = np.load(self.data_files[idx], allow_pickle=True)
        
        reco_hits_true = data['reco_hits_true']
        true_hits = data['true_hits']
        reco_hits = data['reco_hits']
        evis = data['evis'].item()
        ptmiss = data['ptmiss'].item()
        rearcal_energydeposit = data['rearcal_energydeposit'].item()
        rearhcal_energydeposit = data['rearhcal_energydeposit'].item()
        rearmucal_energydeposit = data['rearmucal_energydeposit'].item()
        fasercal_energydeposit = data['fasercal_energydeposit'].item()
        rearhcalmodules = data['rearhcalmodules']
        fasercalmodules = data['fasercalmodules']
        out_lepton_momentum = data['out_lepton_momentum']
        out_lepton_energy = data['out_lepton_energy'].item()
        jet_momentum = data['jet_momentum']

        if len(reco_hits) == 0:
            os.remove(self.data_files[idx])
            print(f"File '{idx}' has been deleted successfully.")
            aux = np.array([])
            return {"pdg": aux, "x": aux, "y": aux, "z": aux.reshape(0, 2), "q": aux,
                    "evis": aux, "ptmiss": aux,
                    "rearcal_energydeposit": aux,
                    "rearhcal_energydeposit": aux,
                    "rearmucal_energydeposit": aux,
                    "fasercal_energydeposit": aux,
                    "rearhcalmodules": aux,
                    "fasercalmodules": aux,
                    "out_lepton_momentum": aux.reshape(0, 3),
                    "out_lepton_energy": aux,
                    "jet_momentum": aux.reshape(0, 3),
                   }

        try:
            pdg = np.unique(np.concatenate([true_hits[reco_hit_true if isinstance(reco_hit_true, list) else reco_hit_true.astype(int)][:, 3] for reco_hit, reco_hit_true in zip(reco_hits, reco_hits_true)]))
        except:
            assert False, idx
            
        x = np.unique(data['reco_hits'][:, 0])
        y = np.unique(data['reco_hits'][:, 1])
        z = np.unique(np.stack((data['reco_hits'][:, 2], data['reco_hits'][:, 3]), axis=1), axis=0)
        #q = np.log(data['reco_hits'][:, 4])
        q = data['reco_hits'][:, 4]
        evis = np.array([evis])
        ptmiss = np.array([ptmiss])
        rearcal_energydeposit = np.array([rearcal_energydeposit])
        rearhcal_energydeposit = np.array([rearhcal_energydeposit])
        rearmucal_energydeposit = np.array([rearmucal_energydeposit])
        fasercal_energydeposit = np.array([fasercal_energydeposit])
        out_lepton_momentum = out_lepton_momentum.reshape(1, 3)
        out_lepton_energy = np.array([out_lepton_energy])
        jet_momentum = jet_momentum.reshape(1, 3)
        
        if x.shape[0] == 0:
            print(idx)
            assert False

        return {"pdg": pdg, "x": x, "y": y, "z": z, "q": q, "evis": evis, "ptmiss": ptmiss,
                "rearcal_energydeposit": rearcal_energydeposit,
                "rearhcal_energydeposit": rearhcal_energydeposit,
                "rearmucal_energydeposit": rearmucal_energydeposit,
                "fasercal_energydeposit": fasercal_energydeposit,
                "rearhcalmodules": rearhcalmodules,
                "fasercalmodules": fasercalmodules,
                "out_lepton_momentum": out_lepton_momentum,
                "out_lepton_energy": out_lepton_energy,
                "jet_momentum": jet_momentum
               }

dataset = SparseFASERCALDataset("/scratch2/salonso/faser/events_v3.5")

def collate(batch):
    pdg = np.unique(np.concatenate([x['pdg'] for x in batch]))
    x = np.unique(np.concatenate([x['x'] for x in batch]))
    y = np.unique(np.concatenate([x['y'] for x in batch]))
    z = np.unique(np.concatenate([x['z'] for x in batch]), axis=0)
    q = np.concatenate([x['q'] for x in batch])
    evis = np.concatenate([x['evis'] for x in batch])
    ptmiss = np.concatenate([x['ptmiss'] for x in batch])
    rearcal_energydeposit = np.concatenate([x['rearcal_energydeposit'] for x in batch])
    rearhcal_energydeposit = np.concatenate([x['rearhcal_energydeposit'] for x in batch])
    rearmucal_energydeposit = np.concatenate([x['rearmucal_energydeposit'] for x in batch])
    fasercal_energydeposit = np.concatenate([x['fasercal_energydeposit'] for x in batch])
    rearhcalmodules = np.concatenate([x['rearhcalmodules'] for x in batch])
    fasercalmodules = np.concatenate([x['fasercalmodules'] for x in batch])
    out_lepton_momentum = np.concatenate([x['out_lepton_momentum'] for x in batch])
    out_lepton_energy = np.concatenate([x['out_lepton_energy'] for x in batch])
    jet_momentum = np.concatenate([x['jet_momentum'] for x in batch])
    
    return {"pdg": pdg, "x": x, "y": y, "z": z, "q": q, "evis": evis, "ptmiss": ptmiss,
            "rearcal_energydeposit": rearcal_energydeposit,
            "rearhcal_energydeposit": rearhcal_energydeposit,
            "rearmucal_energydeposit": rearmucal_energydeposit,
            "fasercal_energydeposit": fasercal_energydeposit,
            "rearhcalmodules": rearhcalmodules,
            "fasercalmodules": fasercalmodules,
            "out_lepton_momentum": out_lepton_momentum,
            "out_lepton_energy": out_lepton_energy,
            "jet_momentum": jet_momentum
           }
    
loader = DataLoader(dataset, collate_fn=collate, batch_size=10, num_workers=10, drop_last=False, shuffle=False)

pdg = []
x = []
y = []
z = []
q = []
evis = []
ptmiss = []
rearcal_energydeposit = []
rearhcal_energydeposit = []
rearmucal_energydeposit = []
fasercal_energydeposit = []
rearhcalmodules = []
fasercalmodules = []
out_lepton_momentum = []
out_lepton_energy = []
jet_momentum = []

t = tqdm(enumerate(loader), total=len(loader), desc="Loading", disable=False)
for i, batch in t:
    pdg.append(batch["pdg"])
    x.append(batch["x"])
    y.append(batch["y"])
    z.append(batch["z"])
    q.append(batch["q"])
    evis.append(batch["evis"])
    ptmiss.append(batch["ptmiss"])
    rearcal_energydeposit.append(batch["rearcal_energydeposit"])
    rearhcal_energydeposit.append(batch["rearhcal_energydeposit"])
    rearmucal_energydeposit.append(batch["rearmucal_energydeposit"])
    fasercal_energydeposit.append(batch["fasercal_energydeposit"])
    rearhcalmodules.append(batch["rearhcalmodules"])
    fasercalmodules.append(batch["fasercalmodules"])
    out_lepton_momentum.append(batch["out_lepton_momentum"])
    out_lepton_energy.append(batch["out_lepton_energy"])
    jet_momentum.append(batch["jet_momentum"])

pdg = np.unique(np.concatenate(pdg))
x = np.unique(np.concatenate(x))
y = np.unique(np.concatenate(y))
z = np.unique(np.concatenate(z), axis=0)
q = np.concatenate(q)
evis = np.concatenate(evis)
ptmiss = np.concatenate(ptmiss)
rearcal_energydeposit = np.concatenate(rearcal_energydeposit)
rearhcal_energydeposit = np.concatenate(rearhcal_energydeposit)
rearmucal_energydeposit = np.concatenate(rearmucal_energydeposit)
fasercal_energydeposit = np.concatenate(fasercal_energydeposit)
rearhcalmodules = np.concatenate(rearhcalmodules)
fasercalmodules = np.concatenate(fasercalmodules)
out_lepton_momentum = np.concatenate(out_lepton_momentum)
out_lepton_energy = np.concatenate(out_lepton_energy)
jet_momentum = np.concatenate(jet_momentum)
out_lepton_momentum_magnitude = np.linalg.norm(out_lepton_momentum, axis=1)
jet_momentum_magnitude = np.linalg.norm(jet_momentum, axis=1)

import pickle as pk

ghost_pdg = [-10000]
muonic_pdg = [-13, 13]
electromagnetic_pdg = [-11, 11, -15, 15, 22]
hadronic_pdg = [x for x in pdg if x not in ghost_pdg]
hadronic_pdg = [x for x in hadronic_pdg if x not in muonic_pdg]
hadronic_pdg = [x for x in hadronic_pdg if x not in electromagnetic_pdg]

dic = {'x': x, 'y': y, 'z': z,
       'rearcal_energydeposit': get_dict(rearcal_energydeposit),
       'rearhcal_energydeposit': get_dict(rearhcal_energydeposit),
       'rearmucal_energydeposit': get_dict(rearmucal_energydeposit),
       'fasercal_energydeposit': get_dict(fasercal_energydeposit),
       'rearhcalmodules': get_dict(rearhcalmodules),
       'fasercalmodules': get_dict(fasercalmodules),
       'out_lepton_momentum': get_dict(out_lepton_momentum),
       'out_lepton_momentum_magnitude': get_dict(out_lepton_momentum_magnitude),
       'out_lepton_energy': get_dict(out_lepton_energy),
       'jet_momentum': get_dict(jet_momentum),
       'jet_momentum_magnitude': get_dict(jet_momentum_magnitude),
       'q': get_dict(q),
       'evis': get_dict(evis),
       'ptmiss': get_dict(ptmiss),
       'ghost_pdg': set(ghost_pdg),
       'muonic_pdg': set(muonic_pdg), 
       'electromagnetic_pdg': set(electromagnetic_pdg),
       'hadronic_pdg': set(hadronic_pdg)
      }

with open("/scratch2/salonso/faser/events_v3.5/metadata.pkl", "wb") as fd:
    pk.dump(dic, fd)

