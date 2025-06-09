"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: script to generate metadata.
"""


import torch
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
from collections import Counter
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
            np.savez(self.data_files[idx], filename=data['filename'], hits=filtered_hits)
            return filtered_hits[:, 7].max()
        else:
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

        is_cc = data['is_cc'].item()
        reco_hits_true = data['reco_hits_true']
        true_hits = data['true_hits']
        reco_hits = data['reco_hits']
        e_vis = data['e_vis'].item()
        pt_miss = data['pt_miss'].item()
        faser_cal_energy = data['faser_cal_energy'].item()
        faser_cal_modules = data['faser_cal_modules']#.sum()
        rear_cal_energy = data['rear_cal_energy'].item()
        rear_cal_modules = data['rear_cal_modules']#.sum()
        rear_hcal_energy = data['rear_hcal_energy'].item()
        rear_hcal_modules = data['rear_hcal_modules']#.sum()
        rear_mucal_energy = data['rear_mucal_energy'].item()
        out_lepton_momentum = data['out_lepton_momentum']
        in_neutrino_energy = data['in_neutrino_energy'].item()
        out_lepton_energy = data['out_lepton_energy'].item()
        jet_momentum = data['jet_momentum']
        try:
            pdg = np.unique(np.concatenate([true_hits[reco_hit_true if isinstance(reco_hit_true, list) else reco_hit_true.astype(int)][:, 3] for reco_hit, reco_hit_true in zip(reco_hits, reco_hits_true)]))
        except:
            assert False, idx
            
        x = np.unique(data['reco_hits'][:, 0])
        y = np.unique(data['reco_hits'][:, 1])
        z = np.unique(np.stack((data['reco_hits'][:, 2], data['reco_hits'][:, 3]), axis=1), axis=0)
        q = data['reco_hits'][:, 4].round().astype(int)
        e_vis = np.array([e_vis])
        if is_cc:
            e_vis_cc = e_vis
            e_vis_nc = np.array([])
            pt_miss = np.array([])
            out_lepton_momentum = out_lepton_momentum.reshape(1, 3)
        else:
            e_vis_cc = np.array([])
            e_vis_nc = e_vis
            pt_miss = np.array([pt_miss])
            out_lepton_momentum = np.array([]).reshape(0, 3)
        jet_momentum = jet_momentum.reshape(1, 3)

        rear_cal_energy = np.array([rear_cal_energy])
        rear_hcal_energy = np.array([rear_hcal_energy])
        rear_mucal_energy = np.array([rear_mucal_energy])
        faser_cal_energy = np.array([faser_cal_energy])
        in_neutrino_energy = np.array([in_neutrino_energy])
        out_lepton_energy = np.array([out_lepton_energy])

        if x.shape[0] == 0:
            print(idx)
            assert False

        return {"pdg": pdg, "x": x, "y": y, "z": z, "q": q, 
                "e_vis": e_vis, "e_vis_cc": e_vis_cc, "e_vis_nc": e_vis_nc, 
                "pt_miss": pt_miss,
                "out_lepton_momentum": out_lepton_momentum,
                "jet_momentum": jet_momentum,
                "faser_cal_energy": faser_cal_energy,
                "faser_cal_modules": faser_cal_modules,
                "rear_cal_energy": rear_cal_energy,
                "rear_cal_modules": rear_cal_modules,
                "rear_hcal_energy": rear_hcal_energy,
                "rear_hcal_modules": rear_hcal_modules,
                "rear_mucal_energy": rear_mucal_energy,
                "in_neutrino_energy": in_neutrino_energy,
                "out_lepton_energy": out_lepton_energy,
               }

dataset = SparseFASERCALDataset("/scratch/salonso/sparse-nns/faser/events_v5.1b")

def collate(batch):
    pdg = np.unique(np.concatenate([x['pdg'] for x in batch]))
    x = np.unique(np.concatenate([x['x'] for x in batch]))
    y = np.unique(np.concatenate([x['y'] for x in batch]))
    z = np.unique(np.concatenate([x['z'] for x in batch]), axis=0)
    q = np.concatenate([x['q'] for x in batch])
    e_vis = np.concatenate([x['e_vis'] for x in batch])
    e_vis_cc = np.concatenate([x['e_vis_cc'] for x in batch])
    e_vis_nc = np.concatenate([x['e_vis_nc'] for x in batch])
    pt_miss = np.concatenate([x['pt_miss'] for x in batch])
    out_lepton_momentum = np.concatenate([x['out_lepton_momentum'] for x in batch])
    jet_momentum = np.concatenate([x['jet_momentum'] for x in batch])
    faser_cal_energy = np.concatenate([x['faser_cal_energy'] for x in batch])
    faser_cal_modules = np.concatenate([x['faser_cal_modules'] for x in batch])
    rear_cal_energy = np.concatenate([x['rear_cal_energy'] for x in batch])
    rear_cal_modules = np.concatenate([x['rear_cal_modules'] for x in batch])
    rear_hcal_energy = np.concatenate([x['rear_hcal_energy'] for x in batch])
    rear_hcal_modules = np.concatenate([x['rear_hcal_modules'] for x in batch])
    rear_mucal_energy = np.concatenate([x['rear_mucal_energy'] for x in batch])
    in_neutrino_energy = np.concatenate([x['in_neutrino_energy'] for x in batch])
    out_lepton_energy = np.concatenate([x['out_lepton_energy'] for x in batch])
    
    
    return {"pdg": pdg, "x": x, "y": y, "z": z, "q": q, 
            "e_vis": e_vis, "e_vis_cc": e_vis_cc, "e_vis_nc": e_vis_nc,
            "pt_miss": pt_miss,
            "out_lepton_momentum": out_lepton_momentum,
            "jet_momentum": jet_momentum,
            "faser_cal_energy": faser_cal_energy,
            "faser_cal_modules": faser_cal_modules,
            "rear_cal_energy": rear_cal_energy,
            "rear_cal_modules": rear_cal_modules,
            "rear_hcal_energy": rear_hcal_energy,
            "rear_hcal_modules": rear_hcal_modules,
            "rear_mucal_energy": rear_mucal_energy,
            "in_neutrino_energy": in_neutrino_energy,
            "out_lepton_energy": out_lepton_energy,
           }
    
loader = DataLoader(dataset, collate_fn=collate, batch_size=10, num_workers=10, drop_last=False, shuffle=False)

pdg = []
x = []
y = []
z = []
q_counter = Counter()  # otherwise it explodes
e_vis = []
e_vis_cc = []
e_vis_nc = []
pt_miss = []
out_lepton_momentum = []
jet_momentum = []
faser_cal_energy = []
faser_cal_modules = []
rear_cal_energy = []
rear_cal_modules = []
rear_hcal_energy = []
rear_hcal_modules = []
rear_mucal_energy = []
in_neutrino_energy = []
out_lepton_energy = []

t = tqdm(enumerate(loader), total=len(loader), disable=False)
for i, batch in t:
    pdg.append(batch["pdg"])
    x.append(batch["x"])
    y.append(batch["y"])
    z.append(batch["z"])
    q_counter.update(batch["q"])
    e_vis.append(batch["e_vis"])
    e_vis_cc.append(batch["e_vis_cc"])
    e_vis_nc.append(batch["e_vis_nc"])
    pt_miss.append(batch["pt_miss"])
    out_lepton_momentum.append(batch["out_lepton_momentum"])
    jet_momentum.append(batch["jet_momentum"])
    faser_cal_energy.append(batch["faser_cal_energy"])
    faser_cal_modules.append(batch["faser_cal_modules"])
    rear_cal_energy.append(batch["rear_cal_energy"])
    rear_cal_modules.append(batch["rear_cal_modules"])
    rear_hcal_energy.append(batch["rear_hcal_energy"])
    rear_hcal_modules.append(batch["rear_hcal_modules"])
    rear_mucal_energy.append(batch["rear_mucal_energy"])
    in_neutrino_energy.append(batch["in_neutrino_energy"])
    out_lepton_energy.append(batch["out_lepton_energy"])
    
print("Done with loader")

pdg = np.unique(np.concatenate(pdg))
x = np.unique(np.concatenate(x))
y = np.unique(np.concatenate(y))
z = np.unique(np.concatenate(z), axis=0)
e_vis = np.concatenate(e_vis)
e_vis_cc = np.concatenate(e_vis_cc)
e_vis_nc = np.concatenate(e_vis_nc)
pt_miss = np.concatenate(pt_miss)
out_lepton_momentum = np.concatenate(out_lepton_momentum)
out_lepton_momentum_magnitude = np.linalg.norm(out_lepton_momentum, axis=1)
jet_momentum = np.concatenate(jet_momentum)
jet_momentum_magnitude = np.linalg.norm(jet_momentum, axis=1)
faser_cal_energy = np.concatenate(faser_cal_energy)
faser_cal_modules = np.concatenate(faser_cal_modules)
rear_cal_energy = np.concatenate(rear_cal_energy)
rear_cal_modules = np.concatenate(rear_cal_modules)
rear_hcal_energy = np.concatenate(rear_hcal_energy)
rear_hcal_modules = np.concatenate(rear_hcal_modules)
rear_mucal_energy = np.concatenate(rear_mucal_energy)
in_neutrino_energy = np.concatenate(in_neutrino_energy)
out_lepton_energy = np.concatenate(out_lepton_energy)

print("Done with concat")

def get_dict(x):
    return {'mean': x.mean(),
            'median': np.median(x),
            'std': x.std(),
            'min': x.min(),
            'max': x.max()}

# assemble base metadata
base_keys = ['e_vis', 'e_vis_cc', 'e_vis_nc', 'pt_miss',
             'out_lepton_momentum_magnitude', 'jet_momentum_magnitude', 
             'in_neutrino_energy', 'out_lepton_energy',
             'faser_cal_energy', 'faser_cal_modules',
             'rear_cal_energy', 'rear_cal_modules',
             'rear_hcal_energy', 'rear_hcal_modules',
             'rear_mucal_energy',
            ]

metadata = {}
for key in base_keys:
    arr = locals()[key]
    metadata[key] = get_dict(arr)
    metadata[f"{key}_log1p"] = get_dict(np.log1p(arr))
    metadata[f"{key}_sqrt"] = get_dict(np.sqrt(arr))

# … inside your script, after “Done with loader” …
# q_counter has been built via q_counter.update(batch["q"])
total_hits = sum(q_counter.values())

# 1) compute raw-q stats
q_mean = sum(val * cnt for val, cnt in q_counter.items()) / total_hits
q2_mean = sum((val**2) * cnt for val, cnt in q_counter.items()) / total_hits
q_std = np.sqrt(q2_mean - q_mean**2)

# find median, min, max
cum = 0
median_pos = total_hits / 2
for val in sorted(q_counter):
    cum += q_counter[val]
    if cum >= median_pos:
        q_median = val
        break
q_min = min(q_counter)
q_max = max(q_counter)

# 2) compute log1p-q stats
#    E[log1p(q)], E[(log1p(q))^2], etc.
q_lp_mean = sum(np.log1p(val)    * cnt for val, cnt in q_counter.items()) / total_hits
q_lp2_mean= sum((np.log1p(val)**2)* cnt for val, cnt in q_counter.items()) / total_hits
q_lp_std  = np.sqrt(q_lp2_mean - q_lp_mean**2)
q_lp_median = np.log1p(q_median)
q_lp_min    = np.log1p(q_min)
q_lp_max    = np.log1p(q_max)

# 3) compute sqrt-q stats
#    E[sqrt(q)], E[(sqrt(q))^2], etc.
q_s_mean = sum(np.sqrt(val)    * cnt for val, cnt in q_counter.items()) / total_hits
q_s2_mean= sum((np.sqrt(val)**2)* cnt for val, cnt in q_counter.items()) / total_hits
q_s_std   = np.sqrt(q_s2_mean - q_s_mean**2)
q_s_median = np.sqrt(q_median)
q_s_min    = np.sqrt(q_min)
q_s_max    = np.sqrt(q_max)

# 4) inject into your metadata dict
metadata['q'] = {
    'mean':   q_mean,
    'median': q_median,
    'std':    q_std,
    'min':    q_min,
    'max':    q_max
}
metadata['q_log1p'] = {
    'mean':   q_lp_mean,
    'median': q_lp_median,
    'std':    q_lp_std,
    'min':    q_lp_min,
    'max':    q_lp_max
}
metadata['q_sqrt'] = {
    'mean':   q_s_mean,
    'median': q_s_median,
    'std':    q_s_std,
    'min':    q_s_min,
    'max':    q_s_max
}

# include coordinate and pdg info
metadata.update({'x': x, 'y': y, 'z': z,
                 'ghost_pdg': set([-10000]),
                 'muonic_pdg': set([-13,13]),
                 'electromagnetic_pdg': set([-11,11,-15,15,22]),
                 'hadronic_pdg': set([p for p in pdg if p not in [-10000,-13,13,-11,11,-15,15,22]])})

# save metadata
import pickle as pk
with open("/scratch2/salonso/faser/events_v5.1b/metadata.pkl", "wb") as fd:
    pk.dump(metadata, fd)

print("Metadata with log1p and sqrt statistics saved.")
