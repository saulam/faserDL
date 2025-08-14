"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: script to generate metadata.
"""


import torch
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt

# ---------- robust helpers ----------
def _mad(x, axis=None):
    med = np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(x - med), axis=axis)

def _madn(x, axis=None, eps=1e-12):
    return np.maximum(_mad(x, axis=axis) / 0.6745, eps)

def _select_cc_leptons(p_lep_true: np.ndarray, is_cc=None, eps=1e-12):
    """
    Prefer CC-only rows using is_cc. If not provided, drop exact-zeros (NC) by norm.
    Falls back to all rows if the selection would be empty.
    """
    if is_cc is not None:
        m = np.asarray(is_cc).astype(bool)
        sel = p_lep_true[m]
        if sel.shape[0] > 0:
            return sel
    norms = np.linalg.norm(p_lep_true, axis=1)
    sel = p_lep_true[norms > eps]
    return sel if sel.shape[0] > 0 else p_lep_true

# ---------- core stats ----------
@dataclass
class VectorStatsLog1p:
    # log1p/expm1 parameters for pT and pz (both >=0)
    k_T: float;   mu_uT: float; sigma_uT: float   # for uT = log1p(pT/k_T)
    k_Z: float;   mu_uZ: float; sigma_uZ: float   # for uZ = log1p(pz/k_Z)
    # robust loss scales
    s_xyz: Tuple[float, float, float]            # MADN per component (px,py,pz)
    s_mag: float                                 # MADN for ||p||
    # residual floors (for (true-reco)/max(true, tau))
    tau_ptmiss: float
    tau_evis: float

def compute_vector_stats_from_cartesian(
    p_true: np.ndarray,
    use_robust: bool = True,
    residual_floor_pct: float = 5.0,
    enforce_nonneg: bool = True,
) -> VectorStatsLog1p:
    """
    Compute stats for a 3-vector whose transverse and z components are non-negative scalars:
      - pT stats via uT = log1p(pT / k_T)  (so pT = k_T * expm1(uT))
      - pz stats via uZ = log1p(pz / k_Z)  (so pz = k_Z * expm1(uZ))
    Also returns robust scales for the loss (component-wise MADN + magnitude MADN)
    and residual floors (percentiles) for ptmiss and evis.

    IMPORTANT:
      * For visible momentum: call with ALL events (pz >= 0 by construction).
      * For lepton momentum: call with CC-ONLY rows (mask NC where lepton = 0).
    """
    assert p_true.ndim == 2 and p_true.shape[1] == 3, "p_true must be shape (N,3)"
    px, py, pz = p_true[:, 0], p_true[:, 1], p_true[:, 2]

    # Guard: these heads assume non-negative pz. If tiny negatives exist, clamp them.
    if enforce_nonneg:
        pz = np.maximum(pz, 0.0)

    pT  = np.sqrt(px**2 + py**2)
    mag = np.sqrt(px**2 + py**2 + pz**2)

    # --- log1p/expm1 parameterization stats for pT ---
    if use_robust:
        k_T = float(max(_madn(pT), 1e-8))
    else:
        k_T = float(max(np.std(pT), 1e-8))
    uT = np.log1p(pT / k_T)
    if use_robust:
        mu_uT   = float(np.median(uT))
        sigma_uT= float(max(_madn(uT), 1e-8))
    else:
        mu_uT   = float(np.mean(uT))
        sigma_uT= float(max(np.std(uT), 1e-8))

    # --- log1p/expm1 parameterization stats for pz (>=0) ---
    if use_robust:
        k_Z = float(max(_madn(pz), 1e-8))
    else:
        k_Z = float(max(np.std(pz), 1e-8))
    uZ = np.log1p(pz / k_Z)
    if use_robust:
        mu_uZ   = float(np.median(uZ))
        sigma_uZ= float(max(_madn(uZ), 1e-8))
    else:
        mu_uZ   = float(np.mean(uZ))
        sigma_uZ= float(max(np.std(uZ), 1e-8))

    # --- robust loss scales (used by Huber on Cartesian residuals) ---
    if use_robust:
        s_xyz = tuple(_madn(p_true, axis=0).astype(np.float64))
        s_mag = float(_madn(mag))
    else:
        s_xyz = tuple((np.std(p_true, axis=0) + 1e-12).astype(np.float64))
        s_mag = float(np.std(mag) + 1e-12)
    s_xyz = tuple(max(float(v), 1e-8) for v in s_xyz)
    s_mag = max(float(s_mag), 1e-8)

    # --- residual floors (for analysis-aligned scalar residuals) ---
    tau_ptmiss = float(np.percentile(pT,  residual_floor_pct))
    tau_evis   = float(np.percentile(mag, residual_floor_pct))

    return VectorStatsLog1p(
        k_T=k_T, mu_uT=mu_uT, sigma_uT=sigma_uT,
        k_Z=k_Z, mu_uZ=mu_uZ, sigma_uZ=sigma_uZ,
        s_xyz=s_xyz, s_mag=s_mag,
        tau_ptmiss=tau_ptmiss, tau_evis=tau_evis,
    )


# ---------- extended all-in-one ----------
def compute_all_stats(
    p_vis_true: np.ndarray,
    p_lep_true: np.ndarray,
    *,
    p_jet_true: Optional[np.ndarray] = None,   # optional
    is_cc: Optional[np.ndarray] = None,        # optional mask aligned with rows
    # robust / floors config
    use_robust: bool = True,
    residual_floor_pct: float = 5.0,
    # Option A eps for log(pT + eps)
    eps_T_vis: float = 1e-6,
    eps_T_lep: float = 1e-6,
    eps_T_jet: float = 1e-6,
    # jet stats controls
    jet_scales_cc_only: bool = True,          # CC-only loss scales for jet (recommended)
    compute_jet_inversion_stats: bool = False # set True only if you plan to predict jet directly
) -> Dict[str, Dict]:
    """
    Returns a dict with:
      - 'vis': VectorStats as dict
      - 'lep': VectorStats as dict
      - (optional) 'jet_loss_scales': {'s_xyz': (..), 's_mag': ..}
      - (optional) 'jet': VectorStats as dict (if compute_jet_inversion_stats=True)
      - (optional) class-specific vis floors: vis_tau_ptmiss_cc/nc, vis_tau_evis_cc/nc
    """
    out: Dict[str, Dict] = {}

    # --- vis & lep (always computed) ---
    vis_stats = compute_vector_stats_from_cartesian(
        p_vis_true, use_robust=use_robust, 
        residual_floor_pct=residual_floor_pct, enforce_nonneg=True,
    )
    p_lep_cc = _select_cc_leptons(p_lep_true, is_cc=is_cc)
    lep_stats = compute_vector_stats_from_cartesian(
        p_lep_cc, use_robust=use_robust, 
        residual_floor_pct=residual_floor_pct, enforce_nonneg=True,
    )
    out["vis"] = asdict(vis_stats)
    out["lep"] = asdict(lep_stats)

    # --- class-specific floors for vis (optional but handy) ---
    if is_cc is not None:
        mask = np.asarray(is_cc).astype(bool)
        for tag, arr in [("cc", p_vis_true[mask]), ("nc", p_vis_true[~mask])]:
            if arr.shape[0] > 0:
                px, py, pz = arr[:,0], arr[:,1], arr[:,2]
                pT = np.sqrt(px**2 + py**2)
                mag = np.sqrt(px**2 + py**2 + pz**2)
                out[f"vis_tau_ptmiss_{tag}"] = float(np.percentile(pT,  residual_floor_pct))
                out[f"vis_tau_evis_{tag}"]   = float(np.percentile(mag, residual_floor_pct))
            else:
                # fallback to global
                out[f"vis_tau_ptmiss_{tag}"] = out["vis"]["tau_ptmiss"]
                out[f"vis_tau_evis_{tag}"]   = out["vis"]["tau_evis"]

    # --- jet (optional) ---
    if p_jet_true is not None:
        assert p_jet_true.shape == p_vis_true.shape, "p_jet_true must be (N,3) aligned with p_vis_true"

        # Loss scales for jet (recommended if adding jet aux loss)
        if jet_scales_cc_only:
            if is_cc is None:
                raise ValueError("is_cc mask is required when jet_scales_cc_only=True.")
            mask = np.asarray(is_cc).astype(bool)
            pJ = p_jet_true[mask]
            if pJ.shape[0] == 0:
                pJ = p_jet_true   # graceful fallback
        else:
            pJ = p_jet_true

        if use_robust:
            sJ_xyz = tuple(_madn(pJ, axis=0).astype(np.float64))
            sJ_mag = float(_madn(np.linalg.norm(pJ, axis=1)))
        else:
            sJ_xyz = tuple((np.std(pJ, axis=0) + 1e-12).astype(np.float64))
            sJ_mag = float(np.std(np.linalg.norm(pJ, axis=1)) + 1e-12)

        sJ_xyz = tuple(max(float(v), 1e-8) for v in sJ_xyz)
        sJ_mag = max(float(sJ_mag), 1e-8)

        out["jet_loss_scales"] = {"s_xyz": sJ_xyz, "s_mag": sJ_mag}

        # Optional inversion stats if ever predicting jet directly (usually not needed)
        if compute_jet_inversion_stats:
            jet_stats = compute_vector_stats_from_cartesian(
                p_jet_true, use_robust=use_robust, 
                residual_floor_pct=residual_floor_pct, enforce_nonneg=False
            )
            out["jet"] = asdict(jet_stats)

    return out


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
        faser_cal_energy = data['faser_cal_energy'].item()
        faser_cal_modules = data['faser_cal_modules']#.sum()
        rear_cal_energy = data['rear_cal_energy'].item()
        rear_cal_modules = data['rear_cal_modules']#.sum()
        rear_hcal_energy = data['rear_hcal_energy'].item()
        rear_hcal_modules = data['rear_hcal_modules']#.sum()
        rear_mucal_energy = data['rear_mucal_energy'].item()
        vis_sp_momentum = data['vis_sp_momentum']
        out_lepton_momentum = data['out_lepton_momentum']
        in_neutrino_pdg = data['in_neutrino_pdg'].item()
        in_neutrino_energy = data['in_neutrino_energy'].item()
        out_lepton_energy = data['out_lepton_energy'].item()
        jet_momentum = data['jet_momentum']
        tauvis_momentum = data['tauvis_momentum']

        try:
            pdg = np.unique(np.concatenate([true_hits[reco_hit_true if isinstance(reco_hit_true, list) else reco_hit_true.astype(int)][:, 3] for reco_hit, reco_hit_true in zip(reco_hits, reco_hits_true)]))
        except:
            assert False, idx
            
        x = np.unique(data['reco_hits'][:, 0])
        y = np.unique(data['reco_hits'][:, 1])
        z = np.unique(np.stack((data['reco_hits'][:, 2], data['reco_hits'][:, 3]), axis=1), axis=0)
        q = data['reco_hits'][:, 4].round().astype(int)
        if is_cc:
            out_lepton_momentum = out_lepton_momentum.reshape(1, 3)
        else:
            out_lepton_momentum = np.zeros(shape=(1, 3))
        if is_cc and in_neutrino_pdg in [-16, 16]:  # nutau
            out_lepton_momentum = tauvis_momentum.reshape(1, 3)
        vis_sp_momentum = vis_sp_momentum.reshape(1, 3)
        jet_momentum = jet_momentum.reshape(1, 3)

        module_hits = np.bincount(reco_hits[:, 3].astype(int))
        module_hits = module_hits[module_hits>0]
        event_hits = np.array([reco_hits.shape[0]])
        rear_cal_energy = np.array([rear_cal_energy])
        rear_hcal_energy = np.array([rear_hcal_energy])
        rear_mucal_energy = np.array([rear_mucal_energy])
        faser_cal_energy = np.array([faser_cal_energy])
        in_neutrino_energy = np.array([in_neutrino_energy])
        out_lepton_energy = np.array([out_lepton_energy])
        is_cc = np.array([is_cc], dtype=np.bool_)

        if x.shape[0] == 0:
            print(idx)
            assert False

        return {"pdg": pdg, "x": x, "y": y, "z": z, "q": q, 
                "vis_sp_momentum": vis_sp_momentum,
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
                "is_cc": is_cc,
                "module_hits": module_hits,
                "event_hits": event_hits,
               }

dataset = SparseFASERCALDataset("/scratch/salonso/sparse-nns/faser/events_v5.1b")

def collate(batch):
    pdg = np.unique(np.concatenate([x['pdg'] for x in batch]))
    x = np.unique(np.concatenate([x['x'] for x in batch]))
    y = np.unique(np.concatenate([x['y'] for x in batch]))
    z = np.unique(np.concatenate([x['z'] for x in batch]), axis=0)
    q = np.concatenate([x['q'] for x in batch])
    vis_sp_momentum = np.concatenate([x['vis_sp_momentum'] for x in batch])
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
    is_cc = np.concatenate([x['is_cc'] for x in batch])
    module_hits = np.concatenate([x['module_hits'] for x in batch])
    event_hits = np.concatenate([x['event_hits'] for x in batch])
    
    
    return {"pdg": pdg, "x": x, "y": y, "z": z, "q": q, 
            "vis_sp_momentum": vis_sp_momentum,
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
            "is_cc": is_cc,
            "module_hits": module_hits,
            "event_hits": event_hits,
           }
    
loader = DataLoader(dataset, collate_fn=collate, batch_size=10, num_workers=10, drop_last=False, shuffle=False)

pdg = []
x = []
y = []
z = []
q_counter = Counter()  # otherwise it explodes
vis_sp_momentum = []
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
is_cc = []
module_hits = []
event_hits = []

t = tqdm(enumerate(loader), total=len(loader), disable=False)
for i, batch in t:
    pdg.append(batch["pdg"])
    x.append(batch["x"])
    y.append(batch["y"])
    z.append(batch["z"])
    q_counter.update(batch["q"])
    vis_sp_momentum.append(batch["vis_sp_momentum"])
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
    is_cc.append(batch["is_cc"])
    module_hits.append(batch["module_hits"])
    event_hits.append(batch["event_hits"])
    
print("Done with loader")

pdg = np.unique(np.concatenate(pdg))
x = np.unique(np.concatenate(x))
y = np.unique(np.concatenate(y))
z = np.unique(np.concatenate(z), axis=0)
vis_sp_momentum = np.concatenate(vis_sp_momentum)
out_lepton_momentum = np.concatenate(out_lepton_momentum)
jet_momentum = np.concatenate(jet_momentum)
faser_cal_energy = np.concatenate(faser_cal_energy)
faser_cal_modules = np.concatenate(faser_cal_modules)
rear_cal_energy = np.concatenate(rear_cal_energy)
rear_cal_modules = np.concatenate(rear_cal_modules)
rear_hcal_energy = np.concatenate(rear_hcal_energy)
rear_hcal_modules = np.concatenate(rear_hcal_modules)
rear_mucal_energy = np.concatenate(rear_mucal_energy)
in_neutrino_energy = np.concatenate(in_neutrino_energy)
out_lepton_energy = np.concatenate(out_lepton_energy)
is_cc = np.concatenate(is_cc)
module_hits = np.concatenate(module_hits)
event_hits = np.concatenate(event_hits)

print("Done with concat")

stats = compute_all_stats(
        p_vis_true=vis_sp_momentum,
        p_lep_true=out_lepton_momentum,
        p_jet_true=jet_momentum,
        is_cc=is_cc,
        use_robust=True,
        residual_floor_pct=5.0,
        jet_scales_cc_only=True,
        compute_jet_inversion_stats=False  # flip to True only if you plan a direct jet head
    )

def get_dict(x, axis=None):
    return {
        'mean':   x.mean(axis=axis),
        'median': np.median(x, axis=axis),
        'std':    x.std(axis=axis),
        'min':    x.min(axis=axis),
        'max':    x.max(axis=axis),
    }

# assemble base metadata
base_keys = ['in_neutrino_energy', 'out_lepton_energy',
             'faser_cal_energy', 'faser_cal_modules',
             'rear_cal_energy', 'rear_cal_modules',
             'rear_hcal_energy', 'rear_hcal_modules',
             'rear_mucal_energy', 'module_hits', 'event_hits',
            ]

metadata = {}
for key in base_keys:
    arr = locals()[key]
    axis = None
    metadata[key] = get_dict(arr, axis=axis)
    metadata[f"{key}_log1p"] = get_dict(np.log1p(arr), axis=axis)
    metadata[f"{key}_sqrt"]  = get_dict(np.sqrt(arr), axis=axis)

# q_counter has been built via q_counter.update(batch["q"])
total_hits = sum(q_counter.values())

# compute raw-q stats
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

# compute log1p-q stats
#    E[log1p(q)], E[(log1p(q))^2], etc.
q_lp_mean = sum(np.log1p(val)    * cnt for val, cnt in q_counter.items()) / total_hits
q_lp2_mean= sum((np.log1p(val)**2)* cnt for val, cnt in q_counter.items()) / total_hits
q_lp_std  = np.sqrt(q_lp2_mean - q_lp_mean**2)
q_lp_median = np.log1p(q_median)
q_lp_min    = np.log1p(q_min)
q_lp_max    = np.log1p(q_max)

# compute sqrt-q stats
#    E[sqrt(q)], E[(sqrt(q))^2], etc.
q_s_mean = sum(np.sqrt(val)    * cnt for val, cnt in q_counter.items()) / total_hits
q_s2_mean= sum((np.sqrt(val)**2)* cnt for val, cnt in q_counter.items()) / total_hits
q_s_std   = np.sqrt(q_s2_mean - q_s_mean**2)
q_s_median = np.sqrt(q_median)
q_s_min    = np.sqrt(q_min)
q_s_max    = np.sqrt(q_max)

# inject into metadata dict
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

metadata = {**metadata, **stats}  # for Python < 3.9 compatibility

# save metadata
import pickle as pk
with open("/scratch/salonso/sparse-nns/faser/events_v5.1b2/metadata_stats.pkl", "wb") as fd:
    pk.dump(metadata, fd)

print("Metadata saved.")
