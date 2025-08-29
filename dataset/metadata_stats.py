"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 08.25

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
import pickle as pk

# -------------------------
# Robust stats helpers
# -------------------------

MADN_CONST = 1.482602218505602  # normalizing constant so MADN ~= std for Gaussian

def _mad(x, axis=None):
    med = np.median(x, axis=axis, keepdims=True)
    return np.median(np.abs(x - med), axis=axis)

def _madn(x, axis=None, eps=1e-12):
    return np.maximum(_mad(x, axis=axis) * MADN_CONST, eps)

def _transform_lookup(name: str):
    if name == "identity":
        return lambda x: x
    if name == "log1p":
        return np.log1p
    if name == "sqrt":
        return np.sqrt
    raise ValueError(f"Unknown transform: {name}")

# -------------------------
# Robust params from Counter
# -------------------------

def _N(counter: Counter) -> int:
    return sum(counter.values())

def _weighted_median(counter: Counter):
    """Lower weighted median: first value where cumulative count >= N/2."""
    N = _N(counter)
    if N == 0:
        return 0.0
    median_pos = N / 2
    cum = 0
    for val in sorted(counter):
        cum += counter[val]
        if cum >= median_pos:
            return float(val)
    return float(next(iter(counter)))

def _mad_from_counter(counter: Counter, center: float):
    """Weighted median of absolute deviations |x - center| (no expansion)."""
    N = _N(counter)
    if N == 0:
        return 0.0
    devs = {}
    for v, cnt in counter.items():
        d = abs(v - center)
        devs[d] = devs.get(d, 0) + cnt
    median_pos = N / 2
    cum = 0
    for d in sorted(devs):
        cum += devs[d]
        if cum >= median_pos:
            return float(d)
    return 0.0

def _madn_from_counter(counter: Counter, center: float):
    return _mad_from_counter(counter, center) * MADN_CONST

def compute_robust_params_for_transform(q_counter: Counter, transform: str, eps: float = 1e-8):
    """
    Robust two-stage standardization:
      1) k = MADN(q)  (robust scale in original space)
      2) u = f(q / k) where f is monotone (identity|log1p|sqrt)
      3) mu = median(u); sigma = MADN(u)
    """
    N = _N(q_counter)
    if N == 0:
        return {
            "transform": transform,
            "k": eps, "mu": 0.0, "sigma": 1.0,
            "orig_median": 0.0, "orig_min": 0.0, "orig_max": 0.0,
            "u_median": 0.0, "u_min": 0.0, "u_max": 0.0,
        }

    f = _transform_lookup(transform)

    # k in original space (robust)
    q_med = _weighted_median(q_counter)
    k = max(_madn_from_counter(q_counter, q_med), eps)

    # transformed stats u = f(q / k)
    u_med = f(q_med / k)

    # MAD in u-space (weighted)
    devs_u = {}
    for v, cnt in q_counter.items():
        du = abs(f(v / k) - u_med)
        devs_u[du] = devs_u.get(du, 0) + cnt
    median_pos = N / 2
    cum = 0
    mad_u = 0.0
    for d in sorted(devs_u):
        cum += devs_u[d]
        if cum >= median_pos:
            mad_u = float(d)
            break
    sigma = max(mad_u * MADN_CONST, eps)

    q_min = min(q_counter)
    q_max = max(q_counter)
    u_min = f(q_min / k)
    u_max = f(q_max / k)

    return {
        "transform": transform,
        "k": float(k),
        "mu": float(u_med),
        "sigma": float(sigma),
        "orig_median": float(q_med),
        "orig_min": float(q_min),
        "orig_max": float(q_max),
        "u_median": float(u_med),
        "u_min": float(u_min),
        "u_max": float(u_max),
    }

def add_robust_standardization_metadata(q_counter: Counter, metadata: dict, key_prefix: str = "q"):
    """
    Fills metadata with robust standardization parameters for:
      - identity (no transform): key f"{key_prefix}"
      - log1p:                     f"{key_prefix}_log1p"
      - sqrt:                      f"{key_prefix}_sqrt"
    """
    meta_identity = compute_robust_params_for_transform(q_counter, "identity")
    meta_log1p    = compute_robust_params_for_transform(q_counter, "log1p")
    meta_sqrt     = compute_robust_params_for_transform(q_counter, "sqrt")

    metadata[f"{key_prefix}"]       = meta_identity
    metadata[f"{key_prefix}_log1p"] = meta_log1p
    metadata[f"{key_prefix}_sqrt"]  = meta_sqrt

    return metadata

# -------------------------
# Robust params from arrays (for base_keys)
# -------------------------

def compute_robust_params_for_array(arr: np.ndarray, transform: str, eps: float = 1e-8):
    """
    Same robust scheme as the Counter version, but for dense arrays:
      1) k = MADN(arr)
      2) u_i = f(arr_i / k)
      3) mu = median(u), sigma = MADN(u)
    Also returns optional mins/max for monitoring.
    """
    arr = np.asarray(arr).ravel()
    if arr.size == 0:
        return {
            "transform": transform,
            "k": eps, "mu": 0.0, "sigma": 1.0,
            "orig_median": 0.0, "orig_min": 0.0, "orig_max": 0.0,
            "u_median": 0.0, "u_min": 0.0, "u_max": 0.0,
        }

    f = _transform_lookup(transform)

    # Robust scale k in original space
    q_med = float(np.median(arr))
    k = max(float(_madn(arr)), eps)

    # Transformed values
    u = f(arr / k)

    mu = float(np.median(u))
    sigma = max(float(_madn(u)), eps)

    q_min = float(np.min(arr))
    q_max = float(np.max(arr))
    u_min = float(f(q_min / k))
    u_max = float(f(q_max / k))

    return {
        "transform": transform,
        "k": float(k),
        "mu": float(mu),
        "sigma": float(sigma),
        "orig_median": float(q_med),
        "orig_min": float(q_min),
        "orig_max": float(q_max),
        "u_median": float(mu),
        "u_min": float(u_min),
        "u_max": float(u_max),
    }

def add_robust_standardization_metadata_array(arr: np.ndarray, metadata: dict, key_prefix: str):
    """
    Mirrors add_robust_standardization_metadata but for arrays.
    Produces:
      - f"{key_prefix}"
      - f"{key_prefix}_log1p"
      - f"{key_prefix}_sqrt"
    """
    metadata[f"{key_prefix}"]       = compute_robust_params_for_array(arr, "identity")
    metadata[f"{key_prefix}_log1p"] = compute_robust_params_for_array(arr, "log1p")
    metadata[f"{key_prefix}_sqrt"]  = compute_robust_params_for_array(arr, "sqrt")
    return metadata

# -------------------------
# Dataset (unchanged behavior)
# -------------------------

def _select_cc_leptons(p_lep_true: np.ndarray, is_cc=None, eps=1e-12):
    """
    Prefer CC-only rows using is_cc. If not provided, drop exact-zeros (NC) by norm.
    """
    if is_cc is not None:
        m = np.asarray(is_cc).astype(bool)
        sel = p_lep_true[m]
        if sel.shape[0] > 0:
            return sel
    norms = np.linalg.norm(p_lep_true, axis=1)
    sel = p_lep_true[norms > eps]
    return sel if sel.shape[0] > 0 else p_lep_true

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

    if enforce_nonneg:
        pz = np.maximum(pz, 0.0)

    pT  = np.sqrt(px**2 + py**2)
    mag = np.sqrt(px**2 + py**2 + pz**2)

    # log1p/expm1 parameterization stats for pT
    if use_robust:
        k_T = float(max(_madn(pT), 1e-8))
        uT = np.log1p(pT / k_T)
        mu_uT   = float(np.median(uT))
        sigma_uT= float(max(_madn(uT), 1e-8))
    else:
        k_T = float(max(np.std(pT), 1e-8))
        uT = np.log1p(pT / k_T)
        mu_uT   = float(np.mean(uT))
        sigma_uT= float(max(np.std(uT), 1e-8))

    # log1p/expm1 parameterization stats for pz (>=0)
    if use_robust:
        k_Z = float(max(_madn(pz), 1e-8))
        uZ = np.log1p(pz / k_Z)
        mu_uZ   = float(np.median(uZ))
        sigma_uZ= float(max(_madn(uZ), 1e-8))
    else:
        k_Z = float(max(np.std(pz), 1e-8))
        uZ = np.log1p(pz / k_Z)
        mu_uZ   = float(np.mean(uZ))
        sigma_uZ= float(max(np.std(uZ), 1e-8))

    # robust loss scales
    if use_robust:
        s_xyz = tuple(_madn(p_true, axis=0).astype(np.float64))
        s_mag = float(_madn(mag))
    else:
        s_xyz = tuple((np.std(p_true, axis=0) + 1e-12).astype(np.float64))
        s_mag = float(np.std(mag) + 1e-12)

    s_xyz = tuple(max(float(v), 1e-8) for v in s_xyz)
    s_mag = max(float(s_mag), 1e-8)

    # residual floors
    tau_ptmiss = float(np.percentile(pT,  residual_floor_pct))
    tau_evis   = float(np.percentile(mag, residual_floor_pct))

    return VectorStatsLog1p(
        k_T=k_T, mu_uT=mu_uT, sigma_uT=sigma_uT,
        k_Z=k_Z, mu_uZ=mu_uZ, sigma_uZ=sigma_uZ,
        s_xyz=s_xyz, s_mag=s_mag,
        tau_ptmiss=tau_ptmiss, tau_evis=tau_evis,
    )

def compute_all_stats(
    p_vis_true: np.ndarray,
    p_lep_true: np.ndarray,
    p_jet_true: Optional[np.ndarray] = None,
    is_cc: Optional[np.ndarray] = None,
    use_robust: bool = True,
    residual_floor_pct: float = 5.0,
    jet_scales_cc_only: bool = True,          # CC-only loss scales for jet (recommended)
    compute_jet_inversion_stats: bool = False # set True only if planning to predict jet directly
) -> Dict[str, Dict]:
    """
    Returns a dict with:
      - 'vis': VectorStats as dict
      - 'lep': VectorStats as dict
      - (optional) 'jet_loss_scales': {'s_xyz': (..), 's_mag': ..}
      - (optional) 'jet': VectorStats as dict (if compute_jet_inversion_stats=True)
      - (optional) class-specific vis floors: vis_tau_ptmiss_cc/nc, vis_tau_evis_cc/nc

    Notes:
      * Output format/keys are intentionally unchanged.
      * Uses robust statistics when use_robust=True.
    """
    out: Dict[str, Dict] = {}

    # --- visible & lepton stats (always computed) ---
    vis_stats = compute_vector_stats_from_cartesian(
        p_vis_true,
        use_robust=use_robust,
        residual_floor_pct=residual_floor_pct,
        enforce_nonneg=True,
    )
    p_lep_cc = _select_cc_leptons(p_lep_true, is_cc=is_cc)
    lep_stats = compute_vector_stats_from_cartesian(
        p_lep_cc,
        use_robust=use_robust,
        residual_floor_pct=residual_floor_pct,
        enforce_nonneg=True,
    )
    out["vis"] = asdict(vis_stats)
    out["lep"] = asdict(lep_stats)

    # --- class-specific floors for vis (optional but useful) ---
    if is_cc is not None:
        mask = np.asarray(is_cc).astype(bool)
        for tag, arr in (("cc", p_vis_true[mask]), ("nc", p_vis_true[~mask])):
            if arr.shape[0] > 0:
                px, py, pz = arr[:, 0], arr[:, 1], arr[:, 2]
                pT  = np.sqrt(px**2 + py**2)
                mag = np.sqrt(px**2 + py**2 + pz**2)
                out[f"vis_tau_ptmiss_{tag}"] = float(np.percentile(pT,  residual_floor_pct))
                out[f"vis_tau_evis_{tag}"]   = float(np.percentile(mag, residual_floor_pct))
            else:
                # fallback to global floors
                out[f"vis_tau_ptmiss_{tag}"] = out["vis"]["tau_ptmiss"]
                out[f"vis_tau_evis_{tag}"]   = out["vis"]["tau_evis"]

    # --- jet-related stats (optional) ---
    if p_jet_true is not None:
        assert p_jet_true.shape == p_vis_true.shape, "p_jet_true must be (N,3) aligned with p_vis_true"

        # Loss scales for jet (used e.g. by auxiliary jet losses)
        if jet_scales_cc_only:
            if is_cc is None:
                raise ValueError("is_cc mask is required when jet_scales_cc_only=True.")
            mask = np.asarray(is_cc).astype(bool)
            pJ = p_jet_true[mask]
            if pJ.shape[0] == 0:
                pJ = p_jet_true  # graceful fallback
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

        # Optional inversion stats (only if you're directly predicting jet)
        if compute_jet_inversion_stats:
            jet_stats = compute_vector_stats_from_cartesian(
                p_jet_true,
                use_robust=use_robust,
                residual_floor_pct=residual_floor_pct,
                enforce_nonneg=False,
            )
            out["jet"] = asdict(jet_stats)

    return out


# -------------------------
# Dataset & Loader
# -------------------------

class SparseFASERCALDataset(Dataset):
    def __init__(self, root, shuffle=False, **kwargs):
        # Normalize root into a list
        if isinstance(root, str):
            root = [root]
        self.roots = root

        self.data_files = self.processed_file_names
        self.train = False
        self.total_events = self.__len__

    @property
    def processed_dirs(self):
        return self.roots
    
    @property
    def processed_file_names(self):
        files = []
        for d in self.processed_dirs:
            files.extend(glob(f'{d}/*.npz'))
        return files
    
    def __len__(self):
        return len(self.data_files)

    def collate_sparse_minkowski(self, batch):
        coords = [d['coords'].int() for d in batch if d['coords'] is not None]
        feats = torch.cat([d['feats'] for d in batch if d['coords'] is not None])
        labels = torch.cat([d['labels'] for d in batch if d['coords'] is not None])
        return {'f': feats, 'c': coords, 'y': labels}
    
    def remove_empty_events(self, idx):
        data = np.load(self.data_files[idx])
        hits = data['hits']
        filtered_hits = hits[hits[:, 7] >= 0.5]
        if filtered_hits.shape[0] > 0:
            np.savez(self.data_files[idx], filename=data['filename'], hits=filtered_hits)
            return filtered_hits[:, 7].max()
        else:
            os.remove(self.data_files[idx])
            return 0
        
    def __getitem__(self, idx):
        data = np.load(self.data_files[idx], allow_pickle=True)

        is_cc = data['is_cc'].item()
        reco_hits_true = data['reco_hits_true']
        true_hits = data['true_hits']
        reco_hits = data['reco_hits']
        faser_cal_energy = data['faser_cal_energy'].item()
        faser_cal_modules = data['faser_cal_modules']
        rear_cal_energy = data['rear_cal_energy'].item()
        rear_cal_modules = data['rear_cal_modules']
        rear_hcal_energy = data['rear_hcal_energy'].item()
        rear_hcal_modules = data['rear_hcal_modules']
        rear_mucal_energy = data['rear_mucal_energy'].item()
        vis_sp_momentum = data['vis_sp_momentum']
        out_lepton_momentum = data['out_lepton_momentum']
        in_neutrino_pdg = data['in_neutrino_pdg'].item()
        in_neutrino_energy = data['in_neutrino_energy'].item()
        out_lepton_energy = data['out_lepton_energy'].item()
        jet_momentum = data['jet_momentum']
        tauvis_momentum = data['tauvis_momentum']

        try:
            pdg = np.unique(np.concatenate([
                true_hits[reco_hit_true if isinstance(reco_hit_true, list) else reco_hit_true.astype(int)][:, 3]
                for reco_hit, reco_hit_true in zip(reco_hits, reco_hits_true)
            ]))
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
        module_hits = module_hits[module_hits > 0]
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

dataset = SparseFASERCALDataset(
    [
        "/scratch/salonso/sparse-nns/faser/events_new_v5.1b",
        "/scratch/salonso/sparse-nns/faser/events_new_v5.1b_2",
        "/scratch/salonso/sparse-nns/faser/events_new_v5.1b_3",
        "/scratch/salonso/sparse-nns/faser/events_new_v5.1b_tau",
        "/scratch/salonso/sparse-nns/faser/events_new_v5.1b_tau_2",
        "/scratch/salonso/sparse-nns/faser/events_new_v5.1b_tau_3",
    ])

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

# -------------------------
# Aggregate
# -------------------------

pdg = []
x = []
y = []
z = []
q_counter = Counter()  # memory-safe counter for q
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

# -------------------------
# Compute vector/jet stats (UNCHANGED OUTPUT SHAPE/KEYS)
# -------------------------
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

# -------------------------
# Assemble metadata
# -------------------------
metadata = {}

# q robust standardization (Counter-based)
add_robust_standardization_metadata(q_counter, metadata, key_prefix="q")

# Robust metadata for base_keys (array-based)
base_keys = [
    'in_neutrino_energy', 'out_lepton_energy',
    'faser_cal_energy', 'faser_cal_modules',
    'rear_cal_energy', 'rear_cal_modules',
    'rear_hcal_energy', 'rear_hcal_modules',
    'rear_mucal_energy', 'module_hits', 'event_hits',
]

for key in base_keys:
    arr = locals()[key]
    add_robust_standardization_metadata_array(arr, metadata, key_prefix=key)

# include coordinate and pdg info
metadata.update({
    'x': x, 'y': y, 'z': z,
    'ghost_pdg': set([-10000]),
    'muonic_pdg': set([-13, 13]),
    'electromagnetic_pdg': set([-11, 11, -15, 15, 22]),
    'hadronic_pdg': set([p for p in pdg if p not in [-10000, -13, 13, -11, 11, -15, 15, 22]])
})

# merge vector/jet stats
metadata = {**metadata, **stats}  # for Python < 3.9 compatibility

# save metadata
with open("/scratch/salonso/sparse-nns/faser/events_v5.1b/metadata_stats.pkl", "wb") as fd:
    pk.dump(metadata, fd)

print("Metadata saved.")
