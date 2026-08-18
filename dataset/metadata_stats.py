"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 08.25

Description: script to generate metadata.
"""

import argparse
import torch
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from itertools import chain
import pickle as pk

# -------------------------
# Robust stats helpers
# -------------------------

MADN_CONST = 1.482602218505602  # normalizing constant so MADN ~= std for Gaussian
TRUE_HIT_COLUMNS = 13
V9_MUSPEC_MIN_POINTS = 2

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
    if name == "asinh":
        return np.arcsinh
    if name == "slog1p":  # signed log1p
        return lambda x: np.sign(x) * np.log1p(np.abs(x))
    raise ValueError(f"Unknown transform: {name}")

def _process_muspec(muspec):
    ntracks = 0
    tracks = []
    if len(muspec) == 0:
        muspec = np.zeros((11, 0))
    for i in range(muspec.shape[1]):
        info = muspec[:, i]
        charge, npoints, px, py, pz, p, chi2, ndof, pval, fperr, fiperr = info
        # v9 has only four muon spectrometer stations total. The old v8-era
        # high-multiplicity cut rejects valid downstream muons, so metadata
        # stats must use the same v9-aware threshold as the dataset loader.
        if npoints >= V9_MUSPEC_MIN_POINTS and chi2 > 0:
            ntracks += 1
            # keep in step with dataset.py::process_muspec
            tracks.append([charge, py, pz, chi2, fperr, npoints])

    return ntracks, np.array(tracks).reshape(ntracks, 6)

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

    q_min = float(np.min(arr))
    q_max = float(np.max(arr))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        u = f(arr / k)

    finite = np.isfinite(u)
    if not finite.any():
        return {
            "transform": transform,
            "k": float(k),
            "mu": 0.0,
            "sigma": 1.0,
            "orig_median": float(q_med),
            "orig_min": float(q_min),
            "orig_max": float(q_max),
            "u_median": 0.0,
            "u_min": 0.0,
            "u_max": 0.0,
        }

    u = u[finite]
    mu = float(np.median(u))
    sigma = max(float(_madn(u)), eps)
    u_min = float(np.min(u))
    u_max = float(np.max(u))

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
    Produces:
      - f"{key_prefix}"
      - f"{key_prefix}_log1p"
      - f"{key_prefix}_sqrt"
      - f"{key_prefix}_asinh"
      - f"{key_prefix}_slog1p"
    """
    metadata[f"{key_prefix}"]         = compute_robust_params_for_array(arr, "identity")
    metadata[f"{key_prefix}_log1p"]   = compute_robust_params_for_array(arr, "log1p")
    metadata[f"{key_prefix}_sqrt"]    = compute_robust_params_for_array(arr, "sqrt")
    metadata[f"{key_prefix}_asinh"]   = compute_robust_params_for_array(arr, "asinh")
    metadata[f"{key_prefix}_slog1p"]  = compute_robust_params_for_array(arr, "slog1p")
    return metadata

def compute_zscore_params_for_array(arr: np.ndarray, eps: float = 1e-8):
    """
    Per-column z-score parameters for vector-valued quantities such as the
    primary vertex. Downstream code standardizes with (x - mean) / std and
    reverses with x * std + mean.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")

    dim = arr.shape[1]
    finite_rows = np.all(np.isfinite(arr), axis=1)
    arr = arr[finite_rows]
    if arr.shape[0] == 0:
        return {
            "transform": "zscore",
            "mean": np.zeros(dim, dtype=np.float32),
            "std": np.ones(dim, dtype=np.float32),
            "min": np.zeros(dim, dtype=np.float32),
            "max": np.zeros(dim, dtype=np.float32),
            "count": 0,
        }

    mean = np.mean(arr, axis=0)
    std = np.maximum(np.std(arr, axis=0), eps)
    return {
        "transform": "zscore",
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "min": np.min(arr, axis=0).astype(np.float32),
        "max": np.max(arr, axis=0).astype(np.float32),
        "count": int(arr.shape[0]),
    }

# -------------------------
# Dataset & Loader
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
    k_T: float;  mu_uT: float; sigma_uT: float   # for uT = log1p(pT/k_T)
    k_Z: float;  mu_uZ: float; sigma_uZ: float   # for uZ = log1p(pz/k_Z)
    # robust loss scales
    s_xyz: Tuple[float, float, float]            # MADN per component (px,py,pz)
    s_pT: float                                  # MADN for pT
    s_mag: float                                 # MADN for ||p||
    # residual floors (for (true-reco)/max(true, tau))
    tau_pt: float
    tau_mag: float

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
        s_pT = float(_madn(pT))
        s_mag = float(_madn(mag))
    else:
        s_xyz = tuple((np.std(p_true, axis=0) + 1e-12).astype(np.float64))
        s_pT = float(np.std(pT) + 1e-12)
        s_mag = float(np.std(mag) + 1e-12)

    s_xyz = tuple(max(float(v), 1e-8) for v in s_xyz)
    s_pT = max(float(s_pT), 1e-8)
    s_mag = max(float(s_mag), 1e-8)

    # residual floors
    tau_pt  = float(np.percentile(pT,  residual_floor_pct))
    tau_mag = float(np.percentile(mag, residual_floor_pct))

    return VectorStatsLog1p(
        k_T=k_T, mu_uT=mu_uT, sigma_uT=sigma_uT,
        k_Z=k_Z, mu_uZ=mu_uZ, sigma_uZ=sigma_uZ,
        s_xyz=s_xyz, s_pT=s_pT, s_mag=s_mag,
        tau_pt=tau_pt, tau_mag=tau_mag,
    )

def compute_all_stats(
    p_vis_true: np.ndarray,
    p_jet_true: np.ndarray,
    p_lep_true: np.ndarray,
    is_cc: Optional[np.ndarray] = None,
    use_robust: bool = True,
    residual_floor_pct: float = 5.0,
) -> Dict[str, Dict]:
    """
    Returns a dict with:
      - 'vis': VectorStats as dict
      - 'lep': VectorStats as dict
      - 'jet': VectorStats as dict
      - (optional) class-specific vis floors: tau_pt_cc/nc, tau_mag_cc/nc

    Notes:
      * Output format/keys are intentionally unchanged.
      * Uses robust statistics when use_robust=True.
    """
    out: Dict[str, Dict] = {}

    # --- visible, jet, and lepton stats (always computed) ---
    vis_stats = compute_vector_stats_from_cartesian(
        p_vis_true,
        use_robust=use_robust,
        residual_floor_pct=residual_floor_pct,
        enforce_nonneg=False,
    )
    jet_stats = compute_vector_stats_from_cartesian(
        p_jet_true,
        use_robust=use_robust,
        residual_floor_pct=residual_floor_pct,
        enforce_nonneg=True,
    )
    p_lep_cc = _select_cc_leptons(p_lep_true, is_cc=is_cc)
    lep_stats = compute_vector_stats_from_cartesian(
        p_lep_cc,
        use_robust=use_robust,
        residual_floor_pct=residual_floor_pct,
        enforce_nonneg=False,
    )
    out["vis"] = asdict(vis_stats)
    out["jet"] = asdict(jet_stats)
    out["lep"] = asdict(lep_stats)

    # Helper to compute CC/NC residual floors for a given 3-vector array
    def _class_floors(
        arr: np.ndarray,
        mask: np.ndarray,
        pct: float,
        fallback_tau_pt: float,
        fallback_tau_mag: float,
    ) -> Dict[str, float]:
        res: Dict[str, float] = {}
        for tag, sub in (("cc", arr[mask]), ("nc", arr[~mask])):
            if sub.shape[0] > 0:
                px, py, pz = sub[:, 0], sub[:, 1], sub[:, 2]
                pT = np.sqrt(px**2 + py**2)
                mag = np.sqrt(px**2 + py**2 + pz**2)
                res[f"tau_pt_{tag}"] = float(np.percentile(pT, pct))
                res[f"tau_mag_{tag}"] = float(np.percentile(mag, pct))
            else:
                # fallback to the group's global floors
                res[f"tau_pt_{tag}"] = float(fallback_tau_pt)
                res[f"tau_mag_{tag}"] = float(fallback_tau_mag)
        return res

    # --- class-specific floors for each group ---
    if is_cc is not None:
        mask = np.asarray(is_cc, dtype=bool)

        # vis
        vis_cls = _class_floors(
            p_vis_true, mask, residual_floor_pct,
            fallback_tau_pt=out["vis"]["tau_pt"],
            fallback_tau_mag=out["vis"]["tau_mag"],
        )
        out["vis"].update(vis_cls)

        # jet
        jet_cls = _class_floors(
            p_jet_true, mask, residual_floor_pct,
            fallback_tau_pt=out["jet"]["tau_pt"],
            fallback_tau_mag=out["jet"]["tau_mag"],
        )
        out["jet"].update(jet_cls)

        # lep (note: NC leptons may be absent; fallback handles it)
        lep_cls = _class_floors(
            p_lep_true, mask, residual_floor_pct,
            fallback_tau_pt=out["lep"]["tau_pt"],
            fallback_tau_mag=out["lep"]["tau_mag"],
        )
        out["lep"].update(lep_cls)

    return out


# -------------------------
# Dataset & Loader
# -------------------------

class SparseFASERCALDataset(Dataset):
    def __init__(self, root, shuffle=False, sample_prob=0.10, **kwargs):
        # Normalize root into a list
        self.root = root
        self.data_files = sorted(
            chain(
                glob(os.path.join(self.root, "*.npz")),
                glob(os.path.join(self.root, "*", "*.npz")),
            ),
            key=str.lower,
        )
        self.data_files = [x for x in self.data_files if np.random.rand() < sample_prob]
        self.train = False
        self.total_events = self.__len__

    @property
    def processed_dirs(self):
        return self.roots
    
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
        reco_hits = data['reco_hits']
        true_index = data['true_index']
        vis_sp_momentum = data['vis_sp_momentum']
        out_lepton_momentum = data['out_lepton_momentum']
        in_neutrino_pdg = data['in_neutrino_pdg'].item()
        in_neutrino_energy = data['in_neutrino_energy'].item()
        out_lepton_energy = data['out_lepton_energy'].item()
        jet_momentum = data['jet_momentum']
        tau_vis_momentum = data['tau_vis_momentum']
        muspec_info = data['muspec_info']
        primary_vertex = np.asarray(data['primary_vertex'], dtype=np.float32).reshape(1, 3)

        if true_index.size > 0:
            true_hits = data['true_hits']
            if np.asarray(true_hits).ndim == 0:
                true_hits = np.empty((0, TRUE_HIT_COLUMNS), dtype=np.float32)
            else:
                true_hits = np.asarray(true_hits, dtype=np.float32)
            pdg = np.unique(true_hits[true_index][:, 3])
        else:
            pdg = np.empty((0,), dtype=np.float32)
            
        x = np.unique(data['reco_hits'][:, 0])
        y = np.unique(data['reco_hits'][:, 1])
        z = np.unique(np.stack((data['reco_hits'][:, 2], data['reco_hits'][:, 3]), axis=1), axis=0)
        q = (data['reco_hits'][:, 4]*10).round().astype(int)
        ecal_raw = data['ecal_hits']
        if ecal_raw.ndim == 2 and ecal_raw.shape[1] == 4:
            # sparse ECAL: (N, 4) with columns (x, y, z, energy)
            ecal_hits = (ecal_raw[:, 3] * 10).round().astype(int)
        else:
            # dense ECAL: (5, 5)
            ecal_hits = ecal_raw.reshape(-1)
        ahcal_hits = (data['ahcal_hits'][:, 3]*10).round().astype(int)
        nb_muspec_tracks, muspec_tracks = _process_muspec(muspec_info)
        muspec_q = muspec_tracks[:, 0]
        muspec_py = muspec_tracks[:, 1]
        muspec_pz = muspec_tracks[:, 2]
        muspec_chi2 = muspec_tracks[:, 3]
        muspec_fperr = muspec_tracks[:, 4]
        muspec_npoints = muspec_tracks[:, 5]

        if is_cc:
            out_lepton_momentum = out_lepton_momentum.reshape(1, 3)
        else:
            out_lepton_momentum = np.zeros(shape=(1, 3))
        if is_cc and in_neutrino_pdg in [-16, 16]:  # nutau
            out_lepton_momentum = tau_vis_momentum.reshape(1, 3)

        vis_sp_momentum = vis_sp_momentum.reshape(1, 3)
        jet_momentum = jet_momentum.reshape(1, 3)

        module_hits = np.bincount(reco_hits[:, 3].astype(int))
        module_hits = module_hits[module_hits > 0]
        event_hits = np.array([reco_hits.shape[0]])
        in_neutrino_energy = np.array([in_neutrino_energy])
        out_lepton_energy = np.array([out_lepton_energy])
        is_cc = np.array([is_cc], dtype=np.bool_)

        return {"pdg": pdg, "x": x, "y": y, "z": z, "q": q, 
                "vis_sp_momentum": vis_sp_momentum,
                "out_lepton_momentum": out_lepton_momentum,
                "jet_momentum": jet_momentum,
                "ecal_hits": ecal_hits,
                "ahcal_hits": ahcal_hits,
                "nb_muspec_tracks": nb_muspec_tracks,
                "muspec_q": muspec_q,
                "muspec_py": muspec_py,
                "muspec_pz": muspec_pz,
                "muspec_chi2": muspec_chi2,
                "muspec_fperr": muspec_fperr,
                "muspec_npoints": muspec_npoints,
                "in_neutrino_energy": in_neutrino_energy,
                "out_lepton_energy": out_lepton_energy,
                "primary_vertex": primary_vertex,
                "is_cc": is_cc,
                "module_hits": module_hits,
                "event_hits": event_hits,
               }

def collate(batch):
    pdg = np.unique(np.concatenate([x['pdg'] for x in batch]))
    x = np.unique(np.concatenate([x['x'] for x in batch]))
    y = np.unique(np.concatenate([x['y'] for x in batch]))
    z = np.unique(np.concatenate([x['z'] for x in batch]), axis=0)
    q = np.concatenate([x['q'] for x in batch])
    vis_sp_momentum = np.concatenate([x['vis_sp_momentum'] for x in batch])
    out_lepton_momentum = np.concatenate([x['out_lepton_momentum'] for x in batch])
    jet_momentum = np.concatenate([x['jet_momentum'] for x in batch])
    ecal_hits = np.concatenate([x['ecal_hits'] for x in batch])
    ahcal_hits = np.concatenate([x['ahcal_hits'] for x in batch])
    nb_muspec_tracks = np.array([x['nb_muspec_tracks'] for x in batch])
    muspec_q = np.concatenate([x['muspec_q'] for x in batch])
    muspec_py = np.concatenate([x['muspec_py'] for x in batch])
    muspec_pz = np.concatenate([x['muspec_pz'] for x in batch])
    muspec_chi2 = np.concatenate([x['muspec_chi2'] for x in batch])
    muspec_fperr = np.concatenate([x['muspec_fperr'] for x in batch])
    muspec_npoints = np.concatenate([x['muspec_npoints'] for x in batch])
    in_neutrino_energy = np.concatenate([x['in_neutrino_energy'] for x in batch])
    out_lepton_energy = np.concatenate([x['out_lepton_energy'] for x in batch])
    primary_vertex = np.concatenate([x['primary_vertex'] for x in batch])
    is_cc = np.concatenate([x['is_cc'] for x in batch])
    module_hits = np.concatenate([x['module_hits'] for x in batch])
    event_hits = np.concatenate([x['event_hits'] for x in batch])
    
    return {"pdg": pdg, "x": x, "y": y, "z": z, "q": q, 
            "vis_sp_momentum": vis_sp_momentum,
            "out_lepton_momentum": out_lepton_momentum,
            "jet_momentum": jet_momentum,
            "ecal_hits": ecal_hits,
            "ahcal_hits": ahcal_hits,
            "nb_muspec_tracks": nb_muspec_tracks,
            "muspec_q": muspec_q,
            "muspec_fperr": muspec_fperr,
            "muspec_npoints": muspec_npoints,
            "muspec_py": muspec_py,
            "muspec_pz": muspec_pz,
            "muspec_chi2": muspec_chi2,
            "in_neutrino_energy": in_neutrino_energy,
            "out_lepton_energy": out_lepton_energy,
            "primary_vertex": primary_vertex,
            "is_cc": is_cc,
            "module_hits": module_hits,
            "event_hits": event_hits,
           }
    
def main():
    parser = argparse.ArgumentParser(description="Build metadata statistics for FASER datasets.")
    parser.add_argument(
        "--dataset_glob",
        type=str,
        default="/scratch/salonso/sparse-nns/faser/events_v7.0*",
        help="Glob that matches dataset directories containing NPZ event files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/scratch/salonso/sparse-nns/faser/events_v7.0_500_npz/metadata_stats.pkl",
        help="Output pickle path for the metadata statistics.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for the metadata dataloader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--sample_prob",
        type=float,
        default=0.10,
        help="Fraction of files to sample when estimating metadata.",
    )
    args = parser.parse_args()

    dataset = SparseFASERCALDataset(args.dataset_glob, sample_prob=args.sample_prob)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No NPZ files matched dataset_glob={args.dataset_glob!r}")

    loader = DataLoader(
        dataset,
        collate_fn=collate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
    )

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
    ecal_hits = []
    ahcal_hits = []
    nb_muspec_tracks = []
    muspec_q = []
    muspec_py = []
    muspec_pz = []
    muspec_chi2 = []
    muspec_fperr = []
    muspec_npoints = []
    in_neutrino_energy = []
    out_lepton_energy = []
    primary_vertex = []
    is_cc = []
    module_hits = []
    event_hits = []

    t = tqdm(enumerate(loader), total=len(loader), disable=False, ascii=True)
    for _, batch in t:
        pdg.append(batch["pdg"])
        x.append(batch["x"])
        y.append(batch["y"])
        z.append(batch["z"])
        q_counter.update(batch["q"])
        vis_sp_momentum.append(batch["vis_sp_momentum"])
        out_lepton_momentum.append(batch["out_lepton_momentum"])
        jet_momentum.append(batch["jet_momentum"])
        ecal_hits.append(batch["ecal_hits"])
        ahcal_hits.append(batch["ahcal_hits"])
        nb_muspec_tracks.append(batch["nb_muspec_tracks"])
        muspec_q.append(batch["muspec_q"])
        muspec_fperr.append(batch["muspec_fperr"])
        muspec_npoints.append(batch["muspec_npoints"])
        muspec_py.append(batch["muspec_py"])
        muspec_pz.append(batch["muspec_pz"])
        muspec_chi2.append(batch["muspec_chi2"])
        in_neutrino_energy.append(batch["in_neutrino_energy"])
        out_lepton_energy.append(batch["out_lepton_energy"])
        primary_vertex.append(batch["primary_vertex"])
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
    ecal_hits = np.concatenate(ecal_hits)
    ahcal_hits = np.concatenate(ahcal_hits)
    nb_muspec_tracks = np.concatenate(nb_muspec_tracks)
    muspec_q = np.concatenate(muspec_q)
    muspec_fperr = np.concatenate(muspec_fperr)
    muspec_npoints = np.concatenate(muspec_npoints)
    muspec_py = np.concatenate(muspec_py)
    muspec_pz = np.concatenate(muspec_pz)
    muspec_chi2 = np.concatenate(muspec_chi2)
    in_neutrino_energy = np.concatenate(in_neutrino_energy)
    out_lepton_energy = np.concatenate(out_lepton_energy)
    primary_vertex = np.concatenate(primary_vertex)
    is_cc = np.concatenate(is_cc)
    module_hits = np.concatenate(module_hits)
    event_hits = np.concatenate(event_hits)

    print("Done with concat")

    # -------------------------
    # Compute vector/jet stats (UNCHANGED OUTPUT SHAPE/KEYS)
    # -------------------------
    stats = compute_all_stats(
        p_vis_true=vis_sp_momentum,
        p_jet_true=jet_momentum,
        p_lep_true=out_lepton_momentum,
        is_cc=is_cc,
        use_robust=True,
        residual_floor_pct=5.0,
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
        'ecal_hits', 'ahcal_hits',
        'nb_muspec_tracks', 'muspec_q', 'muspec_py', 'muspec_pz', 'muspec_chi2',
        'muspec_fperr', 'muspec_npoints',
        'module_hits', 'event_hits',
    ]

    for key in base_keys:
        arr = locals()[key]
        add_robust_standardization_metadata_array(arr, metadata, key_prefix=key)

    metadata["primary_vertex"] = compute_zscore_params_for_array(primary_vertex)

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

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "wb") as fd:
        pk.dump(metadata, fd, protocol=4)

    print(f"Metadata saved to {args.out}")


if __name__ == "__main__":
    main()
