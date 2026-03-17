"""
Build charge preprocessing metadata for transfer-learning datasets.

The resulting pickle contains a `q_log1p` entry compatible with
`transfer_learning.transfer_charge_preprocessing.LogChargePreprocessor`.

The implementation mirrors the original metadata flow: charges are streamed
with PyTorch `DataLoader` workers, scaled by the same x10 factor used during
pretraining, quantized into charge bins, and accumulated in a `Counter` so
memory stays proportional to the number of distinct charge values rather than
the number of voxels.

Dataset-specific convention:
  - PILArNet: pre-crop raw charges from manifest-selected particles, scanned
    event-by-event
  - Zenodo: processed transfer inputs after the dataset crop/dedup logic, but
    before charge standardisation
"""

from __future__ import annotations

import argparse
import os
import pickle as pk
import threading
from collections import Counter
from dataclasses import asdict, dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .transfer_charge_preprocessing import PRETRAIN_CHARGE_SCALE


MADN_CONST = 1.482602218505602
DEFAULT_CHARGE_BIN_WIDTH = {
    "pilarnet": 0.1,
    "zenodo": 1.0,
}
DEGENERATE_STATS_THRESHOLD = 1e-7


def _transform_lookup(name: str):
    if name == "identity":
        return lambda x: x
    if name == "log1p":
        return np.log1p
    if name == "sqrt":
        return np.sqrt
    if name == "asinh":
        return np.arcsinh
    if name == "slog1p":
        return lambda x: np.sign(x) * np.log1p(np.abs(x))
    raise ValueError(f"Unknown transform: {name}")


def _N(counter: Counter) -> int:
    return int(sum(counter.values()))


def _weighted_median(counter: Counter, value_scale: float = 1.0) -> float:
    """Lower weighted median: first value where cumulative count >= N / 2."""
    total = _N(counter)
    if total == 0:
        return 0.0
    median_pos = total / 2
    cum = 0
    for value in sorted(counter):
        cum += counter[value]
        if cum >= median_pos:
            return float(value) * value_scale
    return float(next(iter(counter))) * value_scale


def _mad_from_counter(counter: Counter, center: float, value_scale: float = 1.0) -> float:
    """Weighted median of absolute deviations |x - center|."""
    total = _N(counter)
    if total == 0:
        return 0.0
    devs = Counter()
    for value, count in counter.items():
        devs[abs(float(value) * value_scale - float(center))] += int(count)
    median_pos = total / 2
    cum = 0
    for dev in sorted(devs):
        cum += devs[dev]
        if cum >= median_pos:
            return float(dev)
    return 0.0


def _madn_from_counter(counter: Counter, center: float, value_scale: float = 1.0) -> float:
    return _mad_from_counter(counter, center, value_scale=value_scale) * MADN_CONST


def compute_robust_params_for_transform(
    q_counter: Counter,
    transform: str,
    eps: float = 1e-8,
    value_scale: float = 1.0,
) -> dict:
    """
    Robust two-stage standardization:
      1) k = MADN(q) in original space
      2) u = f(q / k)
      3) mu = median(u), sigma = MADN(u)
    """
    total = _N(q_counter)
    if total == 0:
        return {
            "transform": transform,
            "k": eps,
            "mu": 0.0,
            "sigma": 1.0,
            "orig_median": 0.0,
            "orig_min": 0.0,
            "orig_max": 0.0,
            "u_median": 0.0,
            "u_min": 0.0,
            "u_max": 0.0,
        }

    f = _transform_lookup(transform)

    q_med = _weighted_median(q_counter, value_scale=value_scale)
    k = max(_madn_from_counter(q_counter, q_med, value_scale=value_scale), eps)

    u_med = float(f(q_med / k))
    devs_u = Counter()
    for value, count in q_counter.items():
        q_value = float(value) * value_scale
        devs_u[float(abs(f(q_value / k) - u_med))] += int(count)

    median_pos = total / 2
    cum = 0
    mad_u = 0.0
    for dev in sorted(devs_u):
        cum += devs_u[dev]
        if cum >= median_pos:
            mad_u = float(dev)
            break
    sigma = max(mad_u * MADN_CONST, eps)

    q_min = float(min(q_counter)) * value_scale
    q_max = float(max(q_counter)) * value_scale
    u_min = float(f(q_min / k))
    u_max = float(f(q_max / k))

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


def validate_q_log1p_stats(stats: dict, charge_bin_width: float) -> None:
    if (
        float(stats.get("orig_max", 0.0)) > 0.0
        and float(stats.get("k", 0.0)) <= DEGENERATE_STATS_THRESHOLD
        and float(stats.get("sigma", 0.0)) <= DEGENERATE_STATS_THRESHOLD
    ):
        raise RuntimeError(
            "Degenerate q_log1p metadata: both k and sigma collapsed near zero. "
            f"Current charge_bin_width={charge_bin_width:g} is too coarse for this dataset. "
            "Rebuild metadata with a smaller --charge_bin_width."
        )


@dataclass
class MetadataSummary:
    dataset: str
    samples_seen: int
    voxels_seen: int
    unique_charge_values: int
    charge_scale: float
    charge_bin_width: float
    estimator: str
    split: str | None = None
    manifest_path: str | None = None
    data_dir: str | None = None


def concat_charge_collate(batch: list[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(x, dtype=np.float32).reshape(-1) for x in batch if len(x) > 0]
    if not nonempty:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(nonempty, axis=0)


class PILArNetEventChargeDataset(torch.utils.data.Dataset):
    """Raw pre-crop charge extraction grouped by event for the PILArNet manifest."""

    def __init__(self, manifest_path: str):
        data = np.load(manifest_path, allow_pickle=True)
        self.manifest = data["manifest"]
        self.h5_paths = list(data["h5_paths"])
        self._local = threading.local()
        self._event_specs = self._build_event_index()

    def __len__(self) -> int:
        return len(self._event_specs)

    def _build_event_index(self) -> list[tuple[int, int, np.ndarray]]:
        order = np.lexsort((self.manifest["event_idx"], self.manifest["h5_idx"]))
        sorted_manifest = self.manifest[order]
        if len(sorted_manifest) == 0:
            return []

        h5_idx = sorted_manifest["h5_idx"]
        event_idx = sorted_manifest["event_idx"]
        particle_idx = sorted_manifest["particle_idx"].astype(np.int32, copy=False)

        changes = np.nonzero(
            (h5_idx[1:] != h5_idx[:-1]) | (event_idx[1:] != event_idx[:-1])
        )[0] + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(sorted_manifest)]))

        event_specs = []
        for start, end in zip(starts, ends):
            event_specs.append(
                (
                    int(h5_idx[start]),
                    int(event_idx[start]),
                    np.unique(particle_idx[start:end]),
                )
            )
        return event_specs

    def _get_h5(self, h5_idx: int) -> h5py.File:
        cache = getattr(self._local, "h5_cache", None)
        if cache is None:
            cache = {}
            self._local.h5_cache = cache
        if h5_idx not in cache:
            cache[h5_idx] = h5py.File(self.h5_paths[h5_idx], "r")
        return cache[h5_idx]

    def __getitem__(self, idx: int) -> np.ndarray:
        h5_idx, event_idx, particle_ids = self._event_specs[idx]

        h5_file = self._get_h5(h5_idx)
        voxel_extent = h5_file["Data/sparse3d_data_group/voxel_extents"][event_idx]
        voxel_first, voxel_count = int(voxel_extent["first"]), int(voxel_extent["N"])
        voxels = h5_file["Data/sparse3d_data_group/voxels"][voxel_first : voxel_first + voxel_count]

        group_extent = h5_file["Data/sparse3d_group_group/voxel_extents"][event_idx]
        group_first, group_count = int(group_extent["first"]), int(group_extent["N"])
        groups = h5_file["Data/sparse3d_group_group/voxels"][group_first : group_first + group_count]["value"].astype(np.int32)

        mask = np.isin(groups, particle_ids)
        return voxels[mask]["value"].astype(np.float32, copy=False)


def build_dataset(args: argparse.Namespace):
    if args.dataset == "pilarnet":
        if args.manifest is None:
            raise ValueError("--manifest is required for dataset=pilarnet")
        dataset = PILArNetEventChargeDataset(args.manifest)
        return dataset, concat_charge_collate
    if args.dataset == "zenodo":
        if args.data_dir is None:
            raise ValueError("--data_dir is required for dataset=zenodo")
        try:
            from .transfer_zenodo_pid.zenodo_dataset import (
                ZenodoPIDDataset,
                zenodo_collate_fn,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Zenodo support requires transfer_learning/transfer_zenodo_pid."
            ) from exc
        dataset = ZenodoPIDDataset(
            data_dir=args.data_dir,
            split=args.split,
            spatial_shape=tuple(args.zenodo_spatial_shape),
            augment=False,
            charge_metadata_path=None,
        )
        return dataset, zenodo_collate_fn
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def update_charge_counter(
    q_counter: Counter,
    charges: np.ndarray,
    scale: float,
    charge_bin_width: float,
) -> int:
    arr = np.asarray(charges, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0

    finite_mask = np.isfinite(arr)
    if not np.all(finite_mask):
        arr = arr[finite_mask]
    if arr.size == 0:
        return 0

    scaled = np.clip(arr * float(scale), a_min=0.0, a_max=None)
    binned = np.rint(scaled / float(charge_bin_width)).astype(np.int64, copy=False)
    values, counts = np.unique(binned, return_counts=True)
    q_counter.update({int(value): int(count) for value, count in zip(values.tolist(), counts.tolist())})
    return int(binned.size)


def resolve_charge_bin_width(args: argparse.Namespace) -> float:
    if args.charge_bin_width is not None:
        if args.charge_bin_width <= 0:
            raise ValueError("--charge_bin_width must be positive")
        return float(args.charge_bin_width)
    return float(DEFAULT_CHARGE_BIN_WIDTH[args.dataset])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build transfer charge metadata")
    parser.add_argument("--dataset", choices=["pilarnet", "zenodo"], required=True)
    parser.add_argument(
        "--manifest",
        "--manifest_path",
        dest="manifest",
        type=str,
        default=None,
        help="PILArNet manifest (.npz). Required for --dataset pilarnet.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Zenodo root containing training/ and testing/. Required for --dataset zenodo.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "testing"],
        help="Zenodo split to scan (default: training). Ignored for PILArNet.",
    )
    parser.add_argument("--out", type=str, required=True, help="Output metadata pickle path.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of dataset samples per DataLoader batch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Deprecated and ignored. Metadata is now computed exactly with a Counter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deprecated and ignored. Metadata is now computed exactly with a Counter.",
    )
    parser.add_argument(
        "--charge_scale",
        type=float,
        default=PRETRAIN_CHARGE_SCALE,
        help="Charge scaling factor applied before bin counting and q_log1p stats.",
    )
    parser.add_argument(
        "--charge_bin_width",
        type=float,
        default=None,
        help=(
            "Charge-bin width in the scaled-charge space used for Counter accumulation. "
            "Defaults: 0.1 for PILArNet, 1.0 for Zenodo."
        ),
    )
    parser.add_argument(
        "--zenodo_spatial_shape",
        type=int,
        nargs=3,
        default=[80, 80, 80],
        help="Zenodo crop shape used before stat collection.",
    )
    return parser.parse_args()


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()
    charge_bin_width = resolve_charge_bin_width(args)

    dataset, collate_fn = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    q_counter = Counter()
    samples_seen = 0
    voxels_seen = 0

    progress = tqdm(loader, total=len(loader), desc=f"Scanning {args.dataset}", ascii=True)
    for batch in progress:
        if args.dataset == "zenodo":
            feats = batch["feats"]
            if isinstance(feats, torch.Tensor):
                charges = feats.detach().cpu().numpy().reshape(-1)
            else:
                charges = np.asarray(feats, dtype=np.float32).reshape(-1)
            batch_samples = int(batch.get("batch_size", 0))
        else:
            charges = np.asarray(batch, dtype=np.float32).reshape(-1)
            batch_samples = min(args.batch_size, len(dataset) - samples_seen)

        voxels_seen += update_charge_counter(
            q_counter,
            charges,
            args.charge_scale,
            charge_bin_width,
        )
        samples_seen += batch_samples
        progress.set_postfix(
            samples=f"{samples_seen:,}/{len(dataset):,}",
            voxels=f"{voxels_seen:,}",
            unique_q=f"{len(q_counter):,}",
        )

    stats = compute_robust_params_for_transform(
        q_counter,
        "log1p",
        value_scale=charge_bin_width,
    )
    validate_q_log1p_stats(stats, charge_bin_width)

    metadata = {
        "q_log1p": stats,
        "_transfer_charge_metadata": asdict(
            MetadataSummary(
                dataset=args.dataset,
                samples_seen=samples_seen,
                voxels_seen=voxels_seen,
                unique_charge_values=len(q_counter),
                charge_scale=float(args.charge_scale),
                charge_bin_width=float(charge_bin_width),
                estimator="exact_counter_quantized_scaled_charges",
                split=args.split if args.dataset == "zenodo" else None,
                manifest_path=args.manifest if args.dataset == "pilarnet" else None,
                data_dir=args.data_dir if args.dataset == "zenodo" else None,
            )
        ),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fd:
        pk.dump(metadata, fd)

    print("\nSaved metadata to:", args.out)
    print("q_log1p stats:")
    print(f"  k     = {stats['k']:.6g}")
    print(f"  mu    = {stats['mu']:.6g}")
    print(f"  sigma = {stats['sigma']:.6g}")
    print(f"  orig_median = {stats['orig_median']:.6g}")
    print(f"  orig_min    = {stats['orig_min']:.6g}")
    print(f"  orig_max    = {stats['orig_max']:.6g}")
    print("summary:")
    print(f"  voxels_seen         = {voxels_seen:,}")
    print(f"  unique_charge_bins  = {len(q_counter):,}")
    print(f"  charge_scale        = {float(args.charge_scale):.6g}")
    print(f"  charge_bin_width    = {float(charge_bin_width):.6g}")


if __name__ == "__main__":
    main()
