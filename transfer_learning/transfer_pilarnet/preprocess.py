"""Build the PILArNet particle manifest and the fixed train/val/test split."""

import argparse
import json
import os
import sys

import h5py
import numpy as np
from glob import glob


# Shape labels kept for reference.
SHAPE_NAMES = {0: "shower", 1: "track", 2: "michel", 3: "delta", 4: "low_energy"}

# PDG code to the 5-class PID label used by PILArNet.
PDG_TO_TYPE = {
    22: 0,    # photon
    11: 1,    # electron
    -11: 1,   # positron  → electron class
    13: 2,    # muon-
    -13: 2,   # muon+
    211: 3,   # pion+
    -211: 3,  # pion-
    2212: 4,  # proton
}
TYPE_NAMES = {0: "photon", 1: "electron", 2: "muon", 3: "pion", 4: "proton"}


def build_manifest_for_file(h5_path, min_voxels=5):
    """Return a list of dicts, one per qualifying particle."""
    records = []
    with h5py.File(h5_path, "r") as f:
        particles_ds = f["Data/particle_mcst_group/particles"]
        part_extents = f["Data/particle_mcst_group/extents"]
        group_voxels = f["Data/sparse3d_group_group/voxels"]
        group_vext = f["Data/sparse3d_group_group/voxel_extents"]
        n_events = len(part_extents)

        for ev in range(n_events):
            pf, pn = int(part_extents[ev]["first"]), int(part_extents[ev]["N"])
            gf_raw = group_vext[ev]
            gf, _, gn = int(gf_raw["first"]), int(gf_raw["ID"]), int(gf_raw["N"])

            grp_vals = group_voxels[gf : gf + gn]["value"].astype(np.int32)

            parts = particles_ds[pf : pf + pn]
            for pi in range(pn):
                p = parts[pi]
                n_vox = int((grp_vals == pi).sum())
                if n_vox < min_voxels:
                    continue
                shape_label = int(p["shape"])
                if shape_label < 0 or shape_label > 4:
                    continue
                pdg = int(p["pdg"])
                if pdg not in PDG_TO_TYPE:
                    continue
                type_label = PDG_TO_TYPE[pdg]
                records.append(
                    {
                        "h5_path": h5_path,
                        "event_idx": ev,
                        "particle_idx": pi,
                        "shape_label": shape_label,
                        "type_label": type_label,
                        "pdg": pdg,
                        "num_voxels": n_vox,
                        "px": float(p["px"]),
                        "py": float(p["py"]),
                        "pz": float(p["pz"]),
                        "energy_init": float(p["energy_init"]),
                        "energy_deposit": float(p["energy_deposit"]),
                    }
                )
    return records


def records_to_structured(records, h5_paths_list):
    """Convert list-of-dicts to a compact NumPy structured array."""
    dt = np.dtype(
        [
            ("h5_idx", np.uint16),
            ("event_idx", np.uint32),
            ("particle_idx", np.uint16),
            ("shape_label", np.uint8),
            ("type_label", np.uint8),
            ("pdg", np.int32),
            ("num_voxels", np.uint32),
            ("px", np.float32),
            ("py", np.float32),
            ("pz", np.float32),
            ("energy_init", np.float32),
            ("energy_deposit", np.float32),
        ]
    )
    path_to_idx = {p: i for i, p in enumerate(h5_paths_list)}
    arr = np.empty(len(records), dtype=dt)
    for i, r in enumerate(records):
        arr[i] = (
            path_to_idx[r["h5_path"]],
            r["event_idx"],
            r["particle_idx"],
            r["shape_label"],
            r["type_label"],
            r["pdg"],
            r["num_voxels"],
            r["px"],
            r["py"],
            r["pz"],
            r["energy_init"],
            r["energy_deposit"],
        )
    return arr


def save_manifest(out_path, manifest, h5_paths_list):
    np.savez_compressed(
        out_path,
        manifest=manifest,
        h5_paths=np.array(h5_paths_list, dtype=object),
    )


def print_type_summary(manifest, header):
    labels, counts = np.unique(manifest["type_label"], return_counts=True)
    print(f"\n{header}")
    print(f"  Total particles: {len(manifest)}")
    for lab, cnt in zip(labels, counts):
        name = TYPE_NAMES.get(int(lab), f"unknown_{lab}")
        print(f"  {name} ({lab}): {cnt}  ({100*cnt/len(manifest):.1f}%)")


def count_unique_events(manifest):
    if len(manifest) == 0:
        return 0
    _, starts, _ = _event_group_indices(manifest)
    return len(starts)


def _event_group_indices(manifest):
    order = np.lexsort((manifest["event_idx"], manifest["h5_idx"]))
    sorted_manifest = manifest[order]

    h5_idx = sorted_manifest["h5_idx"]
    event_idx = sorted_manifest["event_idx"]
    changes = np.nonzero(
        (h5_idx[1:] != h5_idx[:-1]) | (event_idx[1:] != event_idx[:-1])
    )[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(order)]))
    return order, starts, ends


def _select_events(manifest, order, starts, ends, event_positions):
    if len(event_positions) == 0:
        return manifest[:0].copy()

    record_indices = np.concatenate(
        [order[starts[pos] : ends[pos]] for pos in event_positions]
    )
    return manifest[np.sort(record_indices)]


def build_paper_splits(
    manifest,
    *,
    train_events=80000,
    val_events=2000,
    test_events=18000,
    split_seed=42,
):
    order, starts, ends = _event_group_indices(manifest)
    n_events = len(starts)
    expected_total = train_events + val_events + test_events
    if n_events != expected_total:
        raise ValueError(
            "Paper split protocol expects exactly "
            f"{expected_total} events, found {n_events}."
        )

    train_positions = np.arange(train_events, dtype=np.int64)
    original_test_positions = np.arange(train_events, n_events, dtype=np.int64)

    rng = np.random.default_rng(split_seed)
    val_positions = np.sort(
        rng.choice(original_test_positions, size=val_events, replace=False)
    )
    test_positions = np.setdiff1d(
        original_test_positions,
        val_positions,
        assume_unique=True,
    )

    return {
        "train": _select_events(manifest, order, starts, ends, train_positions),
        "val": _select_events(manifest, order, starts, ends, val_positions),
        "test": _select_events(manifest, order, starts, ends, test_positions),
    }


def main():
    parser = argparse.ArgumentParser(description="Build PILArNet particle manifest.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing larcv3 HDF5 files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="manifest_pilarnet.npz",
        help="Output manifest path.",
    )
    parser.add_argument(
        "--min_voxels",
        type=int,
        default=5,
        help="Minimum voxels for a particle to be included.",
    )
    parser.add_argument(
        "--train_events",
        type=int,
        default=80000,
        help="Number of events assigned to the paper-style training split.",
    )
    parser.add_argument(
        "--val_events",
        type=int,
        default=2000,
        help="Number of events assigned to the paper-style validation split.",
    )
    parser.add_argument(
        "--test_events",
        type=int,
        default=18000,
        help="Number of events assigned to the paper-style test split.",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed used when carving validation events from the original test pool.",
    )
    args = parser.parse_args()

    h5_files = sorted(glob(os.path.join(args.data_dir, "*.h5")))
    if not h5_files:
        print(f"No HDF5 files found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(h5_files)} HDF5 files in {args.data_dir}")
    all_records = []
    for fi, h5_path in enumerate(h5_files):
        print(f"  [{fi+1}/{len(h5_files)}] {os.path.basename(h5_path)} ...", end=" ", flush=True)
        recs = build_manifest_for_file(h5_path, min_voxels=args.min_voxels)
        print(f"{len(recs)} particles")
        all_records.extend(recs)

    arr = records_to_structured(all_records, h5_files)
    print_type_summary(arr, "Full manifest summary")
    save_manifest(args.out, arr, h5_files)
    print(f"\nFull manifest saved to {args.out}")

    split_manifests = build_paper_splits(
        arr,
        train_events=args.train_events,
        val_events=args.val_events,
        test_events=args.test_events,
        split_seed=args.split_seed,
    )
    base, ext = os.path.splitext(args.out)
    for split_name, split_manifest in split_manifests.items():
        split_path = f"{base}_{split_name}{ext}"
        save_manifest(split_path, split_manifest, h5_files)
        print_type_summary(
            split_manifest,
            f"{split_name} split summary (events={count_unique_events(split_manifest)})",
        )
        print(f"{split_name.capitalize()} manifest saved to {split_path}")


if __name__ == "__main__":
    main()
