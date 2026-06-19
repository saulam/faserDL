from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
from collections import Counter
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_config, pipeline_path
from .schema import (
    charm_label,
    flavour_label,
    lepton_target,
    load_event,
    remove_primary_origin_artifact,
)
from .statistics import RunningMoments, target_statistics
from .tokenization import (
    PATCH_ENCODING,
    PROJECTION_GEOMETRY,
    count_nonempty_patches,
)


def collect_files(patterns: list[str]) -> list[str]:
    files: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(os.path.expanduser(pattern))
        for match in matches:
            path = Path(match)
            if path.is_dir():
                files.update(str(item.resolve()) for item in path.glob("*.npz"))
            elif path.suffix == ".npz":
                files.add(str(path.resolve()))
    if not files:
        raise FileNotFoundError(f"No NPZ files matched: {patterns}")
    return sorted(files, key=str.lower)


def canonical_split(
    files: list[str], seed: int, fractions: list[float]
) -> dict[str, list[str]]:
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must contain three values summing to one")
    permutation = torch.randperm(
        len(files), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    train_length = int(len(files) * fractions[0])
    val_length = int(len(files) * fractions[1])
    return {
        "train": [files[index] for index in permutation[:train_length]],
        "val": [
            files[index]
            for index in permutation[train_length : train_length + val_length]
        ],
        "test": [files[index] for index in permutation[train_length + val_length :]],
    }


def _moments_payload(features: np.ndarray) -> dict[str, Any]:
    moments = RunningMoments.from_array(features)
    return {
        "count": moments.count,
        "mean": moments.mean,
        "m2": moments.m2,
    }


def _log_charge(projection: np.ndarray) -> np.ndarray:
    if not len(projection):
        return np.empty((0, 1), dtype=np.float32)
    return np.log1p(projection[:, 2].astype(np.float32, copy=False)).reshape(-1, 1)


def inspect_for_metadata(
    arguments: tuple[str, bool, int, int, bool, bool]
) -> dict[str, Any]:
    (
        path,
        training,
        patch_size,
        expected_xy_views,
        require_empty_tail,
        remove_origin,
    ) = arguments
    try:
        event = load_event(
            path,
            expected_xy_views=expected_xy_views,
            require_empty_xy_tail=require_empty_tail,
        )
        removed_origin = {}
        if remove_origin:
            event, removed_origin = remove_primary_origin_artifact(event)
        result: dict[str, Any] = {
            "path": path,
            "valid": True,
            "projection_shapes": {
                "xz": (
                    (event["xz_proj"][:, :2].max(axis=0) + 1).astype(int).tolist()
                    if len(event["xz_proj"])
                    else [0, 0]
                ),
                "yz": (
                    (event["yz_proj"][:, :2].max(axis=0) + 1).astype(int).tolist()
                    if len(event["yz_proj"])
                    else [0, 0]
                ),
                "xy": [0, 0],
            },
            "xy_length": len(event["xy_projs"]),
            "xy_nonempty": sum(
                bool(len(event["xy_projs"][index])) for index in range(expected_xy_views)
            ),
            "removed_origin": removed_origin,
        }
        xy_nonempty = [
            event["xy_projs"][index]
            for index in range(expected_xy_views)
            if len(event["xy_projs"][index])
        ]
        if xy_nonempty:
            all_xy = np.concatenate(xy_nonempty, axis=0)
            result["projection_shapes"]["xy"] = (
                all_xy[:, :2].max(axis=0) + 1
            ).astype(int).tolist()
        if training:
            if patch_size != PATCH_ENCODING["xz_yz_patch_shape"][0]:
                raise ValueError(
                    f"Geometry-aware tokenization requires patch_size="
                    f"{PATCH_ENCODING['xz_yz_patch_shape'][0]}"
                )
            xz_charge = _log_charge(event["xz_proj"])
            yz_charge = _log_charge(event["yz_proj"])
            xy_charge = np.concatenate(
                [
                    _log_charge(event["xy_projs"][index])
                    for index in range(expected_xy_views)
                ],
                axis=0,
            )
            xz_patches = count_nonempty_patches(event["xz_proj"], view="xz")
            yz_patches = count_nonempty_patches(event["yz_proj"], view="yz")
            xy_patches = sum(
                count_nonempty_patches(
                    event["xy_projs"][index], view="xy", xy_layer=index
                )
                for index in range(expected_xy_views)
            )
            result.update(
                {
                    "moments": {
                        "xz": _moments_payload(xz_charge),
                        "yz": _moments_payload(yz_charge),
                        "xy": _moments_payload(xy_charge),
                    },
                    "patch_counts": {
                        "xz": xz_patches,
                        "yz": yz_patches,
                        "xy": xy_patches,
                    },
                    "targets": {
                        "visible": np.asarray(
                            event["vis_sp_momentum"], dtype=np.float32
                        ),
                        "jet": np.asarray(event["jet_momentum"], dtype=np.float32),
                        "lepton": lepton_target(event),
                        "vertex": np.asarray(event["primary_vertex"], dtype=np.float32),
                        "is_cc": bool(np.asarray(event["is_cc"]).item()),
                        "flavour": flavour_label(event),
                        "charm": charm_label(event),
                    },
                }
            )
        return result
    except Exception as exc:
        return {"path": path, "valid": False, "error": f"{type(exc).__name__}: {exc}"}


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
    os.replace(temporary, path)


def quantiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create validated manifests and training-only normalization metadata"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--max-files",
        type=int,
        help="Bounded diagnostic build. Do not use this metadata for a full experiment.",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    data_config = config["data"]
    files = collect_files(data_config["globs"])
    split_seed = int(data_config["split_seed"])
    if args.max_files is not None and args.max_files < len(files):
        order = torch.randperm(
            len(files), generator=torch.Generator().manual_seed(split_seed)
        ).tolist()
        files = sorted([files[index] for index in order[: args.max_files]], key=str.lower)
    splits = canonical_split(files, split_seed, list(data_config["split_fractions"]))
    output_dir = pipeline_path(data_config["manifests_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_size = int(data_config["patch_size"])
    expected_xy = int(data_config["expected_xy_views"])
    require_empty_tail = bool(data_config.get("require_empty_xy_tail", True))
    remove_origin = bool(data_config.get("remove_primary_origin_pixel", True))
    moments = {view: RunningMoments.empty(1) for view in ("xz", "yz", "xy")}
    projection_shapes = {view: np.zeros(2, dtype=int) for view in ("xz", "yz", "xy")}
    patch_counts = {"xz": [], "yz": [], "xy": [], "two_view": [], "three_view": []}
    target_values: dict[str, list[Any]] = {
        "visible": [],
        "jet": [],
        "lepton": [],
        "vertex": [],
        "is_cc": [],
    }
    class_counts = {"flavour": Counter(), "charm": Counter()}
    xy_lengths = Counter()
    xy_nonempty_counts: list[int] = []
    rejected: list[dict[str, str]] = []
    removed_origin_counts = Counter()
    removed_origin_values: dict[str, list[float]] = {
        "xz_proj": [],
        "yz_proj": [],
        "xy_0": [],
    }
    valid_splits: dict[str, list[str]] = {}

    for split_name, split_files in splits.items():
        tasks = (
            (
                path,
                split_name == "train",
                patch_size,
                expected_xy,
                require_empty_tail,
                remove_origin,
            )
            for path in split_files
        )
        valid_paths = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(inspect_for_metadata, tasks, chunksize=32)
            for result in results:
                if not result["valid"]:
                    rejected.append({"path": result["path"], "error": result["error"]})
                    continue
                valid_paths.append(result["path"])
                xy_lengths[result["xy_length"]] += 1
                xy_nonempty_counts.append(result["xy_nonempty"])
                for view, value in result["removed_origin"].items():
                    removed_origin_counts[view] += 1
                    removed_origin_values[view].append(value)
                for view in projection_shapes:
                    projection_shapes[view] = np.maximum(
                        projection_shapes[view],
                        np.asarray(result["projection_shapes"][view], dtype=int),
                    )
                if split_name != "train":
                    continue
                for view in moments:
                    payload = result["moments"][view]
                    moments[view].combine(
                        RunningMoments(
                            int(payload["count"]),
                            np.asarray(payload["mean"], dtype=np.float64),
                            np.asarray(payload["m2"], dtype=np.float64),
                        )
                    )
                counts = result["patch_counts"]
                for view in ("xz", "yz", "xy"):
                    patch_counts[view].append(counts[view])
                patch_counts["two_view"].append(counts["xz"] + counts["yz"])
                patch_counts["three_view"].append(
                    counts["xz"] + counts["yz"] + counts["xy"]
                )
                targets = result["targets"]
                for key in target_values:
                    target_values[key].append(targets[key])
                class_counts["flavour"][targets["flavour"]] += 1
                class_counts["charm"][targets["charm"]] += 1
        valid_splits[split_name] = valid_paths
        write_lines(output_dir / f"{split_name}.txt", valid_paths)
        print(f"{split_name}: {len(valid_paths)} valid / {len(split_files)} snapshot files")

    with (output_dir / "rejected.jsonl").open("w", encoding="utf-8") as handle:
        for item in rejected:
            handle.write(json.dumps(item) + "\n")
    train_targets = {
        key: np.asarray(values) for key, values in target_values.items()
    }
    metadata = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_subset": args.max_files is not None,
        "source_globs": data_config["globs"],
        "snapshot_files": len(files),
        "split_seed": split_seed,
        "split_fractions": data_config["split_fractions"],
        "split_counts": {key: len(value) for key, value in valid_splits.items()},
        "rejected_count": len(rejected),
        "patch_size": patch_size,
        "expected_xy_views": expected_xy,
        "remove_primary_origin_pixel": remove_origin,
        "projection_shapes": {
            view: shape.astype(int).tolist() for view, shape in projection_shapes.items()
        },
        "projection_geometry": PROJECTION_GEOMETRY,
        "patch_encoding": PATCH_ENCODING,
        "input_normalization": {
            **{view: summary.result() for view, summary in moments.items()},
            "clip": float(data_config.get("normalization_clip", 12.0)),
        },
        "patch_count_statistics": {
            key: quantiles(values) for key, values in patch_counts.items()
        },
        "xy_array_lengths": dict(xy_lengths),
        "xy_nonempty_view_statistics": quantiles(xy_nonempty_counts),
        "removed_primary_origin": {
            "counts": dict(removed_origin_counts),
            "charge_statistics": {
                view: quantiles(values)
                for view, values in removed_origin_values.items()
                if values
            },
        },
        "target_stats": target_statistics(
            train_targets["visible"],
            train_targets["jet"],
            train_targets["lepton"],
            train_targets["vertex"],
            train_targets["is_cc"],
        ),
        "class_counts": {
            key: {str(label): count for label, count in counts.items()}
            for key, counts in class_counts.items()
        },
    }
    metadata_path = pipeline_path(data_config["metadata_path"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(metadata_path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    os.replace(temporary, metadata_path)
    print(f"Metadata: {metadata_path}")
    print(f"Manifests: {output_dir}")
    print(f"Rejected files: {len(rejected)}")


if __name__ == "__main__":
    main()
