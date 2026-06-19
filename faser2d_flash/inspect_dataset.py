from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .build_metadata import collect_files
from .schema import REQUIRED_KEYS, load_event, remove_primary_origin_artifact
from .tokenization import (
    PATCH_ENCODING,
    PROJECTION_GEOMETRY,
    TRANSVERSE_PATCH,
    count_nonempty_patches,
)


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect representative 2D event files")
    parser.add_argument("--glob", action="append", required=True, dest="globs")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--patch-size", type=int, default=12)
    parser.add_argument("--expected-xy-views", type=int, default=10)
    parser.add_argument("--keep-primary-origin", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.patch_size != TRANSVERSE_PATCH:
        raise ValueError(
            f"Geometry-aware tokenization requires --patch-size {TRANSVERSE_PATCH}"
        )
    files = collect_files(args.globs)
    rng = np.random.default_rng(7)
    selected = [files[index] for index in rng.choice(len(files), min(args.samples, len(files)), replace=False)]

    invalid = []
    keys = Counter()
    format_example: dict[str, Any] | None = None
    coord_min: dict[str, np.ndarray] = {}
    coord_max: dict[str, np.ndarray] = {}
    value_min: dict[str, float] = {}
    value_max: dict[str, float] = {}
    counts: dict[str, list[float]] = defaultdict(list)
    xy_lengths = Counter()
    xy_nonempty_index = Counter()
    nonfinite_fields = Counter()
    removed_origin_counts = Counter()
    removed_origin_values: dict[str, list[float]] = defaultdict(list)
    repeated_primary_origin = 0

    for path in selected:
        try:
            event = load_event(path, expected_xy_views=args.expected_xy_views)
            keys[tuple(sorted(event))] += 1
            if format_example is None:
                format_example = {}
                with np.load(path, allow_pickle=True) as raw:
                    for key in raw.files:
                        value = raw[key]
                        if key == "xy_projs":
                            format_example[key] = {
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "layer_shapes": [list(np.asarray(layer).shape) for layer in value],
                                "layer_dtypes": [str(np.asarray(layer).dtype) for layer in value],
                            }
                        else:
                            array = np.asarray(value)
                            format_example[key] = {
                                "shape": list(array.shape),
                                "dtype": str(array.dtype),
                            }
            cleaned, removed = remove_primary_origin_artifact(event)
            for view, value in removed.items():
                removed_origin_counts[view] += 1
                removed_origin_values[view].append(value)
            if {"xz_proj", "yz_proj", "xy_0"}.issubset(removed):
                values = np.asarray(
                    [removed["xz_proj"], removed["yz_proj"], removed["xy_0"]],
                    dtype=np.float64,
                )
                if np.allclose(values, values[0], rtol=0.0, atol=1e-5):
                    repeated_primary_origin += 1
            if not args.keep_primary_origin:
                event = cleaned
            views = [("xz", event["xz_proj"]), ("yz", event["yz_proj"])]
            xy_lengths[len(event["xy_projs"])] += 1
            for layer in range(args.expected_xy_views):
                projection = event["xy_projs"][layer]
                if len(projection):
                    xy_nonempty_index[layer] += 1
                views.append(("xy", projection))
            pixel_by_view = {"xz": len(event["xz_proj"]), "yz": len(event["yz_proj"])}
            pixel_by_view["xy"] = sum(len(item[1]) for item in views[2:])
            counts["pixels_two_view"].append(pixel_by_view["xz"] + pixel_by_view["yz"])
            counts["pixels_three_view"].append(sum(pixel_by_view.values()))
            patch_by_view = {}
            for view in ("xz", "yz"):
                patch_by_view[view] = count_nonempty_patches(
                    event[f"{view}_proj"], view=view
                )
            patch_by_view["xy"] = sum(
                count_nonempty_patches(
                    event["xy_projs"][layer], view="xy", xy_layer=layer
                )
                for layer in range(args.expected_xy_views)
            )
            counts["patches_two_view"].append(patch_by_view["xz"] + patch_by_view["yz"])
            counts["patches_three_view"].append(sum(patch_by_view.values()))
            counts["xy_nonempty_views"].append(
                sum(bool(len(event["xy_projs"][layer])) for layer in range(args.expected_xy_views))
            )
            for view, projection in views:
                if not len(projection):
                    continue
                minimum = projection[:, :2].min(axis=0)
                maximum = projection[:, :2].max(axis=0)
                coord_min[view] = np.minimum(coord_min.get(view, minimum), minimum)
                coord_max[view] = np.maximum(coord_max.get(view, maximum), maximum)
                value_min[view] = min(value_min.get(view, float("inf")), float(projection[:, 2].min()))
                value_max[view] = max(value_max.get(view, float("-inf")), float(projection[:, 2].max()))
            for key, value in event.items():
                if key == "xy_projs":
                    continue
                array = np.asarray(value)
                if (
                    array.dtype.fields is None
                    and np.issubdtype(array.dtype, np.number)
                    and not np.isfinite(array).all()
                ):
                    nonfinite_fields[key] += 1
        except Exception as exc:
            invalid.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})

    temporary_files = []
    for pattern in args.globs:
        for directory in Path(pattern).parent.glob(Path(pattern).name):
            if directory.is_dir():
                with os.scandir(directory) as entries:
                    temporary_files.extend(
                        entry.path for entry in entries if ".tmp-" in entry.name
                    )
    report = {
        "source_globs": args.globs,
        "available_files": len(files),
        "sampled_files": len(selected),
        "valid_files": len(selected) - len(invalid),
        "invalid_files": invalid,
        "required_keys": sorted(REQUIRED_KEYS),
        "observed_key_sets": len(keys),
        "example_format": format_example,
        "coordinate_ranges": {
            view: {"min": coord_min[view].tolist(), "max": coord_max[view].tolist()}
            for view in coord_min
        },
        "value_ranges": {
            view: {"min": value_min[view], "max": value_max[view]}
            for view in value_min
        },
        "xy_array_lengths": {str(key): value for key, value in xy_lengths.items()},
        "xy_nonempty_index_counts": {
            str(key): value for key, value in sorted(xy_nonempty_index.items())
        },
        "statistics": {key: describe(value) for key, value in counts.items()},
        "nonfinite_numeric_fields": dict(nonfinite_fields),
        "temporary_files_found": temporary_files,
        "patch_size": args.patch_size,
        "patch_encoding": PATCH_ENCODING,
        "projection_geometry": PROJECTION_GEOMETRY,
        "primary_origin_policy": (
            "kept" if args.keep_primary_origin else "removed from XZ, YZ, and XY0"
        ),
        "primary_origin_artifact": {
            "counts": dict(removed_origin_counts),
            "identical_in_xz_yz_xy0": repeated_primary_origin,
            "charge_statistics": {
                view: describe(values) for view, values in removed_origin_values.items()
            },
        },
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
