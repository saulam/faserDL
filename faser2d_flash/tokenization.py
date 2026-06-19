from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


TRANSVERSE_SIZE = 48
LONGITUDINAL_SIZE = 236
NUM_MODULES = 10
MODULE_ACTIVE_DEPTH = 20
MODULE_GAP = 4
MODULE_PITCH = MODULE_ACTIVE_DEPTH + MODULE_GAP
TRANSVERSE_PATCH = 12
LONGITUDINAL_PATCH = 5
XY_PATCH = 12
MAX_PATCH_PIXELS = XY_PATCH * XY_PATCH
PATCH_VECTOR_SIZE = 2 * MAX_PATCH_PIXELS

PATCH_ENCODING = {
    "type": "learned_local_cnn",
    "channels": ["standardized_log1p_charge", "occupancy"],
    "xz_yz_patch_shape": [TRANSVERSE_PATCH, LONGITUDINAL_PATCH],
    "xy_patch_shape": [XY_PATCH, XY_PATCH],
    "max_patch_pixels": MAX_PATCH_PIXELS,
    "vector_size": PATCH_VECTOR_SIZE,
}

PROJECTION_GEOMETRY = {
    "transverse_size": TRANSVERSE_SIZE,
    "longitudinal_size": LONGITUDINAL_SIZE,
    "num_modules": NUM_MODULES,
    "module_active_depth": MODULE_ACTIVE_DEPTH,
    "module_gap": MODULE_GAP,
    "module_pitch": MODULE_PITCH,
}

VIEW_TO_ID = {"xz": 0, "yz": 1, "xy": 2}


@dataclass
class TokenizedEvent:
    features: np.ndarray
    positions: np.ndarray
    view_ids: np.ndarray
    token_kinds: np.ndarray
    patch_count: int


def _longitudinal_layout(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = projection[:, :2].astype(np.int64, copy=False)
    transverse = coordinates[:, 0]
    z = coordinates[:, 1]
    module = np.minimum(z // MODULE_PITCH, NUM_MODULES - 1)
    local_z = z - module * MODULE_PITCH
    active = (module >= 0) & (module < NUM_MODULES) & (local_z >= 0) & (
        local_z < MODULE_ACTIVE_DEPTH
    )
    if not active.all():
        bad = coordinates[~active][:8].tolist()
        raise ValueError(f"Projection contains pixels in module gaps or outside geometry: {bad}")
    patch_ids = np.column_stack(
        (module, transverse // TRANSVERSE_PATCH, local_z // LONGITUDINAL_PATCH)
    )
    local_pixel = (transverse % TRANSVERSE_PATCH) * LONGITUDINAL_PATCH + (
        local_z % LONGITUDINAL_PATCH
    )
    return patch_ids, local_pixel, coordinates


def _xy_layout(projection: np.ndarray, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = projection[:, :2].astype(np.int64, copy=False)
    u, v = coordinates[:, 0], coordinates[:, 1]
    patch_ids = np.column_stack(
        (
            np.full(len(projection), layer, dtype=np.int64),
            u // XY_PATCH,
            v // XY_PATCH,
        )
    )
    local_pixel = (u % XY_PATCH) * XY_PATCH + (v % XY_PATCH)
    return patch_ids, local_pixel, coordinates


def count_nonempty_patches(
    projection: np.ndarray,
    *,
    view: str,
    xy_layer: int | None = None,
) -> int:
    if len(projection) == 0:
        return 0
    if view in {"xz", "yz"}:
        patch_ids, _, _ = _longitudinal_layout(projection)
    else:
        if xy_layer is None:
            raise ValueError("xy_layer is required for XY projections")
        patch_ids, _, _ = _xy_layout(projection, xy_layer)
    return len(np.unique(patch_ids, axis=0))


def _patch_positions(unique_ids: np.ndarray, view: str) -> np.ndarray:
    if view in {"xz", "yz"}:
        module = unique_ids[:, 0].astype(np.float32)
        transverse_patch = unique_ids[:, 1].astype(np.float32)
        z_patch = unique_ids[:, 2].astype(np.float32)
        transverse = (
            transverse_patch * TRANSVERSE_PATCH + 0.5 * TRANSVERSE_PATCH
        ) / TRANSVERSE_SIZE
        z = (
            module * MODULE_PITCH
            + z_patch * LONGITUDINAL_PATCH
            + 0.5 * LONGITUDINAL_PATCH
        ) / LONGITUDINAL_SIZE
        zero = np.zeros_like(transverse)
        return np.column_stack(
            (transverse, zero, z) if view == "xz" else (zero, transverse, z)
        ).astype(np.float32)

    layer = unique_ids[:, 0].astype(np.float32)
    patch_u = unique_ids[:, 1].astype(np.float32)
    patch_v = unique_ids[:, 2].astype(np.float32)
    u = (patch_u * XY_PATCH + 0.5 * XY_PATCH) / TRANSVERSE_SIZE
    v = (patch_v * XY_PATCH + 0.5 * XY_PATCH) / TRANSVERSE_SIZE
    z = (layer * MODULE_PITCH + 0.5 * MODULE_ACTIVE_DEPTH) / LONGITUDINAL_SIZE
    return np.column_stack((u, v, z)).astype(np.float32)


def learned_patch_inputs(
    projection: np.ndarray,
    *,
    view: str,
    metadata: dict[str, Any],
    xy_layer: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(projection) == 0:
        return (
            np.empty((0, PATCH_VECTOR_SIZE), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
        )
    if view in {"xz", "yz"}:
        patch_ids, local_pixel, _ = _longitudinal_layout(projection)
        local_pixels = TRANSVERSE_PATCH * LONGITUDINAL_PATCH
    else:
        if xy_layer is None:
            raise ValueError("xy_layer is required for XY projections")
        patch_ids, local_pixel, _ = _xy_layout(projection, xy_layer)
        local_pixels = XY_PATCH * XY_PATCH

    unique_ids, inverse = np.unique(patch_ids, axis=0, return_inverse=True)
    charge = np.zeros((len(unique_ids), MAX_PATCH_PIXELS), dtype=np.float32)
    np.add.at(
        charge,
        (inverse, local_pixel),
        projection[:, 2].astype(np.float32, copy=False),
    )
    occupied = charge[:, :local_pixels] > 0
    log_charge = np.log1p(charge[:, :local_pixels])
    stats = metadata["input_normalization"][view]
    mean = float(np.asarray(stats["mean"]).reshape(-1)[0])
    std = float(np.asarray(stats["std"]).reshape(-1)[0])
    clip = float(metadata["input_normalization"].get("clip", 12.0))
    standardized = np.zeros_like(log_charge)
    standardized[occupied] = np.clip(
        (log_charge[occupied] - mean) / std, -clip, clip
    )

    features = np.zeros((len(unique_ids), PATCH_VECTOR_SIZE), dtype=np.float32)
    features[:, :local_pixels] = standardized
    features[:, MAX_PATCH_PIXELS : MAX_PATCH_PIXELS + local_pixels] = occupied
    return features, _patch_positions(unique_ids, view)


def _anchor_position(view: str, layer: int | None) -> np.ndarray:
    if view == "xz":
        return np.array([[0.5, 0.0, 0.5]], dtype=np.float32)
    if view == "yz":
        return np.array([[0.0, 0.5, 0.5]], dtype=np.float32)
    z = (float(layer) * MODULE_PITCH + 0.5 * MODULE_ACTIVE_DEPTH) / LONGITUDINAL_SIZE
    return np.array([[0.5, 0.5, z]], dtype=np.float32)


def tokenize_event(
    event: dict[str, Any],
    metadata: dict[str, Any],
    *,
    input_mode: str,
    patch_size: int,
    expected_xy_views: int,
    rng: np.random.Generator | None = None,
    gain_log_sigma: float = 0.0,
) -> TokenizedEvent:
    if patch_size != TRANSVERSE_PATCH:
        raise ValueError(
            f"This geometry-aware tokenizer requires patch_size={TRANSVERSE_PATCH}, "
            f"got {patch_size}"
        )
    gain = 1.0
    if rng is not None and gain_log_sigma > 0:
        gain = float(np.exp(rng.normal(0.0, gain_log_sigma)))

    views: list[tuple[str, int | None, np.ndarray]] = [
        ("xz", None, event["xz_proj"]),
        ("yz", None, event["yz_proj"]),
    ]
    if input_mode == "xz_yz_xy":
        views.extend(
            ("xy", layer, event["xy_projs"][layer])
            for layer in range(expected_xy_views)
        )

    feature_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    view_parts: list[np.ndarray] = []
    kind_parts: list[np.ndarray] = []
    patch_count = 0

    for view, layer, raw_projection in views:
        projection = raw_projection
        if gain != 1.0 and len(projection):
            projection = projection.copy()
            projection[:, 2] *= gain

        feature_parts.append(np.zeros((1, PATCH_VECTOR_SIZE), dtype=np.float32))
        position_parts.append(_anchor_position(view, layer))
        view_parts.append(np.array([VIEW_TO_ID[view]], dtype=np.int64))
        kind_parts.append(np.array([1], dtype=np.int64))

        features, positions = learned_patch_inputs(
            projection,
            view=view,
            metadata=metadata,
            xy_layer=layer,
        )
        feature_parts.append(features)
        position_parts.append(positions)
        view_parts.append(np.full(len(features), VIEW_TO_ID[view], dtype=np.int64))
        kind_parts.append(np.zeros(len(features), dtype=np.int64))
        patch_count += len(features)

    return TokenizedEvent(
        features=np.concatenate(feature_parts, axis=0),
        positions=np.concatenate(position_parts, axis=0),
        view_ids=np.concatenate(view_parts, axis=0),
        token_kinds=np.concatenate(kind_parts, axis=0),
        patch_count=patch_count,
    )
