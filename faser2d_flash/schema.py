from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = {
    "run_number",
    "event_id",
    "is_cc",
    "is_tau",
    "is_charmed",
    "charm_decay",
    "vis_sp_momentum",
    "jet_momentum",
    "out_lepton_momentum",
    "tau_vis_momentum",
    "primary_vertex",
    "in_neutrino_pdg",
    "tau_decay_mode",
    "xz_proj",
    "yz_proj",
    "xy_projs",
}

TARGET_VECTOR_KEYS = (
    "vis_sp_momentum",
    "jet_momentum",
    "out_lepton_momentum",
    "tau_vis_momentum",
    "primary_vertex",
)


class InvalidEventError(RuntimeError):
    pass


def _projection(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 3:
        raise InvalidEventError(f"{name} must have shape (N, 3), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise InvalidEventError(f"{name} must be numeric, got {array.dtype}")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise InvalidEventError(f"{name} contains non-finite values")
    if array.size and ((array[:, :2] < 0).any() or (array[:, 2] < 0).any()):
        raise InvalidEventError(f"{name} contains negative coordinates or charge")
    return array


def validate_event(
    data: dict[str, Any],
    *,
    expected_xy_views: int = 10,
    require_empty_xy_tail: bool = True,
) -> dict[str, Any]:
    missing = REQUIRED_KEYS.difference(data)
    if missing:
        raise InvalidEventError(f"Missing keys: {sorted(missing)}")

    event = dict(data)
    event["xz_proj"] = _projection(event["xz_proj"], "xz_proj")
    event["yz_proj"] = _projection(event["yz_proj"], "yz_proj")

    xy_raw = np.asarray(event["xy_projs"], dtype=object)
    if xy_raw.ndim != 1:
        raise InvalidEventError(f"xy_projs must be one-dimensional, got {xy_raw.shape}")
    if len(xy_raw) < expected_xy_views:
        raise InvalidEventError(
            f"xy_projs has {len(xy_raw)} entries; expected at least {expected_xy_views}"
        )
    xy = [_projection(layer, f"xy_projs[{index}]") for index, layer in enumerate(xy_raw)]
    if require_empty_xy_tail:
        nonempty_tail = [index for index in range(expected_xy_views, len(xy)) if len(xy[index])]
        if nonempty_tail:
            raise InvalidEventError(
                f"Found non-empty XY projections beyond index {expected_xy_views - 1}: "
                f"{nonempty_tail[:8]}"
            )
    event["xy_projs"] = xy

    for key in TARGET_VECTOR_KEYS:
        value = np.asarray(event[key], dtype=np.float32)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise InvalidEventError(f"{key} must be a finite vector with shape (3,), got {value}")
        event[key] = value

    charm = np.asarray(event["charm_decay"])
    if charm.dtype.fields is None or "pdg" not in charm.dtype.fields:
        raise InvalidEventError("charm_decay must be a structured array with a 'pdg' field")
    event["charm_decay"] = charm.copy()
    return event


def load_event(
    path: str | Path,
    *,
    expected_xy_views: int = 10,
    require_empty_xy_tail: bool = True,
    retries: int = 2,
    retry_delay: float = 0.1,
) -> dict[str, Any]:
    path = Path(path)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with np.load(path, allow_pickle=True) as archive:
                payload = {key: archive[key] for key in archive.files}
            return validate_event(
                payload,
                expected_xy_views=expected_xy_views,
                require_empty_xy_tail=require_empty_xy_tail,
            )
        except (OSError, EOFError, ValueError, zipfile.BadZipFile, InvalidEventError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay)
    raise InvalidEventError(f"Could not load {path}: {last_error}") from last_error


def remove_primary_origin_artifact(
    event: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    cleaned = dict(event)
    removed: dict[str, float] = {}

    for key in ("xz_proj", "yz_proj"):
        projection = event[key]
        mask = (projection[:, 0] == 0) & (projection[:, 1] == 0)
        if mask.any():
            removed[key] = float(projection[mask, 2].sum())
            cleaned[key] = projection[~mask].copy()

    xy = list(event["xy_projs"])
    if xy:
        projection = xy[0]
        mask = (projection[:, 0] == 0) & (projection[:, 1] == 0)
        if mask.any():
            removed["xy_0"] = float(projection[mask, 2].sum())
            xy[0] = projection[~mask].copy()
    cleaned["xy_projs"] = xy
    return cleaned, removed


def flavour_label(event: dict[str, Any]) -> int:
    is_cc = bool(np.asarray(event["is_cc"]).item())
    pdg = abs(int(np.asarray(event["in_neutrino_pdg"]).item()))
    mode = int(np.asarray(event["tau_decay_mode"]).item())
    if not is_cc:
        return 5
    if pdg == 12:
        return 0
    if pdg == 14:
        return 1
    if pdg == 16:
        if mode <= 0:
            raise InvalidEventError("CC tau event has no valid tau_decay_mode")
        return 2 if mode == 1 else 3 if mode == 2 else 4
    raise InvalidEventError(f"Unsupported CC incoming neutrino PDG: {pdg}")


def charm_label(event: dict[str, Any]) -> int:
    if not bool(np.asarray(event["is_charmed"]).item()):
        return 0
    pdgs = set(np.asarray(event["charm_decay"])["pdg"].astype(int).tolist())
    if pdgs.intersection({-13, 13}):
        return 2
    if pdgs.intersection({-11, 11}):
        return 1
    return 3


def lepton_target(event: dict[str, Any]) -> np.ndarray:
    is_cc = bool(np.asarray(event["is_cc"]).item())
    if not is_cc:
        return np.zeros(3, dtype=np.float32)
    pdg = abs(int(np.asarray(event["in_neutrino_pdg"]).item()))
    if pdg == 16:
        return np.asarray(event["tau_vis_momentum"], dtype=np.float32)
    return np.asarray(event["out_lepton_momentum"], dtype=np.float32)
