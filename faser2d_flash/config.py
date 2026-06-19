from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .tokenization import TRANSVERSE_PATCH


PIPELINE_ROOT = Path(__file__).resolve().parent


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_recursive(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Config inheritance cycle at {path}")
    seen.add(path)
    with path.open(encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    parent = current.pop("extends", None)
    if parent is None:
        return current
    parent_path = (path.parent / parent).resolve()
    return _merge(_load_recursive(parent_path, seen), current)


def _parse_override(value: str) -> Any:
    return yaml.safe_load(value)


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _load_recursive(config_path, set())
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        _set_nested(config, key, _parse_override(value))
    config["_config_path"] = str(config_path)
    config["_pipeline_root"] = str(PIPELINE_ROOT)
    validate_config(config)
    return config


def pipeline_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PIPELINE_ROOT / path


def validate_config(config: dict[str, Any]) -> None:
    mode = config["data"]["input_mode"]
    if mode not in {"xz_yz", "xz_yz_xy"}:
        raise ValueError(f"Unsupported input mode: {mode}")
    backend = config["model"]["attention_backend"]
    if backend not in {"flash", "torch_sdpa"}:
        raise ValueError(f"Unsupported attention backend: {backend}")
    dim = int(config["model"]["embed_dim"])
    heads = int(config["model"]["num_heads"])
    if dim % heads:
        raise ValueError("model.embed_dim must be divisible by model.num_heads")
    if int(config["data"]["patch_size"]) != TRANSVERSE_PATCH:
        raise ValueError(
            f"Geometry-aware tokenization requires data.patch_size={TRANSVERSE_PATCH}"
        )
    if int(config["model"].get("patch_encoder_channels", 16)) <= 0:
        raise ValueError("model.patch_encoder_channels must be positive")
