from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import (
    InvalidEventError,
    charm_label,
    flavour_label,
    lepton_target,
    load_event,
    remove_primary_origin_artifact,
)
from .tokenization import (
    PATCH_ENCODING,
    PATCH_VECTOR_SIZE,
    PROJECTION_GEOMETRY,
    TokenizedEvent,
    tokenize_event,
)


TASK_NAMES = ("flavour", "charm", "vis", "jet", "vertex")
NUM_TASKS = len(TASK_NAMES)


def read_manifest(path: str | Path) -> list[str]:
    with Path(path).open(encoding="utf-8") as handle:
        files = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    if not files:
        raise ValueError(f"Manifest is empty: {path}")
    return files


def load_metadata(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("patch_encoding") != PATCH_ENCODING:
        raise ValueError(
            "Metadata patch encoding does not match this code. Rebuild metadata "
            "with python -m faser2d_flash.build_metadata."
        )
    if metadata.get("projection_geometry") != PROJECTION_GEOMETRY:
        raise ValueError(
            "Metadata projection geometry does not match this code. Rebuild metadata."
        )
    return metadata


class ProjectionDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        metadata: dict[str, Any],
        data_config: dict[str, Any],
        *,
        training: bool,
        seed: int,
    ):
        self.files = read_manifest(manifest)
        self.metadata = metadata
        self.config = data_config
        self.training = training
        self.seed = int(seed)
        self.epoch = 0
        self.repeat_factor = int(self.config.get("repeat_factor", 1))
        if self.repeat_factor <= 0:
            raise ValueError("data.repeat_factor must be positive")

    def __len__(self) -> int:
        return len(self.files) * self.repeat_factor

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, path: str) -> np.random.Generator:
        material = f"{self.seed}:{self.epoch}:{path}".encode()
        seed = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "little")
        return np.random.default_rng(seed)

    def __getitem__(self, index: int) -> dict[str, Any] | None:
        path = self.files[index % len(self.files)]
        try:
            event = load_event(
                path,
                expected_xy_views=int(self.config["expected_xy_views"]),
                require_empty_xy_tail=bool(self.config.get("require_empty_xy_tail", True)),
                retries=int(self.config.get("load_retries", 2)),
                retry_delay=float(self.config.get("retry_delay", 0.1)),
            )
            if bool(self.config.get("remove_primary_origin_pixel", True)):
                event, _ = remove_primary_origin_artifact(event)
            rng = self._rng(path) if self.training else None
            tokens = tokenize_event(
                event,
                self.metadata,
                input_mode=self.config["input_mode"],
                patch_size=int(self.config["patch_size"]),
                expected_xy_views=int(self.config["expected_xy_views"]),
                rng=rng,
                gain_log_sigma=float(
                    self.config.get("augmentation", {}).get("gain_log_sigma", 0.0)
                )
                if self.training
                else 0.0,
            )
            is_cc = bool(np.asarray(event["is_cc"]).item())
            vertex_stats = self.metadata["target_stats"]["primary_vertex"]
            vertex_mean = np.asarray(vertex_stats["mean"], dtype=np.float32)
            vertex_std = np.asarray(vertex_stats["std"], dtype=np.float32)
            return {
                "tokens": tokens,
                "targets": {
                    "flavour_label": flavour_label(event),
                    "charm_label": charm_label(event),
                    "vis_sp_momentum": np.asarray(event["vis_sp_momentum"], dtype=np.float32),
                    "jet_momentum": np.asarray(event["jet_momentum"], dtype=np.float32),
                    "out_lepton_momentum": lepton_target(event),
                    "primary_vertex": (
                        np.asarray(event["primary_vertex"], dtype=np.float32) - vertex_mean
                    )
                    / vertex_std,
                    "is_cc": float(is_cc),
                },
                "run_number": int(np.asarray(event["run_number"]).item()),
                "event_id": int(np.asarray(event["event_id"]).item()),
                "path": path,
            }
        except InvalidEventError:
            if self.config.get("invalid_policy", "error") == "skip":
                return None
            raise


@dataclass
class ProjectionBatch:
    features: torch.Tensor
    positions: torch.Tensor
    view_ids: torch.Tensor
    token_kinds: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    task_indices: torch.Tensor
    targets: dict[str, torch.Tensor]
    run_numbers: torch.Tensor
    event_ids: torch.Tensor
    paths: list[str]
    patch_counts: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.task_indices.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: torch.device, non_blocking: bool = True) -> "ProjectionBatch":
        return ProjectionBatch(
            features=self.features.to(device, non_blocking=non_blocking),
            positions=self.positions.to(device, non_blocking=non_blocking),
            view_ids=self.view_ids.to(device, non_blocking=non_blocking),
            token_kinds=self.token_kinds.to(device, non_blocking=non_blocking),
            cu_seqlens=self.cu_seqlens.to(device, non_blocking=non_blocking),
            max_seqlen=self.max_seqlen,
            task_indices=self.task_indices.to(device, non_blocking=non_blocking),
            targets={
                key: value.to(device, non_blocking=non_blocking)
                for key, value in self.targets.items()
            },
            run_numbers=self.run_numbers,
            event_ids=self.event_ids,
            paths=self.paths,
            patch_counts=self.patch_counts,
        )


def collate_events(
    samples: list[dict[str, Any] | None],
    *,
    require_all_valid: bool = False,
) -> ProjectionBatch:
    valid = [sample for sample in samples if sample is not None]
    if require_all_valid and len(valid) != len(samples):
        raise RuntimeError(
            "A validated DDP manifest produced an invalid sample. Rebuild metadata "
            "or set data.invalid_policy=error to expose the source file."
        )
    if not valid:
        raise RuntimeError("Every sample in this batch was invalid")

    features: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    view_ids: list[torch.Tensor] = []
    token_kinds: list[torch.Tensor] = []
    task_indices: list[torch.Tensor] = []
    lengths: list[int] = []
    patch_counts: list[int] = []
    offset = 0

    for sample in valid:
        tokens: TokenizedEvent = sample["tokens"]
        sequence_length = NUM_TASKS + len(tokens.features)
        features.append(
            torch.cat(
                (
                    torch.zeros(NUM_TASKS, PATCH_VECTOR_SIZE),
                    torch.from_numpy(tokens.features),
                ),
                dim=0,
            )
        )
        positions.append(
            torch.cat((torch.zeros(NUM_TASKS, 3), torch.from_numpy(tokens.positions)), dim=0)
        )
        view_ids.append(
            torch.cat(
                (
                    torch.full((NUM_TASKS,), 3, dtype=torch.long),
                    torch.from_numpy(tokens.view_ids),
                )
            )
        )
        token_kinds.append(
            torch.cat(
                (
                    torch.full((NUM_TASKS,), 2, dtype=torch.long),
                    torch.from_numpy(tokens.token_kinds),
                )
            )
        )
        task_indices.append(torch.arange(offset, offset + NUM_TASKS, dtype=torch.long))
        lengths.append(sequence_length)
        patch_counts.append(tokens.patch_count)
        offset += sequence_length

    cu = torch.zeros(len(valid) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    target_keys = valid[0]["targets"].keys()
    targets = {
        key: torch.as_tensor(
            np.stack([sample["targets"][key] for sample in valid])
            if np.asarray(valid[0]["targets"][key]).ndim
            else [sample["targets"][key] for sample in valid]
        )
        for key in target_keys
    }
    targets["flavour_label"] = targets["flavour_label"].long()
    targets["charm_label"] = targets["charm_label"].long()
    for key in (
        "vis_sp_momentum",
        "jet_momentum",
        "out_lepton_momentum",
        "primary_vertex",
        "is_cc",
    ):
        targets[key] = targets[key].float()

    return ProjectionBatch(
        features=torch.cat(features, dim=0).float(),
        positions=torch.cat(positions, dim=0).float(),
        view_ids=torch.cat(view_ids, dim=0).long(),
        token_kinds=torch.cat(token_kinds, dim=0).long(),
        cu_seqlens=cu,
        max_seqlen=max(lengths),
        task_indices=torch.stack(task_indices),
        targets=targets,
        run_numbers=torch.tensor([sample["run_number"] for sample in valid], dtype=torch.long),
        event_ids=torch.tensor([sample["event_id"] for sample in valid], dtype=torch.long),
        paths=[sample["path"] for sample in valid],
        patch_counts=torch.tensor(patch_counts, dtype=torch.long),
    )
