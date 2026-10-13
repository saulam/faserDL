"""Create the canonical test CSV for a JEPA-initialized v7 fine-tune.

This is the v7 counterpart of the supplied ``create_csv_v9_ft.py`` driver.
Unlike the exported v9 run, v7 has no split manifest: training creates its
85/5/10 split with ``torch.utils.data.random_split`` and seed 7.  This driver
repeats that exact operation over the exact ordered file inventory recorded in
the fine-tuning hparams, then evaluates only the resulting test files.

The script deliberately refuses incomplete, scratch, MAE-initialized, partial-
coverage, or wrong-backbone runs.  It also writes a test-file manifest and a
JSON provenance sidecar next to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from glob import glob
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import random_split

from dataset import SparseFASERCALMapDataset
from utils import create_loader

from .inference_common import (
    RESULT_COLUMNS,
    build_eval_args,
    checkpoint_metadata,
    load_models_from_paths,
    resolve_task_checkpoints,
    run_inference,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints_final" / "finetune_jepa_target"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs_final" / "logs"
SPLIT_RATIOS = (0.85, 0.05, 0.10)
SPLIT_SEED = 7

TASK_VALIDATION_METRICS = {
    "flavour": "loss_cls/flavour/val",
    "charm": "loss_cls/charm/val",
    "vis_p": "loss_vis/geom/val",
    "jet_p": "loss_jet/geom/val",
    "vertex": "loss_vertex/val",
}

_MODEL_RE = re.compile(r"\bvit_(?P<size>tiny|base)\b")


def read_hparams(path: Path) -> dict[str, object]:
    """Read top-level Lightning hparams without constructing Python YAML tags."""
    values = {}
    for line in path.read_text().splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        raw_value = raw_value.strip()
        if key == "model":
            match = _MODEL_RE.search(raw_value)
            if not match:
                raise RuntimeError(f"Cannot identify model architecture in {path}: {line}")
            values[key] = match.group("size")
            continue
        try:
            values[key] = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            values[key] = raw_value
    return values


def discover_hparams(log_root: Path, run_name: str) -> Path:
    candidates = []
    for path in (log_root / run_name).glob("version_*/hparams.yaml"):
        try:
            version = int(path.parent.name.removeprefix("version_"))
        except ValueError:
            continue
        candidates.append((version, path))
    if not candidates:
        raise FileNotFoundError(
            f"No version_*/hparams.yaml found under {log_root / run_name}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def resolve_from_repo(value: str | os.PathLike) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def validate_jepa_provenance(
    hparams: dict[str, object],
    hparams_path: Path,
    checkpoint_dir: Path,
    *,
    inspect_pretraining_checkpoint: bool = True,
) -> dict[str, object]:
    """Fail closed unless this is an exact-backbone JEPA target fine-tune."""
    errors = []
    source = hparams.get("pretrained_encoder_resolved_source")
    if source != "target":
        errors.append(f"pretrained encoder source is {source!r}, expected 'target'")

    coverage = hparams.get("pretrained_encoder_coverage")
    if coverage is None or not math.isclose(float(coverage), 1.0):
        errors.append(f"pretrained encoder coverage is {coverage!r}, expected 1.0")

    model = hparams.get("model")
    if model not in {"tiny", "base"}:
        errors.append(f"unknown fine-tuning model architecture {model!r}")

    load_checkpoint = hparams.get("load_checkpoint")
    pretraining_path = (
        resolve_from_repo(load_checkpoint) if load_checkpoint else None
    )
    if pretraining_path is None or not pretraining_path.is_file():
        errors.append(f"pretraining checkpoint is missing: {load_checkpoint!r}")

    recorded_name = hparams.get("checkpoint_name")
    if recorded_name != checkpoint_dir.name:
        errors.append(
            f"checkpoint directory is {checkpoint_dir.name!r}, but hparams record "
            f"checkpoint_name={recorded_name!r}"
        )
    recorded_base = hparams.get("checkpoint_path")
    if recorded_base:
        expected_dir = (resolve_from_repo(recorded_base) / str(recorded_name)).resolve()
        if expected_dir != checkpoint_dir.resolve():
            errors.append(
                f"checkpoint directory {checkpoint_dir.resolve()} does not match "
                f"the hparams location {expected_dir}"
            )

    if errors:
        raise RuntimeError(
            "Refusing to evaluate a checkpoint without valid JEPA provenance:\n- "
            + "\n- ".join(errors)
        )

    if inspect_pretraining_checkpoint:
        pretraining = torch.load(
            pretraining_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        keys = tuple(pretraining.get("state_dict", {}))
        has_context = any(key.startswith("model.context_encoder.") for key in keys)
        has_target = any(key.startswith("model.target_encoder.") for key in keys)
        del pretraining
        if not (has_context and has_target):
            raise RuntimeError(
                f"{pretraining_path} is not a JEPA checkpoint with context and target encoders"
            )

    return {
        "model": model,
        "pretrained_encoder_source": source,
        "pretrained_encoder_coverage": float(coverage),
        "pretraining_checkpoint": str(pretraining_path.resolve()),
        "pretraining_checkpoint_epoch": hparams.get("pretrained_checkpoint_epoch"),
        "pretraining_checkpoint_global_step": hparams.get(
            "pretrained_checkpoint_global_step"
        ),
        "hparams": str(hparams_path.resolve()),
    }


def completed_validation_epochs(metrics_path: Path) -> set[int]:
    if not metrics_path.is_file():
        return set()
    completed = set()
    with metrics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("loss_total/val") and row.get("epoch"):
                completed.add(int(float(row["epoch"])))
    return completed


def require_checkpoints_match_metrics(
    checkpoint_paths: dict[str, str],
    metrics_path: Path,
):
    """Require every selected checkpoint to belong to the chosen logger run."""
    recorded = {}
    with metrics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("epoch"):
                continue
            epoch = int(float(row["epoch"]))
            for task, metric in TASK_VALIDATION_METRICS.items():
                if row.get(metric):
                    recorded[(task, epoch)] = float(row[metric])

    errors = []
    for task, path in checkpoint_paths.items():
        info = checkpoint_metadata(path)
        key = (task, info["epoch"])
        expected = recorded.get(key)
        if expected is None:
            errors.append(
                f"{task}: epoch {info['epoch']} is absent from {metrics_path}"
            )
        elif not math.isclose(
            float(info["value"]), expected, rel_tol=0.0, abs_tol=5.1e-7,
        ):
            errors.append(
                f"{task}: filename loss {info['value']} does not match "
                f"logged loss {expected} at epoch {info['epoch']}"
            )
    if errors:
        raise RuntimeError(
            "Selected checkpoints do not belong to the selected completed run:\n- "
            + "\n- ".join(errors)
        )


def require_completed_training(hparams: dict[str, object], hparams_path: Path) -> int:
    expected_epochs = int(hparams["epochs"])
    completed = completed_validation_epochs(hparams_path.with_name("metrics.csv"))
    expected = set(range(expected_epochs))
    if not expected.issubset(completed):
        missing = sorted(expected - completed)
        raise RuntimeError(
            f"Fine-tuning is not complete: validation is missing epochs {missing}; "
            f"expected {expected_epochs} epochs"
        )
    return max(completed)


def discover_event_files(dataset_path: str | os.PathLike) -> list[str]:
    """Match ``SparseFASERCALMapDataset`` discovery and ordering exactly."""
    root = str(dataset_path)
    return sorted(
        chain(
            glob(os.path.join(root, "*.npz")),
            glob(os.path.join(root, "*", "*.npz")),
        ),
        key=str.lower,
    )


def event_identity(path: str | os.PathLike) -> str:
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def file_list_fingerprint(files: list[str]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(event_identity(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def deterministic_split_files(
    files: list[str],
    ratios=SPLIT_RATIOS,
    seed: int = SPLIT_SEED,
) -> tuple[list[str], list[str], list[str]]:
    """Repeat the exact length rounding and ``random_split`` used in training."""
    if len(ratios) != 3 or not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("ratios must contain three values summing to one")
    train_length = int(len(files) * ratios[0])
    validation_length = int(len(files) * ratios[1])
    test_length = len(files) - train_length - validation_length
    subsets = random_split(
        range(len(files)),
        [train_length, validation_length, test_length],
        generator=torch.Generator().manual_seed(seed),
    )
    return tuple(
        [files[index] for index in subset.indices]
        for subset in subsets
    )


def require_same_inventory(reference_files: list[str], candidate_files: list[str]):
    reference = [event_identity(path) for path in reference_files]
    candidate = [event_identity(path) for path in candidate_files]
    if reference != candidate:
        first_difference = next(
            (
                index
                for index, (left, right) in enumerate(zip(reference, candidate))
                if left != right
            ),
            min(len(reference), len(candidate)),
        )
        left = reference[first_difference] if first_difference < len(reference) else None
        right = candidate[first_difference] if first_difference < len(candidate) else None
        raise RuntimeError(
            "Evaluation data do not have the same ordered event inventory as training: "
            f"{len(reference)} versus {len(candidate)} files; first difference at "
            f"index {first_difference}: {left!r} versus {right!r}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_results_frame(frame: pd.DataFrame, expected_rows: int | None = None):
    if tuple(frame.columns) != RESULT_COLUMNS:
        raise RuntimeError(
            "CSV schema mismatch:\n"
            f"expected {list(RESULT_COLUMNS)}\nreceived {list(frame.columns)}"
        )
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(
            f"Inference produced {len(frame)} rows for {expected_rows} test files"
        )
    duplicates = frame.duplicated(subset=["run_number", "event_id"]).sum()
    if duplicates:
        raise RuntimeError(f"Found {duplicates} duplicated (run_number, event_id) rows")

    flavour_probabilities = [
        "CCnue_prob", "CCnumu_prob", "CCnutau2e_prob", "CCnutau2mu_prob",
        "CCnutau2H_prob", "NC_prob",
    ]
    charm_probabilities = [
        "nocharm_prob", "charm2e_prob", "charm2mu_prob", "charm2H_prob",
    ]
    numeric = frame.drop(columns=["flavour_label_true", "charm_label_true", "is_cc"])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("CSV contains non-finite numeric values")
    for columns, name in (
        (flavour_probabilities, "flavour"),
        (charm_probabilities, "charm"),
    ):
        probabilities = frame[columns].to_numpy(dtype=np.float64)
        if (probabilities < -1e-7).any() or (probabilities > 1 + 1e-7).any():
            raise RuntimeError(f"{name} probabilities lie outside [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-5):
            raise RuntimeError(f"{name} probabilities do not sum to one")


def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, frame: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False, na_rep="N/A")
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def default_output(run_name: str, max_batches: int | None) -> Path:
    suffix = "" if max_batches is None else f"_smoke_{max_batches}batches"
    return REPO_ROOT / "results" / f"results_v7_test_{run_name}{suffix}.csv"


def build_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--hparams", type=Path, default=None)
    parser.add_argument("--log_root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--dataset_path", default=None,
        help="optional relocation; ordered event inventory must match the training path",
    )
    parser.add_argument(
        "--metadata", type=Path, default=None,
        help="optional relocation; contents must match the training metadata",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def configure_multiprocessing():
    """Avoid exhausting file descriptors while DataLoader workers share tensors."""
    torch.multiprocessing.set_sharing_strategy("file_system")


def main(argv=None):
    cli = build_cli().parse_args(argv)
    configure_multiprocessing()
    checkpoint_dir = cli.checkpoint_dir.resolve()
    hparams_path = (
        cli.hparams.resolve()
        if cli.hparams is not None
        else discover_hparams(cli.log_root.resolve(), checkpoint_dir.name)
    )
    hparams = read_hparams(hparams_path)
    provenance = validate_jepa_provenance(hparams, hparams_path, checkpoint_dir)
    completed_epoch = require_completed_training(hparams, hparams_path)

    training_dataset_path = str(hparams["dataset_path"])
    evaluation_dataset_path = cli.dataset_path or training_dataset_path
    training_files = discover_event_files(training_dataset_path)
    evaluation_files = discover_event_files(evaluation_dataset_path)
    if not training_files:
        raise RuntimeError(f"No training data files found at {training_dataset_path}")
    require_same_inventory(training_files, evaluation_files)

    training_metadata = resolve_from_repo(str(hparams["metadata_path"]))
    evaluation_metadata = (
        cli.metadata.resolve() if cli.metadata is not None else training_metadata
    )
    if not training_metadata.is_file() or not evaluation_metadata.is_file():
        raise FileNotFoundError(
            f"Metadata missing: training={training_metadata}, evaluation={evaluation_metadata}"
        )
    training_metadata_hash = sha256_file(training_metadata)
    evaluation_metadata_hash = sha256_file(evaluation_metadata)
    if training_metadata_hash != evaluation_metadata_hash:
        raise RuntimeError("Evaluation metadata do not match the fine-tuning metadata")

    # train/finetune.py calls split_dataset without a seed argument, so the
    # split always uses split_dataset's seed-7 default.  The training seed in
    # hparams controls model/data-loader randomness but does not change this.
    train_files, validation_files, test_files = deterministic_split_files(
        evaluation_files,
        seed=SPLIT_SEED,
    )
    print(
        f"Run: {checkpoint_dir.name}\n"
        f"  hparams: {hparams_path}\n"
        f"  architecture: vit_{provenance['model']}\n"
        f"  JEPA encoder: {provenance['pretrained_encoder_source']} "
        f"(coverage={provenance['pretrained_encoder_coverage']:.1f})\n"
        f"  dataset: {evaluation_dataset_path}\n"
        f"  split: {len(train_files)} train / {len(validation_files)} val / "
        f"{len(test_files)} test (seed={SPLIT_SEED})"
    )

    output = cli.out or default_output(checkpoint_dir.name, cli.max_batches)
    output = output.resolve()
    manifest_path = output.with_suffix(".manifest.txt")
    metadata_path = output.with_suffix(".metadata.json")
    existing = [path for path in (output, manifest_path, metadata_path) if path.exists()]
    if existing and not cli.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output(s): {existing}; pass --overwrite explicitly"
        )
    if cli.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if cli.num_workers < 0:
        raise ValueError("--num_workers cannot be negative")
    if cli.max_batches is not None and cli.max_batches <= 0:
        raise ValueError("--max_batches must be positive")

    args = build_eval_args(
        str(evaluation_metadata),
        str(evaluation_dataset_path),
        model=str(provenance["model"]),
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        dropout=float(hparams.get("dropout", 0.0)),
        attn_dropout=float(hparams.get("attn_dropout", 0.0)),
        drop_path_rate=float(hparams.get("drop_path_rate", 0.0)),
        head_init=float(hparams.get("head_init", 2e-5)),
        head_dropout_cls=float(hparams.get("head_dropout_cls", 0.0)),
        head_dropout_reg=float(hparams.get("head_dropout_reg", 0.0)),
        preprocessing_input=str(hparams.get("preprocessing_input", "log")),
        preprocessing_output=str(hparams.get("preprocessing_output", "log")),
        label_smoothing=float(hparams.get("label_smoothing", 0.02)),
    )
    dataset = SparseFASERCALMapDataset(args)
    require_same_inventory(evaluation_files, dataset.data_files)
    dataset.data_files = test_files
    loader = create_loader(
        dataset,
        shuffle=False,
        drop_last=False,
        args=args,
    )

    if cli.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass --cpu only for diagnostic use")
        if cli.gpu < 0 or cli.gpu >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {cli.gpu} is invalid; visible device count is {torch.cuda.device_count()}"
            )
        device = torch.device(f"cuda:{cli.gpu}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(device)}")

    checkpoint_paths = resolve_task_checkpoints(str(checkpoint_dir))
    require_checkpoints_match_metrics(
        checkpoint_paths,
        hparams_path.with_name("metrics.csv"),
    )
    print("Loading best raw-weight checkpoint per task:")
    models = load_models_from_paths(
        checkpoint_paths, args, dataset.metadata, device,
    )
    results = run_inference(
        models,
        loader,
        dataset,
        device,
        max_batches=cli.max_batches,
    )
    frame = pd.DataFrame(results, columns=RESULT_COLUMNS)
    frame = frame.sort_values(["run_number", "event_id"]).reset_index(drop=True)
    validate_results_frame(
        frame,
        expected_rows=len(test_files) if cli.max_batches is None else None,
    )

    manifest_text = "".join(f"{path}\n" for path in test_files)
    checkpoint_report = {}
    for task, path_string in checkpoint_paths.items():
        path = Path(path_string)
        checkpoint_report[task] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            **checkpoint_metadata(str(path)),
        }
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv": str(output),
        "partial": cli.max_batches is not None,
        "max_batches": cli.max_batches,
        "rows": len(frame),
        "columns": list(RESULT_COLUMNS),
        "dataset_path": str(evaluation_dataset_path),
        "dataset_events": len(evaluation_files),
        "dataset_inventory_sha256": file_list_fingerprint(evaluation_files),
        "metadata_path": str(evaluation_metadata.resolve()),
        "metadata_sha256": evaluation_metadata_hash,
        "split": {
            "algorithm": "torch.utils.data.random_split",
            "ratios": list(SPLIT_RATIOS),
            "seed": SPLIT_SEED,
            "train_events": len(train_files),
            "validation_events": len(validation_files),
            "test_events": len(test_files),
            "test_inventory_sha256": file_list_fingerprint(test_files),
            "manifest": str(manifest_path),
            "csv_order": ["run_number", "event_id"],
        },
        "fine_tuning": {
            **provenance,
            "completed_epoch": completed_epoch,
            "training_seed": hparams.get("seed"),
            "weights": "raw state_dict selected by raw validation loss",
        },
        "task_checkpoints": checkpoint_report,
    }

    _atomic_write_csv(output, frame)
    _atomic_write_text(manifest_path, manifest_text)
    _atomic_write_text(metadata_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(frame)} rows and {len(frame.columns)} columns:\n"
        f"  CSV:      {output}\n"
        f"  manifest: {manifest_path}\n"
        f"  metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()
