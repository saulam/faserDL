from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .checkpoint import ExponentialMovingAverage
from .config import PIPELINE_ROOT, load_config, pipeline_path, validate_config
from .dataset import ProjectionDataset, load_metadata
from .evaluate import component_state, resolve_checkpoint
from .model import ProjectionTransformer
from .runtime import require_flash_runtime, runtime_report
from .train import autocast_context, make_loader, seed_everything


CSV_COLUMNS = (
    "run_number",
    "event_id",
    "flavour_label_true",
    "CCnue_prob",
    "CCnumu_prob",
    "CCnutau2e_prob",
    "CCnutau2mu_prob",
    "CCnutau2H_prob",
    "NC_prob",
    "charm_label_true",
    "nocharm_prob",
    "charm2e_prob",
    "charm2mu_prob",
    "charm2H_prob",
    "evis_true",
    "evis_reco",
    "ptmiss_true",
    "ptmiss_reco",
    "vis_px_true",
    "vis_py_true",
    "vis_pz_true",
    "vis_px_reco",
    "vis_py_reco",
    "vis_pz_reco",
    "out_lepton_px_true",
    "out_lepton_py_true",
    "out_lepton_pz_true",
    "out_lepton_px_reco",
    "out_lepton_py_reco",
    "out_lepton_pz_reco",
    "jet_px_true",
    "jet_py_true",
    "jet_pz_true",
    "jet_px_reco",
    "jet_py_reco",
    "jet_pz_reco",
    "primary_vertex_x_true",
    "primary_vertex_y_true",
    "primary_vertex_z_true",
    "primary_vertex_x_reco",
    "primary_vertex_y_reco",
    "primary_vertex_z_reco",
)

TASK_CHECKPOINTS = {
    "flavour": "best_flavour",
    "charm": "best_charm",
    "vis": "best_vis",
    "jet": "best_jet",
    "vertex": "best_vertex",
}

FLAVOUR_NAMES = ("nue", "numu", "nutau->e", "nutau->mu", "nutau->had", "NC")
CHARM_NAMES = ("no_charm", "charm->e", "charm->mu", "charm->had")
EVENT_NAME = re.compile(r"^run_(\d+)_event_(\d+)\.npz$")


def event_sort_key(path: str) -> tuple[int, int]:
    match = EVENT_NAME.match(Path(path).name)
    if match is None:
        raise ValueError(
            f"Cannot sort {path!r} by run and event; expected run_<run>_event_<event>.npz"
        )
    return int(match.group(1)), int(match.group(2))


def load_recorded_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    run_dir = (
        pipeline_path(config["experiment"]["output_dir"])
        / config["experiment"]["name"]
    )
    record_path = run_dir / "run_config.json"
    if record_path.exists():
        with record_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        recorded = payload.get("config")
        if isinstance(recorded, dict):
            config = recorded
            config["_config_path"] = str(Path(config_path).resolve())
            config["_pipeline_root"] = str(PIPELINE_ROOT)
            validate_config(config)
    return config, run_dir


def load_task_model(
    checkpoint_path: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    device: torch.device,
    *,
    use_ema: bool,
) -> ProjectionTransformer:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ProjectionTransformer(config["model"], metadata)
    model.load_state_dict(component_state(checkpoint, "model"))
    if use_ema:
        if "ema" not in checkpoint:
            raise KeyError(f"Checkpoint has no EMA state: {checkpoint_path}")
        ema = ExponentialMovingAverage(
            model, float(config["training"]["ema_decay"])
        )
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    print(
        f"{checkpoint_path.stem:16s} "
        f"epoch={int(checkpoint.get('epoch', -1))} "
        f"global_step={int(checkpoint.get('global_step', -1))}"
    )
    del checkpoint
    return model.to(device).eval()


def load_task_models(
    checkpoint_dir: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    device: torch.device,
    *,
    use_ema: bool,
) -> dict[str, ProjectionTransformer]:
    return {
        task: load_task_model(
            resolve_checkpoint(checkpoint_dir, stem),
            config,
            metadata,
            device,
            use_ema=use_ema,
        )
        for task, stem in TASK_CHECKPOINTS.items()
    }


def combine_outputs(
    task_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "out_flavour": task_outputs["flavour"]["out_flavour"],
        "out_charm": task_outputs["charm"]["out_charm"],
        "out_vis": task_outputs["vis"]["out_vis"],
        "out_jet": task_outputs["jet"]["out_jet"],
        "out_vertex": task_outputs["vertex"]["out_vertex"],
    }


def prediction_rows(
    batch,
    outputs: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    flavour_prob = outputs["out_flavour"].softmax(-1).float().cpu().numpy()
    charm_prob = outputs["out_charm"].softmax(-1).float().cpu().numpy()
    vis_reco = outputs["out_vis"]["p_cart"].float().cpu().numpy()
    jet_reco = outputs["out_jet"]["p_cart"].float().cpu().numpy()
    lepton_reco = vis_reco - jet_reco

    vis_true = batch.targets["vis_sp_momentum"].float().cpu().numpy()
    jet_true = batch.targets["jet_momentum"].float().cpu().numpy()
    lepton_true = batch.targets["out_lepton_momentum"].float().cpu().numpy()
    flavour_true = batch.targets["flavour_label"].cpu().numpy()
    charm_true = batch.targets["charm_label"].cpu().numpy()

    vertex_stats = metadata["target_stats"]["primary_vertex"]
    vertex_mean = np.asarray(vertex_stats["mean"], dtype=np.float32)
    vertex_std = np.asarray(vertex_stats["std"], dtype=np.float32)
    vertex_true = batch.targets["primary_vertex"].float().cpu().numpy()
    vertex_true = vertex_true * vertex_std + vertex_mean
    vertex_reco = outputs["out_vertex"].float().cpu().numpy()
    vertex_reco = vertex_reco * vertex_std + vertex_mean

    evis_true = np.linalg.norm(vis_true, axis=1)
    evis_reco = np.linalg.norm(vis_reco, axis=1)
    ptmiss_true = np.linalg.norm(vis_true[:, :2], axis=1)
    ptmiss_reco = np.linalg.norm(vis_reco[:, :2], axis=1)

    rows = []
    for index in range(batch.batch_size):
        row = {
            "run_number": int(batch.run_numbers[index]),
            "event_id": int(batch.event_ids[index]),
            "flavour_label_true": FLAVOUR_NAMES[int(flavour_true[index])],
            "CCnue_prob": float(flavour_prob[index, 0]),
            "CCnumu_prob": float(flavour_prob[index, 1]),
            "CCnutau2e_prob": float(flavour_prob[index, 2]),
            "CCnutau2mu_prob": float(flavour_prob[index, 3]),
            "CCnutau2H_prob": float(flavour_prob[index, 4]),
            "NC_prob": float(flavour_prob[index, 5]),
            "charm_label_true": CHARM_NAMES[int(charm_true[index])],
            "nocharm_prob": float(charm_prob[index, 0]),
            "charm2e_prob": float(charm_prob[index, 1]),
            "charm2mu_prob": float(charm_prob[index, 2]),
            "charm2H_prob": float(charm_prob[index, 3]),
            "evis_true": float(evis_true[index]),
            "evis_reco": float(evis_reco[index]),
            "ptmiss_true": float(ptmiss_true[index]),
            "ptmiss_reco": float(ptmiss_reco[index]),
        }
        for prefix, truth, prediction in (
            ("vis", vis_true, vis_reco),
            ("out_lepton", lepton_true, lepton_reco),
            ("jet", jet_true, jet_reco),
            ("primary_vertex", vertex_true, vertex_reco),
        ):
            for component, axis in enumerate(("x", "y", "z")):
                row[f"{prefix}_p{axis}_true" if prefix != "primary_vertex" else f"{prefix}_{axis}_true"] = float(
                    truth[index, component]
                )
                row[f"{prefix}_p{axis}_reco" if prefix != "primary_vertex" else f"{prefix}_{axis}_reco"] = float(
                    prediction[index, component]
                )
        rows.append({column: row[column] for column in CSV_COLUMNS})
    return rows


def generate_csv(
    config_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    batch_size: int | None,
    num_workers: int | None,
    device: torch.device,
    use_ema: bool,
    max_batches: int | None,
) -> int:
    config, run_dir = load_recorded_config(config_path)
    if batch_size is not None:
        config["training"]["batch_size"] = int(batch_size)
    if num_workers is not None:
        config["data"]["num_workers"] = int(num_workers)
    seed_everything(int(config["experiment"]["seed"]))

    report = (
        require_flash_runtime(config["training"]["precision"])
        if config["model"]["attention_backend"] == "flash"
        else runtime_report()
    )
    metadata = load_metadata(pipeline_path(config["data"]["metadata_path"]))
    manifest_dir = pipeline_path(config["data"]["manifests_dir"])
    dataset = ProjectionDataset(
        manifest_dir / f"{split}.txt",
        metadata,
        config["data"],
        training=False,
        seed=int(config["experiment"]["seed"]),
    )
    dataset.files.sort(key=event_sort_key)
    loader = make_loader(dataset, config, training=False)
    checkpoint_dir = run_dir / "checkpoints"

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Input mode: {config['data']['input_mode']}")
    print(f"Split: {split} ({len(dataset)} events)")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Attention backend: {report.get('attention_backend', 'flash_attn_varlen')}")
    models = load_task_models(
        checkpoint_dir,
        config,
        metadata,
        device,
        use_ema=use_ema,
    )
    active_backends = {model.attention_backend for model in models.values()}
    if len(active_backends) != 1:
        raise RuntimeError(f"Task models use different attention backends: {active_backends}")
    print(f"Active model backend: {next(iter(active_backends))}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp-{os.getpid()}")
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
    row_count = 0

    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            progress = tqdm(
                enumerate(loader),
                total=total_batches,
                ascii=True,
                desc=config["experiment"]["name"],
            )
            with torch.inference_mode():
                for batch_index, batch in progress:
                    if max_batches is not None and batch_index >= max_batches:
                        break
                    batch = batch.to(device)
                    with autocast_context(device, config["training"]["precision"]):
                        task_outputs = {
                            task: model(batch) for task, model in models.items()
                        }
                        outputs = combine_outputs(task_outputs)
                    rows = prediction_rows(batch, outputs, metadata)
                    writer.writerows(rows)
                    row_count += len(rows)
                    progress.set_postfix(rows=row_count)
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Wrote {row_count} rows -> {output_path}")
    del models, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate legacy-compatible result CSVs from the 2D-view models"
    )
    parser.add_argument("--mode", choices=("two", "three", "both"), default="both")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--two-config",
        default=str(PIPELINE_ROOT / "configs" / "two_view.yaml"),
    )
    parser.add_argument(
        "--three-config",
        default=str(PIPELINE_ROOT / "configs" / "three_view.yaml"),
    )
    parser.add_argument(
        "--two-output",
        default=str(PIPELINE_ROOT / "results" / "results_v8.0_xz_yz.csv"),
    )
    parser.add_argument(
        "--three-output",
        default=str(PIPELINE_ROOT / "results" / "results_v8.0_xz_yz_xy.csv"),
    )
    parser.add_argument("--two-batch-size", type=int)
    parser.add_argument("--three-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Run on a GPU node and set "
            "CUDA_VISIBLE_DEVICES, or pass --device cpu only with a torch_sdpa config."
        )

    if args.mode in {"two", "both"}:
        generate_csv(
            args.two_config,
            args.two_output,
            split=args.split,
            batch_size=args.two_batch_size,
            num_workers=args.num_workers,
            device=device,
            use_ema=args.use_ema,
            max_batches=args.max_batches,
        )
    if args.mode in {"three", "both"}:
        generate_csv(
            args.three_config,
            args.three_output,
            split=args.split,
            batch_size=args.three_batch_size,
            num_workers=args.num_workers,
            device=device,
            use_ema=args.use_ema,
            max_batches=args.max_batches,
        )


if __name__ == "__main__":
    main()
