from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import ExponentialMovingAverage
from .config import load_config, pipeline_path
from .dataset import ProjectionDataset, collate_events, load_metadata
from .losses import MultiTaskObjective
from .metrics import EvaluationAccumulator
from .model import ProjectionTransformer
from .runtime import require_flash_runtime, runtime_report
from .train import autocast_context, make_loader, seed_everything


TASK_CHECKPOINTS = {
    "flavour": "best_flavour",
    "charm": "best_charm",
    "vis": "best_vis",
    "jet": "best_jet",
    "vertex": "best_vertex",
}


def resolve_checkpoint(checkpoint_dir: Path, stem: str) -> Path:
    for suffix in (".ckpt", ".pt"):
        path = checkpoint_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No checkpoint found for {stem!r} in {checkpoint_dir}; expected .ckpt or .pt"
    )


def component_state(
    checkpoint: dict[str, Any],
    component: str,
) -> dict[str, torch.Tensor]:
    if component in checkpoint:
        return checkpoint[component]
    prefix = f"{component}."
    state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise KeyError(f"Checkpoint contains no {component!r} state")
    return state


def load_model(
    path: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    device: torch.device,
    *,
    use_ema: bool,
) -> tuple[ProjectionTransformer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ProjectionTransformer(config["model"], metadata).to(device)
    model.load_state_dict(component_state(checkpoint, "model"))
    if use_ema:
        ema = ExponentialMovingAverage(model, float(config["training"]["ema_decay"]))
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    model.eval()
    return model, checkpoint


def combine_taskwise(
    outputs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "out_flavour": outputs["flavour"]["out_flavour"],
        "out_charm": outputs["charm"]["out_charm"],
        "out_vis": outputs["vis"]["out_vis"],
        "out_jet": outputs["jet"]["out_jet"],
        "out_vertex": outputs["vertex"]["out_vertex"],
    }


def prediction_rows(
    batch,
    outputs: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    flavour_prob = outputs["out_flavour"].softmax(-1).float().cpu().numpy()
    charm_prob = outputs["out_charm"].softmax(-1).float().cpu().numpy()
    vis_pred = outputs["out_vis"]["p_cart"].float().cpu().numpy()
    jet_pred = outputs["out_jet"]["p_cart"].float().cpu().numpy()
    lepton_pred = vis_pred - jet_pred
    vertex_stats = metadata["target_stats"]["primary_vertex"]
    vertex_pred = outputs["out_vertex"].float().cpu().numpy()
    vertex_pred = vertex_pred * np.asarray(vertex_stats["std"]) + np.asarray(
        vertex_stats["mean"]
    )
    vertex_true = batch.targets["primary_vertex"].float().cpu().numpy()
    vertex_true = vertex_true * np.asarray(vertex_stats["std"]) + np.asarray(
        vertex_stats["mean"]
    )
    vis_true = batch.targets["vis_sp_momentum"].float().cpu().numpy()
    jet_true = batch.targets["jet_momentum"].float().cpu().numpy()
    lepton_true = batch.targets["out_lepton_momentum"].float().cpu().numpy()
    rows = []
    for index in range(batch.batch_size):
        row = {
            "run_number": int(batch.run_numbers[index]),
            "event_id": int(batch.event_ids[index]),
            "flavour_label_true": int(batch.targets["flavour_label"][index].cpu()),
            "charm_label_true": int(batch.targets["charm_label"][index].cpu()),
        }
        row.update(
            {f"flavour_prob_{class_index}": float(flavour_prob[index, class_index]) for class_index in range(6)}
        )
        row.update(
            {f"charm_prob_{class_index}": float(charm_prob[index, class_index]) for class_index in range(4)}
        )
        for prefix, truth, prediction in (
            ("vis", vis_true, vis_pred),
            ("jet", jet_true, jet_pred),
            ("lepton", lepton_true, lepton_pred),
            ("vertex", vertex_true, vertex_pred),
        ):
            for component, name in enumerate(("x", "y", "z")):
                row[f"{prefix}_{name}_true"] = float(truth[index, component])
                row[f"{prefix}_{name}_pred"] = float(prediction[index, component])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a 2D projection experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--split", choices=("val", "test", "both"), default="both")
    parser.add_argument(
        "--selection", choices=("total", "last", "taskwise"), default="taskwise"
    )
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    seed_everything(int(config["experiment"]["seed"]))
    precision = config["training"]["precision"]
    report = (
        require_flash_runtime(precision)
        if config["model"]["attention_backend"] == "flash"
        else runtime_report()
    )
    device = torch.device(config["training"].get("device", "cuda"))
    metadata = load_metadata(pipeline_path(config["data"]["metadata_path"]))
    if bool(metadata.get("remove_primary_origin_pixel", False)) != bool(
        config["data"].get("remove_primary_origin_pixel", True)
    ):
        raise ValueError(
            "Config origin-pixel policy does not match metadata. Rebuild metadata "
            "after changing data.remove_primary_origin_pixel."
        )
    manifest_dir = pipeline_path(config["data"]["manifests_dir"])
    run_dir = (
        pipeline_path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
    )
    checkpoint_dir = run_dir / "checkpoints"

    if args.selection == "taskwise":
        models = {}
        for task, stem in TASK_CHECKPOINTS.items():
            models[task], _ = load_model(
                resolve_checkpoint(checkpoint_dir, stem),
                config,
                metadata,
                device,
                use_ema=args.use_ema,
            )
        total_checkpoint = torch.load(
            resolve_checkpoint(checkpoint_dir, "best_total"),
            map_location=device,
            weights_only=False,
        )
    else:
        stem = "best_total" if args.selection == "total" else "last"
        model, total_checkpoint = load_model(
            resolve_checkpoint(checkpoint_dir, stem),
            config,
            metadata,
            device,
            use_ema=args.use_ema,
        )
        models = {"total": model}

    objective = MultiTaskObjective(
        metadata,
        label_smoothing=float(config["training"]["label_smoothing"]),
        kendall_weight_min=float(config["training"]["kendall_weight_min"]),
        kendall_weight_max=float(config["training"]["kendall_weight_max"]),
    ).to(device)
    if total_checkpoint is not None:
        objective.load_state_dict(component_state(total_checkpoint, "objective"))
    objective.eval()

    splits = ("val", "test") if args.split == "both" else (args.split,)
    evaluation_dir = run_dir / "evaluation" / args.selection
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        dataset = ProjectionDataset(
            manifest_dir / f"{split}.txt",
            metadata,
            config["data"],
            training=False,
            seed=int(config["experiment"]["seed"]),
        )
        loader = make_loader(dataset, config, training=False)
        accumulator = EvaluationAccumulator(metadata)
        prediction_path = evaluation_dir / f"{split}_predictions.csv"
        prediction_handle = None
        writer = None
        if args.write_predictions:
            prediction_handle = prediction_path.open("w", newline="", encoding="utf-8")

        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                batch = batch.to(device)
                with autocast_context(device, precision):
                    if args.selection == "taskwise":
                        task_outputs = {
                            task: model(batch) for task, model in models.items()
                        }
                        outputs = combine_taskwise(task_outputs)
                    else:
                        outputs = models["total"](batch)
                    _, losses = objective(outputs, batch.targets)
                accumulator.update(outputs, batch.targets, losses)
                if prediction_handle is not None:
                    rows = prediction_rows(batch, outputs, metadata)
                    if writer is None:
                        writer = csv.DictWriter(prediction_handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                    writer.writerows(rows)
        if prediction_handle is not None:
            prediction_handle.close()
        metrics = accumulator.compute()
        metrics["selection"] = args.selection
        metrics["use_ema"] = args.use_ema
        metrics["attention_backend"] = next(iter(models.values())).attention_backend
        metrics["runtime"] = report
        output_path = evaluation_dir / f"{split}_metrics.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"{split}: {metrics['num_events']} events -> {output_path}")


if __name__ == "__main__":
    main()
