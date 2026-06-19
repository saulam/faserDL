from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .checkpoint import (
    ExponentialMovingAverage,
    atomic_torch_save,
    restore_rng_state,
    rng_state,
)
from .config import load_config, pipeline_path
from .dataset import ProjectionDataset, collate_events, load_metadata
from .losses import MultiTaskObjective
from .model import ProjectionTransformer, parameter_counts
from .runtime import require_flash_runtime, runtime_report


MONITORS = {
    "total": "loss_total",
    "flavour": "loss_cls/flavour",
    "charm": "loss_cls/charm",
    "vis": "loss_vis/geom",
    "jet": "loss_jet/geom",
    "lepton": "loss_lep/geom",
    "vertex": "loss_vertex",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: ProjectionDataset,
    config: dict[str, Any],
    *,
    training: bool,
) -> DataLoader:
    workers = int(config["data"]["num_workers"])
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=training,
        num_workers=workers,
        pin_memory=bool(config["data"].get("pin_memory", True)),
        persistent_workers=workers > 0,
        drop_last=training,
        collate_fn=collate_events,
        generator=generator,
    )


def optimizer_groups(
    model: ProjectionTransformer,
    objective: MultiTaskObjective,
    weight_decay: float,
) -> list[dict[str, Any]]:
    no_decay_names = model.no_weight_decay()
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name in no_decay_names:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
        {
            "params": objective.parameters(),
            "weight_decay": 0.0,
            "lr_scale": 0.1,
        },
    ]


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_epoch(
    *,
    model: ProjectionTransformer,
    objective: MultiTaskObjective,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    training: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    scaler: torch.amp.GradScaler | None = None,
    accumulation: int = 1,
    gradient_clip: float = 1.0,
    ema: ExponentialMovingAverage | None = None,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    model.train(training)
    objective.train(training)
    totals: dict[str, float] = {}
    events = 0
    tokens = 0
    started = time.perf_counter()
    if training:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
    batch_limit = len(loader) if max_batches is None else min(len(loader), max_batches)

    for batch_index, batch in enumerate(loader):
        if batch_index >= batch_limit:
            break
        batch = batch.to(device)
        with torch.set_grad_enabled(training), autocast_context(device, precision):
            outputs = model(batch)
            loss, metrics = objective(outputs, batch.targets)
            scaled_loss = loss / accumulation
        if training:
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_limit
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(
                    list(model.parameters()) + list(objective.parameters()), gradient_clip
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

        batch_size = batch.batch_size
        events += batch_size
        tokens += batch.num_tokens
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size

    elapsed = max(time.perf_counter() - started, 1e-9)
    averaged = {name: total / max(events, 1) for name, total in totals.items()}
    throughput = {
        "events_per_second": events / elapsed,
        "tokens_per_second": tokens / elapsed,
        "events": events,
        "tokens": tokens,
        "seconds": elapsed,
    }
    return averaged, throughput


def checkpoint_payload(
    *,
    config: dict[str, Any],
    model: ProjectionTransformer,
    objective: MultiTaskObjective,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler | None,
    ema: ExponentialMovingAverage,
    epoch: int,
    best: dict[str, float],
) -> dict[str, Any]:
    return {
        "config": config,
        "model": model.state_dict(),
        "objective": objective.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict(),
        "epoch": epoch,
        "best": best,
        "rng": rng_state(),
    }


def append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 2D projection transformer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)

    backend = config["model"]["attention_backend"]
    precision = config["training"]["precision"]
    report = require_flash_runtime(precision) if backend == "flash" else runtime_report()
    device_name = config["training"].get("device", "cuda")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda but CUDA is unavailable")
    torch.set_float32_matmul_precision("high")

    metadata_path = pipeline_path(config["data"]["metadata_path"])
    manifest_dir = pipeline_path(config["data"]["manifests_dir"])
    output_dir = pipeline_path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(metadata_path)
    if int(metadata["patch_size"]) != int(config["data"]["patch_size"]):
        raise ValueError("Config patch size does not match metadata")
    if bool(metadata.get("remove_primary_origin_pixel", False)) != bool(
        config["data"].get("remove_primary_origin_pixel", True)
    ):
        raise ValueError(
            "Config origin-pixel policy does not match metadata. Rebuild metadata "
            "after changing data.remove_primary_origin_pixel."
        )

    train_set = ProjectionDataset(
        manifest_dir / "train.txt", metadata, config["data"], training=True, seed=seed
    )
    val_set = ProjectionDataset(
        manifest_dir / "val.txt", metadata, config["data"], training=False, seed=seed
    )
    train_loader = make_loader(train_set, config, training=True)
    val_loader = make_loader(val_set, config, training=False)

    model = ProjectionTransformer(config["model"], metadata).to(device)
    objective = MultiTaskObjective(
        metadata,
        label_smoothing=float(config["training"]["label_smoothing"]),
        kendall_weight_min=float(config["training"]["kendall_weight_min"]),
        kendall_weight_max=float(config["training"]["kendall_weight_max"]),
    ).to(device)
    counts = parameter_counts(model)

    accumulation = int(config["training"]["accumulation_steps"])
    effective_batch = int(config["training"]["batch_size"]) * accumulation
    learning_rate = (
        float(config["training"]["base_learning_rate"]) * effective_batch / 256.0
    )
    groups = optimizer_groups(
        model, objective, weight_decay=float(config["training"]["weight_decay"])
    )
    for group in groups:
        group["lr"] = learning_rate * float(group.pop("lr_scale", 1.0))
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(config["training"]["betas"]),
        eps=float(config["training"]["epsilon"]),
    )
    epochs = int(config["training"]["epochs"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * int(config["training"]["warmup_epochs"])
    scheduler = make_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_ratio=float(config["training"]["min_lr_ratio"]),
    )
    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_scaler else None
    ema = ExponentialMovingAverage(model, float(config["training"]["ema_decay"]))

    start_epoch = 0
    best = {name: float("inf") for name in MONITORS}
    resume = args.resume or config["training"].get("resume")
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        objective.load_state_dict(checkpoint["objective"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        ema.load_state_dict(checkpoint["ema"])
        best = checkpoint["best"]
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng"])

    run_summary = {
        "attention_backend": model.attention_backend,
        "runtime": report,
        "parameters": counts,
        "effective_batch_size": effective_batch,
        "learning_rate": learning_rate,
        "steps_per_epoch": steps_per_epoch,
        "config": config,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, default=str)
    print(json.dumps({key: value for key, value in run_summary.items() if key != "config"}, indent=2))

    patience = int(config["training"]["early_stopping_patience"])
    stale_epochs = 0
    max_train_batches = config["training"].get("max_train_batches")
    max_val_batches = config["training"].get("max_val_batches")
    for epoch in range(start_epoch, epochs):
        train_set.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics, train_rate = run_epoch(
            model=model,
            objective=objective,
            loader=train_loader,
            device=device,
            precision=precision,
            training=True,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            accumulation=accumulation,
            gradient_clip=float(config["training"]["gradient_clip"]),
            ema=ema,
            max_batches=int(max_train_batches) if max_train_batches is not None else None,
        )
        val_metrics, val_rate = run_epoch(
            model=model,
            objective=objective,
            loader=val_loader,
            device=device,
            precision=precision,
            training=False,
            max_batches=int(max_val_batches) if max_val_batches is not None else None,
        )
        memory = (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "gpu_peak_gib": memory,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_metrics.items()},
            **{f"train_throughput/{key}": value for key, value in train_rate.items()},
            **{f"val_throughput/{key}": value for key, value in val_rate.items()},
        }
        append_log(output_dir / "metrics.csv", row)
        improved_total = False
        improved_names = []
        for name, metric_key in MONITORS.items():
            value = val_metrics[metric_key]
            if value < best[name]:
                best[name] = value
                improved_names.append(name)
                if name == "total":
                    improved_total = True
        payload = checkpoint_payload(
            config=config,
            model=model,
            objective=objective,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            epoch=epoch,
            best=best,
        )
        atomic_torch_save(payload, output_dir / "checkpoints" / "last.pt")
        for name in improved_names:
            atomic_torch_save(payload, output_dir / "checkpoints" / f"best_{name}.pt")
        stale_epochs = 0 if improved_total else stale_epochs + 1
        print(
            f"epoch={epoch} train_loss={train_metrics['loss_total']:.6f} "
            f"val_loss={val_metrics['loss_total']:.6f} "
            f"events/s={train_rate['events_per_second']:.1f} "
            f"tokens/s={train_rate['tokens_per_second']:.0f} peak_gib={memory:.2f}"
        )
        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping after {stale_epochs} epochs without total-loss improvement")
            break


if __name__ == "__main__":
    main()
