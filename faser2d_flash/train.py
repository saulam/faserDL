from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from .checkpoint import ExponentialMovingAverage
from .config import load_config, pipeline_path
from .dataset import ProjectionDataset, collate_events, load_metadata
from .losses import MultiTaskObjective
from .model import ProjectionTransformer, parameter_counts
from .runtime import require_flash_runtime, runtime_report


MONITORS = {
    "total": "val/loss_total",
    "flavour": "val/loss_cls/flavour",
    "charm": "val/loss_cls/charm",
    "vis": "val/loss_vis/geom",
    "jet": "val/loss_jet/geom",
    "lepton": "val/loss_lep/geom",
    "vertex": "val/loss_vertex",
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
    require_all_valid: bool = False,
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
        collate_fn=partial(collate_events, require_all_valid=require_all_valid),
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


class ProjectionLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: ProjectionTransformer,
        objective: MultiTaskObjective,
        config: dict[str, Any],
    ):
        super().__init__()
        self.model = model
        self.objective = objective
        self.config = config
        self.ema: ExponentialMovingAverage | None = None
        self._loaded_ema_state: dict[str, Any] | None = None

    def forward(self, batch):
        return self.model(batch)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def _shared_step(self, batch, prefix: str):
        outputs = self.model(batch)
        loss, metrics = self.objective(outputs, batch.targets)
        batch_size = batch.batch_size
        for name, value in metrics.items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name == "loss_total",
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        if prefix == "train":
            self.log(
                "train/loss_total_step",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_fit_start(self) -> None:
        if self.config["model"]["attention_backend"] == "flash":
            report = require_flash_runtime(self.config["training"]["precision"])
        else:
            report = runtime_report()
        if self.trainer.is_global_zero:
            self.ema = ExponentialMovingAverage(
                self.model, float(self.config["training"]["ema_decay"])
            )
            if self._loaded_ema_state is not None:
                self.ema.load_state_dict(self._loaded_ema_state)
            print(json.dumps({"runtime": report}, indent=2))

    def on_train_epoch_start(self) -> None:
        dataset = getattr(self.trainer.train_dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        if self.trainer.is_global_zero and self.ema is not None:
            self.ema.update(self.model)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()
        checkpoint["pipeline_config"] = self.config
        checkpoint["attention_backend"] = self.model.attention_backend

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self._loaded_ema_state = checkpoint.get("ema")

    def configure_optimizers(self):
        training = self.config["training"]
        effective_batch = (
            int(training["batch_size"])
            * int(training["accumulation_steps"])
            * self.trainer.world_size
        )
        learning_rate = (
            float(training["base_learning_rate"]) * effective_batch / 256.0
        )
        groups = optimizer_groups(
            self.model,
            self.objective,
            weight_decay=float(training["weight_decay"]),
        )
        for group in groups:
            group["lr"] = learning_rate * float(group.pop("lr_scale", 1.0))
        optimizer = torch.optim.AdamW(
            groups,
            betas=tuple(training["betas"]),
            eps=float(training["epsilon"]),
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = math.ceil(total_steps / int(training["epochs"]))
        warmup_steps = steps_per_epoch * int(training["warmup_epochs"])
        scheduler = make_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_ratio=float(training["min_lr_ratio"]),
        )
        if self.trainer.is_global_zero:
            print(
                json.dumps(
                    {
                        "world_size": self.trainer.world_size,
                        "per_device_batch_size": int(training["batch_size"]),
                        "accumulation_steps": int(training["accumulation_steps"]),
                        "effective_batch_size": effective_batch,
                        "learning_rate": learning_rate,
                        "total_steps": total_steps,
                        "steps_per_epoch": steps_per_epoch,
                        "warmup_steps": warmup_steps,
                    },
                    indent=2,
                )
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def _is_global_zero() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _device_count(devices: int | list[int]) -> int:
    return len(devices) if isinstance(devices, list) else int(devices)


def checkpoint_callbacks(output_dir: Path) -> list[ModelCheckpoint]:
    callbacks = []
    for name, monitor in MONITORS.items():
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename=f"best_{name}",
                monitor=monitor,
                mode="min",
                save_top_k=1,
                save_last=name == "total",
                auto_insert_metric_name=False,
                save_weights_only=False,
                enable_version_counter=False,
            )
        )
    return callbacks


class CudaMemoryReport(Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_end(self, trainer, pl_module) -> None:
        if pl_module.device.type != "cuda":
            return
        allocated = torch.cuda.max_memory_allocated(pl_module.device) / 2**30
        reserved = torch.cuda.max_memory_reserved(pl_module.device) / 2**30
        print(
            f"CUDA_MEMORY rank={trainer.global_rank} "
            f"device={torch.cuda.get_device_name(pl_module.device)} "
            f"peak_allocated_gib={allocated:.3f} peak_reserved_gib={reserved:.3f}"
        )


class AsciiProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        progress = super().init_train_tqdm()
        progress.ascii = True
        return progress

    def init_validation_tqdm(self):
        progress = super().init_validation_tqdm()
        progress.ascii = True
        return progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 2D projection transformer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    seed = int(config["experiment"]["seed"])
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    metadata_path = pipeline_path(config["data"]["metadata_path"])
    manifest_dir = pipeline_path(config["data"]["manifests_dir"])
    output_dir = (
        pipeline_path(config["experiment"]["output_dir"]) / config["experiment"]["name"]
    )
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

    distributed_config = config.get("distributed", {})
    devices = distributed_config.get("devices", 1)
    num_nodes = int(distributed_config.get("num_nodes", 1))
    distributed = _device_count(devices) * num_nodes > 1
    train_set = ProjectionDataset(
        manifest_dir / "train.txt", metadata, config["data"], training=True, seed=seed
    )
    val_set = ProjectionDataset(
        manifest_dir / "val.txt", metadata, config["data"], training=False, seed=seed
    )
    train_loader = make_loader(
        train_set,
        config,
        training=True,
        require_all_valid=distributed,
    )
    val_loader = make_loader(
        val_set,
        config,
        training=False,
        require_all_valid=distributed,
    )

    model = ProjectionTransformer(config["model"], metadata)
    objective = MultiTaskObjective(
        metadata,
        label_smoothing=float(config["training"]["label_smoothing"]),
        kendall_weight_min=float(config["training"]["kendall_weight_min"]),
        kendall_weight_max=float(config["training"]["kendall_weight_max"]),
    )
    lightning_model = ProjectionLightningModule(model, objective, config)
    counts = parameter_counts(model)
    csv_logger = CSVLogger(save_dir=output_dir, name="logs")
    tensorboard_logger = TensorBoardLogger(save_dir=output_dir, name="tensorboard")
    enable_checkpointing = bool(
        config["training"].get("enable_checkpointing", True)
    )
    callbacks: list[Any] = [
        AsciiProgressBar(),
        CudaMemoryReport(),
        LearningRateMonitor(logging_interval="step"),
    ]
    if enable_checkpointing:
        callbacks.extend(checkpoint_callbacks(output_dir))
        callbacks.append(
            EarlyStopping(
                monitor=MONITORS["total"],
                mode="min",
                patience=int(config["training"]["early_stopping_patience"]),
            )
        )

    device_name = str(config["training"].get("device", "cuda"))
    accelerator = "gpu" if device_name.startswith("cuda") else "cpu"
    precision = config["training"]["precision"]
    lightning_precision = (
        "bf16-mixed"
        if accelerator == "gpu" and precision == "bf16"
        else "16-mixed"
        if accelerator == "gpu"
        else "32-true"
    )
    strategy = (
        DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=bool(
                distributed_config.get("gradient_as_bucket_view", True)
            ),
            static_graph=bool(distributed_config.get("static_graph", False)),
        )
        if distributed
        else "auto"
    )
    trainer_kwargs: dict[str, Any] = {
        "max_epochs": int(config["training"]["epochs"]),
        "accelerator": accelerator,
        "devices": devices if accelerator == "gpu" else 1,
        "num_nodes": num_nodes,
        "strategy": strategy,
        "precision": lightning_precision,
        "accumulate_grad_batches": int(config["training"]["accumulation_steps"]),
        "gradient_clip_val": float(config["training"]["gradient_clip"]),
        "gradient_clip_algorithm": "norm",
        "callbacks": callbacks,
        "logger": [csv_logger, tensorboard_logger],
        "enable_checkpointing": enable_checkpointing,
        "deterministic": False,
        "use_distributed_sampler": True,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 20,
    }
    if config["training"].get("max_train_batches") is not None:
        trainer_kwargs["limit_train_batches"] = int(
            config["training"]["max_train_batches"]
        )
    if config["training"].get("max_val_batches") is not None:
        trainer_kwargs["limit_val_batches"] = int(
            config["training"]["max_val_batches"]
        )

    if _is_global_zero():
        summary = {
            "parameters": counts,
            "devices": devices,
            "num_nodes": num_nodes,
            "ddp": distributed,
            "attention_backend": model.attention_backend,
            "pytorch_lightning": pl.__version__,
            "runtime": runtime_report(),
            "config": config,
        }
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)
        hyperparameters = {
            "parameters": counts["total"],
            "devices": devices,
            "num_nodes": num_nodes,
            "input_mode": config["data"]["input_mode"],
        }
        csv_logger.log_hyperparams(hyperparameters)
        tensorboard_logger.log_hyperparams(hyperparameters)

    trainer = pl.Trainer(**trainer_kwargs)
    resume = args.resume or config["training"].get("resume")
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume,
    )


if __name__ == "__main__":
    main()
