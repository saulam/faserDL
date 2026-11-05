"""PyTorch Lightning modules for the spconv-free flash-attention JEPA.

FlashJEPAPreTrainer mirrors JEPAPreTrainer (optimiser, schedule, EMA timing);
FlashViTFineTuner reuses ViTFineTuner unchanged (losses, Kendall weighting,
layer-wise LR decay, EMA, schedule) and only adapts the batch format.
"""

import math
from contextlib import contextmanager

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from utils import CombinedScheduler, CustomLambdaLR, arrange_truth
from .flash_vit import TOKEN_KIND_NAMES, EventBatch
from .lightning_model_finetune import ViTFineTuner


def to_device_async(obj, device):
    """move_obj with asynchronous host-to-device copies, so the host queues the
    step's first kernels while the batch is copied.  Safe for the loaders'
    pinned batches: the pinned-memory allocator keeps a buffer until its copy
    has finished.  (Unpinned tensors are copied synchronously, as before.)"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_device_async(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device_async(v, device) for v in obj)
    return obj


def param_groups_weight_decay(model, weight_decay, no_weight_decay_list=()):
    """Same grouping as timm.optim.param_groups_weight_decay (no decay for 1-D/bias/listed)."""
    no_weight_decay_list = set(no_weight_decay_list)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


class DeviceEpochLogging:
    """Cheaper training-step logging for the flash Lightning modules, with the
    same logged values.

    Each self.log call costs ~1 ms of host time, and an on_step value logged
    with sync_dist is all-reduced at every step.  Lightning writes step-level
    metrics only every log_every_n_steps steps, so they are logged only on those
    steps (_writes_step_metrics).  The epoch means are accumulated on the device
    (_accumulate_epoch) and logged once per epoch (_log_epoch_means): per rank
    weighted by batch size, then averaged over ranks, as Lightning's on_epoch
    logging does.
    """

    def _reset_epoch_means(self):
        self._epoch_names = self._epoch_sum = self._epoch_weight = None

    def _writes_step_metrics(self):
        """Whether Lightning writes this training step's step-level metrics to the
        loggers.  Its decision depends on a step counter that only advances
        after the step, so it is already known during the step."""
        trainer = self.trainer
        connector = getattr(trainer, "_logger_connector", None)
        return bool(getattr(connector, "should_update_logs", True)) or bool(trainer.fast_dev_run)

    def _accumulate_epoch(self, values, batch_size, present=None):
        """values: {epoch-level name: 0-dim tensor}, the same names every step;
        present: {name: 0-dim bool tensor} for values that count only in some
        batches (as values Lightning was not given in the others)."""
        names = list(values)
        if getattr(self, "_epoch_names", None) is None:
            self._epoch_names = names
            self._epoch_sum = next(iter(values.values())).new_zeros(len(names), dtype=torch.float32)
            self._epoch_weight = torch.zeros_like(self._epoch_sum)
        elif self._epoch_names != names:
            raise RuntimeError(f"Training metrics changed within an epoch: {self._epoch_names} -> {names}")
        present = present or {}
        one = self._epoch_sum.new_ones(())
        weight = torch.stack([present[name].float() if name in present else one for name in names]) * batch_size
        self._epoch_sum += torch.stack([value.detach().float().reshape(()) for value in values.values()]) * weight
        self._epoch_weight += weight

    def _log_epoch_means(self):
        if getattr(self, "_epoch_names", None) is None:
            return
        # Every name is logged on every rank, so the cross-rank reductions match
        # (a value that counted in no batch gives 0).
        means = self._epoch_sum / self._epoch_weight.clamp_min(1e-12)
        for name, mean in zip(self._epoch_names, means.unbind()):
            self.log(name, mean, on_step=False, on_epoch=True, sync_dist=True)
        self._reset_epoch_means()


class FlashJEPAPreTrainer(DeviceEpochLogging, pl.LightningModule):
    """Train the context encoder/predictor and update the target encoder by EMA."""

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.warmup_epochs = int(args.warmup_epochs)
        self.cosine_annealing_epochs = int(args.cosine_annealing_epochs)
        self.lr = args.lr
        self.blr = args.blr
        self._batch_size = int(args.batch_size)
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = float(args.weight_decay)
        self.eps = float(args.eps)
        self.ema_decay = float(args.jepa_ema_decay)
        self.ema_end_decay = float(args.jepa_ema_end_decay)
        if not 0.0 <= self.ema_decay <= self.ema_end_decay <= 1.0:
            raise ValueError(
                "Expected 0 <= jepa_ema_decay <= jepa_ema_end_decay <= 1, "
                f"got {self.ema_decay} and {self.ema_end_decay}."
            )
        end = getattr(args, "flash_weight_decay_end", None)
        self.weight_decay_end = None if end is None else float(end)
        self.lr_scale_on_resume = float(getattr(args, "flash_lr_scale_on_resume", 1.0))
        if self.lr_scale_on_resume <= 0:
            raise ValueError(f"flash_lr_scale_on_resume must be positive, got {self.lr_scale_on_resume}")
        self.log_grad_norm = bool(getattr(args, "flash_log_grad_norm", False))

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return to_device_async(batch, device)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["flash_encoder_config"] = self.model.encoder_config

    def on_train_start(self):
        """Keep the wrapped optimiser groups accessible on supported Lightning versions."""
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups
        if self.lr_scale_on_resume != 1.0:
            if not self.trainer.ckpt_path:
                raise ValueError("flash_lr_scale_on_resume needs --resume_checkpoint; set --blr for a new run.")
            self._scale_lr_schedule(self.lr_scale_on_resume)

    def _scale_lr_schedule(self, scale):
        """Multiply the restored learning rates and the schedule they follow
        (warm-up and cosine base rates, cosine floor) by ``scale``: the run
        continues on ``scale`` times the original schedule."""
        for group in self.optimizers().param_groups:
            group["lr"] *= scale
            if "initial_lr" in group:
                group["initial_lr"] *= scale
        scheduler = self.lr_schedulers()
        for inner in (getattr(scheduler, "scheduler1", None), getattr(scheduler, "scheduler2", None)):
            if inner is None:
                continue
            inner.base_lrs = [lr * scale for lr in inner.base_lrs]
            if hasattr(inner, "eta_min"):
                inner.eta_min *= scale
            if hasattr(inner, "_last_lr"):
                inner._last_lr = [lr * scale for lr in inner._last_lr]
        if self.trainer.is_global_zero:
            print(f"lr schedule scaled by {scale} on resume: lr = {self.optimizers().param_groups[0]['lr']:.3e}")

    def on_train_epoch_start(self):
        raw = getattr(self.trainer, "train_dataloader", None)
        if raw is None:
            raw = getattr(self.trainer, "train_dataloaders", None)
        loaders = raw if isinstance(raw, (list, tuple)) else [raw]
        for loader in loaders:
            dataset = getattr(loader, "dataset", None)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(self.trainer.current_epoch)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._reset_epoch_means()

    def on_train_epoch_end(self):
        self._log_epoch_means()

    @contextmanager
    def _fixed_val_rng(self, batch_idx: int, val_mask_seed: int = 42):
        rank = int(getattr(self, "global_rank", 0) or 0)
        seed = int(val_mask_seed + batch_idx + 1_000_000 * rank)
        if self.device.type == "cuda":
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            with torch.random.fork_rng(devices=[device_index], enabled=True):
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.default_generators[device_index].manual_seed(seed)
                yield
        else:
            with torch.random.fork_rng(devices=[], enabled=True):
                torch.random.default_generator.manual_seed(seed)
                yield

    def forward(self, batch):
        return self.model(batch)

    def _ema_decay_for_step(self) -> float:
        """Cosine schedule from the configured initial to final target decay."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return self.ema_decay
        total_steps = max(1, int(trainer.estimated_stepping_batches) - 1)
        progress = min(1.0, max(0.0, float(trainer.global_step) / total_steps))
        cosine_progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.ema_decay + (self.ema_end_decay - self.ema_decay) * cosine_progress

    def _weight_decay_for_step(self) -> float:
        """I-JEPA's cosine schedule (CosineWDSchedule) from weight_decay to weight_decay_end."""
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        progress = min(1.0, max(0.0, float(self.trainer.global_step) / total_steps))
        return self.weight_decay_end + (self.weight_decay - self.weight_decay_end) * 0.5 * (
            1.0 + math.cos(math.pi * progress))

    def on_before_optimizer_step(self, optimizer):
        """Gradient norms before clipping (Lightning clips after this hook), on the
        steps whose step-level metrics are written."""
        if not self.log_grad_norm or not self._writes_step_metrics():
            return
        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if name.startswith("predictor."):
                key = "predictor"
            elif name.startswith("context_encoder.blocks."):
                key = f"block{int(name.split('.')[2]) + 1:02d}"
            else:
                key = "encoder_other"  # embeddings and the final norm
            grads.setdefault(key, []).append(param.grad.detach())
        squares = {key: torch.stack(torch._foreach_norm(values)).float().square().sum()
                   for key, values in grads.items()}
        if not squares:
            return
        # DDP has already averaged the gradients: identical on every rank.
        self.log("grad_norm/total", torch.stack(list(squares.values())).sum().sqrt(),
                 on_step=True, on_epoch=False, sync_dist=False)
        for key, value in squares.items():
            self.log(f"grad_norm/{key}", value.sqrt(), on_step=True, on_epoch=False, sync_dist=False)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        if self.weight_decay_end is not None:
            # Set before every step, so a resumed run follows the schedule too.
            groups = getattr(optimizer, "_optimizer", optimizer).param_groups
            if len(groups) != 2:
                raise RuntimeError(f"Expected the no-decay and decay groups, got {len(groups)} groups.")
            groups[1]["weight_decay"] = decay = self._weight_decay_for_step()
            if self._writes_step_metrics():
                self.log("weight_decay", decay, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
        # As in JEPAPreTrainer: the closure (forward/backward) runs inside
        # optimizer.step(), so the EMA update here follows each optimizer step
        # exactly once, with global_step still indexing the step just taken.
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        decay = self._ema_decay_for_step()
        self.model.update_target_encoder(decay)
        # Identical on every rank: no reduction needed.
        if self._writes_step_metrics():
            self.log("jepa/ema_decay", decay, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

    @staticmethod
    def _event_std(tokens, seq):
        """Mean over features of the std across events of mean-pooled token features.

        Tends to zero when every event maps to the same representation.
        """
        pooled = seq.mean(F.layer_norm(tokens.float(), (tokens.size(-1),)))
        if pooled.size(0) < 2:
            return pooled.new_zeros(())
        return pooled.std(dim=0, correction=0).mean()

    def common_step(self, batch):
        output = self.forward(batch)
        prediction, target = output["prediction"], output["target"]
        with torch.no_grad():
            if prediction.numel():
                cosine = F.cosine_similarity(prediction, target, dim=-1).mean()
                prediction_std = prediction.std(dim=0, correction=0).mean()
                target_std = target.std(dim=0, correction=0).mean()
            else:
                cosine = prediction_std = target_std = output["loss"].detach().new_zeros(())
            batch_size = output["token_seq"].batch_size
            target_mask, kind = output["target_mask"], output["token_kind"]
            metrics = {
                "jepa/cosine": cosine,
                "jepa/prediction_std": prediction_std,
                "jepa/target_std": target_std,
                "jepa/context_event_std": self._event_std(output["context_tokens"], output["context_seq"]),
                "jepa/target_event_std": self._event_std(output["target_tokens"], output["token_seq"]),
                "jepa/targets_per_event": target_mask.sum() / batch_size,
                "jepa/context_per_event": (~target_mask).sum() / batch_size,
            }
            for index, name in enumerate(TOKEN_KIND_NAMES):
                selected = kind == index
                metrics[f"mask/{name}_fraction"] = (
                    (target_mask & selected).sum() / selected.sum().clamp_min(1))
            # A detector type's loss counts only in batches with targets of that type.
            present = {}
            for kind, (name, value) in enumerate(output["group_losses"].items()):
                metrics[f"jepa/loss_{name}"] = value
                present[f"jepa/loss_{name}"] = output["group_counts"][kind] > 0
        return output["loss"], metrics, batch_size, present

    def _log_step(self, loss, metrics, batch_size, split, present):
        values = {"loss_total": loss.detach(), **metrics}
        if split == "val":
            for name, value in values.items():
                if name in present and not bool(present[name]):   # (synchronises; validation only)
                    continue
                self.log(f"{name}/val", value, batch_size=batch_size, on_step=False, on_epoch=True,
                         prog_bar=(name == "loss_total"), sync_dist=True)
            return
        # Training.  Per-step values come from each rank's own batch (with
        # sync_dist, Lightning all-reduces every step-level metric at every
        # step, which stalls multi-node jobs); see DeviceEpochLogging for when
        # they and the `/train_epoch` means are logged.
        self._accumulate_epoch({f"{name}/train_epoch": value for name, value in values.items()}, batch_size,
                               {f"{name}/train_epoch": flag for name, flag in present.items()})
        write = self._writes_step_metrics()
        for name, value in values.items():
            if name != "loss_total" and not write:
                continue
            if write and name in present and not bool(present[name]):
                continue
            self.log(f"{name}/train_step", value, batch_size=batch_size, on_step=True, on_epoch=False,
                     prog_bar=(name == "loss_total"), sync_dist=False)

    def training_step(self, batch, batch_idx):
        loss, metrics, batch_size, present = self.common_step(batch)
        self._log_step(loss, metrics, batch_size, "train", present)
        return loss

    def validation_step(self, batch, batch_idx):
        with self._fixed_val_rng(batch_idx):
            loss, metrics, batch_size, present = self.common_step(batch)
        self._log_step(loss, metrics, batch_size, "val", present)
        return loss

    def configure_optimizers(self):
        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = total_steps // self.trainer.max_epochs
        effective_batch_size = (
            self._batch_size * self.trainer.world_size * self.trainer.accumulate_grad_batches
        )
        if self.blr is not None:
            self.lr = self.blr * effective_batch_size / 256.0

        warmup_steps = steps_per_epoch * self.warmup_epochs
        cosine_annealing_steps = steps_per_epoch * self.cosine_annealing_epochs
        start_cosine_step = total_steps - cosine_annealing_steps
        if self.trainer.is_global_zero:
            print(f"lr                = {self.lr}")
            print(f"eff. batch size   = {effective_batch_size}")
            print(f"total_steps       = {total_steps}")
            print(f"steps_per_epoch   = {steps_per_epoch}")
            print(f"warmup_steps      = {warmup_steps}")
            print(f"scheduler_steps   = {cosine_annealing_steps}")
            print(f"start_cosine_step = {start_cosine_step}")
            print(f"jepa_ema_decay    = {self.ema_decay} -> {self.ema_end_decay}")
            print(f"weight_decay      = {self.weight_decay}"
                  + ("" if self.weight_decay_end is None else f" -> {self.weight_decay_end} (cosine)"))

        param_groups = param_groups_weight_decay(
            self.model, self.weight_decay, no_weight_decay_list=self.model.no_weight_decay(),
        )
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=self.betas, eps=self.eps)
        if warmup_steps == 0 and cosine_annealing_steps == 0:
            return optimizer

        warmup_scheduler = CustomLambdaLR(optimizer, warmup_steps) if warmup_steps > 0 else None
        cosine_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=cosine_annealing_steps, eta_min=self.lr * 1e-2,
            )
            if cosine_annealing_steps > 0 else None
        )
        combined_scheduler = CombinedScheduler(
            optimizer=optimizer, scheduler1=warmup_scheduler, scheduler2=cosine_scheduler,
            warmup_steps=warmup_steps, start_cosine_step=start_cosine_step,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": combined_scheduler, "interval": "step"},
        }


class FlashViTFineTuner(DeviceEpochLogging, ViTFineTuner):
    """ViTFineTuner for FlashViT: identical training, packed-token batches, and the
    same logged values with fewer host synchronisations (training_step)."""

    part_losses_as_tensors = True   # no .item() per loss and step

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return to_device_async(batch, device)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._reset_epoch_means()

    def on_train_epoch_end(self):
        self._log_epoch_means()

    def training_step(self, batch, batch_idx):
        """ViTFineTuner.training_step's metrics and keys.  It logs every loss on
        step and on epoch with sync_dist, an all-reduce per loss at every step;
        here the step-level values (still reduced over ranks) are logged on the
        steps Lightning writes, plus the progress-bar loss every step."""
        loss, part_losses, kendall_w, _, batch_size = self.common_step(batch)
        step = {"loss_total/train": loss.detach(), **{f"{key}/train": value for key, value in part_losses.items()}}
        epoch = {f"{name}_epoch": value for name, value in step.items()}
        for key, w in kendall_w.items():
            epoch[f"uncertainty/{key}"] = w
            epoch[f"uncertainty_sigma/{key}"] = (1.0 / torch.sqrt(w.clamp_min(1e-12))).detach()
        self._accumulate_epoch(epoch, batch_size)
        write = self._writes_step_metrics()
        for name, value in step.items():
            if name == "loss_total/train" or write:
                self.log(f"{name}_step", value, batch_size=batch_size, on_step=True, on_epoch=False,
                         prog_bar=(name == "loss_total/train"), sync_dist=True)
        return loss

    def _arrange_batch(self, batch):
        return EventBatch(batch), arrange_truth(batch)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["flash_encoder_config"] = dict(self.model.encoder_config)
