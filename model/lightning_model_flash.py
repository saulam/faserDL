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

from utils import CombinedScheduler, CustomLambdaLR, arrange_truth, move_obj
from .flash_vit import TOKEN_KIND_NAMES, EventBatch
from .lightning_model_finetune import ViTFineTuner


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


class FlashJEPAPreTrainer(pl.LightningModule):
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

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return move_obj(batch, device)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["flash_encoder_config"] = self.model.encoder_config

    def on_train_start(self):
        """Keep the wrapped optimiser groups accessible on supported Lightning versions."""
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # As in JEPAPreTrainer: the closure (forward/backward) runs inside
        # optimizer.step(), so the EMA update here follows each optimizer step
        # exactly once, with global_step still indexing the step just taken.
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        decay = self._ema_decay_for_step()
        self.model.update_target_encoder(decay)
        # Identical on every rank: no reduction needed.
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
            for name, value in output["group_losses"].items():
                metrics[f"jepa/loss_{name}"] = value
        return output["loss"], metrics, batch_size

    def _log_step(self, loss, metrics, batch_size, split):
        values = {"loss_total": loss.detach(), **metrics}
        if split == "val":
            for name, value in values.items():
                self.log(f"{name}/val", value, batch_size=batch_size, on_step=False, on_epoch=True,
                         prog_bar=(name == "loss_total"), sync_dist=True)
            return
        # Same keys as on_step+on_epoch logging, but the per-step values come
        # from each rank's own batch: with sync_dist, Lightning all-reduces
        # every step-level metric at every step, which stalls multi-node jobs.
        # The epoch means are still reduced across ranks (once per epoch).
        for name, value in values.items():
            self.log(f"{name}/train_step", value, batch_size=batch_size, on_step=True, on_epoch=False,
                     prog_bar=(name == "loss_total"), sync_dist=False)
            self.log(f"{name}/train_epoch", value, batch_size=batch_size, on_step=False, on_epoch=True,
                     sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss, metrics, batch_size = self.common_step(batch)
        self._log_step(loss, metrics, batch_size, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        with self._fixed_val_rng(batch_idx):
            loss, metrics, batch_size = self.common_step(batch)
        self._log_step(loss, metrics, batch_size, "val")
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


class FlashViTFineTuner(ViTFineTuner):
    """ViTFineTuner for FlashViT: identical training, packed-token batches."""

    def _arrange_batch(self, batch):
        return EventBatch(batch), arrange_truth(batch)

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["flash_encoder_config"] = dict(self.model.encoder_config)
