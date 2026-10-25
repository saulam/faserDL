"""PyTorch Lightning integration for JEPA-style sparse-detector pretraining."""

import math
from contextlib import contextmanager

import pytorch_lightning as pl
import timm.optim as optim_factory
import torch
from torch.nn import functional as F

from utils import CombinedScheduler, CustomLambdaLR, arrange_input, move_obj


class JEPAPreTrainer(pl.LightningModule):
    """Train the context encoder/predictor and update the target encoder by EMA."""

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.mask_ratio = float(args.mask_ratio)
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

    def forward(self, x, x_glob, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        return self.model(x, x_glob, mask_ratio=mask_ratio)

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
        # Lightning runs the closure (forward/backward) inside optimizer.step().
        # Updating the target here, after the step has returned, applies EMA
        # exactly once per optimizer step with the post-step context weights.
        # (on_before_zero_grad runs inside the closure before backward, i.e.
        # before the step it was meant to follow.)  global_step is incremented
        # only after this hook, so it still indexes the step just taken.  Every
        # DDP rank applies the same update to synchronized context parameters.
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        decay = self._ema_decay_for_step()
        self.model.update_target_encoder(decay)
        self.log(
            "jepa/ema_decay",
            decay,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

    @staticmethod
    def _cross_event_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean per-slot standard deviation across events of valid latent slots.

        Pooling all slots (as ``prediction_std``/``target_std`` do) mixes slot
        identity with event content and stays large for a collapsed encoder;
        this statistic tends to zero when every event maps to the same slot
        representation.
        """
        weights = mask.to(values.dtype).unsqueeze(-1)                     # [B, S, 1]
        counts = weights.sum(dim=0)                                       # [S, 1]
        mean = (values * weights).sum(dim=0) / counts.clamp_min(1.0)
        variance = ((values - mean).square() * weights).sum(dim=0) / counts.clamp_min(1.0)
        valid = counts.squeeze(-1) >= 2
        if not valid.any():
            return values.new_zeros(())
        return variance.sqrt().mean(dim=-1)[valid].mean()

    def common_step(self, batch):
        batch_input, *batch_input_global = arrange_input(batch)
        output = self.forward(batch_input, batch_input_global)
        selected_prediction = output["selected_prediction"]
        selected_target = output["selected_target"]

        with torch.no_grad():
            if selected_prediction.numel():
                cosine = F.cosine_similarity(
                    selected_prediction.float(), selected_target.float(), dim=-1
                ).mean()
                # Population standard deviation stays finite even when a very
                # sparse batch contributes only one target.
                prediction_std = selected_prediction.float().std(dim=0, correction=0).mean()
                target_std = selected_target.float().std(dim=0, correction=0).mean()
            else:
                zero = output["loss"].detach().new_zeros(())
                cosine = prediction_std = target_std = zero
            # Collapse monitors of the encoder latents (the downstream
            # representation), in the normalized space used by the loss.
            context_event_std = self._cross_event_std(
                self.model._normalize(output["context_latents"].detach().float()),
                output["context_latent_keep"],
            )
            target_event_std = self._cross_event_std(
                self.model._normalize(output["target_latents"].float()),
                output["target_latent_keep"],
            )

            fcal_real = output["fcal_patch_keep"].sum().clamp_min(1)
            ahcal_real = output["ahcal_patch_keep"].sum().clamp_min(1)
            targets_per_event = output["targets_per_event"]
            metrics = {
                "jepa/cosine": cosine,
                "jepa/prediction_std": prediction_std,
                "jepa/target_std": target_std,
                "jepa/context_event_std": context_event_std,
                "jepa/target_event_std": target_event_std,
                "jepa/targets_per_event": targets_per_event.mean(),
                "jepa/target_coverage": (
                    targets_per_event / output["target_candidates_per_event"].clamp_min(1.0)
                ).mean(),
                "mask/fcal_fraction": output["fcal_patch_mask"].sum() / fcal_real,
                "mask/ahcal_fraction": output["ahcal_patch_mask"].sum() / ahcal_real,
                "mask/ecal_fraction": output["ecal_drop"].float().mean(),
                "mask/muon_fraction": output["muon_drop"].float().mean(),
            }
            for name, value in output.get("group_losses", {}).items():
                metrics[f"jepa/loss_{name}"] = value
        return output["loss"], metrics, batch_input.batch_size

    def _log_step(self, loss, metrics, batch_size, split):
        self.log(
            f"loss_total/{split}",
            loss.detach(),
            batch_size=batch_size,
            on_step=(split == "train"),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        for name, value in metrics.items():
            self.log(
                f"{name}/{split}",
                value,
                batch_size=batch_size,
                on_step=(split == "train"),
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

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

        if self.blr is not None:
            effective_batch_size = (
                self._batch_size
                * self.trainer.world_size
                * self.trainer.accumulate_grad_batches
            )
            self.lr = self.blr * effective_batch_size / 256.0

        warmup_steps = steps_per_epoch * self.warmup_epochs
        cosine_annealing_steps = steps_per_epoch * self.cosine_annealing_epochs
        start_cosine_step = total_steps - cosine_annealing_steps

        if self.trainer.is_global_zero:
            effective_batch_size = (
                self._batch_size
                * self.trainer.world_size
                * self.trainer.accumulate_grad_batches
            )
            print(f"lr                = {self.lr}")
            print(f"eff. batch size   = {effective_batch_size}")
            print(f"total_steps       = {total_steps}")
            print(f"steps_per_epoch   = {steps_per_epoch}")
            print(f"warmup_steps      = {warmup_steps}")
            print(f"scheduler_steps   = {cosine_annealing_steps}")
            print(f"start_cosine_step = {start_cosine_step}")
            print(f"jepa_ema_decay    = {self.ema_decay} -> {self.ema_end_decay}")

        param_groups = optim_factory.param_groups_weight_decay(
            self.model,
            self.weight_decay,
            no_weight_decay_list=self.model.no_weight_decay(),
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
        )

        if warmup_steps == 0 and cosine_annealing_steps == 0:
            return optimizer

        warmup_scheduler = (
            CustomLambdaLR(optimizer, warmup_steps) if warmup_steps > 0 else None
        )
        cosine_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=cosine_annealing_steps,
                eta_min=self.lr * 1e-2,
            )
            if cosine_annealing_steps > 0
            else None
        )
        combined_scheduler = CombinedScheduler(
            optimizer=optimizer,
            scheduler1=warmup_scheduler,
            scheduler2=cosine_scheduler,
            warmup_steps=warmup_steps,
            start_cosine_step=start_cosine_step,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": combined_scheduler, "interval": "step"},
        }
