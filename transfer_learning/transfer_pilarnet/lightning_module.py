"""Lightning module for PILArNet PID fine-tuning."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
from spconv.pytorch import SparseConvTensor

from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix


class PILArNetFineTuner(pl.LightningModule):
    """Lightning wrapper for particle-level PILArNet training."""

    TYPE_NAMES = ["photon", "electron", "muon", "pion", "proton"]

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.loss_pid = nn.CrossEntropyLoss(
            label_smoothing=getattr(args, "label_smoothing", 0.0),
        )

        self.train_acc = MulticlassAccuracy(num_classes=5, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=5, average="micro")
        self.val_acc_per_class = MulticlassAccuracy(num_classes=5, average="none")
        self.val_confusion = MulticlassConfusionMatrix(num_classes=5)

        self.lr = getattr(args, "lr", 1e-4)
        self.blr = getattr(args, "blr", 5e-4)
        self.warmup_epochs = getattr(args, "warmup_epochs", 5)
        self.cosine_annealing_epochs = getattr(args, "cosine_annealing_epochs", 15)
        self.weight_decay = getattr(args, "weight_decay", 0.05)
        self.layer_decay = getattr(args, "layer_decay", 0.75)
        self.beta1 = getattr(args, "beta1", 0.9)
        self.beta2 = getattr(args, "beta2", 0.999)
        self.eps = getattr(args, "eps", 1e-8)
        self.ema_decay = getattr(args, "ema_decay", 0.9999)
        self._batch_size = getattr(args, "batch_size", 256)
        self.ema = None
        self._ema_applied_for_val = False

    def _build_sparse(self, batch):
        """Construct a SparseConvTensor from the collated batch dict."""
        coords = batch["coords"]
        feats = batch["feats"]
        B = batch["batch_size"]

        if coords.dtype != torch.int32:
            coords = coords.int()

        return SparseConvTensor(
            features=feats,
            indices=coords,
            spatial_shape=list(self.model.spatial_shape),
            batch_size=B,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return _move(batch, device)

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        checkpoint["pretrained_loaded_keys"] = sorted(
            getattr(self.model, "_pretrained_loaded_keys", set())
        )

    def on_load_checkpoint(self, checkpoint):
        self.model._pretrained_loaded_keys = set(
            checkpoint.get("pretrained_loaded_keys", [])
        )
        self.model._patch_embed_loaded = any(
            key.startswith("patch_embed") for key in self.model._pretrained_loaded_keys
        )
        if "ema_state_dict" in checkpoint:
            if self.ema is None:
                self.ema = ExponentialMovingAverage(
                    self.model.parameters(), decay=self.ema_decay
                )
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def on_train_start(self):
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups
        if not self.trainer.is_global_zero:
            return
        if self.ema is None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.ema_decay
            )

    def on_before_zero_grad(self, optimizer):
        if self.ema is not None:
            self.ema.update()

    def on_validation_epoch_start(self):
        if self.ema is None or self.trainer.sanity_checking:
            return
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        self._ema_applied_for_val = True

    def forward(self, x_sp, particle_meta=None):
        return self.model(x_sp, particle_meta)

    def training_step(self, batch, batch_idx):
        x_sp = self._build_sparse(batch)
        outputs = self(x_sp, batch.get("particle_meta"))
        loss = self.loss_pid(outputs["out_pid"], batch["type_label"])

        B = batch["batch_size"]
        preds = outputs["out_pid"].argmax(dim=1)
        self.train_acc.update(preds, batch["type_label"])

        self.log("loss/train", loss.item(), batch_size=B, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        self.log("acc/train", self.train_acc.compute(), prog_bar=True, sync_dist=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x_sp = self._build_sparse(batch)
        outputs = self(x_sp, batch.get("particle_meta"))
        loss = self.loss_pid(outputs["out_pid"], batch["type_label"])

        B = batch["batch_size"]
        preds = outputs["out_pid"].argmax(dim=1)
        self.val_acc.update(preds, batch["type_label"])
        self.val_acc_per_class.update(preds, batch["type_label"])
        self.val_confusion.update(preds, batch["type_label"])

        self.log("loss/val", loss.item(), batch_size=B, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        per_class = self.val_acc_per_class.compute()
        self.log("acc/val", acc, prog_bar=True, sync_dist=True)
        for i, name in enumerate(self.TYPE_NAMES):
            self.log(f"acc_class/{name}/val", per_class[i], sync_dist=True)
        self.val_acc.reset()
        self.val_acc_per_class.reset()
        self.val_confusion.reset()
        if self._ema_applied_for_val:
            self.ema.restore(self.model.parameters())
            self._ema_applied_for_val = False

    def configure_optimizers(self):
        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = max(1, total_steps // self.trainer.max_epochs)

        if self.blr is not None:
            eff_bs = (
                self._batch_size
                * self.trainer.world_size
                * self.trainer.accumulate_grad_batches
            )
            self.lr = self.blr * eff_bs / 256.0

        warmup_steps = steps_per_epoch * self.warmup_epochs
        cosine_steps = steps_per_epoch * self.cosine_annealing_epochs

        if self.trainer.is_global_zero:
            print(f"lr={self.lr:.6f}  total_steps={total_steps}  warmup={warmup_steps}  cosine={cosine_steps}")

        param_groups = self._build_param_groups()
        for pg in param_groups:
            scale = pg.pop("lr_scale", 1.0)
            pg["lr"] = self.lr * scale
        if self.trainer.is_global_zero:
            print("\nParameter groups:")
            for pg in param_groups:
                n_tensors = len(pg["params"])
                n_params = sum(p.numel() for p in pg["params"])
                print(
                    f"  {pg.get('group_name', 'unnamed')}: "
                    f"lr={pg['lr']:.6e} wd={pg['weight_decay']:.4f} "
                    f"tensors={n_tensors} params={n_params}"
                )

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
        )

        schedulers = []
        if warmup_steps > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
            )
            schedulers.append(warmup_sched)
        if cosine_steps > 0:
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_steps, eta_min=self.lr * 1e-2
            )
            schedulers.append(cosine_sched)

        if len(schedulers) == 2:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=[warmup_steps],
            )
        elif len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _build_param_groups(self):
        """Build parameter groups for layer-wise learning-rate decay."""
        loaded_keys = getattr(self.model, "_pretrained_loaded_keys", set())
        reinit_params, pretrained_embed_params = [], []
        block_params, io_params, head_params, other_params = [], [], [], []
        no_wd = self.model.no_weight_decay()

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in loaded_keys:
                reinit_params.append((name, p))
            elif name.startswith("patch_embed") or name.startswith("pos_embed") or name.startswith("cls_token"):
                pretrained_embed_params.append((name, p))
            elif name.startswith("blocks.") or name == "norm.weight" or name == "norm.bias":
                block_params.append((name, p))
            elif name.startswith("lat_") or name.startswith("latent_") or name.startswith("tokens_norm"):
                io_params.append((name, p))
            elif name.startswith("head_") or name.startswith("task_") or name.startswith("latents_norm") or name == "gamma":
                head_params.append((name, p))
            else:
                other_params.append((name, p))

        depth = len(self.model.blocks) + len(self.model.lat_xattn_blocks) * 2 + 2
        ld = self.layer_decay

        def _make_group(params_list, layer_id, label):
            decay_p = [p for n, p in params_list if p.ndim > 1 and n not in no_wd]
            no_decay_p = [p for n, p in params_list if p.ndim <= 1 or n in no_wd]
            scale = ld ** (depth - layer_id)
            groups = []
            if decay_p:
                groups.append(
                    {
                        "params": decay_p,
                        "weight_decay": self.weight_decay,
                        "lr_scale": scale,
                        "group_name": f"{label}/decay",
                    }
                )
            if no_decay_p:
                groups.append(
                    {
                        "params": no_decay_p,
                        "weight_decay": 0.0,
                        "lr_scale": scale,
                        "group_name": f"{label}/no_decay",
                    }
                )
            return groups

        groups = []
        groups.extend(_make_group(pretrained_embed_params, 0, "encoder_pretrained_embed"))
        block_by_layer = {}
        for name, p in block_params:
            if name.startswith("blocks."):
                lid = int(name.split(".")[1]) + 1
            else:
                lid = len(self.model.blocks)
            block_by_layer.setdefault(lid, []).append((name, p))
        for lid, params in sorted(block_by_layer.items()):
            groups.extend(_make_group(params, lid, f"encoder_blocks_l{lid}"))

        tokens_norm_layer = len(self.model.blocks) + 1
        lat_base = len(self.model.blocks) + 2
        io_by_layer = {}
        for name, p in io_params:
            if "lat_xattn_blocks." in name:
                lid = lat_base + int(name.split(".")[1]) * 2
            elif "latent_self_blocks." in name:
                lid = lat_base + int(name.split(".")[1]) * 2 + 1
            else:
                lid = tokens_norm_layer
            io_by_layer.setdefault(lid, []).append((name, p))
        for lid, params in sorted(io_by_layer.items()):
            groups.extend(_make_group(params, lid, f"encoder_io_l{lid}"))

        groups.extend(_make_group(head_params, depth, "head"))
        groups.extend(_make_group(reinit_params, depth, "encoder_reinit"))
        groups.extend(_make_group(other_params, depth, "encoder_other"))
        return groups

def _move(o, device):
    if isinstance(o, SparseConvTensor):
        ind = o.indices.to(device)
        if ind.dtype != torch.int32:
            ind = ind.int()
        return SparseConvTensor(o.features.to(device), ind, o.spatial_shape, o.batch_size)
    if isinstance(o, torch.Tensor):
        return o.to(device)
    if isinstance(o, dict):
        return {k: _move(v, device) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_move(v, device) for v in o)
    return o
