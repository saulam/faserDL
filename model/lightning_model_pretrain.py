"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 1: masked autoencoder.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm.optim as optim_factory
from torch.nn import functional as F
from utils import (
    arrange_input, arrange_truth, csr_keep_rows_torch, bce_with_logits_label_smoothing,
    soft_ce_with_logits_csr, reconstruction_losses_masked_simple,
    CustomLambdaLR, CombinedScheduler, weighted_loss, move_obj,
)


class MAEPreTrainer(pl.LightningModule):
    def __init__(self, model, dataset, args):
        super(MAEPreTrainer, self).__init__()

        self.model = model
        stats = model.metadata
        self.mask_ratio = args.mask_ratio
        self.warmup_steps = args.warmup_steps
        self.start_cosine_step = args.start_cosine_step
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps
        self.dataset = dataset
        self.preprocessing_input = args.preprocessing_input
        self.label_smoothing = args.label_smoothing

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.log_sigma_gho = nn.Parameter(torch.zeros(()))
        self.log_sigma_hie = nn.Parameter(torch.zeros(()))
        self.log_sigma_dec = nn.Parameter(torch.zeros(()))
        self.log_sigma_pid = nn.Parameter(torch.zeros(()))
        self.log_sigma_occ = nn.Parameter(torch.zeros(()))
        self.log_sigma_reg = nn.Parameter(torch.zeros(()))
        self._uncertainty_params = {
            "gho": self.log_sigma_gho,
            "hie": self.log_sigma_hie,
            "dec": self.log_sigma_dec,
            "pid": self.log_sigma_pid,
            "occ": self.log_sigma_occ,
            "reg": self.log_sigma_reg,
        }


    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return move_obj(batch, device)
    

    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups


    def on_train_epoch_start(self):
        raw = getattr(self.trainer, "train_dataloader", None)
        if raw is None:
            raw = getattr(self.trainer, "train_dataloaders", None)
        if raw is None:
            return

        loaders = raw if isinstance(raw, (list, tuple)) else [raw]
        for dl in loaders:
            ds = getattr(dl, "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(self.trainer.current_epoch)

        lr = self.optimizers().param_groups[0]['lr']
        self.log(f"lr", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        

    def forward(self, x, x_glob, mask_ratio):
        return self.model(x, x_glob, mask_ratio)


    def _arrange_batch(self, batch):
        batch_input, *global_params = arrange_input(batch)
        labels = arrange_truth(batch)
        targets = {}
        targets['vis_sp_momentum'] = labels['vis_sp_momentum']
        targets['csr_hie'] = labels['csr_hie_indptr'], labels['csr_hie_ids'], labels['csr_hie_weights']
        targets['csr_dec'] = labels['csr_dec_indptr'], labels['csr_dec_ids'], labels['csr_dec_weights']
        targets['csr_pid'] = labels['csr_pid_indptr'], labels['csr_pid_ids'], labels['csr_pid_weights']
        targets['ghost_mask'] = labels['ghost_mask']
        targets['hit_event_id'] = labels['hit_event_id']

        return batch_input, *global_params, targets


    def mask_and_align_voxels(self, idx_targets):
        """
        idx_targets: [N_tok, P] with -1 for empty slots.
        Returns indices to slice your flat hit tensors; no ghost filtering here.
        """
        valid = idx_targets >= 0
        tok_row, sub_idx = torch.nonzero(valid, as_tuple=True)   # where a voxel is present
        raw_idx = idx_targets[tok_row, sub_idx]                  # [N_valid] indices into hit arrays
        return raw_idx, tok_row, sub_idx
    

    def metric_losses_masked_simple(
        self,
        z_gho: torch.Tensor,             # [N]
        z_hie: torch.Tensor,             # [N, Dp]
        z_dec: torch.Tensor,             # [N, Dp]
        z_pid: torch.Tensor,             # [N, Dp]
        csr_hie: torch.Tensor,           # ([N+1], [L], [L]) int64, float32
        csr_dec: torch.Tensor,           # ([N+1], [L], [L]) int64, float32
        csr_pid: torch.Tensor,           # ([N+1], [L], [L]) int64, float32
        ghost_mask: torch.Tensor,        # [N] bool
    ):
        """
        Computes losses (same-track, same-primary, same-pid) in one call.
        """
        loss_gho = bce_with_logits_label_smoothing(z_gho, ghost_mask.to(z_gho.dtype), label_smoothing=self.label_smoothing)
        loss_hie = soft_ce_with_logits_csr(z_hie, csr_hie, ghost_mask=ghost_mask, label_smoothing=self.label_smoothing)
        loss_dec = soft_ce_with_logits_csr(z_dec, csr_dec, ghost_mask=ghost_mask, label_smoothing=self.label_smoothing)
        loss_pid = soft_ce_with_logits_csr(z_pid, csr_pid, ghost_mask=ghost_mask, label_smoothing=self.label_smoothing)

        part_losses_enc = {
            "gho/total": loss_gho.detach(),
            "hie/total": loss_hie.detach(),
            "dec/total": loss_dec.detach(),
            "pid/total": loss_pid.detach(),
        }

        return loss_gho, loss_hie, loss_dec, loss_pid, part_losses_enc


    def compute_relational_losses(
        self,
        pred_gho: torch.Tensor,
        pred_hie: torch.Tensor,
        pred_dec: torch.Tensor,
        pred_pid: torch.Tensor,
        idx_targets: torch.Tensor,
        csr_hie: torch.Tensor,
        csr_dec: torch.Tensor,
        csr_pid: torch.Tensor,
        ghost_mask: torch.Tensor,
    ):
        raw_idx, tok_row, sub_idx = self.mask_and_align_voxels(idx_targets)

        # Gather embeddings and labels
        z_gho = pred_gho[tok_row, sub_idx]                    # [N_valid]
        z_hie = pred_hie[tok_row, sub_idx, :]                 # [N_valid, D]
        z_dec = pred_dec[tok_row, sub_idx, :]                 # [N_valid, D]
        z_pid = pred_pid[tok_row, sub_idx, :]                 # [N_valid, D]
        csr_hie = csr_keep_rows_torch(*csr_hie, raw_idx)[:3]  # ([N+1], [L], [L])
        csr_dec = csr_keep_rows_torch(*csr_dec, raw_idx)[:3]  # ([N+1], [L], [L])
        csr_pid = csr_keep_rows_torch(*csr_pid, raw_idx)[:3]  # ([N+1], [L], [L])
        ghost = ghost_mask[raw_idx]

        loss_gho, loss_hie, loss_dec, loss_pid, part_losses_enc = self.metric_losses_masked_simple(
            z_gho, z_hie, z_dec, z_pid, csr_hie, csr_dec, csr_pid, ghost,
        )

        return loss_gho, loss_hie, loss_dec, loss_pid, part_losses_enc


    def compute_reconstruction_losses(
        self,
        targ_reg: torch.Tensor,         # [N_hits, C_in]
        pred_occ: torch.Tensor,         # [M, P]
        pred_reg: torch.Tensor,         # [M, P*C_in]
        idx_targets: torch.Tensor,      # [M, P]
        hit_event_id: torch.Tensor,     # [N_hits]
        ghost_mask: torch.Tensor,       # [N_hits]
        per_event_mean: bool = False,
    ):
        p_h, p_w, p_d = self.model.patch_size.tolist()
        loss_occ, loss_reg, part_losses_dec = reconstruction_losses_masked_simple(
            targ_reg=targ_reg,
            pred_occ=pred_occ,
            pred_reg=pred_reg,
            idx_targets=idx_targets,
            ghost_mask=ghost_mask,
            hit_event_id=hit_event_id,
            patch_shape=(p_h, p_w, p_d),
            dataset=self.dataset,
            preprocessing_input=self.preprocessing_input,
            label_smoothing=self.label_smoothing,
            per_event_mean=per_event_mean,
        )
        return loss_occ, loss_reg, part_losses_dec


    def compute_losses(
        self,
        preds: dict,
        targ_reg: torch.Tensor,
        rel_idx_targets: torch.Tensor,
        rec_idx_targets: torch.Tensor,
        labels: dict,
    ):
        pred_gho=preds["gho"]
        pred_hie=preds["hie"]
        pred_dec=preds["dec"]
        pred_pid=preds["pid"]
        pred_occ=preds["occ"]
        pred_reg=preds["reg"]
        csr_hie=labels['csr_hie']
        csr_dec=labels['csr_dec']
        csr_pid=labels['csr_pid']
        ghost_mask=labels['ghost_mask']
        hit_event_id=labels['hit_event_id']

        loss_gho, loss_hie, loss_dec, loss_pid, part_enc = self.compute_relational_losses(
            pred_gho, pred_hie, pred_dec, pred_pid, rel_idx_targets, csr_hie, csr_dec, csr_pid, ghost_mask,
        )
        loss_occ, loss_reg, part_dec = self.compute_reconstruction_losses(
            targ_reg, pred_occ, pred_reg, rec_idx_targets, hit_event_id, ghost_mask,
        )

        # Kendall et al. aggregation
        part_losses = {**part_enc, **part_dec}
        def _weight(loss, attr, kind):
            ls = getattr(self, attr, None)
            return weighted_loss(loss, ls, kind) if ls is not None else loss

        total_loss = (
            _weight(loss_gho, "log_sigma_gho", kind="ce")  +
            _weight(loss_hie, "log_sigma_hie", kind="ce")  +
            _weight(loss_dec, "log_sigma_dec", kind="ce")  +
            _weight(loss_pid, "log_sigma_pid", kind="ce")  +
            _weight(loss_occ, "log_sigma_occ", kind="ce")  +
            _weight(loss_reg, "log_sigma_reg", kind="huber")
        )

        return total_loss, part_losses


    def common_step(self, batch):
        batch_input, *batch_input_global, labels = self._arrange_batch(batch)
        batch_size = batch_input.batch_size

        # Forward pass
        preds, rel_idx_targets, rec_idx_targets, _, _, = self.forward(
            batch_input, batch_input_global, mask_ratio=self.mask_ratio)

        loss, part_losses = self.compute_losses(
            preds=preds,
            targ_reg=batch_input.features,
            rel_idx_targets=rel_idx_targets,
            rec_idx_targets=rec_idx_targets,
            labels=labels,
        )

        return loss, part_losses, batch_size
   

    def training_step(self, batch, batch_idx):
        loss, part_losses, batch_size = self.common_step(batch)

        self.log(
            f"loss_total/train",
            loss.detach(), 
            batch_size=batch_size, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True, 
            sync_dist=True
        )
        for key, value in part_losses.items():
            self.log(
                "{}/train".format(key),
                value, 
                batch_size=batch_size, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=False, 
                sync_dist=True
            )

        # log the actual sigmas (exp(-log_sigma))
        for key, log_sigma in self._uncertainty_params.items():
            uncertainty = torch.exp(-log_sigma).detach()
            self.log(
                f'uncertainty/{key}',
                uncertainty,
                batch_size=batch_size,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

        return loss


    def validation_step(self, batch, batch_idx):
        loss, part_losses, batch_size = self.common_step(batch)

        self.log(
            f"loss_total/val",
            loss.detach(),
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        for key, value in part_losses.items():
            self.log(
                "{}/val".format(key),
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

        return loss


    def configure_optimizers(self):
        """Configure and initialize the optimizer and learning rate scheduler."""
        param_groups = optim_factory.param_groups_weight_decay(
            self.model, self.weight_decay, no_weight_decay_list=self.model.no_weight_decay(),
        )
        param_groups.append({
            'params': list(self._uncertainty_params.values()),
            'lr': self.lr * 0.1,
            'weight_decay': 0.0,
        })
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
        )

        if self.warmup_steps==0 and self.cosine_annealing_steps==0:
            return optimizer

        if self.warmup_steps == 0:
            warmup_scheduler = None
        else:
            # Warm-up scheduler
            warmup_scheduler = CustomLambdaLR(optimizer, self.warmup_steps)
 
        if self.cosine_annealing_steps == 0:
            cosine_scheduler = None
        else:
            # Cosine annealing scheduler
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.cosine_annealing_steps,
                eta_min=0.,
            )

        # Combine both schedulers
        combined_scheduler = CombinedScheduler(
            optimizer=optimizer,
            scheduler1=warmup_scheduler,
            scheduler2=cosine_scheduler,
            warmup_steps=self.warmup_steps,
            start_cosine_step=self.start_cosine_step,
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}
