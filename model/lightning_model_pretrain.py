"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 1: masked autoencoder.
"""

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm.optim as optim_factory
from utils import (
    arrange_input, arrange_truth,
    contrastive_with_ghost_shared, reconstruction_losses_masked_simple,
    CustomLambdaLR, CombinedScheduler, weighted_loss, move_obj,
)


class MAEPreTrainer(pl.LightningModule):
    def __init__(self, model, dataset, args):
        super(MAEPreTrainer, self).__init__()

        self.model = model
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
        self.log_sigma_trk = nn.Parameter(torch.zeros(()))
        self.log_sigma_pri = nn.Parameter(torch.zeros(()))
        self.log_sigma_pid = nn.Parameter(torch.zeros(()))
        self.log_sigma_occ = nn.Parameter(torch.zeros(()))
        self.log_sigma_reg = nn.Parameter(torch.zeros(()))
        self._uncertainty_params = {
            "trk": self.log_sigma_trk,
            "pri": self.log_sigma_pri,
            "pid": self.log_sigma_pid,
            "occ": self.log_sigma_occ,
            "reg": self.log_sigma_reg,
        }

        # learnable scale logits for contrastive losses (init at 1/0.07 ≈ 14.285  => log ≈ 2.659)
        init = math.log(1/0.07)
        self.logit_scale_trk = nn.Parameter(torch.tensor(init))
        self.logit_scale_pri = nn.Parameter(torch.tensor(init))
        self.logit_scale_pid = nn.Parameter(torch.tensor(init))
        self.logit_scale_trk_ghost = nn.Parameter(torch.tensor(init))
        self.logit_scale_pri_ghost = nn.Parameter(torch.tensor(init))
        self.logit_scale_pid_ghost = nn.Parameter(torch.tensor(init))
        self._logit_scale_params = {
            "trk": self.logit_scale_trk,
            "pri": self.logit_scale_pri,
            "pid": self.logit_scale_pid,
            "trk_ghost": self.logit_scale_trk_ghost,
            "pri_ghost": self.logit_scale_pri_ghost,
            "pid_ghost": self.logit_scale_pid_ghost,
        }


    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return move_obj(batch, device)
    

    @staticmethod
    def _scale(p: torch.Tensor) -> torch.Tensor:
        # clamp exp(logit_scale) in [1/100, 100]
        return p.clamp(math.log(1/100), math.log(100)).exp()
    

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
        hit_track_id = labels['hit_track_id']
        hit_primary_id = labels['hit_primary_id']
        hit_pdg = labels['hit_pdg']
        ghost_mask = labels['ghost_mask']
        hit_event_id = labels['hit_event_id']

        return batch_input, *global_params, (hit_track_id, hit_primary_id, hit_pdg, ghost_mask, hit_event_id)


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
        z_track: torch.Tensor,           # [N, Dt]
        z_primary: torch.Tensor,         # [N, Dp]
        z_pid: torch.Tensor,             # [N, Dp]
        track_id: torch.Tensor,          # [N] int64
        primary_id: torch.Tensor,        # [N] int64
        pid_id: torch.Tensor,            # [N] int64
        event_id: torch.Tensor,          # [N] int64
        ghost_mask: torch.Tensor,        # [N] bool
        normalize: bool = True,
        pushaway_weight: float = 0.05,
        per_event_mean: bool = False,
    ):
        """
        Computes both losses (same-track, same-primary, same-pid) in one call.
        Assumes inputs are already masked/aligned (no ghosts/invisible hits).
        """
        s_trk   = self._scale(self.logit_scale_trk)
        s_pri   = self._scale(self.logit_scale_pri)
        s_pid   = self._scale(self.logit_scale_pid)
        s_trk_g = self._scale(self.logit_scale_trk_ghost)
        s_pri_g = self._scale(self.logit_scale_pri_ghost)
        s_pid_g = self._scale(self.logit_scale_pid_ghost)

        # track
        loss_trk, loss_trk_ghost = contrastive_with_ghost_shared(
            z_track, track_id, event_id, num_neg=16, pool_mult=2,
            normalize=normalize, per_event_mean=per_event_mean, ghost_mask=ghost_mask,
            logit_scale=s_trk, ghost_logit_scale=s_trk_g
        )

        # primary
        loss_pri, loss_pri_ghost = contrastive_with_ghost_shared(
            z_primary, primary_id, event_id, num_neg=8, pool_mult=2,
            normalize=normalize, per_event_mean=per_event_mean, ghost_mask=ghost_mask,
            logit_scale=s_pri, ghost_logit_scale=s_pri_g
        )

        # pid
        pid_event_id = torch.zeros_like(event_id)  # trick: pid loss across events
        loss_pid, loss_pid_ghost = contrastive_with_ghost_shared(
            z_pid, pid_id, pid_event_id, num_neg=6, pool_mult=1,
            normalize=normalize, per_event_mean=per_event_mean, ghost_mask=ghost_mask,
            logit_scale=s_pid, ghost_logit_scale=s_pid_g
        )

        loss_trk_all = loss_trk + pushaway_weight * loss_trk_ghost
        loss_pri_all = loss_pri + pushaway_weight * loss_pri_ghost
        loss_pid_all = loss_pid + pushaway_weight * loss_pid_ghost

        part_losses_enc = {
            "trk/total": loss_trk_all.detach(), "trk/pos": loss_trk.detach(), "trk/ghost": loss_trk_ghost.detach(),
            "pri/total": loss_pri_all.detach(), "pri/pos": loss_pri.detach(), "pri/ghost": loss_pri_ghost.detach(),
            "pid/total": loss_pid_all.detach(), "pid/pos": loss_pid.detach(), "pid/ghost": loss_pid_ghost.detach(),
        }

        return loss_trk_all, loss_pri_all, loss_pid_all, part_losses_enc
    

    def compute_contrastive_losses(
        self,
        pred_track: torch.Tensor,
        pred_primary: torch.Tensor,
        pred_pid: torch.Tensor,
        idx_targets: torch.Tensor,
        hit_track_id: torch.Tensor,
        hit_primary_id: torch.Tensor,
        hit_pdg: torch.Tensor,
        hit_event_id: torch.Tensor,
        ghost_mask: torch.Tensor,
    ):
        raw_idx, tok_row, sub_idx = self.mask_and_align_voxels(idx_targets)

        # Gather embeddings and labels
        z_track = pred_track[tok_row, sub_idx, :]         # [N_valid, D]
        z_primary = pred_primary[tok_row, sub_idx, :]     # [N_valid, D]
        z_pid = pred_pid[tok_row, sub_idx, :]             # [N_valid, D]
        y_track = hit_track_id[raw_idx]
        y_primary = hit_primary_id[raw_idx]
        y_pid = hit_pdg[raw_idx]
        evt = hit_event_id[raw_idx]
        ghost = ghost_mask[raw_idx]

        loss_trk, loss_pri, loss_pid, part_losses_enc = self.metric_losses_masked_simple(
            z_track, z_primary, z_pid, y_track, y_primary, y_pid, evt, ghost, normalize=True,
        )

        return loss_trk, loss_pri, loss_pid, part_losses_enc


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
            label_smoothing=getattr(self, "label_smoothing", 0.0),
            focal_gamma=getattr(self, "focal_gamma", 1.5),
            focal_alpha=getattr(self, "focal_alpha", None),
            occ_dilate=getattr(self, "occ_dilate", 2),
            huber_delta=getattr(self, "huber_delta", 1.0),
            reg_weight_lam=getattr(self, "reg_weight_lam", 1.0),
            reg_weight_alpha=getattr(self, "reg_weight_alpha", 0.5),
            reg_weight_q0=getattr(self, "reg_weight_q0", None),
            reg_weight_wmax=getattr(self, "reg_weight_wmax", None),
            occ_empty_beta=getattr(self, "occ_empty_beta", 0.5),
            per_event_mean=per_event_mean,
        )
        return loss_occ, loss_reg, part_losses_dec


    def compute_losses(
        self,
        pred_track: torch.Tensor,
        pred_primary: torch.Tensor,
        pred_pid: torch.Tensor,
        pred_occ: torch.Tensor,
        pred_reg: torch.Tensor,
        targ_reg: torch.Tensor,
        con_idx_targets: torch.Tensor,
        rec_idx_targets: torch.Tensor,
        hit_track_id: torch.Tensor,
        hit_primary_id: torch.Tensor,
        hit_pdg: torch.Tensor,
        hit_event_id: torch.Tensor,
        ghost_mask: torch.Tensor,
    ):
        loss_trk, loss_pri, loss_pid, part_enc = self.compute_contrastive_losses(
            pred_track, pred_primary, pred_pid, con_idx_targets, hit_track_id, hit_primary_id, hit_pdg, hit_event_id, ghost_mask,
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
            _weight(loss_trk, "log_sigma_trk", kind="nce") +
            _weight(loss_pri, "log_sigma_pri", kind="nce") +
            _weight(loss_pid, "log_sigma_pid", kind="nce") +
            _weight(loss_occ, "log_sigma_occ", kind="ce")  +
            _weight(loss_reg, "log_sigma_reg", kind="huber")
        )

        return total_loss, part_losses


    def common_step(self, batch):
        batch_input, *batch_input_global, labels = self._arrange_batch(batch)
        batch_size = batch_input.batch_size
        hit_track_id, hit_primary_id, hit_pdg, ghost_mask, hit_event_id = labels

        # Forward pass
        pred_track, pred_primary, pred_pid, pred_occ, pred_reg, con_idx_targets, rec_idx_targets, _, _, = self.forward(
            batch_input, batch_input_global, mask_ratio=self.mask_ratio)

        loss, part_losses = self.compute_losses(
            pred_track,
            pred_primary,
            pred_pid,
            pred_occ,
            pred_reg,
            batch_input.features,
            con_idx_targets,
            rec_idx_targets,
            hit_track_id,
            hit_primary_id,
            hit_pdg,
            hit_event_id,
            ghost_mask,
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

        # log the actual logit scales
        for key, logit_scale in self._logit_scale_params.items():
            scale = self._scale(logit_scale).detach()
            self.log(
                f'logit_scale/{key}',
                scale,
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
            'params': list(self._uncertainty_params.values()) + list(self._logit_scale_params.values()),
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


    def lr_scheduler_step(self, scheduler, *args):
        """Perform a learning rate scheduler step."""
        scheduler.step()
