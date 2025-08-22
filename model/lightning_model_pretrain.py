"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 1: masked autoencoder.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
import timm.optim.optim_factory as optim_factory
from utils import (
    arrange_sparse_minkowski, arrange_truth, 
    CustomLambdaLR, CombinedScheduler, weighted_loss
)
from functools import partial
from packaging import version
from typing import Optional
from collections import defaultdict



class MAEPreTrainer(pl.LightningModule):
    def __init__(self, model, metadata, args):
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
        self.metadata = metadata
        self.preprocessing_input = args.preprocessing_input
        self.label_smoothing = args.label_smoothing

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.log_sigma_occ = nn.Parameter(torch.zeros(()))
        self.log_sigma_reg = nn.Parameter(torch.zeros(()))
        self.log_sigma_cls = nn.Parameter(torch.zeros(()))
        self._uncertainty_params = {
            "occ": self.log_sigma_occ,
            "reg": self.log_sigma_reg,
            "cls": self.log_sigma_cls,
        }


    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    
    def forward(self, x, x_glob, mask_ratio):
        return self.model(x, x_glob, mask_ratio)


    def _arrange_batch(self, batch):
        batch_input, *global_params = arrange_sparse_minkowski(batch, self.device)
        seg_labels = arrange_truth(batch)['seg_labels']
        return batch_input, *global_params, seg_labels
    

    def _occ_supervision_mask(
        self,
        idx_targets: torch.Tensor,       # [M, P], -1 => empty sub-voxel, >=0 => raw index of hit
        patch_shape: tuple,              # (p_h, p_w, p_d)
        dilate: int = 1,                 # r voxels around the signal inside each patch
        neg_ratio: float = 6.0,          # |N| ≈ neg_ratio × |P| (per token)
        max_negs_per_token: int = 4096,  # hard cap per token
    ):
        """
        Returns:
        sup_mask: [M, P] bool  — which sub-voxels to supervise for occupancy
        sup_targ: [M, P] float — 1 for positives, 0 for border + sampled negatives
        pos_mask: [M, P] bool  — convenience for your reg/cls heads
        """
        device = idx_targets.device
        M, P   = idx_targets.shape
        p_h, p_w, p_d = patch_shape

        pos_mask = (idx_targets >= 0)                                    # [M, P]

        # --- B: border via voxel dilation (inside each patch) ---
        occ = pos_mask.float().view(M, 1, p_h, p_w, p_d)                 # [M,1,ph,pw,pd]
        if dilate > 0:
            ksz = 2 * dilate + 1
            kernel = torch.ones((1, 1, ksz, ksz, ksz), device=device)    # depthwise over tokens
            dil = F.conv3d(occ, kernel, padding=dilate) > 0              # [M,1,ph,pw,pd] -> bool
        else:
            dil = occ.bool()
        dil = dil.view(M, P)                                             # [M, P]
        border_mask = (~pos_mask) & dil                                  # empty-but-near-signal

        # --- Far empties (eligible negatives) ---
        far_empty = ~dil                                                  # outside dilation

        # Per-token negative budget (variable k_i, clamped by availability and a hard cap)
        pos_counts = pos_mask.sum(dim=1)                                  # [M]
        far_counts = far_empty.sum(dim=1)                                 # [M]
        want_negs  = (pos_counts * neg_ratio).to(torch.long)              # [M]
        want_negs  = torch.minimum(want_negs, far_counts)                 # don’t ask for more than exist
        want_negs  = torch.clamp(want_negs, max=max_negs_per_token)       # hard cap
        K_max = int(want_negs.max().item())

        # --- Sample N negatives per token, vectorized ---
        sampled_neg_mask = torch.zeros_like(pos_mask, dtype=torch.bool)   # [M, P]
        if K_max > 0:
            # Random scores; invalidate non-eligible columns by setting to +inf
            rnd = torch.rand(M, P, device=device)
            rnd = rnd.masked_fill(~far_empty, float("inf"))               # only far empties compete

            # Take K_max smallest per row (equivalent to sampling without replacement)
            # We use topk on -rnd to get "smallest".
            vals, idxs = torch.topk(-rnd, k=K_max, dim=1)                 # idxs: [M, K_max]

            # Per-row mask to keep only the first k_i selections
            keep = (torch.arange(K_max, device=device).unsqueeze(0) <
                    want_negs.unsqueeze(1))                               # [M, K_max] bool

            # Scatter to boolean mask without any Python loops
            rows = torch.arange(M, device=device).unsqueeze(1).expand(-1, K_max)  # [M, K_max]
            sampled_neg_mask[rows[keep], idxs[keep]] = True

        # --- Final supervision set ---
        sup_mask = pos_mask | border_mask | sampled_neg_mask              # [M, P]
        sup_targ = pos_mask.float()                                       # 1 for P, 0 for (B ∪ N)

        return sup_mask, sup_targ, pos_mask


    def compute_losses(self, targ_reg, targ_cls, pred_occ, pred_reg, pred_cls, idx_targets):
        """
        targ_reg: [N_hits, C_in]        ground-truth charge/features at raw voxel ids
        targ_cls: [N_hits] (long)       semantic class per raw voxel id
        pred_occ: [M, P]                occupancy logits per masked token (P = p_h*p_w*p_d)
        pred_reg: [M, P*C_in]
        pred_cls: [M, P*C_out]
        idx_targets: [M, P]             raw voxel ids or -1 for empty
        """
        # ----- OCCUPANCY: supervise only on P ∪ B ∪ N -----
        p_h, p_w, p_d = self.model.patch_size.tolist()
        sup_mask, sup_targ, pos_mask = self._occ_supervision_mask(
            idx_targets,
            patch_shape=(p_h, p_w, p_d),
            dilate=getattr(self, "occ_dilate", 2),
            neg_ratio=getattr(self, "occ_neg_ratio", 3.0),
            max_negs_per_token=getattr(self, "occ_max_negs", 1024),
        )

        if self.model.training and getattr(self, "label_smoothing", 0.0) > 0.0:
            eps = self.label_smoothing
            sup_targ = sup_targ * (1.0 - eps) + 0.5 * eps

        occ_logits  = pred_occ[sup_mask]
        occ_targets = sup_targ[sup_mask]

        loss_occ = F.binary_cross_entropy_with_logits(occ_logits, occ_targets)

        # ----- REG / CLS: only on truly occupied sub-voxels -----
        mask_flat = pos_mask.view(-1)                   # [M*P] bool
        idx_flat  = idx_targets.view(-1)[mask_flat]     # raw voxel ids for occupied positions

        pred_reg_v = pred_reg.view(-1, self.model.in_chans)[mask_flat]
        targ_reg_v = targ_reg[idx_flat]
        loss_reg   = F.mse_loss(pred_reg_v, targ_reg_v)

        pred_cls_v = pred_cls.view(-1, self.model.out_chans)[mask_flat]
        targ_cls_v = targ_cls[idx_flat]
        loss_cls   = F.cross_entropy(pred_cls_v, targ_cls_v)

        # ----- aggregate -----
        part_losses = {'occ': loss_occ, 'reg': loss_reg, 'cls': loss_cls}
        total_loss = (
            weighted_loss(loss_occ, self.log_sigma_occ) +
            weighted_loss(loss_reg, self.log_sigma_reg) +
            weighted_loss(loss_cls, self.log_sigma_cls)
        )
        return total_loss, part_losses

    '''
    def compute_losses(self, targ_reg, targ_cls, pred_occ, pred_reg, pred_cls, idx_targets):
        mask_targets = (idx_targets >= 0)
        mask_flat    = mask_targets.view(-1)
        idx_flat     = idx_targets.view(-1)[mask_flat]

        targ_occ = mask_targets.float()
        if self.model.training and self.label_smoothing > 0:
            targ_occ = targ_occ * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss_occ = F.binary_cross_entropy_with_logits(pred_occ, targ_occ)

        pred_reg   = pred_reg.view(-1, self.model.in_chans)[mask_flat]
        targ_reg   = targ_reg[idx_flat]
        loss_reg   = F.mse_loss(pred_reg, targ_reg)

        pred_cls   = pred_cls.view(-1, self.model.out_chans)[mask_flat]
        targ_cls   = targ_cls[idx_flat]
        loss_cls   = F.cross_entropy(pred_cls, targ_cls)

        part_losses = {
            'occ': loss_occ,
            'reg': loss_reg,
            'cls': loss_cls,
        }
        total_loss = ( 
            weighted_loss(loss_occ, self.log_sigma_occ) +
            weighted_loss(loss_reg, self.log_sigma_reg) +
            weighted_loss(loss_cls, self.log_sigma_cls)
        )

        return total_loss, part_losses
    '''
    
    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, *batch_input_global, cls_labels = self._arrange_batch(batch)

        # Forward pass
        pred_occ, pred_reg, pred_cls, idx_targets, _, _ = self.forward(
            batch_input, batch_input_global, mask_ratio=self.mask_ratio)

        loss, part_losses = self.compute_losses(
            batch_input.F, cls_labels, pred_occ, pred_reg, pred_cls, idx_targets,
        )
  
        lr = self.optimizers().param_groups[0]['lr']
        
        return loss, part_losses, batch_size, lr


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/train_total", loss.item(), batch_size=batch_size, on_step=True, on_epoch=True,prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("loss/train_{}".format(key), value.item(), batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)

        # log the actual sigmas (exp(-log_sigma))
        for key, log_sigma in self._uncertainty_params.items():
            uncertainty = torch.exp(-log_sigma)
            self.log(f'uncertainty/{key}', uncertainty, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
                
        return loss


    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/val_total", loss.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("loss/val_{}".format(key), value.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss


    def configure_optimizers(self):
        """Configure and initialize the optimizer and learning rate scheduler."""
        param_groups = optim_factory.param_groups_weight_decay(
            self.model, self.weight_decay,
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
            lr_decay=1.0
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}


    def lr_scheduler_step(self, scheduler, *args):
        """Perform a learning rate scheduler step."""
        scheduler.step()

