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
        self.standardize_input = args.standardize_input

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

    
    def forward(self, x, x_glob, cls_labels, mask_ratio):
        return self.model(x, x_glob, cls_labels, mask_ratio)


    def _arrange_batch(self, batch):
        batch_input, *global_params = arrange_sparse_minkowski(batch, self.device)
        seg_labels = arrange_truth(batch)['seg_labels']
        return batch_input, *global_params, seg_labels


    def compute_losses(self, targ_reg, targ_cls, pred_occ, pred_reg, pred_cls, idx_targets, smooth=0.1):
        mask_targets = (idx_targets >= 0)
        mask_flat    = mask_targets.view(-1)
        idx_flat     = idx_targets.view(-1)[mask_flat]

        targ_occ = mask_targets.float()
        if self.model.training and smooth > 0:
            targ_occ = targ_occ * (1.0 - smooth) + 0.5 * smooth
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
        
    
    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, *batch_input_global, cls_labels = self._arrange_batch(batch)

        # Forward pass
        pred_occ, pred_reg, pred_cls, idx_targets = self.forward(
            batch_input, batch_input_global, cls_labels, mask_ratio=self.mask_ratio)

        loss, part_losses = self.compute_losses(
            batch_input.F, cls_labels, pred_occ, pred_reg, pred_cls, idx_targets
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

