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
from utils import (
    SigmoidFocalLossWithLogits, arrange_sparse_minkowski, argsort_sparse_tensor, 
    arrange_truth, argsort_coords, CustomLambdaLR, CombinedScheduler, dice_loss
)
from functools import partial
from packaging import version
from typing import Optional
from collections import defaultdict



class SparseMAELightningModel(pl.LightningModule):
    def __init__(self, model, args):
        super(SparseMAELightningModel, self).__init__()

        self.model = model
        self.warmup_steps = args.warmup_steps
        self.start_cosine_step = args.start_cosine_step
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps


    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    
    def forward(self, 
                patches_charge,
                patches_lepton,
                patches_seg,
                patch_ids,
                attn_mask,
                glob,
                mask_ratio=0.5
               ):
        return self.model(patches_charge, patches_lepton, patches_seg, patch_ids, attn_mask, glob, mask_ratio=mask_ratio)


    def _arrange_batch(self, batch):
        patches_charge = batch['patches_charge']
        patches_lepton = batch['patches_lepton']
        patches_seg = batch['patches_seg']
        patch_ids = batch['patch_ids']
        attn_mask = batch['attn_mask']
        faser_cal = batch['faser_cal_modules']
        rear_cal = batch['rear_cal_modules']
        rear_hcal = batch['rear_hcal_modules']
        global_scalars = batch['f_glob']
    
        return patches_charge, patches_lepton, patches_seg, patch_ids, attn_mask, faser_cal, rear_cal, rear_hcal, global_scalars
        
    
    def common_step(self, batch):
        patches_charge, patches_lepton, patches_seg, patch_ids, attn_mask, *glob = self._arrange_batch(batch)

        # Forward pass
        total_loss, _, _, individual_losses = self.forward(
            patches_charge, patches_lepton, patches_seg, patch_ids, attn_mask, glob, mask_ratio=0.5
        )
  
        # Retrieve current batch size and learning rate
        batch_size = len(patches_charge)
        lr = self.optimizers().param_groups[0]['lr']

        return total_loss, individual_losses, batch_size, lr


    def training_step(self, batch, batch_idx):
        #torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/train_total", loss.item(), batch_size=batch_size, on_step=True, on_epoch=True,prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("loss/train_{}".format(key), value.item(), batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        #torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/val_total", loss.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("loss/val_{}".format(key), value.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss


    def configure_optimizers(self):
        """Configure and initialize the optimizer and learning rate scheduler."""
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                (p.ndim == 1)
                or name.endswith(".bias")
            ):
                no_decay.append(p)
            else:
                decay.append(p)
                
        optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
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

