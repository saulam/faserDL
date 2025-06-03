"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 1: semantic segmentation.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from utils import (
    focal_loss, dice_loss, arrange_sparse_minkowski, argsort_sparse_tensor, 
    arrange_truth, argsort_coords, CustomLambdaLR, CombinedScheduler
)
from functools import partial
from packaging import version


pl_version = pl.__version__


class SparseAELightningModel(pl.LightningModule):
    def __init__(self, model, args):
        super(SparseAELightningModel, self).__init__()

        self.model = model
        self.loss_sigmoid = nn.BCEWithLogitsLoss()
        self.loss_softmax = nn.CrossEntropyLoss()
        self.loss_charge = nn.MSELoss()
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


    '''
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # Calculate progress p: global step / max_steps
        # Make sure self.trainer is set (it usually is after a few batches)
        if self.trainer.max_steps:
            total_steps = (self.trainer.max_epochs // 2) * self.trainer.num_training_batches  # div by 2 added
            p = float(self.global_step) / total_steps
            # Here, gamma = 10 is a typical choice, modify as needed
            new_value = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            with torch.no_grad():
                self.model.global_weight.fill_(new_value)
    '''

    
    def forward(self, x, x_glob, module_to_event, module_pos):
        return self.model(x, x_glob, module_to_event, module_pos)


    def _arrange_batch(self, batch):
        batch_input, batch_input_global = arrange_sparse_minkowski(batch, self.device)
        batch_module_to_event, batch_module_pos = batch['module_to_event'], batch['module_pos']
        target = arrange_truth(batch)
        return batch_input, batch_input_global, batch_module_to_event, batch_module_pos, target


    def compute_losses(self, batch_output, target, target_charge):
        # pred
        out_primlepton = batch_output['out_primlepton'].F
        out_seg = batch_output['out_seg'].F
        out_charge = batch_output['out_charge'].F

        # true
        targ_primlepton = target['primlepton_labels']
        targ_seg = target['seg_labels']
        targ_charge = target_charge

        # losses
        loss_primlepton = self.loss_sigmoid(out_primlepton, targ_primlepton)
        loss_seg = self.loss_softmax(out_seg, targ_seg)
        loss_charge = self.loss_charge(out_charge, targ_charge)

        part_losses = {'primlepton': loss_primlepton,
                       'seg': loss_seg,
                       'charge': loss_charge,
                       }
        total_loss = sum(part_losses.values())

        return total_loss, part_losses

    
    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, batch_input_global, batch_module_to_event, batch_module_pos, target = self._arrange_batch(batch)

        # Forward pass
        batch_output = self.forward(batch_input, batch_input_global, batch_module_to_event, batch_module_pos)
        loss, part_losses = self.compute_losses(batch_output, target, batch_input.F)
  
        # Retrieve current learning rate
        lr = self.optimizers().param_groups[0]['lr']

        return loss, part_losses, batch_size, lr


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/train_total", loss.item(), batch_size=batch_size, on_step=True, on_epoch=True,prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("loss/train_{}".format(key), value.item(), batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)

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
        # Optimiser
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
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
                eta_min=0
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

