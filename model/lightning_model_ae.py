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
from typing import Optional


class SparseAELightningModel(pl.LightningModule):
    def __init__(self, model, args):
        super(SparseAELightningModel, self).__init__()

        self.model = model
        self.loss_primlepton = nn.BCEWithLogitsLoss()
        self.loss_seg = nn.CrossEntropyLoss()
        self.loss_charge = nn.MSELoss()
        self.loss_iscc = nn.BCEWithLogitsLoss()
        self.warmup_steps = args.warmup_steps
        self.start_cosine_step = args.start_cosine_step
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps
        self.global_alpha = None


    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups


    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        total = float(self.trainer.max_epochs * self.trainer.num_training_batches)
        p = float(self.trainer.global_step) / total
        raw = (p - 0.1) / 0.1
        self.global_alpha = torch.tensor(raw, device=self.device).clamp(0,1)

    
    def forward(self, 
                x, 
                x_glob,
                module_to_event, 
                module_pos, 
                mask_bool,
                global_alpha=None,
               ):
        return self.model(x, x_glob, module_to_event, module_pos, mask_bool, global_alpha)


    def _arrange_batch(self, batch):
        batch_input, *global_params = arrange_sparse_minkowski(batch, self.device)
        batch_module_to_event, batch_module_pos = batch['module_to_event'], batch['module_pos']
        target = arrange_truth(batch)
        return batch_input, *global_params, batch_module_to_event, batch_module_pos, target


    def compute_losses(self, batch_output, target, target_charge, mask_bool):
        # pred
        out_primlepton = batch_output['out_primlepton'].F
        out_seg = batch_output['out_seg'].F
        out_charge = batch_output['out_charge'].F[mask_bool]
        out_iscc = batch_output['out_iscc'].squeeze(-1)

        # true
        targ_primlepton = target['primlepton_labels']
        targ_seg = target['seg_labels']
        targ_charge = target_charge[mask_bool]
        targ_iscc = target['is_cc']

        # losses
        loss_primlepton = self.loss_primlepton(out_primlepton, targ_primlepton)
        loss_seg = self.loss_seg(out_seg, targ_seg)
        loss_charge = self.loss_charge(out_charge, targ_charge)
        loss_iscc = self.loss_iscc(out_iscc, targ_iscc)

        part_losses = {'primlepton': loss_primlepton,
                       'seg': loss_seg,
                       'charge': loss_charge,
                       'iscc': 0.0001*loss_iscc,
                       }
        total_loss = sum(part_losses.values())

        return total_loss, part_losses


    def create_mask(
        self,
        batch_input,
        batch_module_to_event,
        p_mask: float = 0.3,
        p_module_mask: float = 0.8,
        mask_frac: float = 0.75,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Create a mixed mask over voxels:
        - With probability p_module_mask per event (only if the event
          has ≥2 non-empty modules), mask between 1 and up to
          mask_frac of its modules (never all of them).
        - Otherwise, randomly mask up to p_mask fraction of that event’s voxels.
        """
        N = batch_input.F.shape[0]
        device = batch_input.F.device
        voxel_mod = batch_input.C[:, 0].long()
        voxel_evt = batch_module_to_event[voxel_mod]
        mask = torch.zeros(N, dtype=torch.bool, device=device)
    
        evt_to_mods = {}
        for m, e in enumerate(batch_module_to_event.tolist()):
            evt_to_mods.setdefault(e, []).append(m)
    
        for e, mods in evt_to_mods.items():
            if len(mods) < 2:
                continue
            if torch.rand(1, generator=generator, device=device).item() < p_module_mask:
                k_max = min(int(len(mods) * mask_frac), len(mods) - 1)
                num_to_mask = torch.randint(1, k_max + 1, (1,), generator=generator, device=device).item()
                perm = torch.randperm(len(mods), generator=generator, device=device)
                for i in perm[:num_to_mask]:
                    mask[voxel_mod == mods[i]] = True
    
        for e in voxel_evt.unique():
            evt_idx = (voxel_evt == e).nonzero(as_tuple=True)[0]
            if mask[evt_idx].any():
                # only skip if that event is already masked
                continue
            frac = torch.rand(1, generator=generator, device=device).item() * p_mask
            perm = torch.rand(evt_idx.numel(), generator=generator, device=device)
            mask[evt_idx[perm < frac]] = True
    
        return mask
        
    
    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, *batch_input_global, batch_module_to_event, batch_module_pos, target = self._arrange_batch(batch)
        orig_charge = batch_input.F.clone()
        
        # create the mask
        if self.training:
            mask_bool = self.create_mask(batch_input, batch_module_to_event)
        else:
            # deterministic for validation
            gen = torch.Generator(device=batch_input.F.device)
            gen.manual_seed(0)
            mask_bool = self.create_mask(batch_input, batch_module_to_event, generator=gen)

        # Forward pass
        batch_output = self.forward(
            batch_input, batch_input_global, batch_module_to_event, batch_module_pos, mask_bool, self.global_alpha)
        loss, part_losses = self.compute_losses(batch_output, target, orig_charge, mask_bool)
  
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
        self.log(f"global_alpha", self.global_alpha, batch_size=batch_size, prog_bar=False, sync_dist=True)

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

