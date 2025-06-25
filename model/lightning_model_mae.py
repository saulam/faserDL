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
import pytorch_lightning as pl
from utils import (
    focal_loss, dice_loss, arrange_sparse_minkowski, argsort_sparse_tensor, 
    arrange_truth, argsort_coords, CustomLambdaLR, CombinedScheduler
)
from functools import partial
from packaging import version
from typing import Optional
from collections import defaultdict



class SparseMAELightningModel(pl.LightningModule):
    def __init__(self, model, metadata, args):
        super(SparseMAELightningModel, self).__init__()

        self.model = model
        self.loss_iscc = nn.BCEWithLogitsLoss()
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


    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    
    def forward(self, 
                x, 
                x_glob,
                module_to_event, 
                module_pos, 
                mask_bool,
               ):
        return self.model(x, x_glob, module_to_event, module_pos, mask_bool)


    def _arrange_batch(self, batch):
        batch_input, *global_params = arrange_sparse_minkowski(batch, self.device)
        batch_module_to_event, batch_module_pos = batch['module_to_event'], batch['module_pos']
        target = arrange_truth(batch)
        return batch_input, *global_params, batch_module_to_event, batch_module_pos, target


    def compute_losses(self, batch_input, batch_output, target, mask_bool):
        # true
        coords = batch_input.C
        feats = batch_input.F.squeeze(-1)
        targ_iscc = target['is_cc']

        # keep voxels from masked modules
        orig_mod_ids = coords[:, 0].long()
        masked_mod_ids = mask_bool.nonzero(as_tuple=True)[0]
        keep = torch.isin(orig_mod_ids, masked_mod_ids) 
        coords = coords[keep]
        feats = feats[keep]

        # pred
        out_charge = batch_output['out_charge']
        out_iscc = batch_output['out_iscc'].squeeze(-1)

        # compute c = standardized(0) = (0 - mu) / sigma
        if self.standardize_input == "z-score":
            q_meta = "q"
            if self.preprocessing_input == "sqrt":
                q_meta += "_sqrt"
            elif self.preprocessing_input == "log":
                q_meta += "_log"
            mu, sigma = self.metadata[q_meta]["mean"], self.metadata[q_meta]["std"]
            c = (- mu) / sigma
        else:
            c = 0
        c = torch.as_tensor(c, device=out_charge.device, dtype=out_charge.dtype)
        
        # sum‐of‐squares over every voxel (all zero+nonzero)
        total_ss = (out_charge - c).pow(2).sum()

        # gather predictions at nonzero locations
        local_b_idx = torch.searchsorted(masked_mod_ids, coords[:, 0].long())
        x,y,z = coords[:,1], coords[:,2], coords[:,3]
        pred_feats = out_charge[local_b_idx, x, y, z]

        # swap out pred^2 for (pred−true)^2 at those nonzero locations
        nonzero_ss_pred2 = (pred_feats - c).pow(2).sum()
        nonzero_ss_true  = (pred_feats - feats).pow(2).sum()

        # losses
        loss_charge = (total_ss - nonzero_ss_pred2 + nonzero_ss_true) / out_charge.numel()
        loss_iscc = self.loss_iscc(out_iscc, targ_iscc)

        part_losses = {'charge': loss_charge,
                       'iscc': 0.0001*loss_iscc,
                       }
        total_loss = sum(part_losses.values())

        return total_loss, part_losses


    def create_mask(
        self,
        batch_module_to_event: torch.Tensor,
        p_module_mask: float = 0.8,
        mask_frac: float = 0.75,
        generator: Optional[torch.Generator] = None,
    ) -> torch.BoolTensor:
        """
        Decide which modules to fully mask.
    
        Args:
            batch_module_to_event (Tensor[M]): maps each module index to its event index.
            p_module_mask: probability to apply module‐masking to an eligible event.
            mask_frac: relative cap on number of modules to mask per event (fraction of its modules).
            generator: optional torch.Generator for reproducible draws.
    
        Returns:
            module_mask (BoolTensor[M]): True for modules selected to be fully masked.
        """
        M = batch_module_to_event.size(0)
        device = batch_module_to_event.device
    
        module_mask = torch.zeros(M, dtype=torch.bool, device=device)
    
        # group modules by event
        evt_to_mods = defaultdict(list)
        for m, e in enumerate(batch_module_to_event.tolist()):
            evt_to_mods[e].append(m)
    
        # for each event with ≥2 modules, maybe mask some of them
        for e, mods in evt_to_mods.items():
            if len(mods) < 2:
                continue

            # compute the per-event cap (never all modules)
            k_max = min(
                int(len(mods) * mask_frac),  # fraction-based cap
                len(mods) - 1                # leave at least one module unmasked
            )
            # choose how many to mask (1..k_max)
            num_to_mask = torch.randint(
                1, k_max + 1,
                (1,),
                generator=generator,
                device=device
            ).item()
            # randomly select that many distinct modules
            perm = torch.randperm(len(mods), generator=generator, device=device)
            chosen = [mods[i] for i in perm[:num_to_mask].tolist()]
            module_mask[chosen] = True
    
        return module_mask
        
    
    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, *batch_input_global, batch_module_to_event, batch_module_pos, target = self._arrange_batch(batch)
        orig_charge = batch_input.F.clone()
        
        # create the mask
        if self.training:
            mask_bool = self.create_mask(batch_module_to_event)
        else:
            # deterministic for validation
            gen = torch.Generator(device=batch_input.F.device)
            gen.manual_seed(0)
            mask_bool = self.create_mask(batch_module_to_event, generator=gen)

        # Forward pass
        batch_output = self.forward(
            batch_input, batch_input_global, batch_module_to_event, batch_module_pos, mask_bool)
        loss, part_losses = self.compute_losses(batch_input, batch_output, target, mask_bool)
  
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

