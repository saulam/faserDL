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
    def __init__(self, model, metadata, args):
        super(SparseMAELightningModel, self).__init__()

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
        targ_charge = batch_input.F
        targ_primlepton = target['primlepton_labels']
        targ_seg = target['seg_labels']
        targ_iscc = target['is_cc']

        # keep voxels from masked modules only
        orig_mod_ids = coords[:, 0].long()
        masked_mod_ids = mask_bool.nonzero(as_tuple=True)[0]
        keep = torch.isin(orig_mod_ids, masked_mod_ids) 
        coords = coords[keep]
        targ_charge = targ_charge[keep]
        targ_primlepton = targ_primlepton[keep]
        targ_seg = targ_seg[keep]

        # pred
        out_occupancy = batch_output['out_occupancy']
        out_charge = batch_output['out_charge']
        out_primlepton = batch_output['out_primlepton']
        out_seg = batch_output['out_seg']
        out_iscc = batch_output['out_iscc'].squeeze(-1)

        # gather predictions at nonzero locations
        local_b_idx = torch.searchsorted(masked_mod_ids, coords[:, 0].long())
        x, y, z = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
        out_charge = out_charge[local_b_idx, :, x, y, z]
        out_primlepton = out_primlepton[local_b_idx, :, x, y, z]
        out_seg = out_seg[local_b_idx, :, x, y, z]
        targ_occupancy = torch.zeros_like(out_occupancy)
        targ_occupancy[local_b_idx, :, x, y, z] = 1.

        # weight for occupancy loss (using log of original charge)
        q_meta = "q"
        targ_charge_copy = targ_charge.clone()
        if self.preprocessing_input == "sqrt":
            q_meta += "_sqrt"
        elif self.preprocessing_input == "log":
            q_meta += "_log1p"
        if self.standardize_input is not None:
            stats = self.metadata[q_meta]
            if self.standardize_input == "z-score":
                targ_charge_copy = targ_charge_copy * stats["std"] + stats["mean"]
            elif self.standardize_input == "unit-var":
                targ_charge_copy = targ_charge_copy * stats["std"]
            else:
                rng = stats["max"] - stats["min"]
                targ_charge_copy = targ_charge_copy * rng + stats["min"]
        if self.preprocessing_input == "sqrt":
            targ_charge_copy = targ_charge_copy ** 2
        targ_charge_copy = (torch.log1p(targ_charge_copy) + 1)
        weight_occupancy = torch.ones_like(targ_occupancy, dtype=targ_charge_copy.dtype)
        weight_occupancy[local_b_idx, :, x, y, z] = targ_charge_copy
        
        # losses
        loss_occupancy = F.binary_cross_entropy_with_logits(out_occupancy, targ_occupancy, weight=weight_occupancy)
        loss_charge = self.loss_charge(out_charge, targ_charge)
        loss_primlepton = self.loss_primlepton(out_primlepton, targ_primlepton)
        loss_seg = self.loss_seg(out_seg, targ_seg)
        loss_iscc = self.loss_iscc(out_iscc, targ_iscc)

        part_losses = {'occupancy': loss_occupancy,
                       'charge': loss_charge,
                       'primlepton': loss_primlepton,
                       'seg': loss_seg,
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
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                p.ndim <= 1
                or name.endswith(".bias")
                or "emb" in name
                or "cls_mod" in name
                or "cls_evt" in name
                or "mask_mod_emb" in name
                or name.endswith((".gamma", ".beta"))
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

