"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 2: classification and regression tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from utils import MAPE, CosineLoss, SphericalAngularLoss, StableLogCoshLoss, arrange_sparse_minkowski, argsort_sparse_tensor, arrange_truth, argsort_coords, CustomLambdaLR, CombinedScheduler


class SparseEncLightningModel(pl.LightningModule):
    def __init__(self, model, args):
        super(SparseEncLightningModel, self).__init__()

        self.model = model
        self.loss_flavour = nn.CrossEntropyLoss()
        self.loss_evis = [nn.MSELoss(), MAPE(from_log_scale=True)]
        self.loss_ptmiss = [nn.MSELoss(), MAPE(from_log_scale=True)]
        self.loss_lepton_momentum_mag = [nn.MSELoss(), MAPE(from_log_scale=True)]
        self.loss_lepton_momentum_dir = SphericalAngularLoss()
        self.loss_jet_momentum_mag = [nn.MSELoss(), MAPE(from_log_scale=True)]
        self.loss_jet_momentum_dir = SphericalAngularLoss()
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

    
    def forward(self, x, x_glob, module_to_event, module_pos):
        return self.model(x, x_glob, module_to_event, module_pos)


    def _arrange_batch(self, batch):
        batch_input, batch_input_global = arrange_sparse_minkowski(batch, self.device)
        batch_module_to_event, batch_module_pos = batch['module_to_event'], batch['module_pos']
        target = arrange_truth(batch)
        return batch_input, batch_input_global, batch_module_to_event, batch_module_pos, target


    def compute_losses(self, batch_output, target):
        # pred
        out_flavour = batch_output['out_flavour']
        out_e_vis = batch_output['out_e_vis'].view(-1)
        out_pt_miss = batch_output['out_pt_miss'].view(-1)
        out_lepton_momentum_mag = batch_output['out_lepton_momentum_mag']
        out_lepton_momentum_dir = batch_output['out_lepton_momentum_dir']
        out_jet_momentum_mag = batch_output['out_jet_momentum_mag']
        out_jet_momentum_dir = batch_output['out_jet_momentum_dir']

        # true
        targ_flavour = target['flavour_label']
        targ_e_vis = target['e_vis']
        targ_pt_miss = target['pt_miss']
        targ_lepton_momentum_mag = target['out_lepton_momentum_mag']
        targ_lepton_momentum_dir = target['out_lepton_momentum_dir']
        targ_jet_momentum_mag = target['jet_momentum_mag']
        targ_jet_momentum_dir = target['jet_momentum_dir']

        # Mask primary lepton momentum for NC events
        mask = targ_lepton_momentum_mag.squeeze() > 0
        out_lepton_momentum_mag = out_lepton_momentum_mag[mask]
        out_lepton_momentum_dir = out_lepton_momentum_dir[mask]
        targ_lepton_momentum_mag = targ_lepton_momentum_mag[mask]
        targ_lepton_momentum_dir = targ_lepton_momentum_dir[mask]

        # losses
        loss_flavour = self.loss_flavour(out_flavour, targ_flavour)
        loss_e_vis = self.loss_evis[0](out_e_vis, targ_e_vis)
        loss_e_vis += 0.1 * self.loss_evis[1](out_e_vis, targ_e_vis)
        loss_pt_miss = self.loss_ptmiss[0](out_pt_miss, targ_pt_miss)
        loss_pt_miss += 0.1 * self.loss_ptmiss[1](out_pt_miss, targ_pt_miss)
        loss_lepton_momentum_mag = self.loss_lepton_momentum_mag[0](out_lepton_momentum_mag, targ_lepton_momentum_mag)
        loss_lepton_momentum_mag += 0.1 * self.loss_lepton_momentum_mag[1](out_lepton_momentum_mag, targ_lepton_momentum_mag)
        loss_lepton_momentum_dir = self.loss_lepton_momentum_dir(out_lepton_momentum_dir, targ_lepton_momentum_dir)
        loss_jet_momentum_mag = self.loss_jet_momentum_mag[0](out_jet_momentum_mag, targ_jet_momentum_mag)
        loss_jet_momentum_mag += 0.1 * self.loss_jet_momentum_mag[1](out_jet_momentum_mag, targ_jet_momentum_mag)
        loss_jet_momentum_dir = self.loss_jet_momentum_dir(out_jet_momentum_dir, targ_jet_momentum_dir)
        
        part_losses = {'flavour': loss_flavour,
                       'e_vis': loss_e_vis,
                       'pt_miss': loss_pt_miss,
                       'lepton_momentum_mag': loss_lepton_momentum_mag,
                       'lepton_momentum_dir': loss_lepton_momentum_dir,
                       'jet_momentum_mag': loss_jet_momentum_mag,
                       'jet_momentum_dir': loss_jet_momentum_dir,
                       }
        total_loss = loss_flavour + loss_e_vis + loss_pt_miss + loss_lepton_momentum_mag + loss_lepton_momentum_dir + loss_jet_momentum_mag + loss_jet_momentum_dir

        return total_loss, part_losses

    
    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, batch_input_global, batch_module_to_event, batch_module_pos, target = self._arrange_batch(batch)

        # Forward pass
        batch_output = self.forward(batch_input, batch_input_global, batch_module_to_event, batch_module_pos)
        loss, part_losses = self.compute_losses(batch_output, target)
  
        # Retrieve current learning rate
        lr = self.optimizers().param_groups[0]['lr']

        return loss, part_losses, batch_size, lr


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        if torch.isnan(loss):
            return None

        self.log(f"loss/train_total", loss.item(), batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
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
            #self.model.parameters(),
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

