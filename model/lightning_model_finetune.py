"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 2 (transfer learning from stage 1): classification and regression tasks.
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
from utils import param_groups_lrd, MAPE, SphericalAngularLoss, arrange_sparse_minkowski, arrange_truth, CustomLambdaLR, CombinedScheduler


class ViTFineTuner(pl.LightningModule):
    def __init__(self, model, args):
        super(ViTFineTuner, self).__init__()
        self.model = model
        
        # Loss functions
        self.loss_flavour = nn.CrossEntropyLoss()
        self.loss_charm = nn.BCEWithLogitsLoss()
        self.loss_evis = MAPE(preprocessing=args.preprocessing_output)
        self.loss_ptmiss = MAPE(preprocessing=args.preprocessing_output)
        self.loss_lepton_momentum_mag = MAPE(preprocessing=args.preprocessing_output)
        self.loss_lepton_momentum_dir = SphericalAngularLoss()
        self.loss_jet_momentum_mag = MAPE(preprocessing=args.preprocessing_output)
        self.loss_jet_momentum_dir = SphericalAngularLoss()

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.log_sigma_flavour = nn.Parameter(torch.zeros(()))
        self.log_sigma_charm = nn.Parameter(torch.zeros(()))
        self.log_sigma_e_vis = nn.Parameter(torch.zeros(()))
        self.log_sigma_pt_miss = nn.Parameter(torch.zeros(()))
        self.log_sigma_lepton_momentum_mag = nn.Parameter(torch.zeros(()))
        self.log_sigma_lepton_momentum_dir = nn.Parameter(torch.zeros(()))
        self.log_sigma_jet_momentum_mag = nn.Parameter(torch.zeros(()))
        self.log_sigma_jet_momentum_dir = nn.Parameter(torch.zeros(()))
        self._uncertainty_params = [
            self.log_sigma_flavour,
            self.log_sigma_charm,
            self.log_sigma_e_vis,
            self.log_sigma_pt_miss,
            self.log_sigma_lepton_momentum_mag,
            self.log_sigma_lepton_momentum_dir,
            self.log_sigma_jet_momentum_mag,
            self.log_sigma_jet_momentum_dir,
        ]
        self.log_sigma_params = set(self._uncertainty_params)
        
        self.warmup_steps = args.warmup_steps
        self.start_cosine_step = args.start_cosine_step
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.layer_decay = args.layer_decay
        self.ema_decay = args.ema_decay
        self.ema = None

        # Optimiser params
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps

    
    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    
    def on_load_checkpoint(self, checkpoint):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=self.ema_decay,
            )
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        

    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups
        if not self.trainer.is_global_zero:
            # only keep EMA on rank 0 (for DDP)
            return
        if self.ema is None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=self.ema_decay,
            )
            
    
    def on_before_zero_grad(self, optimizer):
        # called after optimizer.step() but before optimizer.zero_grad()
        # so this is the perfect spot to capture the freshly updated weights
        if self.ema is not None:
            self.ema.update()
        
    
    def forward(self, x, x_glob):
        return self.model(x, x_glob)


    def _arrange_batch(self, batch):
        batch_input, *batch_input_global = arrange_sparse_minkowski(batch, self.device)
        target = arrange_truth(batch)
        return batch_input, *batch_input_global, target


    def compute_losses(self, batch_output, target):
        # pred
        out_flavour = batch_output['out_flavour']
        out_charm = batch_output['out_charm'].squeeze()
        out_e_vis = batch_output['out_e_vis'].squeeze()
        out_pt_miss = batch_output['out_pt_miss'].squeeze()
        out_lepton_momentum_mag = batch_output['out_lepton_momentum_mag'].squeeze()
        out_lepton_momentum_dir = batch_output['out_lepton_momentum_dir']
        out_jet_momentum_mag = batch_output['out_jet_momentum_mag'].squeeze()
        out_jet_momentum_dir = batch_output['out_jet_momentum_dir']

        # true
        targ_flavour = target['flavour_label']
        targ_charm = target['charm']
        targ_e_vis = target['e_vis']
        targ_pt_miss = target['pt_miss']
        targ_lepton_momentum_mag = target['out_lepton_momentum_mag']
        targ_lepton_momentum_dir = target['out_lepton_momentum_dir']
        targ_jet_momentum_mag = target['jet_momentum_mag']
        targ_jet_momentum_dir = target['jet_momentum_dir']

        # CC
        mask_cc = targ_flavour < 3 if targ_flavour.ndim==1 else targ_flavour.argmax(dim=1) < 3
        out_lepton_momentum_mag = out_lepton_momentum_mag[mask_cc]
        out_lepton_momentum_dir = out_lepton_momentum_dir[mask_cc]
        targ_lepton_momentum_mag = targ_lepton_momentum_mag[mask_cc]
        targ_lepton_momentum_dir = targ_lepton_momentum_dir[mask_cc]
        
        # NC
        mask_nc = ~mask_cc
        out_pt_miss = out_pt_miss[mask_nc]
        targ_pt_miss = targ_pt_miss[mask_nc]
        
        def weighted_regression(L, s):
            # (1/2) e^(–2s) L + s
            return 0.5 * torch.exp(-2*s) * L + s
        
        def weighted_classification(L, s):
            # different for classification since it's already a likelihood
            # exp(-s) * L + s
            return torch.exp(-1*s) * L + s

        # losses
        loss_flavour = self.loss_flavour(out_flavour, targ_flavour)
        loss_charm = self.loss_charm(out_charm, targ_charm)
        loss_e_vis = self.loss_evis(out_e_vis, targ_e_vis)
        loss_pt_miss = self.loss_ptmiss(out_pt_miss, targ_pt_miss)
        loss_lepton_momentum_mag = self.loss_lepton_momentum_mag(out_lepton_momentum_mag, targ_lepton_momentum_mag)
        loss_lepton_momentum_dir = self.loss_lepton_momentum_dir(out_lepton_momentum_dir, targ_lepton_momentum_dir)
        loss_jet_momentum_mag = self.loss_jet_momentum_mag(out_jet_momentum_mag, targ_jet_momentum_mag)
        loss_jet_momentum_dir = self.loss_jet_momentum_dir(out_jet_momentum_dir, targ_jet_momentum_dir)
        
        part_losses = {'flavour': loss_flavour,
                       'charm': loss_charm,
                       'e_vis': loss_e_vis,
                       'pt_miss': loss_pt_miss,
                       'lepton_momentum_mag': loss_lepton_momentum_mag,
                       'lepton_momentum_dir': loss_lepton_momentum_dir,
                       'jet_momentum_mag': loss_jet_momentum_mag,
                       'jet_momentum_dir': loss_jet_momentum_dir,
                       }
        total_loss = ( weighted_classification(loss_flavour, self.log_sigma_flavour)
                     + weighted_classification(loss_charm, self.log_sigma_charm)
                     + weighted_regression(loss_e_vis, self.log_sigma_e_vis)
                     + weighted_regression(loss_pt_miss, self.log_sigma_pt_miss)
                     + weighted_regression(loss_lepton_momentum_mag, self.log_sigma_lepton_momentum_mag)
                     + weighted_regression(loss_lepton_momentum_dir, self.log_sigma_lepton_momentum_dir)
                     + weighted_regression(loss_jet_momentum_mag, self.log_sigma_jet_momentum_mag)
                     + weighted_regression(loss_jet_momentum_dir, self.log_sigma_jet_momentum_dir)
                   )

        return total_loss, part_losses

    
    def get_lr_for_module(self, module: torch.nn.Module) -> float:
        """
        Scan through all optimizer param_groups and return the lr for the one
        that holds any parameter of `module`.
        """
        opt = self.optimizers()
        param_ids = {id(p) for p in module.parameters()}
        for pg in opt.param_groups:
            if any(id(p) in param_ids for p in pg['params']):
                return pg['lr']
        raise ValueError(f"No param_group found for module {module}")

    
    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, *batch_input_global, target = self._arrange_batch(batch)

        # Forward pass
        batch_output = self.forward(batch_input, batch_input_global)
        loss, part_losses = self.compute_losses(batch_output, target)
  
        # Retrieve current learning rate
        lr = self.get_lr_for_module(self.model.heads["flavour"])

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
        
        # log the actual sigmas (exp(log_sigma))
        for name, param in self.named_parameters():
            if 'log_sigma' in name:
                log_sigma = -param if any(k in name for k in ['flavour', 'charm']) else -2 * param
                self.log(f'uncertainty/{name}', torch.exp(log_sigma), batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/val_total", loss.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("loss/val_{}".format(key), value.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss


    def configure_optimizers(self):
        """Configure optimiser with LR groups, plus warmup & cosine schedulers."""
        param_groups = param_groups_lrd(
            self.model, 
            weight_decay=self.weight_decay,
            layer_decay=self.layer_decay)

        for param_group in param_groups:
            lr_scale = param_group.pop("lr_scale", 1.0)
            param_group["lr"] = self.lr * lr_scale

        # group uncertainty params
        param_groups.append({
            'params': list(self.log_sigma_params),
            'lr': self.lr * 0.1,
            'weight_decay': 0.0,
        })
        
        optimizer = torch.optim.AdamW(
            param_groups, betas=self.betas, eps=self.eps
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
                eta_min=0.
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

