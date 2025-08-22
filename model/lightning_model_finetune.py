"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 2 (transfer learning from stage 1): classification and regression tasks.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
from torch_ema import ExponentialMovingAverage
from utils import (
    param_groups_lrd, KinematicsMultiTaskLoss,
    arrange_sparse_minkowski, arrange_truth, 
    CustomLambdaLR, CombinedScheduler, weighted_loss
)


class ViTFineTuner(pl.LightningModule):
    def __init__(self, model, args):
        super(ViTFineTuner, self).__init__()
        self.model = model
        self.preprocessing_output = args.preprocessing_output
        stats = model.metadata

        def get_or(d: dict, key: str, default: Any) -> Any:
            return d[key] if key in d else default
        
        # Loss functions
        self.loss_iscc = nn.BCEWithLogitsLoss()
        self.loss_flavour = nn.CrossEntropyLoss()
        self.loss_charm = nn.BCEWithLogitsLoss()
        self.crit = KinematicsMultiTaskLoss(
            s_vis_xyz=stats["vis"]["s_xyz"],  s_vis_mag=stats["vis"]["s_mag"],
            s_lep_xyz=stats["lep"]["s_xyz"],  s_lep_mag=stats["lep"]["s_mag"],
            s_jet_xyz=get_or(stats.get("jet_loss_scales", {}), "s_xyz", stats["vis"]["s_xyz"]),
            s_jet_mag=get_or(stats.get("jet_loss_scales", {}), "s_mag", stats["vis"]["s_mag"]),
            tau_ptmiss_cc=stats.get("vis_tau_ptmiss_cc", stats["vis"]["tau_ptmiss"]),
            tau_ptmiss_nc=stats.get("vis_tau_ptmiss_nc", stats["vis"]["tau_ptmiss"]),
            tau_evis_cc=stats.get("vis_tau_evis_cc", stats["vis"]["tau_evis"]),
            tau_evis_nc=stats.get("vis_tau_evis_nc", stats["vis"]["tau_evis"]),
            zero_attractor_w=0.1,
            jet_aux_w=0.2,           # 0.0 to disable aux jet loss
            latent_prior_w=1e-3,     # tiny; only used if we pass latents
            huber_delta=1.0,
            lam_mag=1.0, lam_dir=1.0
        )

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.log_sigma_iscc = nn.Parameter(torch.zeros(()))
        self.log_sigma_flavour = nn.Parameter(torch.zeros(()))
        self.log_sigma_charm = nn.Parameter(torch.zeros(()))
        self.log_sigma_vis = nn.Parameter(torch.zeros(()))
        self.log_sigma_lep = nn.Parameter(torch.zeros(()))
        self.log_sigma_ptmiss = nn.Parameter(torch.zeros(()))
        self.log_sigma_evis = nn.Parameter(torch.zeros(()))
        self.log_sigma_jet = nn.Parameter(torch.zeros(()))
        self._uncertainty_params = {
            "is_cc":   self.log_sigma_iscc,
            "flavour": self.log_sigma_flavour,
            "charm":   self.log_sigma_charm,
            "vis":     self.log_sigma_vis,
            "lep":     self.log_sigma_lep,
            "ptmiss":  self.log_sigma_ptmiss,
            "evis":    self.log_sigma_evis,
            "jet":     self.log_sigma_jet,
        }
        
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
        # ---- predicted outputs -----
        out_iscc    = batch_output['out_iscc'].squeeze()
        out_flavour = batch_output['out_flavour']
        out_charm   = batch_output['out_charm'].squeeze()
        out_vis     = batch_output['out_vis']
        out_lep     = batch_output['out_lep']

        # ---- true outputs -----
        targ_iscc                = target['is_cc']
        targ_flavour             = target['flavour_label']
        targ_charm               = target['charm']
        targ_vis_sp_momentum     = target['vis_sp_momentum']
        targ_lepton_momentum     = target['out_lepton_momentum']

        outs = self.crit(
            p_vis_hat=out_vis["p_cart"],
            p_lep_hat=out_lep["p_cart"],
            p_vis_true=targ_vis_sp_momentum,
            p_lep_true=targ_lepton_momentum,
            is_cc=targ_iscc>0.5,
            p_jet_true=targ_vis_sp_momentum - targ_lepton_momentum,  # if aux jet loss
            p_jet_hat=None,                                          # derive from vis/lepton
            vis_latents=out_vis.get("latents"),
            lep_latents=out_lep.get("latents"),
        )

        # losses
        m_cc         = (targ_iscc > 0.5).to(out_vis["p_cart"].dtype).view(-1)
        loss_iscc    = self.loss_iscc(out_iscc, targ_iscc)
        loss_flavour = self.loss_flavour(out_flavour[m_cc.bool()], targ_flavour[m_cc.bool()])
        loss_charm   = self.loss_charm(out_charm, targ_charm)
        loss_vis     = outs["L_vis"]
        loss_lep     = outs["L_lep_cc"] + outs["L_lep_zero"]
        loss_ptmiss  = outs["L_ptmiss"]
        loss_evis    = outs["L_evis"]
        loss_jet     = outs["jet_aux_weight"] * outs["L_jet"]

        part_losses = {
            'is_cc': loss_iscc,
            'flavour': loss_flavour,
            'charm': loss_charm,
            'vis_comp': outs["L_vis_comp"].mean(),
            'vis_mag': outs["L_vis_mag"].mean(),
            'vis_dir': outs["L_vis_dir"].mean(),
            'vis': outs["L_vis"].mean(),
            'lep_comp': outs["L_lep_comp"].mean(),
            'lep_mag': outs["L_lep_mag"].mean(),
            'lep_dir': outs["L_lep_dir"].mean(),
            'lep_cc': (outs["L_lep_cc"][m_cc>0.5].mean() if (m_cc>0.5).any() else torch.tensor(0., device=self.device)),
            'lep_zero': outs["L_lep_zero"].mean(),
            'evis': outs["L_evis"].mean(),
            'pt_miss': loss_ptmiss.mean(),
            'jet_comp': outs["L_jet_comp"].mean(),
            'jet_mag': outs["L_jet_mag"].mean(),
            'jet_dir': outs["L_jet_dir"].mean(),
            'jet': (outs["L_jet"][m_cc>0.5].mean() if (outs["jet_aux_weight"]>0 and (m_cc>0.5).any()) else torch.tensor(0., device=self.device)),            
        }
        total_loss = (
            weighted_loss(loss_iscc,    self.log_sigma_iscc) +
            weighted_loss(loss_flavour, self.log_sigma_flavour) +
            weighted_loss(loss_charm,   self.log_sigma_charm) +
            weighted_loss(loss_vis,     self.log_sigma_vis) +
            weighted_loss(loss_lep,     self.log_sigma_lep) +
            weighted_loss(loss_ptmiss,  self.log_sigma_ptmiss) +
            weighted_loss(loss_evis,    self.log_sigma_evis) +
            weighted_loss(loss_jet,     self.log_sigma_jet)
        ).mean()

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
            'params': list(self._uncertainty_params.values()),
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

