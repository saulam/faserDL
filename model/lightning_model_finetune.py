"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 2 (transfer learning from stage 1): classification and regression tasks.
"""

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
from torch_ema import ExponentialMovingAverage
from utils import (
    param_groups_lrd, KinematicsMultiTaskLoss, CylindricalConsistencyLoss,
    arrange_input, arrange_truth,
    CustomLambdaLR, CombinedScheduler, weighted_loss, move_obj
)


class ViTFineTuner(pl.LightningModule):
    def __init__(self, model, args):
        super(ViTFineTuner, self).__init__()
        self.model = model
        stats = model.metadata

        def get_or(d: dict, key: str, default: Any) -> Any:
            return d[key] if key in d else default
        
        # Loss functions
        self.loss_flavour = nn.CrossEntropyLoss(reduction='none')
        self.loss_charm = nn.CrossEntropyLoss(reduction='none')
        self.crit = KinematicsMultiTaskLoss(
            stats=stats,
            huber_delta=1.0, lam_dir_xy=1.0, lam_dir_3d=0.0,
            lep_nc_zero_w=0.05, latent_prior_w=0.0,
            enforce_nonneg_truth_pz=True, decouple_radial=False,
        )
        self.crit = CylindricalConsistencyLoss(stats=stats)

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.kendall_w_min = 0.3
        self.kendall_w_max = 5.0
        self.log_sigma_flavour   = nn.Parameter(torch.zeros(()))
        self.log_sigma_charm     = nn.Parameter(torch.zeros(()))
        self.log_sigma_vis_geom  = nn.Parameter(torch.zeros(()))
        #self.log_sigma_vis_pt    = nn.Parameter(torch.zeros(()))
        #self.log_sigma_vis_mag   = nn.Parameter(torch.zeros(()))
        self.log_sigma_jet_geom  = nn.Parameter(torch.zeros(()))
        #self.log_sigma_jet_pt    = nn.Parameter(torch.zeros(()))
        #self.log_sigma_jet_mag   = nn.Parameter(torch.zeros(()))
        self.log_sigma_lep_geom  = nn.Parameter(torch.zeros(()))
        #self.log_sigma_lep_pt    = nn.Parameter(torch.zeros(()))
        #self.log_sigma_lep_mag   = nn.Parameter(torch.zeros(()))
        self.log_sigma_vertex    = nn.Parameter(torch.zeros(()))
        self._uncertainty_params = {
            "flavour":     self.log_sigma_flavour,
            "charm":       self.log_sigma_charm,
            "vis_geom":    self.log_sigma_vis_geom,
            #"vis_pt":      self.log_sigma_vis_pt,
            #"vis_mag":     self.log_sigma_vis_mag,
            "jet_geom":    self.log_sigma_jet_geom,
            #"jet_pt":      self.log_sigma_jet_pt,
            #"jet_mag":     self.log_sigma_jet_mag,
            "lep_geom":    self.log_sigma_lep_geom,
            #"lep_pt":      self.log_sigma_lep_pt,
            #"lep_mag":     self.log_sigma_lep_mag,
            "vertex":      self.log_sigma_vertex,
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

        def _u_for_w(w, w_min, w_max, eps=1e-6):
            # Map desired initial weight w into u so that:
            # w = w_min + (w_max - w_min) * sigmoid(u)
            p = (w - w_min) / (w_max - w_min)
            p = max(eps, min(1.0 - eps, float(p)))
            return math.log(p / (1.0 - p))
        w0 = 1.0  # desired initial Kendall weight
        u0 = _u_for_w(w0, self.kendall_w_min, self.kendall_w_max)
        for p in self._uncertainty_params.values():
            p.data.fill_(u0)

    
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return move_obj(batch, device)

    
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


    def on_train_epoch_start(self):
        raw = getattr(self.trainer, "train_dataloader", None)
        if raw is None:
            raw = getattr(self.trainer, "train_dataloaders", None)
        if raw is None:
            return

        loaders = raw if isinstance(raw, (list, tuple)) else [raw]
        for dl in loaders:
            ds = getattr(dl, "dataset", None)
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(self.trainer.current_epoch)

        lr = self.get_lr_for_module(self.model.heads["flavour"])
        self.log(f"lr", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            
    
    def on_before_zero_grad(self, optimizer):
        # called after optimizer.step() but before optimizer.zero_grad()
        # so this is the perfect spot to capture the freshly updated weights
        if self.ema is not None:
            self.ema.update()
        
    
    def forward(self, x, x_glob):
        return self.model(x, x_glob)


    def _arrange_batch(self, batch):
        batch_input, *global_params = arrange_input(batch)
        target = arrange_truth(batch)
        return batch_input, *global_params, target


    def compute_losses(self, batch_output, target):
        # predicted outputs
        out_flavour = batch_output['out_flavour']
        out_charm   = batch_output['out_charm']
        out_vis     = batch_output['out_vis']
        out_jet     = batch_output['out_jet']
        out_vertex  = batch_output['out_vertex']

        # true outputs
        targ_iscc            = target['is_cc']
        targ_flavour         = target['flavour_label']
        targ_charm           = target['charm_label']
        targ_vis_sp_momentum = target['vis_sp_momentum']
        targ_jet_momentum    = target['jet_momentum']
        targ_primary_vertex  = target['primary_vertex']

        '''
        outs = self.crit(
            p_vis_hat=out_vis["p_cart"],
            p_jet_hat=out_jet["p_cart"],
            p_vis_true=targ_vis_sp_momentum,
            p_jet_true=targ_jet_momentum,
            is_cc=targ_iscc>0.5,
            vis_latents=out_vis.get("latents"),
            jet_latents=out_jet.get("latents"),
        )
        '''
        outs = self.crit(
            pred_vis=out_vis,
            pred_jet=out_jet,
            true_vis=targ_vis_sp_momentum,
            true_jet=targ_jet_momentum,
        )

        # classification losses
        loss_flavour = self.loss_flavour(out_flavour, targ_flavour)
        loss_charm   = self.loss_charm(out_charm, targ_charm)

        '''
        # regression per-sample tensors from the criterion
        loss_vis_geom    = outs["loss_vis/geom"]
        #loss_vis_pt      = outs["loss_vis/pt"]
        #loss_vis_mag     = outs["loss_vis/mag"]
        loss_jet_geom    = outs["loss_jet/geom"]
        #loss_jet_pt      = outs["loss_jet/pt"]
        #loss_jet_mag     = outs["loss_jet/mag"]
        loss_lep_geom    = outs["loss_lep/geom"]
        #loss_lep_pt      = outs["loss_lep/pt"]
        #loss_lep_mag     = outs["loss_lep/mag"]
        '''
        loss_vis_latent = outs["loss_vis/latent"]
        loss_vis_phi = outs["loss_vis/phi"]
        loss_vis_cart = outs["loss_vis/cart"]
        loss_jet_latent = outs["loss_jet/latent"]
        loss_jet_phi = outs["loss_jet/phi"]
        loss_jet_cart = outs["loss_jet/cart"]
        loss_lep_cons = outs["loss_lep/cons"]
        loss_lep_zero = outs["loss_lep/zero"]

        loss_vis_geom = loss_vis_latent + loss_vis_phi + loss_vis_cart
        loss_jet_geom = loss_jet_latent + loss_jet_phi + loss_jet_cart
        loss_lep_geom = loss_lep_cons + loss_lep_zero
        
        # vertex reconstruction loss (MSE)
        loss_vertex = torch.nn.functional.mse_loss(out_vertex, targ_primary_vertex, reduction='none').mean(dim=1)

        # Create nutau weight mask
        if targ_flavour.dim() == 1:
            # Hard labels (validation/test): shape (batch_size,) - discrete class indices
            # num_classes = 6: nutau classes are 2, 3, 4
            # num_classes = 4: nutau class is 2
            max_class = targ_flavour.max().item()
            if max_class > 3:
                # split_tau=True (6 classes): nutau classes are 2, 3, 4
                nutau_mask = (targ_flavour == 2) | (targ_flavour == 3) | (targ_flavour == 4)
            else:
                # split_tau=False (4 classes): nutau class is 2
                nutau_mask = (targ_flavour == 2)
        else:
            # Soft labels (training with label smoothing): shape (batch_size, num_classes)
            num_classes = targ_flavour.shape[-1]
            if num_classes > 4:
                # split_tau=True: nutau classes are 2, 3, 4
                nutau_mask = (targ_flavour[:, 2] + targ_flavour[:, 3] + targ_flavour[:, 4]) > 0.5
            else:
                # split_tau=False: nutau class is 2
                nutau_mask = targ_flavour[:, 2] > 0.5
        
        # Apply 0.1 weight to nutau samples, 1.0 to others
        sample_weights = torch.where(nutau_mask, 
                                     torch.tensor(1.0, device=targ_flavour.device), 
                                     torch.tensor(1.0, device=targ_flavour.device))

        # Apply sample weights to all per-sample losses
        loss_flavour = loss_flavour * sample_weights
        loss_charm = loss_charm * sample_weights
        loss_vis_geom = loss_vis_geom * sample_weights
        loss_jet_geom = loss_jet_geom * sample_weights
        loss_lep_geom = loss_lep_geom * sample_weights
        loss_vertex = loss_vertex * sample_weights

        kendall_w = {}
        kendall_s = {}

        def _weight(name: str, loss: torch.Tensor) -> torch.Tensor:
            u = self._uncertainty_params[name]
            loss_w, w, s = weighted_loss(loss, u, w_min=self.kendall_w_min, w_max=self.kendall_w_max)
            kendall_w[name] = w.detach()
            kendall_s[name] = s.detach()
            return loss_w

        # Kendall-weighted total
        total_loss = (
            _weight("flavour",  loss_flavour).mean()  +
            _weight("charm",    loss_charm).mean()    +
            _weight("vis_geom", loss_vis_geom).mean() +
            _weight("jet_geom", loss_jet_geom).mean() +
            _weight("lep_geom", loss_lep_geom).mean() +
            _weight("vertex",   loss_vertex).mean()
        )

        part_losses = {
            # classification
            'loss_cls/flavour': loss_flavour.mean().detach().item(),
            'loss_cls/charm':   loss_charm.mean().detach().item(),

            # regression (unmasked/batch means)
            #'loss_vis/geom': loss_vis_geom.mean().detach().item(),
            #'loss_vis/pt':   loss_vis_pt.mean().detach().item(),
            #'loss_vis/mag':  loss_vis_mag.mean().detach().item(),
            #'loss_jet/geom': loss_jet_geom.mean().detach().item(),
            #'loss_jet/pt':   loss_jet_pt.mean().detach().item(),
            #'loss_jet/mag':  loss_jet_mag.mean().detach().item(),
            #'loss_lep/geom': loss_lep_geom.mean().detach().item(),
            #'loss_lep/pt':   loss_lep_pt.mean().detach().item(),
            #'loss_lep/mag':  loss_lep_mag.mean().detach().item(),
            'loss_vis/geom': loss_vis_geom.mean().detach().item(),
            'loss_jet/geom': loss_jet_geom.mean().detach().item(),
            'loss_lep/geom': loss_lep_geom.mean().detach().item(),
            'loss_vis/latent': loss_vis_latent.mean().detach().item(),
            'loss_vis/phi': loss_vis_phi.mean().detach().item(),
            'loss_vis/cart': loss_vis_cart.mean().detach().item(),
            'loss_jet/latent': loss_jet_latent.mean().detach().item(),
            'loss_jet/phi': loss_jet_phi.mean().detach().item(),
            'loss_jet/cart': loss_jet_cart.mean().detach().item(),
            'loss_lep/cons': loss_lep_cons.mean().detach().item(),
            'loss_lep/zero': loss_lep_zero.mean().detach().item(),

            'loss_vertex':   loss_vertex.mean().detach().item(),
        }
        
        return total_loss, part_losses, kendall_w, kendall_s

    
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
        batch_input, *batch_input_global, target = self._arrange_batch(batch)
        batch_size = batch_input.batch_size

        # Forward pass
        batch_output = self.forward(batch_input, batch_input_global)
        loss, part_losses, kendall_w, kendall_s = self.compute_losses(batch_output, target)

        return loss, part_losses, kendall_w, kendall_s, batch_size


    def training_step(self, batch, batch_idx):
        loss, part_losses, kendall_w, kendall_s, batch_size = self.common_step(batch)
        self.log(f"loss_total/train", loss.item(), batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("{}/train".format(key), value, batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # log the effective Kendall weights w
        for key, w in kendall_w.items():
            self.log(
                f"uncertainty/{key}",
                w,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

            # Optional: sigma = 1/sqrt(w)
            sigma = (1.0 / torch.sqrt(w.clamp_min(1e-12))).detach()
            self.log(
                f"uncertainty_sigma/{key}",
                sigma,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

        return loss


    def validation_step(self, batch, batch_idx):
        loss, part_losses, _, _, batch_size = self.common_step(batch)

        self.log(f"loss_total/val", loss.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("{}/val".format(key), value, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss


    def configure_optimizers(self):
        """Configure optimiser with LR groups, plus warmup & cosine schedulers."""
        param_groups = param_groups_lrd(
            self.model, 
            weight_decay=self.weight_decay,
            layer_decay=self.layer_decay,
            no_weight_decay_list=self.model.no_weight_decay(),
        )

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
                eta_min=self.lr * 1e-2,
            )

        # Combine both schedulers
        combined_scheduler = CombinedScheduler(
            optimizer=optimizer,
            scheduler1=warmup_scheduler,
            scheduler2=cosine_scheduler,
            warmup_steps=self.warmup_steps,
            start_cosine_step=self.start_cosine_step,
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}
