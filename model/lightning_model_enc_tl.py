"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 2 (transfer learning from stage 1): classification and regression tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from utils import MAPE, CosineLoss, SphericalAngularLoss, StableLogCoshLoss, arrange_sparse_minkowski, argsort_sparse_tensor, arrange_truth, argsort_coords, CustomLambdaLR, CombinedScheduler


class SparseEncTlLightningModel(pl.LightningModule):
    def __init__(self, model, args):
        super(SparseEncTlLightningModel, self).__init__()
        self.model = model
        
        # Loss functions
        self.loss_flavour = nn.CrossEntropyLoss()
        self.loss_evis = [nn.MSELoss(), MAPE(from_log_scale=True)]
        self.loss_ptmiss = [nn.MSELoss(), MAPE(from_log_scale=True)]
        self.loss_lepton_momentum_mag = [nn.MSELoss(), MAPE(from_log_scale=True)]
        self.loss_lepton_momentum_dir = SphericalAngularLoss()
        self.loss_jet_momentum_mag = [nn.MSELoss(), MAPE(from_log_scale=True)]
        self.loss_jet_momentum_dir = SphericalAngularLoss()
        
        # Fine-tuning phases
        self.phase1_epochs = args.phase1_epochs   # heads only
        self.phase2_epochs = args.phase2_epochs   # + branch modules
        self.phase3_epochs = args.phase3_epochs   # + last shared block
        # Phase 4: full backbone for remaining epochs

        # Learning rates for each phase
        self.lr = args.lr
        self.lr_branch = self.lr * 0.1
        self.lr_last_shared = self.lr * 0.01
        self.lr_backbone = self.lr * 0.001

        # Optimiser params
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps


    '''
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        # Calculate progress p: global step / max_steps
        # Make sure self.trainer is set (it usually is after a few batches)
        if self.trainer.max_steps:
            total_steps = self.trainer.max_epochs * self.trainer.num_training_batches
            p = float(self.global_step) / total_steps
            # Here, gamma = 10 is a typical choice, modify as needed
            self.model.global_weight = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    '''


    def forward(self, x, x_glob):
        return self.model(x, x_glob)


    def _arrange_batch(self, batch):
        batch_input, batch_input_global = arrange_sparse_minkowski(batch, self.device)
        target = arrange_truth(batch)
        return batch_input, batch_input_global, target


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
        batch_input, batch_input_global, target = self._arrange_batch(batch)

        # Forward pass
        batch_output = self.forward(batch_input, batch_input_global)
        loss, part_losses = self.compute_losses(batch_output, target)
  
        # Retrieve current learning rate
        lr = self.optimizers().param_groups[0]['lr']

        return loss, part_losses, batch_size, lr


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        if torch.isnan(loss):
            return None

        self.log(f"loss/train_total", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("loss/train_{}".format(key), value.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)
        self.log(f"global_weight", self.model.global_weight, batch_size=batch_size, prog_bar=False, sync_dist=True)
        self.log(f"lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/val_total", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("loss/val_{}".format(key), value.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)

        return loss


    def configure_optimizers(self):
        """Configure and initialize the optimizer and learning rate scheduler."""
        # Determine current fine-tuning stage by max_epochs of this fit call
        max_ep = self.trainer.max_epochs
        e1 = self.phase1_epochs
        e2 = e1 + self.phase2_epochs
        e3 = e2 + self.phase3_epochs
        if max_ep == e1:
            phase = 1
        elif max_ep == e2:
            phase = 2
        elif max_ep == e3:
            phase = 3
        else:
            phase = 4

        # Build param-groups and set requires_grad
        specs = []
        trainable = []
        # Phase 1+: heads
        heads = []
        for branch in self.model.branches.values():
            heads += list(branch['head'].parameters())
        specs.append({'params': heads, 'lr': self.lr})
        trainable += heads
        # Phase 2+: branch modules
        if phase >= 2:
            branch_mods = []
            for branch in self.model.branches.values():
                for key in ('downsample', 'encoder', 'se'):
                    branch_mods += list(branch[key].parameters())
            specs.append({'params': branch_mods, 'lr': self.lr_branch})
            trainable += branch_mods
        # Phase 3+: last shared block
        if phase >= 3:
            last = len(self.model.shared_encoders) - 1
            last_mods = []
            for mod in (self.model.shared_encoders[last],
                        self.model.shared_se_layers[last]):
                last_mods += list(mod.parameters())
            specs.append({'params': last_mods, 'lr': self.lr_last_shared})
            trainable += last_mods
        # Phase 4: full backbone
        if phase >= 4:
            backbone = []
            # stem & global encoder
            for mod in (self.model.stem_ch, self.model.stem_mod,
                        self.model.stem_ln, self.model.global_feats_encoder):
                backbone += list(mod.parameters())
            # remaining shared blocks
            for i in range(len(self.model.shared_encoders) - 1):
                for mod in (self.model.shared_encoders[i],
                            self.model.shared_se_layers[i],
                            self.model.shared_downsamples[i]):
                    backbone += list(mod.parameters())
            specs.append({'params': backbone, 'lr': self.lr_backbone})
            trainable += backbone

        # Freeze all and unfreeze trainable ones
        trainable_set = set(trainable)
        for p in self.model.parameters():
            p.requires_grad = p in trainable_set

        # Create optimizer with phase-specific param groups
        optimizer = torch.optim.AdamW(
            specs,
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

