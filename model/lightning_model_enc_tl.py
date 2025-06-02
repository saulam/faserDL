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
from utils import MAPE, CosineLoss, SphericalAngularLoss, StableLogCoshLoss, arrange_sparse_minkowski, argsort_sparse_tensor, arrange_truth, argsort_coords, CustomLambdaLR, CombinedScheduler


class SparseEncTlLightningModel(pl.LightningModule):
    def __init__(self, model, args):
        super(SparseEncTlLightningModel, self).__init__()
        self.model = model
        self.preprocessing = args.preprocessing
        
        # Loss functions
        self.loss_flavour = nn.CrossEntropyLoss()
        self.loss_evis = MAPE(preprocessing=args.preprocessing_output)
        self.loss_ptmiss = MAPE(preprocessing=args.preprocessing_output)
        self.loss_lepton_momentum_mag = MAPE(preprocessing=args.preprocessing_output)
        self.loss_lepton_momentum_dir = SphericalAngularLoss()
        self.loss_jet_momentum_mag = MAPE(preprocessing=args.preprocessing_output)
        self.loss_jet_momentum_dir = SphericalAngularLoss()

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.log_sigma_flavour = nn.Parameter(torch.zeros(()))
        self.log_sigma_e_vis = nn.Parameter(torch.zeros(()))
        self.log_sigma_pt_miss = nn.Parameter(torch.zeros(()))
        self.log_sigma_lepton_momentum = nn.Parameter(torch.zeros(()))
        self.log_sigma_jet_momentum = nn.Parameter(torch.zeros(()))
        self._uncertainty_params = [
            self.log_sigma_flavour,
            self.log_sigma_e_vis,
            self.log_sigma_pt_miss,
            self.log_sigma_lepton_momentum,
            self.log_sigma_jet_momentum,
        ]
        self.log_sigma_params = set(self._uncertainty_params)
        
        # Fine-tuning phases
        self.current_phase = 1
        self.lr = args.lr
        self.layer_decay = args.layer_decay

        # Optimiser params
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps

        # Placeholders for param lists
        self.phase1_params = []

        # For layer-wise lr decay
        self.encoder_block_counts = [
            len(self.model.encoder_layers[i]) 
            for i in range(self.model.nb_elayers)
        ]

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
        out_e_vis = batch_output['out_e_vis'].squeeze()
        out_pt_miss = batch_output['out_pt_miss'].squeeze()
        out_lepton_momentum_mag = batch_output['out_lepton_momentum_mag'].squeeze()
        out_lepton_momentum_dir = batch_output['out_lepton_momentum_dir']
        out_jet_momentum_mag = batch_output['out_jet_momentum_mag'].squeeze()
        out_jet_momentum_dir = batch_output['out_jet_momentum_dir']

        # true
        targ_flavour = target['flavour_label']
        targ_e_vis = target['e_vis']
        targ_pt_miss = target['pt_miss']
        targ_lepton_momentum_mag = target['out_lepton_momentum_mag']
        targ_lepton_momentum_dir = target['out_lepton_momentum_dir']
        targ_jet_momentum_mag = target['jet_momentum_mag']
        targ_jet_momentum_dir = target['jet_momentum_dir']
        
        def weighted_regression(L, s):
            # (1/2) e^(–2s) L + s
            return 0.5 * torch.exp(-2*s) * L + s
        
        def weighted_classification(L, s):
            # different for classification since it's already a likelihood
            # exp(-s) * L + s
            return torch.exp(-1*s) * L + s

        # losses
        loss_flavour = self.loss_flavour(out_flavour, targ_flavour)
        loss_e_vis = self.loss_evis(out_e_vis, targ_e_vis)
        loss_pt_miss = self.loss_ptmiss(out_pt_miss, targ_pt_miss)
        loss_lepton_momentum_mag = self.loss_lepton_momentum_mag(out_lepton_momentum_mag, targ_lepton_momentum_mag)
        loss_lepton_momentum_dir = self.loss_lepton_momentum_dir(out_lepton_momentum_dir, targ_lepton_momentum_dir)
        loss_jet_momentum_mag = self.loss_jet_momentum_mag(out_jet_momentum_mag, targ_jet_momentum_mag)
        loss_jet_momentum_dir = self.loss_jet_momentum_dir(out_jet_momentum_dir, targ_jet_momentum_dir)
        
        part_losses = {'flavour': loss_flavour,
                       'e_vis': loss_e_vis,
                       'pt_miss': loss_pt_miss,
                       'lepton_momentum_mag': loss_lepton_momentum_mag,
                       'lepton_momentum_dir': loss_lepton_momentum_dir,
                       'jet_momentum_mag': loss_jet_momentum_mag,
                       'jet_momentum_dir': loss_jet_momentum_dir,
                       }
        total_loss = ( weighted_classification(loss_flavour, self.log_sigma_flavour)
                     + weighted_regression(loss_e_vis, self.log_sigma_e_vis)
                     + weighted_regression(loss_pt_miss, self.log_sigma_pt_miss)
                     + weighted_regression(loss_lepton_momentum_mag + loss_lepton_momentum_dir, self.log_sigma_lepton_momentum)
                     + weighted_regression(loss_jet_momentum_mag + loss_jet_momentum_dir, self.log_sigma_jet_momentum)
                   )

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
        
        # log the actual sigmas (exp(log_sigma))
        for name, param in self.named_parameters():
            if 'log_sigma' in name:
                self.log(f'uncertainty/{name}', torch.exp(param), batch_size=batch_size, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/val_total", loss.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("loss/val_{}".format(key), value.item(), batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return loss


    def configure_optimizers(self):
        """Configure optimizer with phase-aware LR groups, plus warmup & cosine schedulers."""
        # ─── Print current phase & count trainable params ────────────────────────────
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Phase {self.current_phase}] Trainable parameters: {num_trainable:,}")

        if self.current_phase == 1:
            # Phase1: heads + cls tokens only
            decay_params = [p for p in self.phase1_params if p not in self.log_sigma_params]
            nodecay_params = [p for p in self.phase1_params if p in self.log_sigma_params]
            param_groups = []
            if decay_params:
                param_groups.append({'params': decay_params, 'lr': self.lr, 'weight_decay': self.weight_decay})
            if nodecay_params:
                param_groups.append({'params': nodecay_params, 'lr': self.lr, 'weight_decay': 0.0})
        else:
            # Phase2: full unfreeze with layer-wise lr decay
            param_groups = self._get_layerwise_param_groups()

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

    
    def freeze_phase1(self):
        self.current_phase = 1
        for p in self.parameters():
            p.requires_grad = False
        # Unfreeze heads, task-transformer, cls_task, log_sigma
        for head in self.model.branches.values():
            for p in head.parameters():
                p.requires_grad = True
        self.model.cls_task.requires_grad_(True)
        for p in self._uncertainty_params:
            p.requires_grad = True
        self.phase1_params = [p for p in self.parameters() if p.requires_grad]

    
    def unfreeze_phase2(self):
        # unfreeze all remaining layers
        self.current_phase = 2
        for p in self.parameters():
            p.requires_grad = True

        
    def _get_layer_id(self, name: str) -> int:
        """
        Assign a unique integer ID according to the exact forward‐pass order
        """        
        # 1) STEM (one ID = 0)
        if name.startswith("model.stem"):
            return 0

        nb_elayers = self.model.nb_elayers
        counts = self.encoder_block_counts  # e.g. [3, 3, 9, 3]

        # Helper: compute “base” = ID of mod_layer
        def compute_base():
            base = 1  # IDs start at 1 immediately after stem
            for x in range(nb_elayers):
                base += counts[x]
                if x < nb_elayers - 1:
                    base += 1  # one ID reserved for downsample_layers[x]
            return base

        # 2) ENCODER LAYERS (each Block) & DOWNSAMPLE LAYERS interleaved
        #    encoder_layers[i][j] → ID = 1 + (all blocks+dowmsamples before it) + j
        m = re.match(r"model\.encoder_layers\.(\d+)\.(\d+)\.", name)
        if m:
            i = int(m.group(1))
            j = int(m.group(2))
            used = 1  # the first available ID after stem = 1
            for k in range(i):
                used += counts[k]
                if k < nb_elayers - 1:
                    used += 1  # that downsample_layers[k]
            return used + j

        m = re.match(r"model\.downsample_layers\.(\d+)\.", name)
        if m:
            i = int(m.group(1))
            used = 1
            for k in range(i):
                used += counts[k]
                if k < nb_elayers - 1:
                    used += 1
            used += counts[i]
            return used

        # 3) MODULE‐LEVEL TRANSFORMER (mod_layer & its cls_mod) → next single ID
        if name.startswith("model.mod_layer") or name.startswith("model.cls_mod"):
            return compute_base()

        # 4) EVENT‐LEVEL TRANSFORMER (each of its 5 TransformerEncoderLayer’s)
        m = re.match(r"model\.event_transformer\.layers\.(\d+)\.", name)
        if m:
            k = int(m.group(1))  # 0..4
            base = compute_base()
            return base + 1 + k  # IDs = base+1, base+2, …, base+5

        # 5) global_feats_encoder, event_pos, cls_task → ALL share ID = (base + 1 + num_evt_layers)
        if (name.startswith("model.global_feats_encoder")
            or name.startswith("model.event_pos")
            or name.startswith("model.cls_task")):
            base = compute_base()
            num_evt_layers = len(self.model.event_transformer.layers)  # =5
            return base + 1 + num_evt_layers

        # 6) All branches (“model.branches.<any>.*”) → share ID = (base + 1 + num_evt_layers + 1)
        if name.startswith("model.branches") or name.startswith("log_sigma"):
            base = compute_base()
            num_evt_layers = len(self.model.event_transformer.layers)
            return base + 1 + num_evt_layers + 1

        raise Exception("Parameter name undefined: {}".format(name))


    def _get_layerwise_param_groups(self):
        nb_elayers = self.model.nb_elayers
        counts = self.encoder_block_counts 
        base = 1  # IDs start from 1 immediately after stem
        for k in range(nb_elayers):
            base += counts[k]
            if k < nb_elayers - 1:
                base += 1  # one ID for downsample_layers[k]
    
        # max layers
        num_event_layers = len(self.model.event_transformer.layers)
        branch_id = base + 1 + num_event_layers + 1
        max_layer_id = branch_id
    
        groups = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            layer_id = self._get_layer_id(name)
            scale   = self.layer_decay ** (max_layer_id - layer_id)
            if (name.endswith(".bias")
                or "norm" in name.lower()         # any norm layer
                or param in self.log_sigma_params # uncertainty tensors
            ):
                decay = 0.0
            else:
                decay = self.weight_decay
            key = (layer_id, decay)
            if key not in groups:
                groups[key] = {
                    'params': [],
                    'lr': self.lr * scale,
                    'weight_decay': decay
                }
            groups[key]['params'].append(param)
    
        return list(groups.values())


    def lr_scheduler_step(self, scheduler, *args):
        """Perform a learning rate scheduler step."""
        scheduler.step()

