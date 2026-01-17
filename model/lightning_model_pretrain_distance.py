"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 1: masked autoencoder with distance-aware losses.
             
This variant addresses the issue of voxel-level losses heavily penalizing spatially close
but misaligned predictions. Uses distance transforms and soft chamfer losses to provide
smoother gradients for near-miss predictions.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm.optim as optim_factory
from torch.nn import functional as F
from utils import (
    arrange_input, arrange_truth, csr_keep_rows_torch, bce_with_logits_label_smoothing,
    soft_ce_with_logits_csr, reconstruction_losses_masked_simple,
    CustomLambdaLR, CombinedScheduler, weighted_loss, move_obj,
)
from utils.distance_losses import (
    combined_distance_aware_reconstruction_loss,
    focal_distance_transform_loss,
    combined_distance_aware_segmentation_loss,
)


class MAEPreTrainerDistance(pl.LightningModule):
    """
    MAE PreTrainer with distance-aware reconstruction losses.
    
    Key differences from standard MAEPreTrainer:
    1. Uses soft chamfer loss for occupancy (considers spatial proximity)
    2. Uses distance-weighted regression loss (nearby mispredictions penalized less)
    3. Optional focal distance transform loss for smoother gradients
    
    Loss modes:
    - 'hybrid': Combines standard voxel-level + distance-aware losses (recommended for transition)
    - 'distance_only': Uses only distance-aware losses
    - 'focal_dt': Uses focal distance transform for occupancy
    """
    
    def __init__(self, model, dataset, args):
        super(MAEPreTrainerDistance, self).__init__()

        self.model = model
        self.mask_ratio = args.mask_ratio
        self.warmup_steps = args.warmup_steps
        self.start_cosine_step = args.start_cosine_step
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps
        self.dataset = dataset
        self.preprocessing_input = args.preprocessing_input
        self.label_smoothing = args.label_smoothing

        # Distance-aware loss parameters
        self.loss_mode = getattr(args, 'distance_loss_mode', 'hybrid')  # 'hybrid', 'distance_only', 'focal_dt'
        self.max_distance = getattr(args, 'max_distance', 5.0)
        self.gamma_distance = getattr(args, 'gamma_distance', 2.0)
        self.chamfer_weight = getattr(args, 'chamfer_weight', 0.3)
        self.distance_reg_weight = getattr(args, 'distance_reg_weight', 0.3)
        self.temperature_chamfer = getattr(args, 'temperature_chamfer', 1.0)
        self.use_focal_dt = getattr(args, 'use_focal_dt', False)
        
        # Semantic segmentation distance-aware parameters (NEW)
        self.use_distance_semantic = getattr(args, 'use_distance_semantic', True)
        self.semantic_distance_weight = getattr(args, 'semantic_distance_weight', 0.3)
        self.semantic_max_distance = getattr(args, 'semantic_max_distance', 3.0)  # Typically smaller than reconstruction

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.log_sigma_gho = nn.Parameter(torch.zeros(()))
        self.log_sigma_hie = nn.Parameter(torch.zeros(()))
        self.log_sigma_dec = nn.Parameter(torch.zeros(()))
        self.log_sigma_pid = nn.Parameter(torch.zeros(()))
        self.log_sigma_occ = nn.Parameter(torch.zeros(()))
        self.log_sigma_reg = nn.Parameter(torch.zeros(()))
        self.log_sigma_occ_ah = nn.Parameter(torch.zeros(()))
        self.log_sigma_reg_ah = nn.Parameter(torch.zeros(()))
        
        self._uncertainty_params = {
            "gho": self.log_sigma_gho,
            "hie": self.log_sigma_hie,
            "dec": self.log_sigma_dec,
            "pid": self.log_sigma_pid,
            "occ": self.log_sigma_occ,
            "reg": self.log_sigma_reg,
            "occ_ah": self.log_sigma_occ_ah,
            "reg_ah": self.log_sigma_reg_ah,
        }


    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return move_obj(batch, device)
    

    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups


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

        lr = self.optimizers().param_groups[0]['lr']
        self.log(f"lr", lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        

    def forward(self, x, x_glob, mask_ratio):
        return self.model(x, x_glob, mask_ratio)


    def _arrange_batch(self, batch):
        batch_input, *global_params = arrange_input(batch)
        labels = arrange_truth(batch)
        targets = {}
        targets['vis_sp_momentum'] = labels['vis_sp_momentum']
        targets['csr_hie'] = labels['csr_hie_indptr'], labels['csr_hie_ids'], labels['csr_hie_weights']
        targets['csr_dec'] = labels['csr_dec_indptr'], labels['csr_dec_ids'], labels['csr_dec_weights']
        targets['csr_pid'] = labels['csr_pid_indptr'], labels['csr_pid_ids'], labels['csr_pid_weights']
        targets['ghost_mask'] = labels['ghost_mask']
        targets['hit_event_id'] = labels['hit_event_id']
        targets['hit_event_id_ahcal'] = labels['hit_event_id_ahcal']

        return batch_input, *global_params, targets


    def mask_and_align_voxels(self, idx_targets):
        """
        idx_targets: [N_tok, P] with -1 for empty slots.
        Returns indices to slice your flat hit tensors; no ghost filtering here.
        """
        valid = idx_targets >= 0
        tok_row, sub_idx = torch.nonzero(valid, as_tuple=True)   # where a voxel is present
        raw_idx = idx_targets[tok_row, sub_idx]                  # [N_valid] indices into hit arrays
        return raw_idx, tok_row, sub_idx
    

    def metric_losses_masked_simple(
        self,
        z_gho: torch.Tensor,             # [N]
        z_hie: torch.Tensor,             # [N, Dp]
        z_dec: torch.Tensor,             # [N, Dp]
        z_pid: torch.Tensor,             # [N, Dp]
        csr_hie: torch.Tensor,           # ([N+1], [L], [L]) int64, float32
        csr_dec: torch.Tensor,           # ([N+1], [L], [L]) int64, float32
        csr_pid: torch.Tensor,           # ([N+1], [L], [L]) int64, float32
        ghost_mask: torch.Tensor,        # [N] bool
    ):
        """
        Computes losses (same-track, same-primary, same-pid) in one call.
        Standard voxel-level version (kept for ghost loss).
        """
        loss_gho = bce_with_logits_label_smoothing(z_gho, ghost_mask.to(z_gho.dtype), 
                                                   label_smoothing=self.label_smoothing)
        loss_hie = soft_ce_with_logits_csr(z_hie, csr_hie, ghost_mask=ghost_mask, 
                                           label_smoothing=self.label_smoothing*2.5, lambda_cp=5e-3)
        loss_dec = soft_ce_with_logits_csr(z_dec, csr_dec, ghost_mask=ghost_mask, 
                                           label_smoothing=self.label_smoothing*2.5, lambda_cp=5e-3)
        loss_pid = soft_ce_with_logits_csr(z_pid, csr_pid, ghost_mask=ghost_mask, 
                                           label_smoothing=self.label_smoothing, lambda_cp=1e-3)

        part_losses_enc = {
            "gho/total": loss_gho.detach(),
            "hie/total": loss_hie.detach(),
            "dec/total": loss_dec.detach(),
            "pid/total": loss_pid.detach(),
        }

        return loss_gho, loss_hie, loss_dec, loss_pid, part_losses_enc


    def compute_relational_losses_distance_aware(
        self,
        pred_gho: torch.Tensor,         # [M, P]
        pred_hie: torch.Tensor,         # [M, P, num_classes]
        pred_dec: torch.Tensor,         # [M, P, num_classes]
        pred_pid: torch.Tensor,         # [M, P, num_classes]
        idx_targets: torch.Tensor,      # [M, P]
        csr_hie: torch.Tensor,
        csr_dec: torch.Tensor,
        csr_pid: torch.Tensor,
        ghost_mask: torch.Tensor,
    ):
        """
        Compute relational losses with optional distance awareness for semantic tasks.
        
        Ghost loss (gho) is kept as standard BCE (binary classification).
        Semantic tasks (hie, dec, pid) can use distance-aware losses.
        """
        raw_idx, tok_row, sub_idx = self.mask_and_align_voxels(idx_targets)

        # Gather ghost predictions and labels (standard BCE)
        z_gho = pred_gho[tok_row, sub_idx]  # [N_valid]
        ghost = ghost_mask[raw_idx]
        loss_gho = bce_with_logits_label_smoothing(
            z_gho, ghost.to(z_gho.dtype), 
            label_smoothing=self.label_smoothing
        )
        
        part_losses = {
            "gho/total": loss_gho.detach(),
        }
        
        # For semantic segmentation tasks, use distance-aware losses if enabled
        if self.use_distance_semantic and self.semantic_distance_weight > 0:
            # These need the full spatial context, so work with [M, P, D] tensors
            # We need to reconstruct the patch shape
            M, P = idx_targets.shape
            
            # For each semantic task
            for name, pred, csr, lambda_cp, class_threshold in [
                ('hie', pred_hie, csr_hie, 5e-3, 0.01),
                ('dec', pred_dec, csr_dec, 5e-3, 0.005),
                ('pid', pred_pid, csr_pid, 1e-3, 0.005),
            ]:
                loss_semantic, metrics_semantic = combined_distance_aware_segmentation_loss(
                    pred_logits=pred,  # [M, P, num_classes]
                    idx_targets=idx_targets,
                    csr_labels=csr,
                    ghost_mask=ghost_mask,
                    patch_shape=tuple(self.model.fcal_patch_size.tolist()),
                    use_distance_weighting=True,
                    distance_weight=self.semantic_distance_weight,
                    max_distance=self.semantic_max_distance,
                    gamma_distance=self.gamma_distance,
                    label_smoothing=self.label_smoothing,
                    lambda_cp=lambda_cp,
                    class_threshold=class_threshold,
                    exclude_classes_from_dt=None if name == 'pid' else 0,  # exclude "none" class for hie/dec
                )
                
                # Store loss
                if name == 'hie':
                    loss_hie = loss_semantic
                elif name == 'dec':
                    loss_dec = loss_semantic
                else:  # pid
                    loss_pid = loss_semantic
                
                # Store metrics with prefixes
                for k, v in metrics_semantic.items():
                    part_losses[f"{name}/{k.split('/')[-1]}"] = v
        
        else:
            # Standard voxel-level losses (original behavior)
            z_hie = pred_hie[tok_row, sub_idx, :]  # [N_valid, D]
            z_dec = pred_dec[tok_row, sub_idx, :]  # [N_valid, D]
            z_pid = pred_pid[tok_row, sub_idx, :]  # [N_valid, D]
            
            csr_hie_valid = csr_keep_rows_torch(*csr_hie, raw_idx)[:3]
            csr_dec_valid = csr_keep_rows_torch(*csr_dec, raw_idx)[:3]
            csr_pid_valid = csr_keep_rows_torch(*csr_pid, raw_idx)[:3]
            
            loss_hie = soft_ce_with_logits_csr(
                z_hie, csr_hie_valid, ghost_mask=ghost, none_index=0, none_row_weight=0.3,
                class_weights=torch.tensor([0.3, 1.0, 1.0], device=z_hie.device),
                label_smoothing=self.label_smoothing*2.5, lambda_cp=5e-3
            )
            loss_dec = soft_ce_with_logits_csr(
                z_dec, csr_dec_valid, ghost_mask=ghost, none_index=0, none_row_weight=0.3,
                class_weights=torch.tensor([0.3, 1.0, 1.0], device=z_hie.device),
                label_smoothing=self.label_smoothing*2.5, lambda_cp=5e-3
            )
            loss_pid = soft_ce_with_logits_csr(
                z_pid, csr_pid_valid, ghost_mask=ghost,
                label_smoothing=self.label_smoothing, lambda_cp=1e-3
            )
            
            part_losses.update({
                "hie/total": loss_hie.detach(),
                "dec/total": loss_dec.detach(),
                "pid/total": loss_pid.detach(),
            })
        
        return loss_gho, loss_hie, loss_dec, loss_pid, part_losses
        

    def compute_reconstruction_losses_distance_aware(
        self,
        targ_reg: torch.Tensor,         # [N_hits, C_in]
        pred_occ: torch.Tensor,         # [M, P]
        pred_reg: torch.Tensor,         # [M, P*C_in]
        idx_targets: torch.Tensor,      # [M, P]
        hit_event_id: torch.Tensor,     # [N_hits]
        ghost_mask: torch.Tensor,       # [N_hits]
        patch_shape,                    # (p_h, p_w, p_d)
        name_prefix: str = "",          # optional prefix for metrics
        per_event_mean: bool = False,
    ):
        """
        Compute reconstruction losses with distance awareness.
        """
        p_h, p_w, p_d = patch_shape
        
        if self.loss_mode == 'focal_dt':
            # Use focal distance transform for occupancy
            loss_occ, metrics_occ = focal_distance_transform_loss(
                pred_occ=pred_occ,
                idx_targets=idx_targets,
                ghost_mask=ghost_mask,
                patch_shape=(p_h, p_w, p_d),
                alpha=0.25,
                gamma=1.5,
                max_distance=self.max_distance,
                distance_gamma=self.gamma_distance,
            )
            
            # Standard regression (could also add distance weighting here)
            loss_reg, part_losses_reg = reconstruction_losses_masked_simple(
                targ_reg=targ_reg,
                pred_occ=pred_occ,
                pred_reg=pred_reg,
                idx_targets=idx_targets,
                ghost_mask=ghost_mask,
                hit_event_id=hit_event_id,
                patch_shape=(p_h, p_w, p_d),
                dataset=self.dataset,
                preprocessing_input=self.preprocessing_input,
                label_smoothing=self.label_smoothing,
                per_event_mean=per_event_mean,
            )[1:]  # Skip occ loss, use only reg
            
            part_losses_dec = {**metrics_occ, **part_losses_reg}
            
        elif self.loss_mode == 'distance_only':
            # Pure distance-aware losses (no standard voxel-level component)
            loss_occ, loss_reg, part_losses_dec = combined_distance_aware_reconstruction_loss(
                targ_reg=targ_reg,
                pred_occ=pred_occ,
                pred_reg=pred_reg,
                idx_targets=idx_targets,
                ghost_mask=ghost_mask,
                hit_event_id=hit_event_id,
                patch_shape=(p_h, p_w, p_d),
                dataset=self.dataset,
                preprocessing_input=self.preprocessing_input,
                use_chamfer_occ=True,
                use_distance_weighted_reg=True,
                chamfer_weight=1.0,  # Full weight since no standard component
                distance_reg_weight=1.0,
                max_distance=self.max_distance,
                gamma_distance=self.gamma_distance,
                temperature_chamfer=self.temperature_chamfer,
                label_smoothing=self.label_smoothing,
                per_event_mean=per_event_mean,
            )
        else:  # 'hybrid' mode (default)
            # Combine standard + distance-aware losses
            loss_occ, loss_reg, part_losses_dec = combined_distance_aware_reconstruction_loss(
                targ_reg=targ_reg,
                pred_occ=pred_occ,
                pred_reg=pred_reg,
                idx_targets=idx_targets,
                ghost_mask=ghost_mask,
                hit_event_id=hit_event_id,
                patch_shape=(p_h, p_w, p_d),
                dataset=self.dataset,
                preprocessing_input=self.preprocessing_input,
                use_chamfer_occ=True,
                use_distance_weighted_reg=True,
                chamfer_weight=self.chamfer_weight,
                distance_reg_weight=self.distance_reg_weight,
                max_distance=self.max_distance,
                gamma_distance=self.gamma_distance,
                temperature_chamfer=self.temperature_chamfer,
                label_smoothing=self.label_smoothing,
                focal_gamma=1.5,
                focal_alpha=0.25,
                occ_dilate=2,
                huber_delta=1.0,
                reg_weight_lam=1.0,
                reg_weight_alpha=0.5,
                reg_weight_q0=None,
                reg_weight_wmax=None,
                occ_empty_beta=0.5,
                per_event_mean=per_event_mean,
            )
        
        if name_prefix:
            part_losses_dec = {f"{name_prefix}{k}": v for k, v in part_losses_dec.items()}
        
        return loss_occ, loss_reg, part_losses_dec


    def compute_losses(
        self,
        preds: dict,
        targ_reg: torch.Tensor,
        targ_reg_ahcal: torch.Tensor,
        rel_idx_targets: torch.Tensor,
        rec_idx_targets: torch.Tensor,
        rec_idx_targets_ahcal: torch.Tensor,
        labels: dict,
    ):
        # FASERCal predictions
        pred_gho=preds["gho"]
        pred_hie=preds["hie"]
        pred_dec=preds["dec"]
        pred_pid=preds["pid"]
        pred_occ=preds["occ"]
        pred_reg=preds["reg"]

        # AHCAL predictions
        pred_occ_ah = preds["occ_ah"]
        pred_reg_ah = preds["reg_ah"]

        csr_hie=labels['csr_hie']
        csr_dec=labels['csr_dec']
        csr_pid=labels['csr_pid']
        ghost_mask=labels['ghost_mask']
        hit_event_id=labels['hit_event_id']
        hit_event_id_ah=labels['hit_event_id_ahcal']
        ghost_mask_ah = torch.zeros_like(hit_event_id_ah, dtype=torch.bool)

        # Relational losses (with optional distance awareness for semantic tasks)
        loss_gho, loss_hie, loss_dec, loss_pid, part_enc = self.compute_relational_losses_distance_aware(
            pred_gho, pred_hie, pred_dec, pred_pid, rel_idx_targets, csr_hie, csr_dec, csr_pid, ghost_mask,
        )
        
        # Distance-aware reconstruction losses for FASERCal
        loss_occ, loss_reg, part_dec = self.compute_reconstruction_losses_distance_aware(
            targ_reg, pred_occ, pred_reg, rec_idx_targets, hit_event_id, ghost_mask,
            patch_shape=tuple(self.model.fcal_patch_size.tolist()),
            name_prefix="",        # keep original metric names
        )
        
        # Distance-aware reconstruction losses for AHCAL
        loss_occ_ah, loss_reg_ah, part_dec_ah = self.compute_reconstruction_losses_distance_aware(
            targ_reg_ahcal, pred_occ_ah, pred_reg_ah, rec_idx_targets_ahcal, hit_event_id_ah, ghost_mask=ghost_mask_ah,
            patch_shape=tuple(self.model.ahcal_patch_size.tolist()),
            name_prefix="ahcal_",   # metrics logged as ahcal_occ/..., ahcal_reg/...
        )

        # Kendall et al. aggregation
        part_losses = {**part_enc, **part_dec, **part_dec_ah}
        def _weight(loss, attr, kind):
            ls = getattr(self, attr, None)
            return weighted_loss(loss, ls, kind) if ls is not None else loss

        total_loss = (
            _weight(loss_gho,    "log_sigma_gho",    kind="ce")     +
            _weight(loss_hie,    "log_sigma_hie",    kind="ce")     +
            _weight(loss_dec,    "log_sigma_dec",    kind="ce")     +
            _weight(loss_pid,    "log_sigma_pid",    kind="ce")     +
            _weight(loss_occ,    "log_sigma_occ",    kind="ce")     +
            _weight(loss_reg,    "log_sigma_reg",    kind="huber")  +
            _weight(loss_occ_ah, "log_sigma_occ_ah", kind="ce")     +
            _weight(loss_reg_ah, "log_sigma_reg_ah", kind="huber")
        )

        return total_loss, part_losses


    def common_step(self, batch):
        batch_input, *batch_input_global, labels = self._arrange_batch(batch)
        ahcal_sparse = batch_input_global[0]
        batch_size = batch_input.batch_size

        # Forward pass
        (
            preds,
            rel_idx_targets,
            rec_idx_targets_fas,
            _row_evt_fas,
            _row_patch_fas,
            rec_idx_targets_ah,
            _row_evt_ah,
            _row_patch_ah,
        ) = self.forward(
            batch_input, batch_input_global, mask_ratio=self.mask_ratio)

        loss, part_losses = self.compute_losses(
            preds=preds,
            targ_reg=batch_input.features,
            targ_reg_ahcal=ahcal_sparse.features,
            rel_idx_targets=rel_idx_targets,
            rec_idx_targets=rec_idx_targets_fas,
            rec_idx_targets_ahcal=rec_idx_targets_ah,
            labels=labels,
        )

        return loss, part_losses, batch_size
   

    def training_step(self, batch, batch_idx):
        loss, part_losses, batch_size = self.common_step(batch)

        self.log(
            f"loss_total/train",
            loss.detach(), 
            batch_size=batch_size, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True, 
            sync_dist=True
        )
        for key, value in part_losses.items():
            self.log(
                "{}/train".format(key),
                value, 
                batch_size=batch_size, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=False, 
                sync_dist=True
            )

        # log the actual sigmas (exp(-log_sigma))
        for key, log_sigma in self._uncertainty_params.items():
            uncertainty = torch.exp(-log_sigma).detach()
            self.log(
                f'uncertainty/{key}',
                uncertainty,
                batch_size=batch_size,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

        return loss


    def validation_step(self, batch, batch_idx):
        loss, part_losses, batch_size = self.common_step(batch)

        self.log(
            f"loss_total/val",
            loss.detach(),
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        for key, value in part_losses.items():
            self.log(
                "{}/val".format(key),
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

        return loss


    def configure_optimizers(self):
        """Configure and initialize the optimizer and learning rate scheduler."""
        param_groups = optim_factory.param_groups_weight_decay(
            self.model, self.weight_decay, no_weight_decay_list=self.model.no_weight_decay(),
        )
        param_groups.append({
            'params': list(self._uncertainty_params.values()),
            'lr': self.lr * 0.1,
            'weight_decay': 0.0,
        })
        optimizer = torch.optim.AdamW(
            param_groups,
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
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}
