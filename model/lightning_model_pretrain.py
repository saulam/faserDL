"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: PyTorch Lightning model - stage 1: masked autoencoder.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm.optim.optim_factory as optim_factory
from utils import (
    arrange_sparse_minkowski, arrange_truth, soft_focal_cross_entropy, soft_focal_bce_with_logits,
    CustomLambdaLR, CombinedScheduler, weighted_loss
)



class MAEPreTrainer(pl.LightningModule):
    def __init__(self, model, metadata, args):
        super(MAEPreTrainer, self).__init__()

        self.model = model
        self.mask_ratio = args.mask_ratio
        self.warmup_steps = args.warmup_steps
        self.start_cosine_step = args.start_cosine_step
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps
        self.metadata = metadata
        self.preprocessing_input = args.preprocessing_input
        self.label_smoothing = args.label_smoothing

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.log_sigma_occ = nn.Parameter(torch.zeros(()))
        self.log_sigma_reg = nn.Parameter(torch.zeros(()))
        self.log_sigma_cls = nn.Parameter(torch.zeros(()))
        self._uncertainty_params = {
            "occ": self.log_sigma_occ,
            "reg": self.log_sigma_reg,
            "cls": self.log_sigma_cls,
        }


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

    
    def forward(self, x, x_glob, mask_ratio):
        return self.model(x, x_glob, mask_ratio)


    def _arrange_batch(self, batch):
        batch_input, *global_params = arrange_sparse_minkowski(batch, self.device)
        seg_labels = arrange_truth(batch)['seg_labels']
        return batch_input, *global_params, seg_labels
    

    def _occ_supervision_mask(
        self,
        idx_targets: torch.Tensor,          # [M, P], -1 => empty sub-voxel, >=0 => raw index of hit
        patch_shape: Tuple[int, int, int],  # (p_h, p_w, p_d)
        targ_cls_soft: torch.Tensor,        # [N_hits, C] soft labels
        ghost_class_idx: int = 0,           # order: [ghost, em, had, lep]
        dilate: int = 1,
    ):
        """
        Positives: occupied, non-ghost subvoxels
        Border   : dilation around positives
        Negatives: sampled from the remaining voxels, with a GLOBAL budget
                proportional to the number of positives (Option C).

        Returns:
            sup_mask:         [M, P] bool — (positives ∪ border ∪ sampled_negatives)
            sup_targ:         [M, P] float — 1 for positives; 0 otherwise
            pos_mask:         [M, P] bool — non-ghost occupied subvoxels
            sampled_neg_mask: [M, P] bool — negatives actually sampled
        """
        device = idx_targets.device
        M, P   = idx_targets.shape
        p_h, p_w, p_d = patch_shape

        # Positives = occupied and non-ghost (argmax)
        is_occ = (idx_targets >= 0)
        max_c  = torch.zeros_like(idx_targets, dtype=torch.long, device=device)
        if is_occ.any():
            max_c[is_occ] = targ_cls_soft[idx_targets[is_occ]].argmax(dim=-1)
        pos_mask = is_occ & (max_c != ghost_class_idx)  # [M, P]

        # Border via dilation around positives
        occ = pos_mask.float().view(M, 1, p_h, p_w, p_d)
        if dilate > 0:
            ksz    = 2 * dilate + 1
            kernel = torch.ones((1, 1, ksz, ksz, ksz), device=device)
            dil    = F.conv3d(occ, kernel, padding=dilate) > 0
        else:
            dil = occ.bool()
        border_mask = dil.view(M, P) & (~pos_mask)

        # Eligible negatives = everything else (not pos, not border)
        eligible_neg = ~(pos_mask | border_mask)          # [M, P]
        pos_counts   = pos_mask.sum()                     # scalar (# positives in batch)
        neg_counts_r = eligible_neg.sum(dim=1)            # [M] per-token negative counts
        total_neg    = int(neg_counts_r.sum().item())

        # Global budget: proportional to positives
        beta = getattr(self, "occ_empty_beta", 0.75)
        target_total_negs = int(min(total_neg, round(beta * int(pos_counts.item()))))
        
        # Proportional quota per token (floored)
        q_r = (neg_counts_r.float() / max(1, total_neg) * target_total_negs).floor().to(torch.long)
        q_r = torch.minimum(q_r, neg_counts_r)  # can't exceed available
        K_max = int(q_r.max().item())

        sampled_neg_mask = torch.zeros_like(eligible_neg, dtype=torch.bool)
        if K_max > 0:
            # Random scores; pick top-k per row according to q_r
            rnd = torch.rand(M, P, device=device).masked_fill(~eligible_neg, float("inf"))
            _, idxs = torch.topk(-rnd, k=K_max, dim=1)                                   # [M, K_max]
            rows = torch.arange(M, device=device).unsqueeze(1).expand(-1, K_max)
            keep = (torch.arange(K_max, device=device).unsqueeze(0) < q_r.unsqueeze(1))  # [M, K_max]
            sampled_neg_mask[rows[keep], idxs[keep]] = True

        # Final OCC supervision
        sup_mask = pos_mask | border_mask | sampled_neg_mask
        sup_targ = pos_mask.float()

        return sup_mask, sup_targ, pos_mask, border_mask, sampled_neg_mask

   
    def compute_losses(
        self,
        targ_reg: torch.Tensor,         # [N_hits, C_in]
        targ_cls: torch.Tensor,         # [N_hits, C_out] (soft)
        pred_occ: torch.Tensor,         # [M, P]
        pred_reg: torch.Tensor,         # [M, P*C_in]
        pred_cls: torch.Tensor,         # [M, P*C_out]
        idx_targets: torch.Tensor,      # [M, P]
    ):
        C_in       = self.model.in_chans
        C_out      = self.model.out_chans
        p_h, p_w, p_d = self.model.patch_size.tolist()
        focal_gamma = getattr(self, "focal_gamma", 1.5)
        focal_alpha = getattr(self, "focal_alpha", None)

        # Masks & negatives
        sup_mask, sup_targ, pos_mask, border_mask, sampled_neg_mask = self._occ_supervision_mask(
            idx_targets,
            patch_shape=(p_h, p_w, p_d),
            targ_cls_soft=targ_cls,
            ghost_class_idx=getattr(self, "ghost_class_idx", 0),
            dilate=getattr(self, "occ_dilate", 2),
        )

        # OCC (optionally smoothed)
        if self.model.training and getattr(self, "label_smoothing", 0.0) > 0.0:
            eps = self.label_smoothing
            sup_targ = sup_targ * (1.0 - eps) + 0.5 * eps
        occ_logits_sup = pred_occ[sup_mask]
        occ_targ_sup   = sup_targ[sup_mask]
        #occ_losses = F.binary_cross_entropy_with_logits(
        #    occ_logits_sup, occ_targ_sup, reduction='none'
        #)  # [N_sup]
        occ_losses = soft_focal_bce_with_logits(
            occ_logits_sup, occ_targ_sup, gamma=focal_gamma, alpha=focal_alpha, reduction='none'
        )  # [N_sup]
        loss_occ = occ_losses.mean()
        occ_pos_loss = occ_losses[(pos_mask | border_mask)[sup_mask]].mean()
        occ_neg_loss = occ_losses[sampled_neg_mask[sup_mask]].mean()

        # REG/CLS only on positives and sampled negatives
        ghost_idx = getattr(self, "ghost_class_idx", 0)
        flat_idx_targets = idx_targets.view(-1)
        pos_idx = torch.where(pos_mask.view(-1))[0]
        neg_idx = torch.where(sampled_neg_mask.view(-1))[0]
        all_idx = torch.cat([pos_idx, neg_idx], dim=0)               # [N_all]
        N_pos, N_neg = pos_idx.numel(), neg_idx.numel()
        N_all = all_idx.numel()

        # Predictions
        pred_reg_flat = pred_reg.view(-1, C_in)[all_idx]             # [N_all, C_in]
        pred_cls_flat = pred_cls.view(-1, C_out)[all_idx]            # [N_all, C_out]

        # Targets
        reg_empty = targ_reg.amin(dim=0)                             # [C_in]
        targ_reg_flat = reg_empty.unsqueeze(0).expand(N_all, -1).clone()
        if N_pos > 0:
            raw_pos = flat_idx_targets[pos_idx]
            targ_reg_flat[:N_pos] = targ_reg[raw_pos]

        targ_cls_flat = pred_cls_flat.new_zeros(N_all, C_out)
        targ_cls_flat[:, ghost_idx] = 1.0
        if self.model.training and getattr(self, "label_smoothing", 0.0) > 0.0:
            eps = self.label_smoothing
            targ_cls_flat = targ_cls_flat * (1.0 - eps) + eps / targ_cls_flat.shape[-1]
        if N_pos > 0:
            raw_pos = flat_idx_targets[pos_idx]
            targ_cls_flat[:N_pos] = targ_cls[raw_pos]

        # REG
        huber_delta = getattr(self, "huber_delta", 1.0)
        reg_elem = F.smooth_l1_loss(
            pred_reg_flat, targ_reg_flat, beta=huber_delta, reduction='none'
        )  # [N_all, C_in]
        reg_row = reg_elem.sum(dim=1)
        loss_reg = reg_row.sum() / N_all
        reg_pos_loss = reg_row[:N_pos].mean()
        reg_neg_loss = reg_row[N_pos:].mean()

        # CLS
        cls_row = soft_focal_cross_entropy(
            pred_cls_flat, targ_cls_flat, gamma=focal_gamma, alpha=focal_alpha, reduction='none'
        ) # [N_all]
        loss_cls = cls_row.mean()
        cls_pos_loss = cls_row[:N_pos].mean()
        cls_neg_loss = cls_row[N_pos:].mean()

        # Kendall aggregation
        part_losses = {
            'occ': loss_occ, 'occ_pos': occ_pos_loss, 'occ_neg': occ_neg_loss,
            'reg': loss_reg, 'reg_pos': reg_pos_loss, 'reg_neg': reg_neg_loss,
            'cls': loss_cls, 'cls_pos': cls_pos_loss, 'cls_neg': cls_neg_loss,
        }
        total_loss = (
            weighted_loss(loss_occ, self.log_sigma_occ) +
            weighted_loss(loss_reg, self.log_sigma_reg) +
            weighted_loss(loss_cls, self.log_sigma_cls)
        )

        return total_loss, part_losses

    
    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, *batch_input_global, cls_labels = self._arrange_batch(batch)

        # Forward pass
        pred_occ, pred_reg, pred_cls, idx_targets, _, _ = self.forward(
            batch_input, batch_input_global, mask_ratio=self.mask_ratio)

        loss, part_losses = self.compute_losses(
            batch_input.F, cls_labels, pred_occ, pred_reg, pred_cls, idx_targets,
        )
  
        lr = self.optimizers().param_groups[0]['lr']
        
        return loss, part_losses, batch_size, lr


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(
            f"loss/train_total",
            loss.item(), 
            batch_size=batch_size, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True, 
            sync_dist=True
        )
        for key, value in part_losses.items():
            self.log(
                "loss/train_{}".format(key),
                value.item(), 
                batch_size=batch_size, 
                on_step=True, 
                on_epoch=True, 
                prog_bar=False, 
                sync_dist=True
            )
        self.log(f"lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)

        # log the actual sigmas (exp(-log_sigma))
        for key, log_sigma in self._uncertainty_params.items():
            uncertainty = torch.exp(-log_sigma)
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
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(
            f"loss/val_total",
            loss.item(),
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        for key, value in part_losses.items():
            self.log(
                "loss/val_{}".format(key),
                value.item(),
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
            self.model, self.weight_decay,
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
            lr_decay=1.0
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}


    def lr_scheduler_step(self, scheduler, *args):
        """Perform a learning rate scheduler step."""
        scheduler.step()

