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
    arrange_sparse_minkowski, arrange_truth, soft_focal_cross_entropy,
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
        neg_ratio: float = 6.0,
        max_negs_per_token: int = 4096,
        empty_neg_quota_frac: float = 0.10,
    ):
        """
        Positives are occupied, non-ghost sub-voxels.
        Ghost sub-voxels are treated as empty for OCC, i.e. target=0 (can still be sampled as negatives).
        Returns:
            sup_mask:        [M, P] bool  — which sub-voxels to supervise for occupancy (P ∪ B ∪ N)
            sup_targ:        [M, P] float — 1 for positives; 0 for border + sampled negatives
            pos_mask:        [M, P] bool  — true, non-ghost occupied sub-voxels (for reg/cls positives)
            sampled_neg_mask:[M, P] bool  — negatives we sampled (useful for reg/cls extra supervision)
        """
        device = idx_targets.device
        M, P   = idx_targets.shape
        p_h, p_w, p_d = patch_shape

        # Identify positives excluding ghosts
        is_occ  = (idx_targets >= 0)                                  # [M, P]
        # probability of ghost at those raw ids
        ghost_p = torch.zeros_like(idx_targets, dtype=torch.float, device=device)
        if is_occ.any():
            ghost_p[is_occ] = targ_cls_soft[idx_targets[is_occ], ghost_class_idx]

        # build argmax only where occupied to avoid indexing with -1
        max_c = torch.zeros_like(idx_targets, dtype=torch.long, device=device)
        if is_occ.any():
            max_c[is_occ] = targ_cls_soft[idx_targets[is_occ]].argmax(dim=-1)
        is_ghost = is_occ & (max_c == ghost_class_idx)

        pos_mask = is_occ & (~is_ghost)                               # [M, P]

        # Border via dilation around positives
        occ = pos_mask.float().view(M, 1, p_h, p_w, p_d)
        if dilate > 0:
            ksz    = 2 * dilate + 1
            kernel = torch.ones((1, 1, ksz, ksz, ksz), device=device)
            dil    = F.conv3d(occ, kernel, padding=dilate) > 0        # [M,1,ph,pw,pd]
        else:
            dil = occ.bool()
        dil = dil.view(M, P)
        border_mask = (~pos_mask) & dil

        # Negatives (1): far-empty in partially-occupied tokens
        far_empty          = ~dil
        partially_occupied = (pos_mask.sum(dim=1) > 0)                # [M]
        eligible_far       = far_empty & (~pos_mask) & (~border_mask)
        eligible_far       = eligible_far & partially_occupied.unsqueeze(1)

        pos_counts     = pos_mask.sum(dim=1)
        far_counts     = eligible_far.sum(dim=1)
        want_negs_far  = (pos_counts * neg_ratio).to(torch.long)
        want_negs_far  = torch.minimum(want_negs_far, far_counts)
        want_negs_far  = torch.clamp(want_negs_far, max=max_negs_per_token)
        K_far          = int(want_negs_far.max().item())

        sampled_far = torch.zeros_like(pos_mask, dtype=torch.bool)
        if K_far > 0:
            rnd = torch.rand(M, P, device=device).masked_fill(~eligible_far, float("inf"))
            _, idxs = torch.topk(-rnd, k=K_far, dim=1)                # [M, K_far]
            keep = (torch.arange(K_far, device=device).unsqueeze(0) <
                    want_negs_far.unsqueeze(1))                       # [M, K_far]
            rows = torch.arange(M, device=device).unsqueeze(1).expand(-1, K_far)
            sampled_far[rows[keep], idxs[keep]] = True

        # Negatives (2): fully empty tokens
        fully_empty     = (pos_mask.sum(dim=1) == 0)
        eligible_empty  = fully_empty.unsqueeze(1).expand(M, P) & (~border_mask)
        quota_empty     = int(min(max_negs_per_token, max(1, round(empty_neg_quota_frac * P))))
        sampled_empty   = torch.zeros_like(pos_mask, dtype=torch.bool)
        if quota_empty > 0 and fully_empty.any():
            rnd2 = torch.rand(M, P, device=device).masked_fill(~eligible_empty, float("inf"))
            k2   = min(quota_empty, P)
            _, idxs2 = torch.topk(-rnd2, k=k2, dim=1)                 # [M, k2]
            rows2 = torch.arange(M, device=device).unsqueeze(1).expand(-1, k2)
            keep_rows = torch.isin(rows2, fully_empty.nonzero(as_tuple=True)[0])
            sampled_empty[rows2[keep_rows], idxs2[keep_rows]] = True

        sampled_neg_mask = sampled_far | sampled_empty

        # Final occupancy supervision
        sup_mask = pos_mask | border_mask | sampled_neg_mask
        sup_targ = pos_mask.float()   # 1 for non-ghost occupancy, 0 otherwise

        return sup_mask, sup_targ, pos_mask, sampled_neg_mask


    def compute_losses(
        self,
        targ_reg: torch.Tensor,         # [N_hits, C_in]
        targ_cls: torch.Tensor,         # [N_hits, C_out]  <-- SOFT labels
        pred_occ: torch.Tensor,         # [M, P]
        pred_reg: torch.Tensor,         # [M, P*C_in]
        pred_cls: torch.Tensor,         # [M, P*C_out]
        idx_targets: torch.Tensor,      # [M, P] raw voxel ids or -1
    ):
        device     = pred_occ.device
        C_in       = self.model.in_chans
        C_out      = self.model.out_chans
        p_h, p_w, p_d = self.model.patch_size.tolist()

        # build masks (ghosts are NOT positives for occupancy)
        sup_mask, sup_targ, pos_mask, sampled_neg_mask = self._occ_supervision_mask(
            idx_targets,
            patch_shape=(p_h, p_w, p_d),
            targ_cls_soft=targ_cls,
            ghost_class_idx=getattr(self, "ghost_class_idx", 0),
            dilate=getattr(self, "occ_dilate", 2),
            neg_ratio=getattr(self, "occ_neg_ratio", 3.0),
            max_negs_per_token=getattr(self, "occ_max_negs", 1024),
            empty_neg_quota_frac=getattr(self, "occ_empty_neg_quota_frac", 0.10),
        )

        # optional label smoothing for OCC only
        if self.model.training and getattr(self, "label_smoothing", 0.0) > 0.0:
            eps = self.label_smoothing
            sup_targ = sup_targ * (1.0 - eps) + 0.5 * eps

        # OCC
        loss_occ = F.binary_cross_entropy_with_logits(pred_occ[sup_mask], sup_targ[sup_mask])

        # REG / CLS on TRUE occupied (non-ghost) sub-voxels
        flat_pos       = pos_mask.view(-1)                                  # [M*P]
        flat_idx_pos   = idx_targets.view(-1)[flat_pos]                     # raw ids

        pred_reg_pos   = pred_reg.view(-1, C_in)[flat_pos]                  # [N_pos, C_in]
        targ_reg_pos   = targ_reg[flat_idx_pos]                             # [N_pos, C_in]
        huber_delta    = getattr(self, "huber_delta", 1.0)
        loss_reg_pos   = F.smooth_l1_loss(pred_reg_pos, targ_reg_pos, beta=huber_delta)

        pred_cls_pos   = pred_cls.view(-1, C_out)[flat_pos]                 # [N_pos, C_out]
        targ_cls_pos   = targ_cls[flat_idx_pos]                             # [N_pos, C_out] soft
        focal_gamma    = getattr(self, "focal_gamma", 2.0)
        focal_alpha    = getattr(self, "focal_alpha", None)                 # None or [C]
        loss_cls_pos   = soft_focal_cross_entropy(
            pred_cls_pos, targ_cls_pos, gamma=focal_gamma, alpha=focal_alpha
        )

        # EXTRA NEGATIVES for REG / CLS (from the OCC sampler)
        neg_cls_weight = getattr(self, "neg_cls_weight", 1.0)
        neg_reg_weight = getattr(self, "neg_reg_weight", 1.0)

        loss_cls_neg = torch.tensor(0.0, device=device)
        loss_reg_neg = torch.tensor(0.0, device=device)

        if sampled_neg_mask.any():
            flat_neg     = sampled_neg_mask.view(-1)
            # classification: force "ghost"
            pred_cls_neg = pred_cls.view(-1, C_out)[flat_neg]               # [N_neg, C_out]
            targ_cls_neg = pred_cls_neg.new_zeros(pred_cls_neg.shape)       # one-hot ghost
            targ_cls_neg[:, getattr(self, "ghost_class_idx", 0)] = 1.0
            loss_cls_neg = soft_focal_cross_entropy(
                pred_cls_neg, targ_cls_neg, gamma=focal_gamma, alpha=focal_alpha
            )

            # regression: force near-zero energy (min of batch targets, already standardized/log1p)
            reg_empty    = targ_reg.amin(dim=0)                             # [C_in]
            pred_reg_neg = pred_reg.view(-1, C_in)[flat_neg]
            targ_reg_neg = reg_empty.unsqueeze(0).expand_as(pred_reg_neg)
            loss_reg_neg = F.smooth_l1_loss(pred_reg_neg, targ_reg_neg, beta=huber_delta)

        # combine pos+neg
        loss_reg = loss_reg_pos + neg_reg_weight * loss_reg_neg
        loss_cls = loss_cls_pos + neg_cls_weight * loss_cls_neg

        # ---- aggregate with Kendall’s uncertainty weights ----
        part_losses = {'occ': loss_occ, 'reg': loss_reg, 'cls': loss_cls}
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

        self.log(f"loss/train_total", loss.item(), batch_size=batch_size, on_step=True, on_epoch=True,prog_bar=True, sync_dist=True)
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

