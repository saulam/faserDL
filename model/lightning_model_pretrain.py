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
    prototype_contrastive_loss_vectorized,  ghost_pushaway_loss,
    CustomLambdaLR, CombinedScheduler, weighted_loss
)


class MAEPreTrainer(pl.LightningModule):
    def __init__(self, model, dataset, args):
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
        self.dataset = dataset
        self.preprocessing_input = args.preprocessing_input
        self.label_smoothing = args.label_smoothing

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.log_sigma_trk = nn.Parameter(torch.zeros(()))
        self.log_sigma_pri = nn.Parameter(torch.zeros(()))
        self.log_sigma_pid = nn.Parameter(torch.zeros(()))
        self.log_sigma_occ = nn.Parameter(torch.zeros(()))
        self.log_sigma_reg = nn.Parameter(torch.zeros(()))
        self._uncertainty_params = {
            "trk": self.log_sigma_trk,
            "pri": self.log_sigma_pri,
            "pid": self.log_sigma_pid,
            "occ": self.log_sigma_occ,
            "reg": self.log_sigma_reg,
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
        labels = arrange_truth(batch)
        hit_track_id = labels['hit_track_id']
        hit_primary_id = labels['hit_primary_id']
        hit_pdg = labels['hit_pdg']
        ghost_mask = labels['ghost_mask']
        hit_event_id = labels['hit_event_id']

        return batch_input, *global_params, (hit_track_id, hit_primary_id, hit_pdg, ghost_mask, hit_event_id)


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
        z_track: torch.Tensor,           # [N, Dt]
        z_primary: torch.Tensor,         # [N, Dp]
        z_pid: torch.Tensor,             # [N, Dp]
        track_id: torch.Tensor,          # [N] int64
        primary_id: torch.Tensor,        # [N] int64
        pid_id: torch.Tensor,            # [N] int64
        event_id: torch.Tensor,          # [N] int64
        ghost_mask: torch.Tensor,        # [N] bool
        num_neg: int = 16,
        temperature: float = 0.07,
        normalize: bool = True,
        pushaway_weight: float = 0.05,
    ):
        """
        Computes both losses (same-track, same-primary, same-pid) in one call.
        Assumes inputs are already masked/aligned (no ghosts/invisible hits).
        """
        loss_trk = prototype_contrastive_loss_vectorized(
            z_track, track_id, event_id, num_neg=num_neg, 
            temperature=temperature, normalize=normalize,
        )
        loss_pri = prototype_contrastive_loss_vectorized(
            z_primary, primary_id, event_id, num_neg=num_neg, 
            temperature=temperature, normalize=normalize,
        )
        loss_pid = prototype_contrastive_loss_vectorized(
            z_pid, pid_id, event_id, num_neg=num_neg,
            temperature=temperature, normalize=normalize,
        )
        loss_trk_ghost = ghost_pushaway_loss(
            z_track, track_id, event_id, ghost_mask, num_neg=num_neg, 
            temperature=temperature, normalize=normalize
        )
        loss_pri_ghost = ghost_pushaway_loss(
            z_primary, primary_id, event_id, ghost_mask, num_neg=num_neg, 
            temperature=temperature, normalize=normalize
        )
        loss_pid_ghost = ghost_pushaway_loss(
            z_pid, pid_id, event_id, ghost_mask, num_neg=num_neg,
            temperature=temperature, normalize=normalize
        )
        loss_trk_all = loss_trk + pushaway_weight * loss_trk_ghost
        loss_pri_all = loss_pri + pushaway_weight * loss_pri_ghost
        loss_pid_all = loss_pid + pushaway_weight * loss_pid_ghost

        part_losses_enc = {
            'trk/pos': loss_trk, 'trk/ghost': loss_trk_ghost,
            'pri/pos': loss_pri, 'pri/ghost': loss_pri_ghost,
            'pid/pos': loss_pid, 'pid/ghost': loss_pid_ghost,
        }

        return loss_trk_all, loss_pri_all, loss_pid_all, part_losses_enc


    def _occ_supervision_mask(
        self,
        idx_targets: torch.Tensor,          # [M, P], -1 => empty sub-voxel, >=0 => raw index of hit
        patch_shape: Tuple[int, int, int],  # (p_h, p_w, p_d)
        ghost_mask: torch.Tensor,
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

        # Positives = occupied and non-ghost
        is_occ = (idx_targets >= 0)
        is_ghost  = torch.zeros_like(idx_targets, dtype=torch.bool, device=device)
        is_ghost[is_occ]  = ghost_mask[idx_targets[is_occ]]
        pos_mask = is_occ & ~is_ghost  # [M, P]

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
        beta = getattr(self, "occ_empty_beta", 0.5)
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
    

    def compute_encoder_losses(
        self,
        pred_track: torch.Tensor,
        pred_primary: torch.Tensor,
        pred_pid: torch.Tensor,
        enc_idx_targets: torch.Tensor,
        hit_track_id: torch.Tensor,
        hit_primary_id: torch.Tensor,
        hit_pdg: torch.Tensor,
        hit_event_id: torch.Tensor,
        ghost_mask: torch.Tensor,
    ):
        raw_idx, tok_row, sub_idx = self.mask_and_align_voxels(enc_idx_targets)

        # Gather embeddings and labels
        z_track = pred_track[tok_row, sub_idx, :]         # [N_valid, D]
        z_primary = pred_primary[tok_row, sub_idx, :]     # [N_valid, D]
        z_pid = pred_pid[tok_row, sub_idx, :]             # [N_valid, D]
        y_track = hit_track_id[raw_idx]
        y_primary = hit_primary_id[raw_idx]
        y_pid = hit_pdg[raw_idx]
        hit_event_id = hit_event_id[raw_idx]
        ghost_mask = ghost_mask[raw_idx]

        loss_trk, loss_pri, loss_pid, part_losses_enc = self.metric_losses_masked_simple(
            z_track, z_primary, z_pid, y_track, y_primary, y_pid, hit_event_id, ghost_mask, normalize=True,
        )

        return loss_trk, loss_pri, loss_pid, part_losses_enc


    def compute_decoder_losses(
        self,
        targ_reg: torch.Tensor,         # [N_hits, C_in]
        pred_occ: torch.Tensor,         # [M, P]
        pred_reg: torch.Tensor,         # [M, P*C_in]
        idx_targets: torch.Tensor,      # [M, P]
        ghost_mask: torch.Tensor,
    ):
        C_in       = self.model.in_chans
        p_h, p_w, p_d = self.model.patch_size.tolist()
        focal_gamma = getattr(self, "focal_gamma", 1.5)
        focal_alpha = getattr(self, "focal_alpha", None)

        # Masks & negatives
        sup_mask, sup_targ, pos_mask, border_mask, sampled_neg_mask = self._occ_supervision_mask(
            idx_targets,
            patch_shape=(p_h, p_w, p_d),
            ghost_mask=ghost_mask,
            dilate=getattr(self, "occ_dilate", 2),
        )

        # OCC (optionally smoothed)
        if self.model.training and getattr(self, "label_smoothing", 0.0) > 0.0:
            eps = self.label_smoothing
            sup_targ = sup_targ * (1.0 - eps) + 0.5 * eps
        occ_logits_sup = pred_occ[sup_mask]
        occ_targ_sup   = sup_targ[sup_mask]
        occ_losses = soft_focal_bce_with_logits(
            occ_logits_sup, occ_targ_sup, gamma=focal_gamma, alpha=focal_alpha, reduction='none'
        )  # [N_sup]
        loss_occ = occ_losses.mean()
        occ_pos_loss = occ_losses[(pos_mask | border_mask)[sup_mask]].mean()
        occ_neg_loss = occ_losses[sampled_neg_mask[sup_mask]].mean()

        # REG only on positives and sampled negatives
        flat_idx_targets = idx_targets.view(-1)
        pos_idx = torch.where(pos_mask.view(-1))[0]
        neg_idx = torch.where(sampled_neg_mask.view(-1))[0]
        all_idx = torch.cat([pos_idx, neg_idx], dim=0)               # [N_all]
        N_pos, N_neg = pos_idx.numel(), neg_idx.numel()
        N_all = all_idx.numel()

        # Predictions
        pred_reg_flat = pred_reg.view(-1, C_in)[all_idx]             # [N_all, C_in]
        
        # Targets
        reg_empty = targ_reg.amin(dim=0)                             # [C_in]
        targ_reg_flat = reg_empty.unsqueeze(0).expand(N_all, -1).clone()
        if N_pos > 0:
            raw_pos = flat_idx_targets[pos_idx]
            targ_reg_flat[:N_pos] = targ_reg[raw_pos]

        # REG
        huber_delta = getattr(self, "huber_delta", 1.0)
        reg_elem = F.smooth_l1_loss(
            pred_reg_flat, targ_reg_flat, beta=huber_delta, reduction='none'
        )  # [N_all, C_in]
        reg_row = reg_elem.sum(dim=1)

        # charge-aware weights (positives only)
        w = torch.ones_like(reg_row)
        if N_pos > 0:
            # recover original charges for positives
            # targ_reg[raw_pos]: [N_pos, C_in] in preprocessed (log/standardized) space
            # unpreprocess back to original units; assume channel 0 is charge if C_in>1
            q_pos_orig = self.dataset.unpreprocess(
                targ_reg[raw_pos], 'q', preprocessing=self.preprocessing_input
            ).squeeze(-1)  # [N_pos]

            # robust scale q0 and weight function
            lam   = getattr(self, "reg_weight_lam", 1.0)
            alpha = getattr(self, "reg_weight_alpha", 0.5)
            q0_cfg = getattr(self, "reg_weight_q0", None)

            eps = 1e-6
            if q0_cfg is None:
                # 75th percentile within the batch positives
                q0 = torch.quantile(q_pos_orig.detach(), 0.75).clamp_min(eps)
            else:
                q0 = torch.as_tensor(q0_cfg, dtype=q_pos_orig.dtype, device=q_pos_orig.device).clamp_min(eps)

            w_pos = 1.0 + lam * (q_pos_orig / q0).clamp_min(0.).pow(alpha)
            wmax = getattr(self, "reg_weight_wmax", None)
            if wmax is not None:
                w_pos = torch.clamp(w_pos, max=float(wmax))

            w[:N_pos] = w_pos

        # Weighted loss (positives upweighted; negatives weight=1)
        loss_reg = (w * reg_row).sum() / w.sum()
        reg_pos_loss = (w[:N_pos] * reg_row[:N_pos]).sum() / (w[:N_pos].sum() + 1e-12)
        reg_neg_loss = reg_row[N_pos:].mean()

        part_losses_dec = {
            'occ/total': loss_occ, 'occ/pos': occ_pos_loss, 'occ/neg': occ_neg_loss,
            'reg/total': loss_reg, 'reg/pos': reg_pos_loss, 'reg/neg': reg_neg_loss,
        }
        return loss_occ, loss_reg, part_losses_dec
    

    def compute_losses(
        self,
        pred_track: torch.Tensor,       # encoder
        pred_primary: torch.Tensor,     # encoder
        pred_pid: torch.Tensor,         # encoder
        pred_occ: torch.Tensor,         # decoder
        pred_reg: torch.Tensor,         # decoder
        targ_reg: torch.Tensor,         # [N_hits, C_in]
        enc_idx_targets: torch.Tensor,  # encoder
        hit_track_id: torch.Tensor,     # encoder
        hit_primary_id: torch.Tensor,   # encoder
        hit_pdg: torch.Tensor,          # encoder
        hit_event_id: torch.Tensor,     # encoder
        idx_targets: torch.Tensor,      # decoder
        ghost_mask: torch.Tensor,       # encoder
    ):
        loss_trk, loss_pri, loss_pid, part_enc = self.compute_encoder_losses(
            pred_track, pred_primary, pred_pid, enc_idx_targets, hit_track_id, hit_primary_id, hit_pdg, hit_event_id, ghost_mask
        )
        loss_occ, loss_reg, part_dec = self.compute_decoder_losses(
            targ_reg, pred_occ, pred_reg, idx_targets, ghost_mask
        )

        # Kendall et al. aggregation
        part_losses = {**part_enc, **part_dec}
        def _weight(loss, attr):
            ls = getattr(self, attr, None)
            return weighted_loss(loss, ls) if ls is not None else loss

        total_loss = (
            _weight(loss_trk, "log_sigma_trk") +
            _weight(loss_pri, "log_sigma_pri") +
            _weight(loss_pid, "log_sigma_pid") +
            _weight(loss_occ, "log_sigma_occ") +
            _weight(loss_reg, "log_sigma_reg")
        )

        return total_loss, part_losses


    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, *batch_input_global, labels = self._arrange_batch(batch)
        hit_track_id, hit_primary_id, hit_pdg, ghost_mask, hit_event_id = labels

        # Forward pass
        pred_track, pred_primary, pred_pid, pred_occ, pred_reg, idx_targets, enc_idx_targets, _, _, = self.forward(
            batch_input, batch_input_global, mask_ratio=self.mask_ratio)

        loss, part_losses = self.compute_losses(
            pred_track,
            pred_primary,
            pred_pid,
            pred_occ,
            pred_reg,
            batch_input.F,
            enc_idx_targets,
            hit_track_id,
            hit_primary_id,
            hit_pdg,
            hit_event_id,
            idx_targets,
            ghost_mask,
        )

        lr = self.optimizers().param_groups[0]['lr']
        return loss, part_losses, batch_size, lr
   

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(
            f"loss_total/train",
            loss.item(), 
            batch_size=batch_size, 
            on_step=True, 
            on_epoch=True,
            prog_bar=True, 
            sync_dist=True
        )
        for key, value in part_losses.items():
            self.log(
                "{}/train".format(key),
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
            f"loss_total/val",
            loss.item(),
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        for key, value in part_losses.items():
            self.log(
                "{}/val".format(key),
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
            lr_decay=1.0
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}


    def lr_scheduler_step(self, scheduler, *args):
        """Perform a learning rate scheduler step."""
        scheduler.step()
