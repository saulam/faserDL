"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.26

Description: PyTorch Lightning model - stage 1: masked autoencoder with distance-aware losses.
             
This variant addresses the issue of voxel-level losses heavily penalizing spatially close
but misaligned predictions. Uses distance transforms and soft chamfer losses to provide
smoother gradients for near-miss predictions.
"""

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm.optim as optim_factory
from contextlib import contextmanager
from torch.nn import functional as F
from utils import (
    arrange_input, arrange_truth, bce_with_logits_label_smoothing,
    soft_ce_with_logits_csr,
    CustomLambdaLR, CombinedScheduler, weighted_loss, move_obj,
)
from utils.distance_losses import (
    unified_reconstruction_loss,
    unified_semantic_segmentation_loss,
)


class MAEPreTrainer(pl.LightningModule):
    """
    MAE PreTrainer with distance-aware reconstruction losses.
    """
    
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

        # Reconstruction distance-aware loss parameters
        self.reconstruction_loss_mode = args.reconstruction_loss_mode
        self.reconstruction_chamfer_weight = args.reconstruction_chamfer_weight
        self.reconstruction_distance_reg_weight = args.reconstruction_distance_reg_weight
        self.reconstruction_max_distance_fcal = args.reconstruction_max_distance_fcal
        self.reconstruction_max_distance_ahcal = args.reconstruction_max_distance_ahcal
        self.reconstruction_gamma_distance = args.reconstruction_gamma_distance
        
        # Semantic segmentation distance-aware parameters
        self.semantic_loss_mode = args.semantic_loss_mode
        self.semantic_distance_weight = args.semantic_distance_weight
        self.semantic_max_distance = args.semantic_max_distance
        self.semantic_gamma_distance = args.semantic_gamma_distance

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.kendall_w_min = 0.3
        self.kendall_w_max = 5.0
        self.log_sigma_gho = nn.Parameter(torch.zeros(()))
        self.log_sigma_hie = nn.Parameter(torch.zeros(()))
        self.log_sigma_dec = nn.Parameter(torch.zeros(()))
        self.log_sigma_pid = nn.Parameter(torch.zeros(()))
        self.log_sigma_occ = nn.Parameter(torch.zeros(()))
        self.log_sigma_reg = nn.Parameter(torch.zeros(()))
        self.log_sigma_occ_ah = nn.Parameter(torch.zeros(()))
        self.log_sigma_reg_ah = nn.Parameter(torch.zeros(()))
        self.log_sigma_ecal = nn.Parameter(torch.zeros(()))
        self.log_sigma_muon = nn.Parameter(torch.zeros(()))
        
        self._uncertainty_params = {
            "gho": self.log_sigma_gho,
            "hie": self.log_sigma_hie,
            "dec": self.log_sigma_dec,
            "pid": self.log_sigma_pid,
            "occ": self.log_sigma_occ,
            "reg": self.log_sigma_reg,
            "occ_ah": self.log_sigma_occ_ah,
            "reg_ah": self.log_sigma_reg_ah,
            "ecal": self.log_sigma_ecal,
            "muon": self.log_sigma_muon,
        }

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


    @contextmanager
    def _fixed_val_rng(self, batch_idx: int, val_mask_seed: int = 42):
        rank = int(getattr(self, "global_rank", 0) or 0)
        seed = int(val_mask_seed + batch_idx + 1_000_000 * rank)

        if self.device.type == "cuda":
            dev = self.device.index
            if dev is None:
                dev = torch.cuda.current_device()

            with torch.random.fork_rng(devices=[dev], enabled=True):
                torch.random.default_generator.manual_seed(seed)
                torch.cuda.default_generators[dev].manual_seed(seed)
                yield
        else:
            with torch.random.fork_rng(devices=[], enabled=True):
                torch.random.default_generator.manual_seed(seed)
                yield
        

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
        Returns indices to slice the flat hit tensors; no ghost filtering here.
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
        
        # Semantic segmentation tasks using unified loss function
        M, P = idx_targets.shape
        
        for name, pred, csr in [
            ('hie', pred_hie, csr_hie),
            ('dec', pred_dec, csr_dec),
            ('pid', pred_pid, csr_pid),
        ]:
            # Determine exclude class for hie/dec (not for pid)
            exclude_class = 0 if name != 'pid' else None

            loss_semantic, metrics_semantic = unified_semantic_segmentation_loss(
                pred_logits=pred, 
                idx_targets=idx_targets,
                csr_labels=csr,
                ghost_mask=ghost_mask,
                patch_shape=tuple(self.model.fcal_patch_size.tolist()),
                loss_mode=self.semantic_loss_mode,  # "standard" | "hybrid" | "distance"
                distance_weight=self.semantic_distance_weight,
                max_distance=self.semantic_max_distance,
                gamma_distance=self.semantic_gamma_distance,
                exclude_classes_from_dt=exclude_class,
                label_smoothing=self.label_smoothing,
                voxel_keep_prob=1-self.mask_ratio,  # key, avoids overfitting on semantic tasks
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
        max_distance: float = None,     # optional override for max_distance
    ):
        """
        Compute reconstruction losses with distance awareness using unified interface.
        """
        p_h, p_w, p_d = patch_shape
        
        # Use provided max_distance or fall back to FASERCal default
        if max_distance is None:
            max_distance = self.reconstruction_max_distance_fcal
        
        loss_occ, loss_reg, part_losses_dec = unified_reconstruction_loss(
            targ_reg=targ_reg,
            pred_occ=pred_occ,
            pred_reg=pred_reg,
            idx_targets=idx_targets,
            ghost_mask=ghost_mask,
            hit_event_id=hit_event_id,
            patch_shape=(p_h, p_w, p_d),
            dataset=self.dataset,
            preprocessing_input=self.preprocessing_input,
            loss_mode=self.reconstruction_loss_mode,
            chamfer_weight=self.reconstruction_chamfer_weight,
            distance_reg_weight=self.reconstruction_distance_reg_weight,
            max_distance=max_distance,
            gamma_distance=self.reconstruction_gamma_distance,
            occ_label_smoothing=self.label_smoothing,
        )
        
        if name_prefix:
            part_losses_dec = {f"{name_prefix}{k}": v for k, v in part_losses_dec.items()}
        
        return loss_occ, loss_reg, part_losses_dec


    def compute_global_losses(
        self,
        preds: dict,
        glob_targets: dict,
        glob_masks: dict,
    ):
        """
        Compute global reconstruction losses for ECAL energy and muon momentum.
        """
        ecal_drop = glob_masks["ecal_drop"].float().unsqueeze(-1)
        muon_drop = glob_masks["muon_drop"].float().unsqueeze(-1)
        
        # ECAL loss (per-dim mean, only when dropped)
        ecal_pred = preds["ecal_rec"]                                            # [B, 25]
        ecal_tgt  = glob_targets["ecal_tgt"]                                     # [B, 25]
        loss_ecal_raw = F.smooth_l1_loss(ecal_pred, ecal_tgt, reduction="none")  # [B, 25]
        loss_ecal_evt = loss_ecal_raw.mean(dim=-1, keepdim=True)                 # [B, 1]
        loss_ecal = (loss_ecal_evt * ecal_drop).sum() / ecal_drop.sum().clamp_min(1.0)
        
        # Muon loss (per-dim masked mean, only when dropped)
        muon_pred = preds["muon_rec"]                                            # [B, 6]
        muon_tgt  = glob_targets["muon_tgt"]                                     # [B, 6]
        has = (muon_tgt[:, 0:1] > 0).float()                                     # [B, 1]  bool: has tracks if count > 0
        p_sample = 0.25                                                          # supervise muon loss on ~25% of dropped samples
        p_has    = 0.8                                                           # supervise "has" fairly often
        p_means  = 0.2                                                           # supervise each mean dim less often
        # base mask
        dim_mask = torch.zeros_like(muon_pred)                                   # [B,6]
        dim_mask[:, 0]  = 1.0
        dim_mask[:, 1:] = has
        # per-dim stochastic gate
        dim_gate = torch.zeros_like(muon_pred)
        dim_gate[:, 0]  = (torch.rand(muon_pred.size(0), device=muon_pred.device) < p_has).float()
        dim_gate[:, 1:] = (torch.rand_like(muon_pred[:, 1:]) < p_means).float()
        dim_gate[:, 1:] *= has
        # per-sample stochastic gate (only matters when dropped)
        gate = (torch.rand_like(muon_drop.float()) < p_sample).float()
        drop = (muon_drop.float() * gate)  # [B,1]
        w = dim_mask * dim_gate * drop
        loss_raw = F.smooth_l1_loss(muon_pred, muon_tgt, reduction="none")
        loss_muon = (loss_raw * w).sum() / w.sum().clamp_min(1.0)
        
        part_losses_glob = {
            "ecal/total": loss_ecal.detach(),
            "muon/total": loss_muon.detach(),
            "ecal/drop_frac": ecal_drop.mean().detach(),
            "muon/drop_frac": muon_drop.mean().detach(),
            "muon/has_frac": has.mean().detach(),
        }
        
        return loss_ecal, loss_muon, part_losses_glob


    def compute_losses(
        self,
        preds: dict,
        targ_reg: torch.Tensor,
        targ_reg_ahcal: torch.Tensor,
        idx_targets_fas: torch.Tensor,
        idx_targets_ahcal: torch.Tensor,
        labels: dict,
        glob_targets: dict,
        glob_masks: dict,
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

        # Both relational and reconstruction tasks now operate on the same masked patches
        # Relational losses (with optional distance awareness for semantic tasks)
        loss_gho, loss_hie, loss_dec, loss_pid, part_enc = self.compute_relational_losses_distance_aware(
            pred_gho, pred_hie, pred_dec, pred_pid, idx_targets_fas, csr_hie, csr_dec, csr_pid, ghost_mask,
        )
        
        # Distance-aware reconstruction losses for FASERCal
        loss_occ, loss_reg, part_dec = self.compute_reconstruction_losses_distance_aware(
            targ_reg, pred_occ, pred_reg, idx_targets_fas, hit_event_id, ghost_mask,
            patch_shape=tuple(self.model.fcal_patch_size.tolist()),
            name_prefix="",        # keep original metric names
            max_distance=self.reconstruction_max_distance_fcal,
        )
        
        # Distance-aware reconstruction losses for AHCAL
        loss_occ_ah, loss_reg_ah, part_dec_ah = self.compute_reconstruction_losses_distance_aware(
            targ_reg_ahcal, pred_occ_ah, pred_reg_ah, idx_targets_ahcal, hit_event_id_ah, ghost_mask=ghost_mask_ah,
            patch_shape=tuple(self.model.ahcal_patch_size.tolist()),
            name_prefix="ahcal_",   # metrics logged as ahcal_occ/..., ahcal_reg/...
            max_distance=self.reconstruction_max_distance_ahcal,
        )
        
        # Global reconstruction losses
        loss_ecal, loss_muon, part_glob = self.compute_global_losses(
            preds, glob_targets, glob_masks
        )

        # Kendall et al. aggregation
        part_losses = {**part_enc, **part_dec, **part_dec_ah, **part_glob}
        kendall_w = {}
        kendall_s = {}

        def _weight(name: str, loss: torch.Tensor) -> torch.Tensor:
            u = self._uncertainty_params[name]
            loss_w, w, s = weighted_loss(loss, u, w_min=self.kendall_w_min, w_max=self.kendall_w_max)
            kendall_w[name] = w.detach()
            kendall_s[name] = s.detach()
            return loss_w

        total_loss = (
            _weight("gho",    loss_gho)    +
            _weight("hie",    loss_hie)    +
            _weight("dec",    loss_dec)    +
            _weight("pid",    loss_pid)    +
            _weight("occ",    loss_occ)    +
            _weight("reg",    loss_reg)    +
            _weight("occ_ah", loss_occ_ah) +
            _weight("reg_ah", loss_reg_ah) +
            _weight("ecal",   loss_ecal)   +
            _weight("muon",   loss_muon)
        )

        return total_loss, part_losses, kendall_w, kendall_s


    def common_step(self, batch):
        batch_input, *batch_input_global, labels = self._arrange_batch(batch)
        ahcal_sparse = batch_input_global[0]
        batch_size = batch_input.batch_size

        # Forward pass
        (
            preds,
            idx_targets_fas,
            idx_targets_ah,
            glob_tgts,
            glob_masks,
        ) = self.forward(
            batch_input, batch_input_global, mask_ratio=self.mask_ratio)

        loss, part_losses, kendall_w, kendall_s = self.compute_losses(
            preds=preds,
            targ_reg=batch_input.features,
            targ_reg_ahcal=ahcal_sparse.features,
            idx_targets_fas=idx_targets_fas,
            idx_targets_ahcal=idx_targets_ah,
            labels=labels,
            glob_targets=glob_tgts,
            glob_masks=glob_masks,
        )

        return loss, part_losses, kendall_w, kendall_s, batch_size
   

    def training_step(self, batch, batch_idx):
        loss, part_losses, kendall_w, kendall_s, batch_size = self.common_step(batch)
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

        # log the effective Kendall weights w
        for key, w in kendall_w.items():
            self.log(
                f"uncertainty/{key}",
                w,
                batch_size=batch_size,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

            # Optional: log sigma = 1/sqrt(w)
            sigma = (1.0 / torch.sqrt(w.clamp_min(1e-12))).detach()
            self.log(
                f"uncertainty_sigma/{key}",
                sigma,
                batch_size=batch_size,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True
            )

        return loss


    def validation_step(self, batch, batch_idx):
        with self._fixed_val_rng(batch_idx):
            # Ensure that the masking is the same across epochs and resumes
            loss, part_losses, _, _, batch_size = self.common_step(batch)

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
