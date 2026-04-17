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
        self.warmup_epochs = args.warmup_epochs
        self.cosine_annealing_epochs = args.cosine_annealing_epochs
        self.lr = args.lr
        self.blr = args.blr
        self._batch_size = args.batch_size
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps
        self.dataset = dataset
        self.preprocessing_input = args.preprocessing_input
        self.label_smoothing = args.label_smoothing
        self.sparse_ecal = args.sparse_ecal

        # Reconstruction distance-aware loss parameters
        self.reconstruction_loss_mode = args.reconstruction_loss_mode
        self.reconstruction_chamfer_weight = args.reconstruction_chamfer_weight
        self.reconstruction_distance_reg_weight = args.reconstruction_distance_reg_weight
        self.reconstruction_max_distance_fcal = args.reconstruction_max_distance_fcal
        self.reconstruction_max_distance_ahcal = args.reconstruction_max_distance_ahcal
        self.reconstruction_gamma_distance = args.reconstruction_gamma_distance

        # Relational pass
        self.relational_pass_prob = args.relational_pass_prob
        self.relational_mask_ratio = args.relational_mask_ratio
        self.relational_voxel_keep_prob = args.relational_voxel_keep_prob
        self.relational_pass_seed = args.relational_pass_seed
        self.relational_loss_mode = args.relational_loss_mode
        self.relational_distance_weight = args.relational_distance_weight
        self.relational_max_distance = args.relational_max_distance
        self.relational_gamma_distance = args.relational_gamma_distance

        # One learnable log-sigma per head (https://arxiv.org/pdf/1705.07115)
        self.kendall_w_min = 1e-2
        self.kendall_w_max = 5.0
        self.u_gho = nn.Parameter(torch.zeros(()))
        self.u_hie = nn.Parameter(torch.zeros(()))
        self.u_pid = nn.Parameter(torch.zeros(()))
        self.u_occ = nn.Parameter(torch.zeros(()))
        self.u_reg = nn.Parameter(torch.zeros(()))
        self.u_occ_ah = nn.Parameter(torch.zeros(()))
        self.u_reg_ah = nn.Parameter(torch.zeros(()))
        self.u_ecal = nn.Parameter(torch.zeros(()))
        self.u_muon = nn.Parameter(torch.zeros(()))

        if self.sparse_ecal:
            self.u_occ_ec = nn.Parameter(torch.zeros(()))
            self.u_reg_ec = nn.Parameter(torch.zeros(()))

        # define a min_max dict for each uncertainty param
        self._uncertainty_params_desired_max = {
            "gho": (1.0, self.kendall_w_max),
            "hie": (0.05, 1.0),
            "pid": (0.05, 0.8),
            "occ": (1.0, self.kendall_w_max),
            "reg": (1.0, self.kendall_w_max),
            "occ_ah": (1.0, self.kendall_w_max),
            "reg_ah": (1.0, self.kendall_w_max),
            "muon": (0.02, 0.05),
        }
        if self.sparse_ecal:
            self._uncertainty_params_desired_max["occ_ec"] = (1.0, self.kendall_w_max)
            self._uncertainty_params_desired_max["reg_ec"] = (1.0, self.kendall_w_max)
        else:
            self._uncertainty_params_desired_max["ecal"] = (0.05, 1.0)
        
        self._uncertainty_params = {
            "gho": self.u_gho,
            "hie": self.u_hie,
            "pid": self.u_pid,
            "occ": self.u_occ,
            "reg": self.u_reg,
            "occ_ah": self.u_occ_ah,
            "reg_ah": self.u_reg_ah,
            "muon": self.u_muon,
        }
        if self.sparse_ecal:
            self._uncertainty_params["occ_ec"] = self.u_occ_ec
            self._uncertainty_params["reg_ec"] = self.u_reg_ec
        else:
            self._uncertainty_params["ecal"] = self.u_ecal

        def _u_for_w(w, w_min, w_max, eps=1e-6):
            # Map desired initial weight w into u so that:
            # w = w_min + (w_max - w_min) * sigmoid(u)
            p = (w - w_min) / (w_max - w_min)
            p = max(eps, min(1.0 - eps, float(p)))
            return math.log(p / (1.0 - p))
        for key, p in self._uncertainty_params.items():
            w0, w_max = self._uncertainty_params_desired_max[key]
            p.data.fill_(_u_for_w(w0, self.kendall_w_min, w_max))


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
        
        
    def _should_run_relational(self) -> bool:
        """
        DDP-safe stochastic gating: same decision across ranks, does not perturb global RNG.
        Uses a local CPU generator seeded by (seed + global_step).
        """
        p = float(self.relational_pass_prob)
        if p >= 1.0: return True
        if p <= 0.0: return False
        step = int(getattr(self.trainer, "global_step", 0))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.relational_pass_seed + step))
        return torch.rand((), generator=g).item() < p
    

    def _should_compute_muon(self) -> bool:
        """
        Compute muon loss every other iteration.
        DDP-safe: same decision across ranks based on global_step.
        """
        step = int(getattr(self.trainer, "global_step", 0))
        return (step % 2) == 0
    
    
    def _get_muon_reg_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate regularization mask for muon loss (~50% of rows).
        DDP-safe: deterministic mask based on global_step.
        """
        step = int(getattr(self.trainer, "global_step", 0))
        g = torch.Generator(device="cpu")
        g.manual_seed(42000 + step)
        rand_mask = torch.rand(batch_size, generator=g, device="cpu")
        return (rand_mask < 0.5).float().to(device)
        

    def forward(self, x, x_glob, mask_ratio, do_relational, relational_mask_ratio):
        return self.model(
            x, x_glob,
            mask_ratio=mask_ratio,
            do_relational=do_relational,
            relational_mask_ratio=relational_mask_ratio,
        )


    def _arrange_batch(self, batch):
        batch_input, *global_params = arrange_input(batch)
        labels = arrange_truth(batch)
        targets = {}
        targets['vis_sp_momentum'] = labels['vis_sp_momentum']
        targets['csr_hie'] = labels['csr_hie_indptr'], labels['csr_hie_ids'], labels['csr_hie_weights']
        targets['csr_pid'] = labels['csr_pid_indptr'], labels['csr_pid_ids'], labels['csr_pid_weights']
        targets['ghost_mask'] = labels['ghost_mask']
        targets['hit_event_id'] = labels['hit_event_id']
        targets['hit_event_id_ahcal'] = labels['hit_event_id_ahcal']
        if self.sparse_ecal:
            targets['hit_event_id_ecal'] = labels['hit_event_id_ecal']

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
        z_pid: torch.Tensor,             # [N, Dp]
        csr_hie: torch.Tensor,           # ([N+1], [L], [L]) int64, float32
        csr_pid: torch.Tensor,           # ([N+1], [L], [L]) int64, float32
        ghost_mask: torch.Tensor,        # [N] bool
    ):
        """
        Computes losses (same-track, same-pid) in one call.
        Standard voxel-level version (kept for ghost loss).
        """
        loss_gho = bce_with_logits_label_smoothing(z_gho, ghost_mask.to(z_gho.dtype), 
                                                   label_smoothing=self.label_smoothing)
        loss_hie = soft_ce_with_logits_csr(z_hie, csr_hie, ghost_mask=ghost_mask, 
                                           label_smoothing=self.label_smoothing*2.5, lambda_cp=5e-3)
        loss_pid = soft_ce_with_logits_csr(z_pid, csr_pid, ghost_mask=ghost_mask, 
                                           label_smoothing=self.label_smoothing, lambda_cp=1e-3)

        part_losses_enc = {
            "gho/total": loss_gho.detach(),
            "hie/total": loss_hie.detach(),
            "pid/total": loss_pid.detach(),
        }

        return loss_gho, loss_hie, loss_pid, part_losses_enc


    def compute_relational_losses_distance_aware(
        self,
        pred_gho: torch.Tensor,         # [M, P]
        pred_hie: torch.Tensor,         # [M, P, num_classes]
        pred_pid: torch.Tensor,         # [M, P, num_classes]
        idx_targets: torch.Tensor,      # [M, P]
        csr_hie: torch.Tensor,
        csr_pid: torch.Tensor,
        ghost_mask: torch.Tensor,
        voxel_keep_prob: float,
    ):
        """
        Compute relational losses with optional distance awareness for semantic tasks.
        
        Ghost loss (gho) is kept as standard BCE (binary classification).
        Semantic tasks (hie, pid) can use distance-aware losses.
        """
        if idx_targets.numel() == 0:
            zero = torch.tensor(0.0, device=pred_gho.device)
            part_losses = {
                "gho/total": zero.detach(),
                "hie/total": zero.detach(),
                "pid/total": zero.detach(),
            }
            return zero, zero, zero, part_losses

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
            ('pid', pred_pid, csr_pid),
        ]:
            # Determine exclude class for hie (not for pid)
            exclude_class = 0 if name != 'pid' else None

            loss_semantic, metrics_semantic = unified_semantic_segmentation_loss(
                pred_logits=pred, 
                idx_targets=idx_targets,
                csr_labels=csr,
                ghost_mask=ghost_mask,
                patch_shape=tuple(self.model.fcal_patch_size.tolist()),
                loss_mode=self.relational_loss_mode,  # "standard" | "hybrid" | "distance"
                distance_weight=self.relational_distance_weight,
                max_distance=self.relational_max_distance,
                gamma_distance=self.relational_gamma_distance,
                exclude_classes_from_dt=exclude_class,
                label_smoothing=self.label_smoothing,
                voxel_keep_prob=voxel_keep_prob,  # key, avoids overfitting on semantic tasks
            )
            
            # Store loss
            if name == 'hie':
                loss_hie = loss_semantic
            else:  # pid
                loss_pid = loss_semantic
            
            # Store metrics with prefixes
            for k, v in metrics_semantic.items():
                part_losses[f"{name}/{k.split('/')[-1]}"] = v
        
        return loss_gho, loss_hie, loss_pid, part_losses
        

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
        if idx_targets.numel() == 0:
            zero = torch.tensor(0.0, device=pred_occ.device)
            part_losses_dec = {
                "occ/total": zero.detach(),
                "occ/pos": zero.detach(),
                "occ/neg": zero.detach(),
                "reg/total": zero.detach(),
                "reg/pos": zero.detach(),
                "reg/neg": zero.detach(),
            }
            if name_prefix:
                part_losses_dec = {f"{name_prefix}{k}": v for k, v in part_losses_dec.items()}
            return zero, zero, part_losses_dec

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
        Compute global reconstruction losses for ECAL energy (dense only) and muon momentum.
        """
        part_losses_glob = {}
        loss_ecal = torch.tensor(0.0, device=self.device)

        if not self.sparse_ecal:
            ecal_drop = glob_masks["ecal_drop"].float().unsqueeze(-1)
            ecal_pred = preds["ecal_rec"]
            ecal_tgt  = glob_targets["ecal_tgt"]
            loss_ecal_raw = F.smooth_l1_loss(ecal_pred, ecal_tgt, reduction="none")
            loss_ecal_evt = loss_ecal_raw.mean(dim=-1, keepdim=True)
            loss_ecal = (loss_ecal_evt * ecal_drop).sum() / ecal_drop.sum().clamp_min(1.0)
            part_losses_glob["ecal/total"] = loss_ecal.detach()
            part_losses_glob["ecal/drop_frac"] = ecal_drop.mean().detach()

        muon_drop = glob_masks["muon_drop"].float().unsqueeze(-1)
        
        # Muon loss (per-dim masked mean, only when dropped)
        muon_pred = preds["muon_rec"]                                            # [B, 5]
        muon_tgt  = glob_targets["muon_tgt"]                                     # [B, 5]
        has_tgt   = muon_tgt[:, 0]                                               # [B] float 0/1
        has_logit = muon_pred[:, 0]                                              # [B] logits
        means_pred = muon_pred[:, 1:]                                            # [B, 4]
        means_tgt  = muon_tgt[:, 1:]                                             # [B, 4]
        drop_1d = muon_drop.squeeze(1)                                           # [B]
        drop_1d = drop_1d * self._get_muon_reg_mask(
            drop_1d.shape[0], drop_1d.device
        )                                                                        # extra regularization: ~50% of rows
        
        eps = self.label_smoothing * 2.5
        if eps > 0.0:
            has_tgt_smooth = has_tgt * (1.0 - eps) + 0.5 * eps
        else:
            has_tgt_smooth = has_tgt
        loss_has_raw = F.binary_cross_entropy_with_logits(
            has_logit, has_tgt_smooth, reduction="none"
        )
        loss_has = (loss_has_raw * drop_1d).sum() / drop_1d.sum().clamp_min(1.0)
        means_on = (has_tgt > 0.5).float() * drop_1d                             # [B]
        w_means = means_on.unsqueeze(1).expand_as(means_pred)                    # [B, 4]
        loss_means_raw = F.smooth_l1_loss(
            means_pred, means_tgt, reduction="none"
        )
        loss_means = (loss_means_raw * w_means).sum() / w_means.sum().clamp_min(1.0)
        loss_muon = loss_has + loss_means
        
        part_losses_glob.update({
            "muon/total": loss_muon.detach(),
            "muon/has_loss": loss_has.detach(),
            "muon/means_loss": loss_means.detach(),
            "muon/drop_frac": muon_drop.mean().detach(),
            "muon/has_frac": has_tgt.mean().detach(),
        })
        
        return loss_ecal, loss_muon, part_losses_glob


    def compute_losses(
        self,
        preds: dict,
        targ_reg: torch.Tensor,
        targ_reg_ahcal: torch.Tensor,
        targ_reg_ecal: torch.Tensor,
        idx_targets_fas_masked: torch.Tensor,
        idx_targets_fas_rel,
        idx_targets_ahcal: torch.Tensor,
        idx_targets_ecal,
        labels: dict,
        glob_targets: dict,
        glob_masks: dict,
        did_relational: bool,
        relational_mask_ratio: float,
        do_muon: bool = True,
    ):
        # Reconstruction losses
        pred_occ = preds["occ"]
        pred_reg = preds["reg"]
        pred_occ_ah = preds["occ_ah"]
        pred_reg_ah = preds["reg_ah"]

        ghost_mask = labels["ghost_mask"]
        hit_event_id = labels["hit_event_id"]
        hit_event_id_ah = labels["hit_event_id_ahcal"]
        ghost_mask_ah = torch.zeros_like(hit_event_id_ah, dtype=torch.bool)

        loss_occ, loss_reg, part_dec = self.compute_reconstruction_losses_distance_aware(
            targ_reg, pred_occ, pred_reg, idx_targets_fas_masked, hit_event_id, ghost_mask,
            patch_shape=tuple(self.model.fcal_patch_size.tolist()),
            max_distance=self.reconstruction_max_distance_fcal,
        )

        loss_occ_ah, loss_reg_ah, part_dec_ah = self.compute_reconstruction_losses_distance_aware(
            targ_reg_ahcal, pred_occ_ah, pred_reg_ah, idx_targets_ahcal, hit_event_id_ah, ghost_mask=ghost_mask_ah,
            patch_shape=tuple(self.model.ahcal_patch_size.tolist()),
            name_prefix="ahcal_",
            max_distance=self.reconstruction_max_distance_ahcal,
        )

        # ECAL sparse reconstruction losses
        part_dec_ec = {}
        loss_occ_ec = torch.tensor(0.0, device=self.device)
        loss_reg_ec = torch.tensor(0.0, device=self.device)
        if self.sparse_ecal and idx_targets_ecal is not None:
            pred_occ_ec = preds["occ_ecal"]
            pred_reg_ec = preds["reg_ecal"]
            hit_event_id_ec = labels["hit_event_id_ecal"]
            ghost_mask_ec = torch.zeros_like(hit_event_id_ec, dtype=torch.bool)

            loss_occ_ec, loss_reg_ec, part_dec_ec = self.compute_reconstruction_losses_distance_aware(
                targ_reg_ecal, pred_occ_ec, pred_reg_ec, idx_targets_ecal, hit_event_id_ec, ghost_mask=ghost_mask_ec,
                patch_shape=tuple(self.model.ecal_patch_size.tolist()),
                name_prefix="ecal_",
                max_distance=self.reconstruction_max_distance_ahcal,  # same grid as AHCAL
            )

        # Relational losses (optional)
        part_rel = {}
        if did_relational and (idx_targets_fas_rel is not None):
            pred_gho = preds["gho"]
            pred_hie = preds["hie"]
            pred_pid = preds["pid"]

            csr_hie = labels["csr_hie"]
            csr_pid = labels["csr_pid"]

            loss_gho, loss_hie, loss_pid, part_rel = self.compute_relational_losses_distance_aware(
                pred_gho, pred_hie, pred_pid,
                idx_targets_fas_rel,
                csr_hie, csr_pid,
                ghost_mask,
                self.relational_voxel_keep_prob,  # regularisation knob
            )
        
        # Global reconstruction losses
        loss_ecal, loss_muon, part_glob = self.compute_global_losses(
            preds, glob_targets, glob_masks
        )

        # Kendall et al. aggregation
        # Only include muon losses in part_losses if computed this iteration
        if do_muon:
            part_losses = {**part_dec, **part_dec_ah, **part_dec_ec, **part_glob, **part_rel}
        else:
            part_glob_no_muon = {k: v for k, v in part_glob.items() if not k.startswith('muon/')}
            part_losses = {**part_dec, **part_dec_ah, **part_dec_ec, **part_glob_no_muon, **part_rel}
        kendall_w, kendall_s = {}, {}

        def _weight(name: str, loss: torch.Tensor) -> torch.Tensor:
            u = self._uncertainty_params[name]
            loss_w, w, s = weighted_loss(
                loss, u, 
                w_min=self.kendall_w_min, 
                w_max=self._uncertainty_params_desired_max[name][1]
            )
            kendall_w[name] = w.detach()
            kendall_s[name] = s.detach()
            return loss_w

        total_loss = (
            _weight("occ",    loss_occ)    +
            _weight("reg",    loss_reg)    +
            _weight("occ_ah", loss_occ_ah) +
            _weight("reg_ah", loss_reg_ah)
        )
        if self.sparse_ecal:
            total_loss = total_loss + _weight("occ_ec", loss_occ_ec) + _weight("reg_ec", loss_reg_ec)
        else:
            total_loss = total_loss + _weight("ecal", loss_ecal)
        if do_muon:
            total_loss = total_loss + _weight("muon", loss_muon)
        
        if did_relational and (idx_targets_fas_rel is not None):
            total_loss = total_loss + (
                _weight("gho", loss_gho) +
                _weight("hie", loss_hie) +
                _weight("pid", loss_pid)
            )

        # basis regularization
        lambda_basis_fcal = 1e-4
        lambda_basis_ah   = 3e-5
        basis_reg_fcal = self.model.fasercal_sep_basis.orthonorm_reg(w_within=1.0, w_across=0.1)
        basis_reg_ahcal = self.model.ahcal_sep_basis.orthonorm_reg(w_within=1.0, w_across=0.1)
        total_loss = total_loss + lambda_basis_fcal * basis_reg_fcal + lambda_basis_ah * basis_reg_ahcal
        part_losses["basis/reg_fcal"] = (lambda_basis_fcal * basis_reg_fcal).detach()
        part_losses["basis/reg_ahcal"] = (lambda_basis_ah * basis_reg_ahcal).detach()
        if self.sparse_ecal:
            lambda_basis_ec = 3e-5
            basis_reg_ecal = self.model.ecal_sep_basis.orthonorm_reg(w_within=1.0, w_across=0.1)
            total_loss = total_loss + lambda_basis_ec * basis_reg_ecal
            part_losses["basis/reg_ecal"] = (lambda_basis_ec * basis_reg_ecal).detach()

        return total_loss, part_losses, kendall_w, kendall_s


    def common_step(self, batch, is_train: bool):
        batch_input, *batch_input_global, labels = self._arrange_batch(batch)
        ahcal_sparse = batch_input_global[0]
        ecal_input = batch_input_global[1]
        batch_size = batch_input.batch_size

        do_relational = self._should_run_relational() if is_train else True
        do_muon = self._should_compute_muon() if is_train else True

        # Forward pass
        (
            preds,
            idx_targets_fas_masked,
            idx_targets_fas_rel,
            idx_targets_ah,
            idx_targets_ecal,
            glob_tgts,
            glob_masks,
            aux,
        ) = self.forward(
            batch_input, 
            batch_input_global, 
            mask_ratio=self.mask_ratio,
            do_relational=do_relational,
            relational_mask_ratio=self.relational_mask_ratio,
        )

        self.log(
            "relational/did_run",
            float(aux["did_relational"]),
            batch_size=batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True
        )

        # Get ECAL sparse features for reconstruction target
        targ_reg_ecal = ecal_input.features if self.sparse_ecal else None

        loss, part_losses, kendall_w, kendall_s = self.compute_losses(
            preds=preds,
            targ_reg=batch_input.features,
            targ_reg_ahcal=ahcal_sparse.features,
            targ_reg_ecal=targ_reg_ecal,
            idx_targets_fas_masked=idx_targets_fas_masked,
            idx_targets_fas_rel=idx_targets_fas_rel,
            idx_targets_ahcal=idx_targets_ah,
            idx_targets_ecal=idx_targets_ecal,
            labels=labels,
            glob_targets=glob_tgts,
            glob_masks=glob_masks,
            did_relational=aux["did_relational"],
            relational_mask_ratio=aux["relational_mask_ratio"],
            do_muon=do_muon,
        )

        return loss, part_losses, kendall_w, kendall_s, batch_size
   

    def training_step(self, batch, batch_idx):
        loss, part_losses, kendall_w, kendall_s, batch_size = self.common_step(batch, is_train=True)
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
            loss, part_losses, _, _, batch_size = self.common_step(batch, is_train=False)

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
        """Configure and initialize the optimizer and learning rate scheduler.

        Step counts are derived from self.trainer.estimated_stepping_batches,
        which accounts for accumulation, DDP world size, and epoch count exactly.
        LR is linearly scaled from blr if provided.
        """
        # Total optimiser steps over the full training run
        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = total_steps // self.trainer.max_epochs

        # Linear LR scaling: lr = blr * effective_batch_size / 256
        if self.blr is not None:
            eff_bs = (
                self._batch_size
                * self.trainer.world_size
                * self.trainer.accumulate_grad_batches
            )
            self.lr = self.blr * eff_bs / 256.0

        warmup_steps = steps_per_epoch * self.warmup_epochs
        cosine_annealing_steps = steps_per_epoch * self.cosine_annealing_epochs
        start_cosine_step = total_steps - cosine_annealing_steps

        if self.trainer.is_global_zero:
            eff_bs = (
                self._batch_size
                * self.trainer.world_size
                * self.trainer.accumulate_grad_batches
            )
            print(f"lr                = {self.lr}")
            print(f"eff. batch size   = {eff_bs}")
            print(f"total_steps       = {total_steps}")
            print(f"steps_per_epoch   = {steps_per_epoch}")
            print(f"warmup_steps      = {warmup_steps}")
            print(f"scheduler_steps   = {cosine_annealing_steps}")
            print(f"start_cosine_step = {start_cosine_step}")

        # --- Build param groups ---
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

        if warmup_steps == 0 and cosine_annealing_steps == 0:
            return optimizer

        if warmup_steps == 0:
            warmup_scheduler = None
        else:
            # Warm-up scheduler
            warmup_scheduler = CustomLambdaLR(optimizer, warmup_steps)
 
        if cosine_annealing_steps == 0:
            cosine_scheduler = None
        else:
            # Cosine annealing scheduler
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=cosine_annealing_steps,
                eta_min=self.lr * 1e-2,
            )

        # Combine both schedulers
        combined_scheduler = CombinedScheduler(
            optimizer=optimizer,
            scheduler1=warmup_scheduler,
            scheduler2=cosine_scheduler,
            warmup_steps=warmup_steps,
            start_cosine_step=start_cosine_step,
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}
