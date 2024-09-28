import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from utils import arrange_sparse_minkowski, argsort_sparse_tensor, arrange_truth, argsort_coords, CustomLambdaLR, CombinedScheduler
from pytorch_lightning.trainer.supporters import CombinedDataset


class SparseLightningModel(pl.LightningModule):
    def __init__(self, model, loss_fn, args):
        super(SparseLightningModel, self).__init__()

        self.model = model
        self.losses = args.losses
        self.loss_fn = loss_fn
        self.warmup_steps = args.warmup_steps
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps
        self.contrastive = args.contrastive
        self.chunk_size = args.chunk_size


    def on_train_start(self):
        "Fixing bug: https://github.com/Lightning-AI/pytorch-lightning/issues/17296#issuecomment-1726715614"
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups
 

    def on_train_epoch_start(self):
        """Hook to be called at the start of each training epoch."""
        train_loader = self.trainer.train_dataloader
        if isinstance(train_loader.dataset, CombinedDataset):
            train_loader.dataset.datasets.dataset.set_training_mode(True)
        else:
            train_loader.dataset.set_training_mode(True)


    def on_validation_epoch_start(self):
        """Hook to be called at the start of each validation epoch."""
        val_loader = self.trainer.val_dataloaders[0]
        val_loader.dataset.dataset.set_training_mode(False)


    def on_test_epoch_start(self):
        """Hook to be called at the start of each test epoch."""
        test_loader = self.trainer.test_dataloaders[0]
        test_loader.dataset.dataset.set_training_mode(False)


    def forward(self, x, y):
        return self.model(x, y)


    def _arrange_batch(self, batch):
        batch_input = arrange_sparse_minkowski(batch, self.device)
        batch_target = arrange_truth(batch, self.device)
        return batch_input, batch_target


    def compute_losses(self, batch_output, batch_target):
        losses = [0. for _ in range(len(self.losses))]
        part_losses = [[] for _ in range(len(self.losses))]
        for step, (out, tgt) in enumerate(zip(batch_output, batch_target)):
            weight = 1 / (2 ** step)  # inspired by https://arxiv.org/pdf/2310.04110

            '''
            out_coords, out_feats = out.decomposed_coordinates_and_features
            tgt_coords, tgt_feats = tgt.decomposed_coordinates_and_features     
            
            batch_size = len(tgt_coords)
            
            # Compute losses
            for i, (loss, loss_fn) in enumerate(zip(self.losses, self.loss_fn)):
                extra_args = {"gamma": 1.0} if loss == "focal" else {}
                
                loss_ghost, loss_muonic, loss_electromagnetic, loss_hadronic = 0., 0., 0., 0.
                for (c1, f1, c2, f2) in zip(out_coords, out_feats, tgt_coords, tgt_feats):
                    if not (c1==c2).all():
                        sorted_indices_c1 = argsort_coords(c1)
                        sorted_indices_c2 = argsort_coords(c2)
                        assert (c1[sorted_indices_c1] == c2[sorted_indices_c2]).all()
                        f1 = f1[sorted_indices_c1]
                        f2 = f2[sorted_indices_c2]
                    loss_ghost += loss_fn(f1[:, 0], f2[:, 0], **extra_args)
                    mask = f2[:, 0] < 0.5  # ghost mask
                    loss_muonic += loss_fn(f1[mask, 1], f2[mask, 1], **extra_args)
                    loss_electromagnetic += loss_fn(f1[mask, 2], f2[mask, 2], **extra_args)
                    loss_hadronic += loss_fn(f1[mask, 3], f2[mask, 3], **extra_args)
                loss_ghost /= batch_size
                loss_muonic /= batch_size
                loss_electromagnetic /= batch_size
                loss_hadronic /= batch_size
            '''
            sorted_ind_out = argsort_sparse_tensor(out)
            sorted_ind_tgt = argsort_sparse_tensor(tgt)
            
            sorted_coords_out = out.coordinates[sorted_ind_out]
            sorted_coords_tgt = tgt.coordinates[sorted_ind_tgt]

            # Now the coordinates should be aligned; you can sum the features
            assert (sorted_coords_out == sorted_coords_tgt).all(), "Coordinates are still not aligned!"

            sorted_feats_out = out.F[sorted_ind_out]
            sorted_feats_tgt = tgt.F[sorted_ind_tgt]

            # Compute losses
            for i, (loss, loss_fn) in enumerate(zip(self.losses, self.loss_fn)):
                extra_args = {"gamma": 1.0} if loss == "focal" else {}

                loss_ghost = loss_fn(sorted_feats_out[:, 0], sorted_feats_tgt[:, 0], **extra_args)
                mask = sorted_feats_tgt[:, 0] < 0.5  # ghost mask
                loss_muonic = loss_fn(sorted_feats_out[mask, 1], sorted_feats_tgt[mask, 1], **extra_args)
                loss_electromagnetic = loss_fn(sorted_feats_out[mask, 2], sorted_feats_tgt[mask, 2], **extra_args)
                loss_hadronic = loss_fn(sorted_feats_out[mask, 3], sorted_feats_tgt[mask, 3], **extra_args)

                part_losses[i].append([loss_ghost, loss_muonic, loss_electromagnetic, loss_hadronic])
                curr_loss = loss_ghost + loss_muonic + loss_electromagnetic + loss_hadronic
                losses[i] += weight * curr_loss 

        part_losses = torch.tensor(part_losses)
        total_loss = sum(losses)

        return total_loss, part_losses

    
    def compute_losses_contrastive(self, batch_output, batch_target):
        losses = [0.]
        part_losses = [[]] 
        for i, output in enumerate(batch_output):
            assert (output.coordinates == batch_target.coordinates).all()
            output = output.F
            target = batch_target.F[:, i]

            if i > 0:
                # mask ghosts out
                mask = batch_target.F[:, 0] < 0.5
                output = output[mask]
                target = target[mask]

            curr_loss = self.loss_fn(output, target, chunk_size=self.chunk_size)
            
            part_losses[0].append([curr_loss])
            losses[0] += curr_loss

        part_losses = torch.tensor(part_losses)
        total_loss = sum(losses)

        return total_loss, part_losses
            

    def common_step(self, batch):
        batch_size = len(batch["c"])
        batch_input, batch_target = self._arrange_batch(batch)

        # Forward pass
        batch_output, batch_target = self.forward(batch_input, batch_target)

        # Compute loss
        if self.contrastive:
            loss, part_losses = self.compute_losses_contrastive(batch_output, batch_target)
        else:
            loss, part_losses = self.compute_losses(batch_output, batch_target)
  
        # Retrieve current learning rate
        lr = self.optimizers().param_groups[0]['lr']

        return loss, part_losses, batch_size, lr


    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/train_total", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)

        for i in range(part_losses.shape[0]):  # Loop over N
            total_sum_over_mk = part_losses[i, :, :].sum().item()  # Sum over M and K for the i-th N
            self.log(f"loss/train_l{i}", total_sum_over_mk, batch_size=batch_size, prog_bar=True, sync_dist=True)

        for j in range(part_losses.shape[1]):  # Loop over M
            total_sum_over_nk = part_losses[:, j, :].sum().item()  # Sum over N and K for the j-th M
            self.log(f"loss/train_step{j}", total_sum_over_nk, batch_size=batch_size, prog_bar=False, sync_dist=True)

        for k in range(part_losses.shape[2]):  # Loop over K
            total_sum_over_nm = part_losses[:, :, k].sum().item()  # Sum over N and M for the k-th K
            self.log(f"loss/train_o{k}", total_sum_over_nm, batch_size=batch_size, prog_bar=False, sync_dist=True)

        for i in range(part_losses.shape[0]):
            for j in range(part_losses.shape[1]):
                for k in range(part_losses.shape[2]):
                    output = part_losses[i, j, k].item()
                    self.log("loss/train_o{}_l{}_step{}".format(k, i, j), output, batch_size=batch_size, prog_bar=False, sync_dist=True)
        
        self.log(f"lr", lr, batch_size=batch_size, prog_bar=True, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        loss, part_losses, batch_size, lr = self.common_step(batch)

        self.log(f"loss/val_total", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)

        for i in range(part_losses.shape[0]):  # Loop over N
            total_sum_over_mk = part_losses[i, :, :].sum().item()  # Sum over M and K for the i-th N
            self.log(f"loss/val_l{i}", total_sum_over_mk, batch_size=batch_size, prog_bar=False, sync_dist=True)

        for j in range(part_losses.shape[1]):  # Loop over M
            total_sum_over_nk = part_losses[:, j, :].sum().item()  # Sum over N and K for the j-th M
            self.log(f"loss/val_step{j}", total_sum_over_nk, batch_size=batch_size, prog_bar=False, sync_dist=True)

        for k in range(part_losses.shape[2]):  # Loop over K
            total_sum_over_nm = part_losses[:, :, k].sum().item()  # Sum over N and M for the k-th K
            self.log(f"loss/val_o{k}", total_sum_over_nm, batch_size=batch_size, prog_bar=False, sync_dist=True)

        for i in range(part_losses.shape[0]):
            for j in range(part_losses.shape[1]):
                for k in range(part_losses.shape[2]):
                    output = part_losses[i, j, k].item()
                    self.log("loss/val_o{}_l{}_step{}".format(k, i, j), output, batch_size=batch_size, prog_bar=False, sync_dist=True)

        return loss


    def configure_optimizers(self):
        """Configure and initialize the optimizer and learning rate scheduler."""
        # Optimiser
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )

        # Cosine annealing scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.cosine_annealing_steps,
            eta_min=0
        )

        if self.warmup_steps > 0:
            # Warm-up scheduler
            warmup_scheduler = CustomLambdaLR(optimizer, self.warmup_steps)
        
            # Combine both schedulers
            combined_scheduler = CombinedScheduler(
                optimizer=optimizer,
                scheduler1=warmup_scheduler,
                scheduler2=cosine_scheduler,
                warmup_steps=self.warmup_steps,
                lr_decay=1.0
            )
        else:
            # No warm-up
            combined_scheduler = cosine_scheduler

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}


    def lr_scheduler_step(self, scheduler, *args):
        """Perform a learning rate scheduler step."""
        scheduler.step()

