import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from utils import arrange_sparse_minkowski, argsort_sparse_tensor, arrange_truth, argsort_coords, CustomLambdaLR, CombinedScheduler
from pytorch_lightning.trainer.supporters import CombinedDataset


class SparseLightningModel(pl.LightningModule):
    def __init__(self, model, args):
        super(SparseLightningModel, self).__init__()

        self.model = model
        self.sigmoid = args.sigmoid
        self.loss_flavour = nn.CrossEntropyLoss()
        self.loss_evis = nn.MSELoss()
        self.loss_ptmiss = nn.MSELoss()
        self.loss_primlepton = nn.BCEWithLogitsLoss()
        self.loss_seg= nn.CrossEntropyLoss() 
        self.warmup_steps = args.warmup_steps
        self.start_cosine_step = args.start_cosine_step
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps
        self.contrastive = args.contrastive
        self.finetuning = args.finetuning
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


    def forward(self, x, x_glob):
        return self.model(x, x_glob)


    def _arrange_batch(self, batch):
        batch_input, batch_input_global = arrange_sparse_minkowski(batch, self.device)
        target = arrange_truth(batch)
        return batch_input, batch_input_global, target


    def compute_losses(self, batch_output, target):
        # pred
        out_flavour = batch_output['out_flavour']
        out_evis = batch_output['out_evis'].view(-1)
        out_ptmiss = batch_output['out_ptmiss'].view(-1)
        out_primlepton = batch_output['out_primlepton'].F
        out_seg = batch_output['out_seg'].F

        # true
        targ_flavour = target['flavour_label']
        targ_evis = target['evis']
        targ_ptmiss = target['ptmiss']
        targ_primlepton = target['primlepton_labels']
        targ_seg = target['seg_labels']

        # losses
        loss_flavour = self.loss_flavour(out_flavour, targ_flavour)
        loss_evis = self.loss_evis(out_evis, targ_evis)
        loss_ptmiss = self.loss_ptmiss(out_ptmiss, targ_ptmiss)
        loss_primlepton = self.loss_primlepton(out_primlepton, targ_primlepton)
        loss_seg = self.loss_seg(out_seg, targ_seg)
        part_losses = {'flavour': loss_flavour,
                       'evis': loss_evis,
                       'ptmiss': loss_ptmiss,
                       'primlepton': loss_primlepton,
                       'seg': loss_seg,
                       }
        total_loss = loss_flavour + loss_evis + loss_ptmiss + loss_primlepton + loss_seg

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

        self.log(f"loss/train_total", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)
        for key, value in part_losses.items():
            self.log("loss/train_{}".format(key), value.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)
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
        # Optimiser
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            #self.model.parameters(),
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
                start_cosine_step=self.start_cosine_step,
                lr_decay=1.0
            )
        else:
            # No warm-up
            combined_scheduler = cosine_scheduler

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}


    def lr_scheduler_step(self, scheduler, *args):
        """Perform a learning rate scheduler step."""
        scheduler.step()

