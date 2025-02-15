"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 09.24

Description: Training script.
"""


import os
import torch
import pytorch_lightning as pl
import optuna
from functools import partial
from utils import CustomFinetuningReversed, ini_argparse, split_dataset, supervised_pixel_contrastive_loss, focal_loss, dice_loss
from dataset import SparseFASERCALDataset
from model import MinkUNetClsConvNeXtV2, SparseLightningModel
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, TQDMProgressBar


class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used for validation
        return bar


class NaNPruningCallback(Callback):
    def __init__(self, trial):
        self.trial = trial

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        train_loss = trainer.callback_metrics.get("loss/train_total")
        
        # Check if val_dice is NaN
        if train_loss is not None and (train_loss.isnan() or train_loss.isinf()):
            # Log that NaN was encountered and prune the trial
            self.trial.set_user_attr("nan_encountered", True)
            print("NaN encountered in validation metric. Pruning trial.")
            raise optuna.TrialPruned()


torch.multiprocessing.set_sharing_strategy('file_system')
parser = ini_argparse()
args = parser.parse_args()

print("\n- Arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
nb_gpus = len(args.gpus)
gpus = ', '.join(args.gpus) if nb_gpus > 1 else str(args.gpus[0])

# Manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
 
# Dataset
dataset = SparseFASERCALDataset(args)
print("- Dataset size: {} events".format(len(dataset)))
train_loader, valid_loader, test_loader = split_dataset(dataset, args, splits=[0.6, 0.1, 0.3]) 

# Calculate arguments for scheduler
nb_batches = len(train_loader)
args.scheduler_steps = nb_batches * args.cosine_annealing_steps // (args.accum_grad_batches * nb_gpus)
args.warmup_steps = nb_batches * args.warmup_steps // (args.accum_grad_batches * nb_gpus)
args.start_cosine_step = (nb_batches * args.epochs // (args.accum_grad_batches * nb_gpus)) - args.scheduler_steps

def objective(trial):
    lr = trial.suggest_float('lr', 1e-6, 1e-3)
    weight_decay = trial.suggest_float('weight_decay', 0, 0.05)

    args.lr = lr
    args.weight_decay = weight_decay

    print(f"args.lr = {args.lr}")
    print(f"args.weight_decay = {args.weight_decay}")

    # Initialize the model
    model = MinkUNetClsConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
    #print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params model (total): {}".format(total_params))

    nan_callback = NaNPruningCallback(trial)
    progress_bar = CustomProgressBar()
    callbacks = [nan_callback, progress_bar]

    # Lightning model
    lightning_model = SparseLightningModel(model=model,
        args=args)

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        max_epochs=1,
        callbacks=callbacks,
        accelerator="gpu",
        devices=nb_gpus,
        strategy="ddp" if nb_gpus > 1 else None,
        deterministic=True,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    # Train and validate the model
    trainer.fit(model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader)

    return trainer.callback_metrics["loss/val_total"].item()


# Create an Optuna study and optimize it
#study = optuna.create_study(study_name='v1', storage='sqlite:///example.db', direction='minimize')
study = optuna.load_study(study_name='v1', storage='sqlite:///example.db')
study.optimize(objective, n_trials=20)

# Display best trial
print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
        print(f"    {key}: {value}")

