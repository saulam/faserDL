import os
import torch
import pytorch_lightning as pl
import optuna
from functools import partial
from utils import (
    ini_argparse,
    split_dataset
)
from dataset import SparseFASERCALDataset
from model import MinkAEConvNeXtV2, SparseAELightningModel
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


torch.set_float32_matmul_precision("medium")
pl_major = int(pl.__version__.split(".")[0])

class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used for validation
        return bar

def objective(trial, args, train_loader, valid_loader, denom, nb_gpus):
    base_lr = trial.suggest_float("base_lr", low=1e-6, high=1e-1, log=True)
    args.lr = base_lr * (args.batch_size * denom) / 256.0

    print(f"Trial {trial.number}: base_lr={base_lr:.2e}, adjusted_lr={args.lr:.2e}")
    
    model = MinkAEConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
    lightning_model = SparseAELightningModel(model=model, args=args)

    progress_bar = CustomProgressBar()

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[progress_bar],
        accelerator="gpu",
        devices=nb_gpus,
        precision="bf16-mixed" if pl_major >= 2 else 32,
        strategy="ddp" if nb_gpus > 1 else "auto",
        logger=False,
        deterministic=True,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    val_loss = trainer.callback_metrics.get("loss/val_total")
    if isinstance(val_loss, torch.Tensor):
        val_loss = val_loss.item()

    return float(val_loss)


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = ini_argparse()
    args = parser.parse_args()

    nb_gpus = len(args.gpus)
    gpus = ", ".join(args.gpus) if nb_gpus > 1 else str(args.gpus[0])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    dataset = SparseFASERCALDataset(args)
    train_loader, valid_loader, _ = split_dataset(dataset, args, splits=[0.6, 0.1, 0.3])

    nb_batches = len(train_loader)
    denom = args.accum_grad_batches * nb_gpus
    args.scheduler_steps = 0
    args.warmup_steps = 0
    args.start_cosine_step = (nb_batches * args.epochs // denom) - args.scheduler_steps

    study = optuna.create_study(
        study_name="lr_search",
        storage="sqlite:///optuna_lr_search.db",
        load_if_exists=True,
        direction="minimize",
    )
    study.optimize(
        func=partial(
            objective,
            args=args,
            train_loader=train_loader,
            valid_loader=valid_loader,
            denom=denom,
            nb_gpus=nb_gpus
        ),
        n_trials=20,
        show_progress_bar=True,
    )

    best_base_lr = study.best_trial.params["base_lr"]
    best_adjusted_lr = best_base_lr * (args.batch_size * denom) / 256.0
    print(f"\nBest base_lr: {best_base_lr:.2e}")
    print(f"Adjusted args.lr: {best_adjusted_lr:.2e}")


if __name__ == "__main__":
    main()
