"""Training entry point for scintillator PID transfer."""

import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from .adapted_model import scintillator_encoder_base, scintillator_encoder_tiny
from .checkpoint_utils import load_pretrained_encoder
from .lightning_module import ScintillatorPIDFineTuner
from .scintillator_dataset import ScintillatorPIDDataset, scintillator_collate_fn

torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

MODEL_FACTORIES = {
    "tiny": scintillator_encoder_tiny,
    "base": scintillator_encoder_base,
}


class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True
        return bar


def parse_args():
    p = argparse.ArgumentParser(description="Scintillator 4-class PID fine-tuning")

    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root of the scintillator dataset (contains training/ and testing/).",
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Fraction of the training set held out for validation.",
    )
    p.add_argument("--spatial_shape", type=int, nargs=3, default=[120, 120, 120])
    p.add_argument(
        "--charge_metadata_path",
        type=str,
        default=None,
        help="Transfer charge metadata pickle containing q_log1p stats.",
    )

    p.add_argument("--model", type=str, choices=list(MODEL_FACTORIES.keys()), default="base")
    p.add_argument("--patch_size", type=int, nargs=3, default=[12, 12, 10])
    p.add_argument("--window_size", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--num_cls", type=int, default=2)
    p.add_argument("--drop_path_rate", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--attn_dropout", type=float, default=0.0)
    p.add_argument("--head_dropout", type=float, default=0.0)
    p.add_argument("--head_init", type=float, default=2e-5)
    p.add_argument("--global_pool", action="store_true", default=True)

    p.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Pretrained FASERCal .ckpt to transfer encoder weights from.",
    )
    p.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Resume training from this Lightning checkpoint.",
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--blr", type=float, default=5e-4)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--cosine_annealing_epochs", type=int, default=25)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--layer_decay", type=float, default=0.75)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--accum_grad_batches", type=int, default=1)
    p.add_argument("--label_smoothing", type=float, default=0.02)

    p.add_argument(
        "--save_dir",
        type=str,
        default="transfer_learning/transfer_scintillator/logs_scintillator",
    )
    p.add_argument("--name", type=str, default="scintillator_pid_base")
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="transfer_learning/transfer_scintillator/checkpoints_scintillator",
    )
    p.add_argument("--checkpoint_name", type=str, default="scintillator_pid_base")
    p.add_argument("--save_top_k", type=int, default=3)
    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--early_stop_patience", type=int, default=10)

    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--nb_nodes", type=int, default=1)

    augment_group = p.add_mutually_exclusive_group()
    augment_group.add_argument("--augment", dest="augment", action="store_true")
    augment_group.add_argument("--no-augment", dest="augment", action="store_false")
    p.set_defaults(augment=True)

    return p.parse_args()


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()

    print("\n--- Arguments ---")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    nb_gpus = len(args.gpus)
    gpus = ",".join(map(str, args.gpus)) if nb_gpus > 1 else str(args.gpus[0])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    ss = tuple(args.spatial_shape)

    full_train_ds = ScintillatorPIDDataset(
        data_dir=args.data_dir,
        split="training",
        spatial_shape=ss,
        augment=args.augment,
        charge_metadata_path=args.charge_metadata_path,
    )

    n_val = int(len(full_train_ds) * args.val_fraction)
    n_train = len(full_train_ds) - n_val
    train_ds, val_indices = torch.utils.data.random_split(
        full_train_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    val_ds_base = ScintillatorPIDDataset(
        data_dir=args.data_dir,
        split="training",
        spatial_shape=ss,
        augment=False,
        charge_metadata_path=args.charge_metadata_path,
    )
    val_ds = torch.utils.data.Subset(val_ds_base, val_indices.indices)

    print(f"Train set: {len(train_ds)}  |  Val set: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=scintillator_collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=scintillator_collate_fn,
        drop_last=False,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    factory = MODEL_FACTORIES[args.model]
    model = factory(
        spatial_shape=ss,
        patch_size=tuple(args.patch_size),
        window_size=tuple(args.window_size),
        drop_rate=args.dropout,
        attn_drop_rate=args.attn_dropout,
        drop_path_rate=args.drop_path_rate,
        num_cls=args.num_cls,
        head_dropout=args.head_dropout,
        head_init=args.head_init,
        global_pool=args.global_pool,
    )

    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        load_pretrained_encoder(model, args.load_checkpoint, verbose=True)
    else:
        print("Training from scratch (no pretrained checkpoint loaded).")

    lightning_model = ScintillatorPIDFineTuner(model=model, args=args)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_path, args.checkpoint_name, "loss_val"),
            filename="epoch={epoch}-loss={loss/val:.4f}",
            auto_insert_metric_name=False,
            save_top_k=args.save_top_k,
            monitor="loss/val",
            mode="min",
            save_last=True,
        ),
        ModelCheckpoint(
            dirpath=os.path.join(args.checkpoint_path, args.checkpoint_name, "acc_val"),
            filename="epoch={epoch}-acc={acc/val:.4f}",
            auto_insert_metric_name=False,
            save_top_k=args.save_top_k,
            monitor="acc/val",
            mode="max",
            save_last=True,
        ),
        EarlyStopping(
            monitor="loss/val",
            patience=args.early_stop_patience,
            mode="min",
        ),
        CustomProgressBar(),
    ]

    csv_logger = CSVLogger(save_dir=os.path.join(args.save_dir, "logs"), name=args.name)
    tb_logger = TensorBoardLogger(save_dir=os.path.join(args.save_dir, "tb_logs"), name=args.name)

    pl_major = int(pl.__version__.split(".")[0])
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        accelerator="gpu",
        devices=nb_gpus,
        num_nodes=args.nb_nodes,
        precision="bf16-mixed" if pl_major >= 2 else 32,
        strategy=(
            DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
            if nb_gpus > 1
            else "auto"
        ),
        logger=[csv_logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    resume_path = (
        args.resume_checkpoint
        if args.resume_checkpoint and os.path.exists(args.resume_checkpoint)
        else None
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_path,
    )


if __name__ == "__main__":
    main()
