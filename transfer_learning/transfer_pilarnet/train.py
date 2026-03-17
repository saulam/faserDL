"""Training entry point for the PILArNet transfer study."""

import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping

from .adapted_model import pilarnet_encoder_tiny, pilarnet_encoder_base
from .checkpoint_utils import load_pretrained_encoder
from .pilarnet_dataset import (
    PILArNetParticleDataset,
    PILArNetMultiParticleDataset,
    PILARNET_SINGLE_PARTICLE_META_DIM,
    pilarnet_collate_fn,
    pilarnet_multiparticle_collate_fn,
)
from .lightning_module import PILArNetFineTuner
from .lightning_module_multiparticle import PILArNetMultiParticleFineTuner
from .multiparticle_model import (
    pilarnet_multiparticle_tiny,
    pilarnet_multiparticle_base,
)

torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

SINGLE_MODEL_FACTORIES = {
    "tiny": pilarnet_encoder_tiny,
    "base": pilarnet_encoder_base,
}

MULTI_MODEL_FACTORIES = {
    "tiny": pilarnet_multiparticle_tiny,
    "base": pilarnet_multiparticle_base,
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
    p = argparse.ArgumentParser(description="PILArNet particle-level fine-tuning")

    # Data
    p.add_argument("--manifest", type=str, required=True, help="Path to manifest .npz")
    p.add_argument("--val_manifest", type=str, default=None,
                   help="Optional separate validation manifest. If not given, a split is used.")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--spatial_shape", type=int, nargs=3, default=[168, 168, 180])
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--charge_metadata_path", type=str, default=None,
                   help="Transfer charge metadata pickle containing q_log1p stats.")
    p.add_argument("--min_voxels", type=int, default=5)
    p.add_argument("--task_mode", type=str, choices=["single", "multi"], default="single")
    p.add_argument("--min_particles_per_event", type=int, default=2)
    p.add_argument("--max_particles_per_event", type=int, default=None)

    # Model
    p.add_argument("--model", type=str, choices=list(SINGLE_MODEL_FACTORIES.keys()), default="base")
    p.add_argument("--patch_size", type=int, nargs=3, default=[12, 12, 10])
    p.add_argument("--window_size", type=int, nargs=3, default=[2, 2, 3])
    p.add_argument("--num_cls", type=int, default=2)
    p.add_argument("--drop_path_rate", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--attn_dropout", type=float, default=0.0)
    p.add_argument("--head_dropout_cls", type=float, default=0.0)
    p.add_argument("--head_init", type=float, default=2e-5)
    p.add_argument("--global_pool", action="store_true", default=True)
    p.add_argument("--context_layers", type=int, default=2)
    p.add_argument("--context_heads", type=int, default=12)
    p.add_argument("--context_dropout", type=float, default=0.1)
    p.add_argument("--meta_dropout", type=float, default=0.1)
    p.add_argument("--single_particle_meta", action="store_true", default=False)

    # Checkpoint
    p.add_argument("--load_checkpoint", type=str, default=None)
    p.add_argument("--resume_checkpoint", type=str, default=None)

    # Training
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--blr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--cosine_annealing_epochs", type=int, default=70)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--layer_decay", type=float, default=0.95)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--accum_grad_batches", type=int, default=1)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # Logging / checkpoints
    p.add_argument("--save_dir", type=str, default="transfer_learning/transfer_pilarnet/logs_pilarnet")
    p.add_argument("--name", type=str, default="pilarnet_pid")
    p.add_argument("--checkpoint_path", type=str, default="transfer_learning/transfer_pilarnet/checkpoints_pilarnet")
    p.add_argument("--checkpoint_name", type=str, default="pilarnet_pid")
    p.add_argument("--save_top_k", type=int, default=1)
    p.add_argument("--log_every_n_steps", type=int, default=10)
    p.add_argument("--early_stop_patience", type=int, default=10)

    # Hardware
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--nb_nodes", type=int, default=1)

    # Augmentations
    p.add_argument("--augment", action="store_true", default=False)

    # Detector grid
    p.add_argument("--grid_n", type=int, default=768,
                   help="Detector voxel grid size (768 for PILArNet).")

    return p.parse_args()


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()

    print("\n--- Arguments ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # GPU setup
    nb_gpus = len(args.gpus)
    gpus = ",".join(map(str, args.gpus)) if nb_gpus > 1 else str(args.gpus[0])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Dataset
    ss = tuple(args.spatial_shape)
    dataset_cls = (
        PILArNetParticleDataset
        if args.task_mode == "single"
        else PILArNetMultiParticleDataset
    )
    dataset_kwargs = dict(
        manifest_path=args.manifest,
        spatial_shape=ss,
        min_voxels=args.min_voxels,
        augment=args.augment,
        cache_dir=args.cache_dir,
        grid_n=args.grid_n,
        charge_metadata_path=args.charge_metadata_path,
    )
    if args.task_mode == "single":
        dataset_kwargs["include_particle_meta"] = args.single_particle_meta
    else:
        dataset_kwargs.update(
            min_particles_per_event=args.min_particles_per_event,
            max_particles_per_event=args.max_particles_per_event,
        )
    full_ds = dataset_cls(**dataset_kwargs)

    if args.val_manifest is not None:
        val_kwargs = dict(dataset_kwargs)
        val_kwargs["manifest_path"] = args.val_manifest
        val_kwargs["augment"] = False
        val_ds = dataset_cls(**val_kwargs)
        train_ds = full_ds
    else:
        n_val = int(len(full_ds) * args.val_fraction)
        n_train = len(full_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        # Disable augmentation on the validation split
        val_kwargs = dict(dataset_kwargs)
        val_kwargs["augment"] = False
        val_ds.dataset = dataset_cls(**val_kwargs)

    split_unit = "particles" if args.task_mode == "single" else "events"
    print(f"Train set: {len(train_ds)} {split_unit}  |  Val set: {len(val_ds)} {split_unit}")
    if args.task_mode == "multi":
        print("Batch size is interpreted as number of events in multi-particle mode.")

    collate_fn = (
        pilarnet_collate_fn
        if args.task_mode == "single"
        else pilarnet_multiparticle_collate_fn
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Model
    model_kwargs = dict(
        spatial_shape=ss,
        patch_size=tuple(args.patch_size),
        window_size=tuple(args.window_size),
        drop_rate=args.dropout,
        attn_drop_rate=args.attn_dropout,
        drop_path_rate=args.drop_path_rate,
        num_cls=args.num_cls,
        head_dropout_cls=args.head_dropout_cls,
        head_init=args.head_init,
        global_pool=args.global_pool,
    )
    if args.task_mode == "single":
        factory = SINGLE_MODEL_FACTORIES[args.model]
        if args.single_particle_meta:
            model_kwargs.update(
                meta_dim=PILARNET_SINGLE_PARTICLE_META_DIM,
                meta_dropout=args.meta_dropout,
            )
    else:
        factory = MULTI_MODEL_FACTORIES[args.model]
        model_kwargs.update(
            meta_dropout=args.meta_dropout,
            context_layers=args.context_layers,
            context_heads=args.context_heads,
            context_dropout=args.context_dropout,
        )
    model = factory(**model_kwargs)

    # Checkpoint loading
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        target_model = model if args.task_mode == "single" else model.encoder
        load_pretrained_encoder(target_model, args.load_checkpoint, verbose=True)
    else:
        print("Training from scratch (no pretrained checkpoint loaded).")

    # Lightning module
    if args.task_mode == "single":
        lightning_model = PILArNetFineTuner(model=model, args=args)
    else:
        lightning_model = PILArNetMultiParticleFineTuner(model=model, args=args)

    # Callbacks
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

    # Loggers
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
