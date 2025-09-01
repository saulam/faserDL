"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: pre-training script.
"""

import json
import os
import torch
import pytorch_lightning as pl
from pathlib import Path
from utils import ini_argparse, split_dataset, create_loader
from dataset import *
from model import *
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping 


torch.set_float32_matmul_precision("medium")
pl_major = int(pl.__version__.split(".")[0])
MODEL_FACTORIES = {
    'tiny':  mae_vit_tiny,
    'base':  mae_vit_base,
    'large': mae_vit_large,
    'huge':  mae_vit_huge,
}


class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used for validation
        return bar


def shard_pattern(split, meta, out_dir):
    n = meta["splits"][split]["num_shards"]
    # If no shards, return an empty pattern
    if n == 0:
        return ""
    # zero-based inclusive brace range, e.g. {0000..0017}
    start = "0000"
    end = f"{n-1:04d}"
    return str(out_dir / f"{split}-{{{start}..{end}}}.tar")


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = ini_argparse(MODEL_FACTORIES)
    args = parser.parse_args()
    print("\n- Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # GPU setup
    nb_gpus = len(args.gpus)
    gpus = ', '.join(args.gpus) if nb_gpus > 1 else str(args.gpus[0])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Dataset
    if args.web_dataset_path is not None:
        print("Iterable dataset")
        args.web_dataset_path = Path(args.web_dataset_path)
        with open(args.web_dataset_path / "metadata.json") as f:
            meta = json.load(f)
        train_pat = shard_pattern("train", meta, args.web_dataset_path)
        val_pat   = shard_pattern("val", meta, args.web_dataset_path)
        train_set = SparseFASERCALIterableDataset(
            args, 'train', meta=meta, shard_pattern=train_pat, shardshuffle=args.shardshuffle, shuffle=args.shuffle
        )
        val_set = SparseFASERCALIterableDataset(
            args, 'val', meta=meta, shard_pattern=val_pat, shardshuffle=args.shardshuffle, shuffle=args.shuffle
        )
        train_loader = create_loader(train_set, shuffle=False, drop_last=True, args=args)
        valid_loader = create_loader(val_set, shuffle=False, drop_last=True, args=args)
        metadata = train_set.metadata
        nb_batches_train = len(train_set) // args.batch_size
        nb_batches_val = len(val_set) // args.batch_size
    else:
        print("Standard dataset")
        dataset = SparseFASERCALMapDataset(args)
        print("- Dataset size: {} events".format(len(dataset)))
        train_loader, valid_loader, _ = split_dataset(
            dataset, args, splits=[0.85, 0.05, 0.1],
        )
        metadata = dataset.metadata
        nb_batches_train = len(train_loader)
        nb_batches_val = len(valid_loader)

    # Calculate arguments for scheduler
    denom = args.accum_grad_batches * nb_gpus
    if args.blr is not None:
        # overwrite lr by linearly-scaled blr if args.blr is defined
        args.lr = args.blr * (args.batch_size * denom) / 256.
    args.scheduler_steps = nb_batches_train * args.cosine_annealing_steps // denom
    args.warmup_steps = nb_batches_train * args.warmup_steps // denom
    args.start_cosine_step = (nb_batches_train * args.epochs // denom) - args.scheduler_steps
    print(f"lr                = {args.lr}")
    print(f"scheduler_steps   = {args.scheduler_steps}")
    print(f"warmup_steps      = {args.warmup_steps}")
    print(f"start_cosine_step = {args.start_cosine_step}")
    print(f"eff. batch size   = {args.batch_size * denom}")

    # Initialise the model
    model = args.model(
        drop_rate = args.dropout,
        attn_drop_rate = args.attn_dropout,
        drop_path_rate = args.drop_path_rate,
        drop_rate_dec = args.dropout_dec,
        attn_drop_rate_dec = args.attn_dropout_dec,
        drop_path_rate_dec = args.drop_path_rate_dec,
    )
    #print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params model (total): {}".format(total_params))

    # Define logger and checkpoint
    logger = CSVLogger(save_dir=args.save_dir + "/logs", name=args.name)
    tb_logger = TensorBoardLogger(save_dir=args.save_dir + "/tb_logs", name=args.name)
    callbacks = []
    monitored_losses = [
            'loss/val_total',
    ]
    for loss_name in monitored_losses:
        checkpoint = ModelCheckpoint(
            dirpath=f"{args.checkpoint_path}/{args.checkpoint_name}/{loss_name.replace('/', '_')}",
            save_top_k=args.save_top_k,
            monitor=loss_name,
            mode="min",
            save_last=True if "total" in loss_name else False 
        )
        callbacks.append(checkpoint)    

    # Rest of callbacks
    progress_bar = CustomProgressBar()
    callbacks.append(progress_bar)
    if args.early_stop_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor='loss/val_total',
            patience=args.early_stop_patience,
            verbose=True,
            mode='min' 
        )
        callbacks.append(early_stop_callback)
    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))

    # Lightning model
    lightning_model = MAEPreTrainer(
        model=model,
        metadata=metadata,
        args=args)
 
    # Initialise PyTorch Lightning trainer
    trainer = pl.Trainer(
        limit_train_batches=nb_batches_train//denom,
        limit_val_batches=nb_batches_val//denom,
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=nb_gpus,
        precision="bf16-mixed" if pl_major >= 2 else 32,
        strategy="ddp" if nb_gpus > 1 else "auto",
        logger=[logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    # Train and validate the model
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args.load_checkpoint if args.load_checkpoint else None,
    )


if __name__ == "__main__":
    main()
