"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: Pre-training script with distance-aware losses.
             Uses balanced parameters for complete training with spatial awareness.
"""

import json
import os
import torch
import pytorch_lightning as pl
from pathlib import Path
from utils import ini_argparse, split_dataset, create_loader, SplitTensorBoardLogger
from dataset import *
from model import *
from model.lightning_model_pretrain import MAEPreTrainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping 


torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision("high")
pl_major = int(pl.__version__.split(".")[0])
MODEL_FACTORIES = {
    'tiny':  mae_vit_tiny,
    'base':  mae_vit_base,
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
        dataset = train_set
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

    # NOTE: LR scaling (blr), warmup steps, and scheduler steps are now computed
    # inside the Lightning model's configure_optimizers() using
    # self.trainer.estimated_stepping_batches.

    # Initialise the model
    model = args.model(
        drop_rate = args.dropout,
        attn_drop_rate = args.attn_dropout,
        drop_path_rate = args.drop_path_rate,
        drop_rate_dec = args.dropout_dec,
        attn_drop_rate_dec = args.attn_dropout_dec,
        metadata = metadata,
        sparse_ecal = args.sparse_ecal,
    )
    #print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params model (total): {}".format(total_params))
    
    # Checkpoint
    callbacks = []
    monitored_losses = [
            'loss_total/val',
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
            monitor='loss_total/val',
            patience=args.early_stop_patience,
            verbose=True,
            mode='min' 
        )
        callbacks.append(early_stop_callback)

    logger = CSVLogger(save_dir=args.save_dir + "/logs", name=args.name)
    tb_logger = SplitTensorBoardLogger(   
        save_dir=f"{args.save_dir}/tb_logs",
        name=f"{args.name}",
        other_target="train",
        strip_suffix=True,
        val_suffix = "_epoch",
    )
    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))

    # Lightning model with distance-aware losses
    lightning_model = MAEPreTrainer(
        model=model,
        dataset=dataset,
        args=args)
 
    # Initialise PyTorch Lightning trainer
    limit_train_batches = None
    if os.getenv("PRETRAIN_LIMIT_TRAIN_BATCHES"):
        limit_train_batches = int(os.environ["PRETRAIN_LIMIT_TRAIN_BATCHES"])
    elif args.web_dataset_path:
        limit_train_batches = nb_batches_train//(nb_gpus*args.nb_nodes)

    limit_val_batches = None
    if os.getenv("PRETRAIN_LIMIT_VAL_BATCHES"):
        limit_val_batches = int(os.environ["PRETRAIN_LIMIT_VAL_BATCHES"])
    elif args.web_dataset_path:
        limit_val_batches = nb_batches_val//(nb_gpus*args.nb_nodes)

    precision = os.getenv("PRETRAIN_PRECISION")
    if precision is None:
        precision = "bf16-mixed" if pl_major >= 2 else 32
    else:
        try:
            precision = int(precision)
        except ValueError:
            pass

    trainer = pl.Trainer(
        # For IterableDataset (webdataset), Lightning cannot determine epoch length
        # automatically, so we provide the number of batches per device explicitly.
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        accelerator="gpu",
        num_nodes=args.nb_nodes,
        devices=nb_gpus,
        precision=precision,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=False
        ) if nb_gpus > 1 else "auto",
        logger=[logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    # Load pre-trained weights (start fresh training)
    if args.load_checkpoint is not None and os.path.exists(args.load_checkpoint):
        checkpoint = torch.load(args.load_checkpoint, map_location='cpu', weights_only=True)
        msg = lightning_model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"Loaded pre-trained weights from: {args.load_checkpoint}")
        print(f"Load msg: {msg}")

    # Resume training from checkpoint (restores optimiser, epoch, etc.)
    resume_path = args.resume_checkpoint if args.resume_checkpoint and os.path.exists(args.resume_checkpoint) else None

    # Train and validate the model
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=resume_path,
    )


if __name__ == "__main__":
    main()
