"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: Training script - stage 1: semantic segmentation.
"""


import os
import torch
import pytorch_lightning as pl
from functools import partial
from utils import CustomFinetuningReversed, ini_argparse, split_dataset
from dataset import SparseFASERCALDataset
from model import MinkMAEConvNeXtV2, SparseMAELightningModel
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping 


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


def main():
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
    denom = args.accum_grad_batches * nb_gpus
    args.lr = args.lr * (args.batch_size * denom) / 256.
    args.scheduler_steps = nb_batches * args.cosine_annealing_steps // denom
    args.warmup_steps = nb_batches * args.warmup_steps // denom
    args.start_cosine_step = (nb_batches * args.epochs // denom) - args.scheduler_steps
    print(f"lr                = {args.lr}")
    print(f"scheduler_steps   = {args.scheduler_steps}")
    print(f"warmup_steps      = {args.warmup_steps}")
    print(f"start_cosine_step = {args.start_cosine_step}")
    print(f"eff. batch size   = {args.batch_size * denom}")

    # Initialize the model
    model = MinkMAEConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
    #print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params model (total): {}".format(total_params))

    # Define logger and checkpoint
    logger = CSVLogger(save_dir=args.save_dir + "/logs", name=args.name)
    tb_logger = TensorBoardLogger(save_dir=args.save_dir + "/tb_logs", name=args.name)
    callbacks = []
    monitored_losses = [
            'loss/val_occupancy',
            'loss/val_charge',
            'loss/val_primlepton',
            'loss/val_seg',
            'loss/val_iscc',
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

    # Lightning model
    lightning_model = SparseMAELightningModel(
        model=model,
        metadata=dataset.metadata,
        args=args)

    # Log the hyperparameters
    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))
 
    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        #num_sanity_val_steps=0,
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=nb_gpus,
        precision="bf16-mixed" if pl_major >= 2 else 32,
        strategy="ddp" if nb_gpus > 1 else "auto",
        logger=[logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
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

