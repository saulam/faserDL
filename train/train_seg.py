"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 09.24

Description: Training script.
"""


import os
import torch
import pytorch_lightning as pl
from functools import partial
from utils import CustomFinetuningReversed, ini_argparse, split_dataset, supervised_pixel_contrastive_loss, focal_loss, dice_loss
from dataset import SparseFASERCALDatasetSeg
from model import MinkUNetConvNeXtV2, SparseSegLightningModel
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = ini_argparse()
    args = parser.parse_args()

    print("\n- Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    nb_gpus = len(args.gpus)
    gpus = [int(gpu) for gpu in args.gpus]

    # Dataset
    dataset = SparseFASERCALDatasetSeg(args)
    print("- Dataset size: {} events".format(len(dataset)))
    train_loader, valid_loader, test_loader = split_dataset(dataset, args, splits=[0.6, 0.1, 0.3]) 

    # Define loss functions
    if args.contrastive and not args.finetuning:
        loss_fn = supervised_pixel_contrastive_loss
    else:
        loss_fn = []
        for loss in args.losses:
            if loss == "focal":
                loss_fn.append(partial(focal_loss, sigmoid=args.sigmoid, reduction="mean"))
            elif loss == "dice":
                loss_fn.append(partial(dice_loss, sigmoid=args.sigmoid, reduction="mean"))
            else:
                raise ValueError("Wrong loss")

    # Calculate arguments for scheduler
    nb_batches = len(train_loader)
    args.scheduler_steps = nb_batches * (args.epochs - args.warmup_steps) // (args.accum_grad_batches * nb_gpus)
    args.warmup_steps = nb_batches * args.warmup_steps // (args.accum_grad_batches * nb_gpus)

    # Initialize the model
    model = MinkUNetConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
    #print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params model (total): {}".format(total_params))

    # Define logger and checkpoint
    logger = CSVLogger(save_dir=args.save_dir + "/logs", name=args.name)
    tb_logger = TensorBoardLogger(save_dir=args.save_dir + "/tb_logs", name=args.name)
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_path + "/" + args.checkpoint_name,
        save_last=True, save_top_k=args.save_top_k, monitor="loss/val_total")
    callbacks = [checkpoint_callback]

    if args.finetuning:
        # load pre-trained model if fine-tuning 
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        # Remove the "model." prefix from the keys in the state_dict
        state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
        filtered_state_dict = {key: value for key, value in state_dict.items() if "cls_layer" not in key}
        model.load_state_dict(filtered_state_dict, strict=False)
        print("Loaded model weights of: {}".format(args.pretrained_path))
        finetuning_callback = CustomFinetuningReversed(unfreeze_at_epoch=args.unfreeze_at_epoch, 
            gradual_unfreeze_steps=args.gradual_unfreeze_steps, lr_factor=args.lr_factor, unfreeze_all=args.unfreeze_all)
        callbacks.append(finetuning_callback)

    # Lightning model
    lightning_model = SparseSegLightningModel(model=model,
        loss_fn=loss_fn,
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
        devices=gpus,
        strategy="ddp" if nb_gpus > 1 else None,
        logger=[logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    # Train and validate the model
    trainer.fit(model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()

