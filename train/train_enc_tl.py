"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description: Updated training script with four .fit() calls for fine-tuning.
"""

import os
import torch
import pytorch_lightning as pl
from functools import partial
from utils import transfer_weights, CustomFinetuningReversed, ini_argparse, split_dataset, supervised_pixel_contrastive_loss, focal_loss, dice_loss
from dataset import SparseFASERCALDataset
from model import MinkUNetConvNeXtV2, MinkEncConvNeXtV2, SparseEncTlLightningModel
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True
        return bar


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = ini_argparse()
    args = parser.parse_args()

    # GPU setup
    nb_gpus = len(args.gpus)
    gpus = ','.join(map(str, args.gpus)) if nb_gpus > 1 else str(args.gpus[0])
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    # Dataset and splits
    dataset = SparseFASERCALDataset(args)
    train_loader, valid_loader, test_loader = split_dataset(dataset, args, splits=[0.6, 0.1, 0.3])

    # Constants for scheduler calculation
    nb_batches = len(train_loader)
    denom = args.accum_grad_batches * nb_gpus

    # Model and LightningModule
    pretrained_model = MinkUNetConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
    base_model = MinkEncConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
    
    # Transfer weights from pretrained model to new model
    assert args.load_checkpoint is not None, "checkpoint not given as argument"
    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
    state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    pretrained_model.load_state_dict(state_dict, strict=True)
    transfer_weights(pretrained_model, base_model)
    del pretrained_model
    print("Weights transferred succesfully")    

    # LightningModule
    lightning_model = SparseEncTlLightningModel(model=base_model, args=args)

    # Logger & Checkpoint
    logger = CSVLogger(save_dir=f"{args.save_dir}/logs", name=args.name)
    tb_logger = TensorBoardLogger(save_dir=f"{args.save_dir}/tb_logs", name=args.name)
    checkpoint = ModelCheckpoint(
        dirpath=f"{args.checkpoint_path}/{args.checkpoint_name}",
        save_top_k=1,
        monitor='loss/val_total',
        mode='min',
        save_last=True
    )
    progress_bar = CustomProgressBar()

    # Define fine-tuning phases
    phases = [
        ('Phase 1: heads only', (args.phase1_epochs, 0.5)),
        ('Phasee 2: + branch modules', (args.phase2_epochs, 0.25)),
        ('Phase 3: + last shared block', (args.phase3_epochs, 0.25)),
        ('Phase 4: full backbone',
         args.epochs - (args.phase1_epochs + args.phase2_epochs + args.phase3_epochs), 0.25),
    ]

    total_epochs = 0
    for name, (e, wu) in phases:
        total_epochs += e
        warmup_epochs = int(e * wu)
        warmup_steps_phase = warmup_epochs * nb_batches // denom
        scheduler_steps_phase = (e - warmup_epochs) * nb_batches // denom
        start_cosine_step_phase = warmup_steps_phase
        lightning_model.warmup_steps = warmup_steps_phase
        lightning_model.cosine_annealing_steps = scheduler_steps_phase
        lightning_model.start_cosine_step = start_cosine_step_phase

        print(f"=== {name}: {e} epochs ===")
        print(f"warmup_steps={warmup_steps_phase}, scheduler_steps={scheduler_steps_phase}, start_cosine_step={start_cosine_step_phase}")

        trainer = pl.Trainer(
            max_epochs=e,
            accelerator='gpu', devices=nb_gpus,
            strategy='ddp' if nb_gpus > 1 else 'auto',
            logger=[logger, tb_logger],
            callbacks=[checkpoint, progress_bar],
            accumulate_grad_batches=args.accum_grad_batches,
            log_every_n_steps=args.log_every_n_steps,
            deterministic=True
        )
        trainer.fit(
            lightning_model,
            train_loader,
            valid_loader
        )
        

if __name__ == '__main__':
    main()
