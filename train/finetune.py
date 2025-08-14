"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 07.25

Description: fine-tuning script.
"""

import os
import torch
import pytorch_lightning as pl
from functools import partial
from utils import ini_argparse, split_dataset
from dataset import SparseFASERCALDataset
from model import *
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


torch.set_float32_matmul_precision("medium")
pl_major = int(pl.__version__.split(".")[0])
MODEL_FACTORIES = {
    'base':  vit_base,
    'large': vit_large,
    'huge':  vit_huge,
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


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = ini_argparse(MODEL_FACTORIES)
    args = parser.parse_args()
    print("\n- Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # GPU setup
    nb_gpus = len(args.gpus)
    gpus = ','.join(map(str, args.gpus)) if nb_gpus > 1 else str(args.gpus[0])
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    # Dataset and splits
    dataset = SparseFASERCALDataset(args)
    extra_dataset = None
    if args.extra_dataset_path is not None:
        args.dataset_path = args.extra_dataset_path
        extra_dataset = SparseFASERCALDataset(args)
        
    print("- Dataset size: {} events".format(len(dataset)))
    train_loader, valid_loader, test_loader = split_dataset(
        dataset, args, splits=[0.6, 0.1, 0.3], extra_dataset=extra_dataset
    )

    # Calculate arguments for scheduler
    nb_batches = len(train_loader)
    denom = args.accum_grad_batches * nb_gpus
    if args.blr is not None:
        # overwrite lr by linearly-scaled blr if args.blr is defined
        args.lr = args.blr * (args.batch_size * denom) / 256.
    args.scheduler_steps = nb_batches * args.cosine_annealing_steps // denom
    args.warmup_steps = nb_batches * args.warmup_steps // denom
    args.start_cosine_step = (nb_batches * args.epochs // denom) - args.scheduler_steps
    print(f"lr                = {args.lr}")
    print(f"scheduler_steps   = {args.scheduler_steps}")
    print(f"warmup_steps      = {args.warmup_steps}")
    print(f"start_cosine_step = {args.start_cosine_step}")
    print(f"eff. batch size   = {args.batch_size * denom}")

    # Transfer weights from pre-trained model
    model = args.model(
        drop_rate = args.dropout,
        drop_path_rate = args.drop_path_rate,
        metadata = dataset.metadata,
    )
    assert args.load_checkpoint is not None, "checkpoint not given as argument"
    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
    state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    filtered = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    mismatched = model.load_state_dict(filtered, strict=False)
    print("missing keys:", mismatched.missing_keys)
    print("unexpected keys:", mismatched.unexpected_keys)
    print("Weights transferred succesfully")

    # define the list of losses to monitor
    monitor_losses = [
        "loss/val_total",
        "loss/val_flavour",
        #"loss/val_charm",
        #"loss/val_vis_sp_momentum_mag",
        #"loss/val_vis_sp_momentum_dir",
        #"loss/val_lepton_momentum_mag",
        #"loss/val_lepton_momentum_dir",
        #"loss/val_e_vis",
        #"loss/val_pt_miss",
        #"loss/val_jet_momentum_dir",
        #"loss/val_jet_momentum_mag",
        #"loss/val_lepton_momentum_dir",
        #"loss/val_lepton_momentum_mag",
    ]
    
    # helper to build a fresh checkpoint callback list
    def make_callbacks():
        cbs = []
        for loss in monitor_losses:
            cbs.append(
                ModelCheckpoint(
                    dirpath=f"{args.checkpoint_path}/{args.checkpoint_name}/{loss.replace('/', '_')}",
                    save_top_k=args.save_top_k,
                    monitor=loss,
                    mode="min",
                    save_last=(loss == "loss/val_total"),
                )
            )
        progress_bar = CustomProgressBar()
        cbs.append(progress_bar)
        return cbs

    # Rest of callbacks
    logger    = CSVLogger(save_dir=f"{args.save_dir}/logs", name=f"{args.name}")
    tb_logger = TensorBoardLogger(save_dir=f"{args.save_dir}/tb_logs", name=f"{args.name}")
    callbacks = make_callbacks()
    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))

    # Lightning model
    lightning_model = ViTFineTuner(model=model, args=args)

    # Initialise PyTorch Lightning trainer
    trainer = pl.Trainer(
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
    )
        

if __name__ == '__main__':
    main()
