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
from model import MinkMAEConvNeXtV2, MinkEncConvNeXtV2, SparseEncTlLightningModel
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


torch.set_float32_matmul_precision("medium")
pl_major = int(pl.__version__.split(".")[0])


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
    train_loader, valid_loader, test_loader = split_dataset(dataset, args, splits=[0.6, 0.1, 0.3])

    # Calculate arguments for scheduler
    nb_batches = len(train_loader)
    denom = args.accum_grad_batches * nb_gpus
    #args.lr = args.lr * (args.batch_size * denom) / 256.
    args.scheduler_steps = nb_batches * args.cosine_annealing_steps // denom
    args.warmup_steps = nb_batches * args.warmup_steps // denom
    args.start_cosine_step = (nb_batches * args.epochs // denom) - args.scheduler_steps
    print(f"lr               = {args.lr}")
    print(f"scheduler_steps  = {args.scheduler_steps}")
    print(f"warmup_steps     = {args.warmup_steps}")
    print(f"start_cosine_step= {args.start_cosine_step}")
    print(f"eff. batch size  = {args.batch_size * denom}")

    # Model and LightningModule
    pretrained_model = MinkMAEConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
    base_model = MinkEncConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
    
    # Transfer weights from pretrained model to new model
    assert args.load_checkpoint is not None, "checkpoint not given as argument"
    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
    state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    pretrained_model.load_state_dict(state_dict, strict=True)
    filtered = {k: v for k, v in pretrained_model.state_dict().items() if k in base_model.state_dict()}
    if "cls_evt" in filtered:
        target_shape = base_model.state_dict()["cls_evt"].shape  # (1, num_cls, emb_dim)
        _, num_cls, _ = target_shape
        filtered["cls_evt"] = filtered["cls_evt"].repeat(1, num_cls, 1).contiguous()
    mismatched = base_model.load_state_dict(filtered, strict=False)
    print("missing keys:", mismatched.missing_keys)
    print("unexpected keys:", mismatched.unexpected_keys)
    del pretrained_model
    print("Weights transferred succesfully")

    # 1) define the list of losses you want to monitor
    monitor_losses = [
        "loss/val_total",
        #"loss/val_flavour",
        #"loss/val_charm",
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
    
    lightning_model = SparseEncTlLightningModel(model=base_model, args=args)
    
    logger    = CSVLogger(save_dir=f"{args.save_dir}/logs", name=f"{args.name}")
    tb_logger = TensorBoardLogger(save_dir=f"{args.save_dir}/tb_logs", name=f"{args.name}")
    callbacks = make_callbacks()
    
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
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
        

if __name__ == '__main__':
    main()
