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
from model import MinkAEConvNeXtV2, MinkEncConvNeXtV2, SparseEncTlLightningModel
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

def compute_scheduler_args(
    nb_batches: int,
    accum_batches: int,
    nb_gpus: int,
    phase_epochs: int,
    phase_warmup: int,
    phase_cosine: int,
):
    """
    Returns (total_steps, warmup_steps, cosine_steps, start_cosine_step)
    for a single fine-tuning phase.
    """
    denom = accum_batches * nb_gpus
    total_steps = nb_batches * phase_epochs // denom
    warmup_steps = nb_batches * phase_warmup // denom
    cosine_steps = nb_batches * phase_cosine // denom
    start_cosine = total_steps - cosine_steps
    return total_steps, warmup_steps, cosine_steps, start_cosine


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = ini_argparse()
    args = parser.parse_args()
    print("\n- Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    base_lr = args.lr

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
    accum = args.accum_grad_batches

    # Model and LightningModule
    pretrained_model = MinkAEConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
    base_model = MinkEncConvNeXtV2(in_channels=1, out_channels=3, D=3, args=args)
    
    # Transfer weights from pretrained model to new model
    assert args.load_checkpoint is not None, "checkpoint not given as argument"
    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
    state_dict = {key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()}
    pretrained_model.load_state_dict(state_dict, strict=True)
    filtered = {k: v for k, v in pretrained_model.state_dict().items() if k in base_model.state_dict()}
    mismatched = base_model.load_state_dict(filtered, strict=False)
    print("missing keys:", mismatched.missing_keys)
    print("unexpected keys:", mismatched.unexpected_keys)
    del pretrained_model
    print("Weights transferred succesfully")

    iscc_token = state_dict['iscc_token']                   # shape (1, 1, d_mod)
    num_tasks = base_model.num_tasks
    d_mod     = iscc_token.size(-1)
    cls_task_weights = iscc_token.repeat(1, num_tasks, 1)   # (1, num_tasks, d_mod)
    with torch.no_grad():
        base_model.cls_task.data.copy_(cls_task_weights)
    print("Cls tokens transferred too")  

    # 1) define the list of losses you want to monitor (same for all phases)
    monitor_losses = [
        "loss/val_total",
        "loss/val_flavour",
        "loss/val_charm",
        "loss/val_e_vis_cc",
        "loss/val_e_vis_nc",
        "loss/val_pt_miss"
        "loss/val_jet_momentum_dir",
        "loss/val_jet_momentum_mag",
        "loss/val_lepton_momentum_dir",
        "loss/val_lepton_momentum_mag",
    ]
    
    # 2) helper to build a fresh checkpoint callback list for each phase
    def make_callbacks(phase_name):
        cbs = []
        for loss in monitor_losses:
            cbs.append(
                ModelCheckpoint(
                    dirpath=f"{args.checkpoint_path}/{args.checkpoint_name}/{phase_name}/{loss.replace('/', '_')}",
                    save_top_k=args.save_top_k,
                    monitor=loss,
                    mode="min",
                    save_last=(loss == "loss/val_total"),
                )
            )
        progress_bar = CustomProgressBar()
        cbs.append(progress_bar)
        return cbs
    
    # 3) Instantiate your LightningModule
    args.lr = base_lr * 10
    lightning_model = SparseEncTlLightningModel(model=base_model, args=args)

    # 4) Phase 1: train only the new heads + task‐transformer
    _, w1, c1, s1 = compute_scheduler_args(
        nb_batches, accum, nb_gpus,
        phase_epochs=args.phase1_epochs,
        phase_warmup=args.phase1_epochs//2,
        phase_cosine=args.phase1_epochs//2,
    )
    lightning_model.warmup_steps = w1
    lightning_model.cosine_annealing_steps = c1
    lightning_model.start_cosine_step = s1
    lightning_model.freeze_phase1()
    
    logger1   = CSVLogger(save_dir=f"{args.save_dir}/logs",     name=f"{args.name}_phase1")
    tb_logger1= TensorBoardLogger(save_dir=f"{args.save_dir}/tb_logs", name=f"{args.name}_phase1")
    callbacks1= make_callbacks("phase1")
    
    trainer1 = pl.Trainer(
        max_epochs=args.phase1_epochs,
        callbacks=callbacks1,
        accelerator="gpu",
        devices=nb_gpus,
        precision="bf16-mixed" if pl_major >= 2 else 32,
        strategy="ddp" if nb_gpus > 1 else "auto",
        logger=[logger1, tb_logger1],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        accumulate_grad_batches=args.accum_grad_batches,
    )
    trainer1.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    
    # 5) Phase 2: unfreeze rest of the model
    _, w2, c2, s2 = compute_scheduler_args(
        nb_batches, accum, nb_gpus,
        phase_epochs=args.phase2_epochs,
        phase_warmup=3,
        phase_cosine=args.phase2_epochs-3,
    )
    lightning_model.lr = base_lr
    lightning_model.warmup_steps = w2
    lightning_model.cosine_annealing_steps = c2
    lightning_model.start_cosine_step = s2
    lightning_model.unfreeze_phase2()
    
    logger2    = CSVLogger(save_dir=f"{args.save_dir}/logs",     name=f"{args.name}_phase2")
    tb_logger2 = TensorBoardLogger(save_dir=f"{args.save_dir}/tb_logs", name=f"{args.name}_phase2")
    callbacks2 = make_callbacks("phase2")
    
    trainer2 = pl.Trainer(
        max_epochs=args.phase2_epochs,
        callbacks=callbacks2,
        accelerator="gpu",
        devices=nb_gpus,
        precision="bf16-mixed" if pl_major >= 2 else 32,
        strategy="ddp" if nb_gpus > 1 else "auto",
        logger=[logger2, tb_logger2],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        accumulate_grad_batches=args.accum_grad_batches,
    )
    trainer2.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
        

if __name__ == '__main__':
    main()
