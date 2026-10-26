"""
Description: I-JEPA-style pretraining of the spconv-free flash-attention
encoder (model/flash_jepa.py).  Runs without spconv, timm or torch_ema, e.g.
in the neutrino-jepa environment.
"""

import os
import torch
import pytorch_lightning as pl
from utils import (
    ini_argparse,
    add_flash_args,
    split_dataset,
    SplitTensorBoardLogger,
    resolve_checkpoint_path,
)
from dataset import SparseFASERCALMapDataset
from model.flash_vit import FLASH_ENCODER_PRESETS, collate_tokens
from model.flash_jepa import FlashJEPA
from model.lightning_model_flash import FlashJEPAPreTrainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping


torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
MODEL_PRESETS = {name: name for name in FLASH_ENCODER_PRESETS}


class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True
        return bar


def build_flash_jepa(args):
    return FlashJEPA(
        **FLASH_ENCODER_PRESETS[args.model],
        fcal_patch_size=args.flash_fcal_patch_size,
        ahcal_patch_size=args.flash_ahcal_patch_size,
        drop_rate=args.dropout,
        attn_drop_rate=args.attn_dropout,
        drop_path_rate=args.drop_path_rate,
        predictor_embed_dim=args.flash_predictor_embed_dim,
        predictor_depth=args.flash_predictor_depth,
        predictor_num_heads=args.flash_predictor_num_heads,
        mask_num_blocks=args.flash_mask_num_blocks,
        mask_block_scale=tuple(args.flash_mask_block_scale),
        mask_block_aspect=tuple(args.flash_mask_block_aspect),
        mask_global_prob=args.flash_mask_global_prob,
    )


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = add_flash_args(ini_argparse(MODEL_PRESETS))
    args = parser.parse_args()
    if args.web_dataset_path is not None:
        raise NotImplementedError("The flash-attention models read the map-style dataset (--dataset_path).")
    args.backbone = "flash"
    args.pretraining_objective = "jepa"
    pl.seed_everything(args.seed, workers=True)

    load_path = resolve_checkpoint_path(args.load_checkpoint, "Load")
    resume_path = resolve_checkpoint_path(args.resume_checkpoint, "Resume")

    print("\n- Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # GPU setup
    nb_gpus = len(args.gpus)
    gpus = ', '.join(args.gpus) if nb_gpus > 1 else str(args.gpus[0])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Dataset: JEPA reads only the detector inputs, so the relational voxel
    # labels of the MAE (stage 1) are not built.
    dataset = SparseFASERCALMapDataset(args)
    dataset.stage1 = False
    print("- Dataset size: {} events".format(len(dataset)))
    train_loader, valid_loader, _ = split_dataset(
        dataset, args, splits=[0.85, 0.05, 0.1], collate_fn=collate_tokens,
    )

    model = build_flash_jepa(args)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.context_encoder.parameters())
    print("Total trainable params model: {}".format(trainable_params))
    print("Encoder params: {}".format(encoder_params))
    args.flash_encoder_config = model.encoder_config

    # The EMA target changes throughout training, so the validation loss is not
    # comparable across epochs: keep last.ckpt (save_top_k=0 by default).
    callbacks = [
        ModelCheckpoint(
            dirpath=f"{args.checkpoint_path}/{args.checkpoint_name}/loss_total_val",
            save_top_k=args.save_top_k,
            monitor="loss_total/val",
            mode="min",
            save_last=True,
        ),
        CustomProgressBar(),
    ]
    if args.early_stop_patience > 0:
        callbacks.append(EarlyStopping(
            monitor='loss_total/val', patience=args.early_stop_patience, verbose=True, mode='min'))

    logger = CSVLogger(save_dir=args.save_dir + "/logs", name=args.name)
    tb_logger = SplitTensorBoardLogger(
        save_dir=f"{args.save_dir}/tb_logs",
        name=f"{args.name}",
        other_target="train",
        strip_suffix=True,
        val_suffix="_epoch",
    )
    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))

    lightning_model = FlashJEPAPreTrainer(model=model, args=args)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        accelerator="gpu",
        num_nodes=args.nb_nodes,
        devices=nb_gpus,
        precision="bf16-mixed",
        strategy=DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=False,
        ) if nb_gpus > 1 else "auto",
        logger=[logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,
        accumulate_grad_batches=args.accum_grad_batches,
        fast_dev_run=args.fast_dev_run,
    )

    # Load pre-trained weights (start fresh training)
    if load_path is not None:
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)
        msg = lightning_model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"Loaded pre-trained weights from: {load_path}")
        print(f"Load msg: {msg}")

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=resume_path,
    )


if __name__ == "__main__":
    main()
