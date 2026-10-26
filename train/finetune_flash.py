"""
Description: fine-tuning of the spconv-free flash-attention encoder
(model/flash_vit.py) with the published recipe: the same ViTFineTuner losses,
task weighting, layer-wise LR decay, EMA and checkpoint selection as
train/finetune.py.  Runs without spconv, timm or torch_ema.
"""

import os
import torch
import pytorch_lightning as pl
from utils import (
    ini_argparse,
    add_flash_args,
    split_dataset,
    SplitTensorBoardLogger,
    load_mae_encoder,
    resolve_checkpoint_path,
    checkpoint_monitors,
)
from dataset import SparseFASERCALMapDataset
from model.flash_vit import FLASH_ENCODER_PRESETS, FlashViT, collate_tokens
from model.lightning_model_flash import FlashViTFineTuner
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


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


def build_flash_vit(args, metadata):
    return FlashViT(
        **FLASH_ENCODER_PRESETS[args.model],
        fcal_patch_size=args.flash_fcal_patch_size,
        ahcal_patch_size=args.flash_ahcal_patch_size,
        drop_rate=args.dropout,
        attn_drop_rate=args.attn_dropout,
        drop_path_rate=args.drop_path_rate,
        head_init=args.head_init,
        head_dropout_cls=args.head_dropout_cls,
        head_dropout_reg=args.head_dropout_reg,
        metadata=metadata,
    )


def require_matching_encoder(model, checkpoint, path):
    """The fine-tuned encoder must be exactly the pretrained architecture."""
    stored = checkpoint.get("flash_encoder_config")
    if stored is None:
        raise RuntimeError(
            f"{path} is not a flash-attention JEPA checkpoint (no flash_encoder_config); "
            "spconv MAE/JEPA checkpoints are fine-tuned with train/finetune.py.")
    if dict(stored) != model.encoder_config:
        raise RuntimeError(
            f"The pretrained encoder {dict(stored)} differs from the fine-tuning encoder "
            f"{model.encoder_config}; pass the same --model and --flash_*_patch_size as in pretraining.")


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = add_flash_args(ini_argparse(MODEL_PRESETS))
    args = parser.parse_args()
    if args.web_dataset_path is not None:
        raise NotImplementedError("The flash-attention models read the map-style dataset (--dataset_path).")
    args.backbone = "flash"
    pl.seed_everything(args.seed, workers=True)
    load_path = resolve_checkpoint_path(args.load_checkpoint, "Pretraining")
    resume_path = resolve_checkpoint_path(args.resume_checkpoint, "Resume")
    print("\n- Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # GPU setup
    nb_gpus = len(args.gpus)
    gpus = ','.join(map(str, args.gpus)) if nb_gpus > 1 else str(args.gpus[0])
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    # Dataset (same split as train/finetune.py)
    dataset = SparseFASERCALMapDataset(args)
    print("- Dataset size: {} events".format(len(dataset)))
    train_loader, valid_loader, _ = split_dataset(
        dataset, args, splits=[0.85, 0.05, 0.1], collate_fn=collate_tokens,
    )
    metadata = dataset.metadata

    model = build_flash_vit(args, metadata)
    args.flash_encoder_config = model.encoder_config

    # Load the pretrained encoder and record transfer provenance in hparams.
    args.pretrained_encoder_resolved_source = None
    args.pretrained_encoder_coverage = None
    args.pretrained_checkpoint_epoch = None
    args.pretrained_checkpoint_global_step = None
    args.pretrained_checkpoint_type = None
    if load_path is not None:
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)
        require_matching_encoder(model, checkpoint, load_path)
        load_report = load_mae_encoder(
            model,
            checkpoint,
            encoder_source=args.pretrained_encoder_source,
            require_full_encoder=True,
        )
        args.pretrained_encoder_resolved_source = load_report["source"]
        args.pretrained_encoder_coverage = load_report["encoder_coverage"]
        args.pretrained_checkpoint_epoch = checkpoint.get("epoch")
        args.pretrained_checkpoint_global_step = checkpoint.get("global_step")
        args.pretrained_checkpoint_type = (
            "jepa" if load_report["source"] in {"context", "target"} else "mae"
        )
    else:
        print("Training from scratch!")

    def make_callbacks():
        cbs = []
        for loss in checkpoint_monitors(args.checkpoint_monitors):
            safe_name = loss.replace('/', '_')
            cbs.append(
                ModelCheckpoint(
                    dirpath=f"{args.checkpoint_path}/{args.checkpoint_name}/{safe_name}",
                    filename=f"epoch={{epoch}}-{safe_name}={{{loss}:.6f}}",
                    auto_insert_metric_name=False,
                    save_top_k=args.save_top_k,
                    monitor=loss,
                    mode="min",
                    save_last=bool(args.save_last),
                )
            )
        cbs.append(CustomProgressBar())
        return cbs

    logger = CSVLogger(save_dir=f"{args.save_dir}/logs", name=f"{args.name}")
    tb_logger = SplitTensorBoardLogger(
        save_dir=f"{args.save_dir}/tb_logs",
        name=f"{args.name}",
        other_target="train",
        strip_suffix=True,
        val_suffix="_epoch",
    )
    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))

    lightning_model = FlashViTFineTuner(model=model, args=args)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=make_callbacks(),
        accelerator="gpu",
        devices=nb_gpus,
        num_nodes=args.nb_nodes,
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

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=resume_path,
    )


if __name__ == '__main__':
    main()
