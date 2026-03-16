import copy
import os
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from dataset import SparseFASERCALMapDataset
from model import ViTFineTuner, vit_base, vit_tiny
from utils import SplitTensorBoardLogger, create_loader, ini_argparse, load_mae_encoder
from utils.funcs import collate


torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
pl_major = int(pl.__version__.split(".")[0])
MODEL_FACTORIES = {
    "tiny": vit_tiny,
    "base": vit_base,
}
MONITOR_LOSSES = [
    "loss_total/val",
    "loss_cls/flavour/val",
    "loss_cls/charm/val",
    "loss_vis/geom/val",
    "loss_jet/geom/val",
    "loss_lep/geom/val",
    "loss_vertex/val",
]


class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True
        return bar


def read_manifest(path):
    with open(path, encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def build_callbacks(args):
    callbacks = []
    for loss in MONITOR_LOSSES:
        safe_name = loss.replace("/", "_")
        callbacks.append(
            ModelCheckpoint(
                dirpath=f"{args.checkpoint_path}/{args.checkpoint_name}/{safe_name}",
                filename=f"epoch={{epoch}}-{safe_name}={{{loss}:.6f}}",
                auto_insert_metric_name=False,
                save_top_k=args.save_top_k,
                monitor=loss,
                mode="min",
                save_last=True,
            )
        )

    callbacks.append(CustomProgressBar())
    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="loss_total/val",
                patience=args.early_stop_patience,
                mode="min",
                verbose=True,
            )
        )

    return callbacks


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = ini_argparse(MODEL_FACTORIES)
    parser.add_argument("--train_manifest", type=str, required=True, help="Training manifest")
    parser.add_argument("--val_manifest", type=str, required=True, help="Validation manifest")
    parser.add_argument("--pl_seed", type=int, default=None, help="PyTorch Lightning seed")
    args = parser.parse_args()

    if args.pl_seed is not None:
        pl.seed_everything(args.pl_seed, workers=True)

    print("\n- Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    nb_gpus = len(args.gpus)
    visible_devices = ",".join(map(str, args.gpus)) if nb_gpus > 1 else str(args.gpus[0])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    print("Standard dataset (manifest-based)")
    dataset = SparseFASERCALMapDataset(args)
    metadata = dataset.metadata

    train_files = read_manifest(args.train_manifest)
    val_files = read_manifest(args.val_manifest)
    if not train_files:
        raise ValueError(f"Training manifest is empty: {args.train_manifest}")
    if not val_files:
        raise ValueError(f"Validation manifest is empty: {args.val_manifest}")

    print(f"- Train manifest: {len(train_files)} events from {args.train_manifest}")
    print(f"- Val manifest:   {len(val_files)} events from {args.val_manifest}")

    train_set = copy.deepcopy(dataset)
    val_set = copy.deepcopy(dataset)
    train_set.data_files = train_files
    val_set.data_files = val_files
    train_set.augmentations_enabled = args.augmentations_enabled

    collate_fn = partial(collate, test=False)
    train_loader = create_loader(
        train_set,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        args=args,
    )
    valid_loader = create_loader(
        val_set,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
        args=args,
    )

    nb_batches_train = len(train_loader)
    nb_batches_val = len(valid_loader)
    print(f"- Train batches: {nb_batches_train}, Val batches: {nb_batches_val}")

    model = args.model(
        drop_rate=args.dropout,
        attn_drop_rate=args.attn_dropout,
        drop_path_rate=args.drop_path_rate,
        head_init=args.head_init,
        head_dropout_cls=args.head_dropout_cls,
        head_dropout_reg=args.head_dropout_reg,
        metadata=metadata,
    )

    if args.load_checkpoint is not None:
        if not os.path.exists(args.load_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location='cpu', weights_only=True)
        load_mae_encoder(model, checkpoint)
    else:
        print("Initialising fine-tuner from scratch.")

    logger = CSVLogger(save_dir=f"{args.save_dir}/logs", name=f"{args.name}")
    tb_logger = SplitTensorBoardLogger(
        save_dir=f"{args.save_dir}/tb_logs",
        name=f"{args.name}",
        other_target="train",
        strip_suffix=True,
        val_suffix="_epoch",
    )
    callbacks = build_callbacks(args)

    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))

    lightning_model = ViTFineTuner(model=model, args=args)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        accelerator="gpu",
        devices=nb_gpus,
        num_nodes=args.nb_nodes,
        precision="bf16-mixed" if pl_major >= 2 else 32,
        strategy=DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=False,
        ) if nb_gpus > 1 else "auto",
        logger=[logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    resume_path = None
    if args.resume_checkpoint is not None:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        resume_path = args.resume_checkpoint

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=resume_path,
    )


if __name__ == "__main__":
    main()
