"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 07.25

Description:
    Arguments.
"""

import argparse
from typing import Callable, Dict


def ini_argparse(
    model_factories: Dict[str, Callable[[], object]]
) -> argparse.ArgumentParser:
    """
    Creates an ArgumentParser.
    """
    def model_type(name: str) -> Callable[[], object]:
        if name not in model_factories:
            valid = ', '.join(model_factories)
            raise argparse.ArgumentTypeError(f"Invalid model '{name}'. Choose from: {valid}")
        return model_factories[name]
            
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=True, help="set if training")
    parser.add_argument("--test", action="store_false", dest="train", help="set if testing")
    parser.add_argument("--stage1", action="store_true", default=True, help="set if stage 1 (voxel tagging)")
    parser.add_argument("--stage2", action="store_false", dest="stage1", help="set if stage 2 (flavour/regression)")
    parser.add_argument("--preprocessing_input", type=str, default=None, help="input data preprocessing (log or sqrt)")
    parser.add_argument("--preprocessing_output", type=str, default=None, help="output data preprocessing (log or sqrt)")
    parser.add_argument("--augmentations_enabled", action="store_true", default=True, help="set to allow augmentations")
    parser.add_argument("--augmentations_disabled", action="store_false", dest="augmentations_enabled", help="set to disable augmentations")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="label smoothing factor")
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="mixup alpha")
    parser.add_argument("-d", "--dataset_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--web_dataset_path", type=str, default=None, help="Web dataset path")
    parser.add_argument("--shardshuffle", type=int, default=200, help="shard shuffling if iterable dataset")
    parser.add_argument("--shuffle", type=int, default=2000, help="sample shuffling if iterable dataset")
    parser.add_argument("--metadata_path", type=str, default=None, required=True, help="Path for metadata")
    parser.add_argument('--model', type=model_type, default='base', help='Which model size to use (base / large / huge')
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Percentage of masked patches during pre-training")
    parser.add_argument("--eps", type=float, default=1e-12, help="value to prevent division by zero")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--layer_decay", type=float, default=0.9, help="layer-wise lr decay during finetuning")
    parser.add_argument("-w", "--num_workers", type=int, default=16, help="number of loader workers")
    parser.add_argument("--lr", type=float, default=None, help="learning rate of the optimiser")
    parser.add_argument("--blr", type=float, default=None, help="base learning rate of the optimiser")
    parser.add_argument("-ag", "--accum_grad_batches", type=int, default=1, help="batches for gradient accumulation")
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='number of warmup steps')
    parser.add_argument('--cosine_annealing_steps', type=int, default=0, help='number of cosine annealing steps')
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.05, help="weight_decay of the optimiser")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="AdamW first beta value")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="AdamW second beta value")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay")
    parser.add_argument("--head_init", type=float, default=0.001, help="Fine-tuning head(s) initialisation value")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (encoder)")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention dropout rate (encoder)")
    parser.add_argument("--drop_path_rate", type=float, default=0.0, help="Drop path rate (encoder)")
    parser.add_argument("--dropout_dec", type=float, default=0.0, help="Dropout rate (decoder)")
    parser.add_argument("--attn_dropout_dec", type=float, default=0.0, help="Attention dropout rate (decoder)")
    parser.add_argument("--drop_path_rate_dec", type=float, default=0.0, help="Drop path rate (decoder)")
    parser.add_argument("--save_dir", type=str, default="/scratch/salonso/sparse-nns/faser/deep_learning/faserDL", help="log save directory")
    parser.add_argument("--name", type=str, default="v1", help="model name")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="steps between logs")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="early stopping patience (0 means no early stopping)")
    parser.add_argument("--save_top_k", type=int, default=1, help="save top k checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="/scratch/salonso/sparse-nns/faser/deep_learning/faserDL/checkpoints", help="Checkpoint path")
    parser.add_argument("--checkpoint_name", type=str, default="v1", help="checkpoint name")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="name of the checkpoint to load")
    parser.add_argument('--gpus', nargs='*',  # 'nargs' can be '*' or '+' depending on your needs
                        default=[0],  # Default list
                        help='list of GPUs to use (more than 1 GPU will run the training in parallel)'
                        )

    return parser

