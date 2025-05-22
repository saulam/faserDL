"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 01.25

Description:
    Arguments.
"""

import argparse

'''
Parameters
'''
def ini_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=True, help="set if training")
    parser.add_argument("--test", action="store_false", dest="train", help="set if testing")
    parser.add_argument("--stage1", action="store_true", default=True, help="set if stage 1 (voxel tagging)")
    parser.add_argument("--stage2", action="store_false", dest="stage1", help="set if stage 2 (flavour/regression)")
    parser.add_argument("--augmentations_enabled", action="store_true", default=True, help="set to allow augmentations")
    parser.add_argument("--augmentations_disabled", action="store_false", dest="augmentations_enabled", help="set to disable augmentations")
    parser.add_argument("-d", "--dataset_path", type=str, default="/scratch/salonso/sparse-nns/faser/events_v3_new", help="Dataset path")
    parser.add_argument("--sets_path", type=str, default=None, help="path of pickle file with training/val/test splits")
    parser.add_argument("--load_seg", action="store_true", default=False, help="whether to load results from segmentation network.")
    parser.add_argument("--eps", type=float, default=1e-12, help="value to prevent division by zero")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--phase1_epochs", type=int, default=5, help="number of phase 1 epochs (transfer learning)")
    parser.add_argument("--phase2_epochs", type=int, default=10, help="number of phase 2 epochs (transfer learning)")
    parser.add_argument("--phase3_epochs", type=int, default=10, help="number of phase 3 epochs (transfer learning)")
    parser.add_argument("--layer_decay", type=float, default=0.8, help="lr layer decay during finetuning")
    parser.add_argument("-w", "--num_workers", type=int, default=16, help="number of loader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of the optimiser")
    parser.add_argument("-ag", "--accum_grad_batches", type=int, default=1, help="batches for gradient accumulation")
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='number of warmup steps')
    parser.add_argument('--cosine_annealing_steps', type=int, default=0, help='number of cosine annealing steps')
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.05, help="weight_decay of the optimiser")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="AdamW first beta value")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="AdamW second beta value")
    parser.add_argument('--losses', nargs='*',  # 'nargs' can be '*' or '+' depending on your needs
                        default=["focal", "dice"],  # Default list
                        help='kist of losses to use (options: "focal", "dice")'
                        )
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

