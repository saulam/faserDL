"""
Author: Dr. Saul Alonso-Monsalve
Email: salonso(at)ethz.ch, saul.alonso.monsalve(at)cern.ch
Date: 09.24

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
    parser.add_argument("-d", "--dataset_path", type=str, default="/scratch/salonso/sparse-nns/medical_data/data_kits23_{}_{}_{}_good/*", help="Dataset path")
    parser.add_argument("--eps", type=float, default=1e-12, help="value to prevent division by zero")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=16, help="number of loader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of the optimiser")
    parser.add_argument("-ag", "--accum_grad_batches", type=int, default=1, help="batches for gradient accumulation")
    parser.add_argument('-ws', '--warmup_steps', type=int, default=0, help='Maximum number of warmup steps')
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.05, help="weight_decay of the optimiser")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="AdamW first beta value")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="AdamW second beta value")
    parser.add_argument('--losses', nargs='*',  # 'nargs' can be '*' or '+' depending on your needs
                        default=["focal", "dice"],  # Default list
                        help='List of losses to use (options: "focal", "dice")'
                        )
    parser.add_argument("--save_dir", type=str, default="/scratch/salonso/sparse-nns/faser/deep_learning/faserDL", help="Log save directory")
    parser.add_argument("--name", type=str, default="v1", help="model name")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="steps between logs")
    parser.add_argument("--save_top_k", type=int, default=1, help="Save top k checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="/scratch/salonso/sparse-nns/faser/deep_learning/faserDL/checkpoints", help="Checkpoint path")
    parser.add_argument("--checkpoint_name", type=str, default="v1", help="Checkpoint name")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Name of the checkpoint to load")
    parser.add_argument('--gpus', nargs='*',  # 'nargs' can be '*' or '+' depending on your needs
                        default=[0],  # Default list
                        help='List of GPUs to use (more than 1 GPU will run the training in parallel)'
                        )

    return parser
