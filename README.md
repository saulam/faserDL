Minkowski Engine UNet Training Script

This script trains a Minkowski Engine UNet model using a dataset of sparse data. The script supports various configurations via command-line arguments.

## Requirements

- Python 3.7+
- PyTorch
- MinkowskiEngine (version with depthwise convolutions: )
- tensorboardX
- torchvision
- tqdm
- numpy
- argparse

## Usage

```
python -m train.train [args]
```

### Arguments

- `--train`: Enables the training mode. Set this flag when you want to train the model. This is a boolean flag (default: `True`).
  
- `--test`: Enables the testing mode by disabling training. If this flag is set, the script will run in test mode (overrides `--train` flag).
  
- `-d, --dataset_path`: Specifies the path to the dataset. This should include the path pattern to the training data. The default path is `/scratch/salonso/sparse-nns/faser/events_v3_new`.
  
- `--eps`: A small constant value to prevent division by zero during calculations (default: `1e-12`).
  
- `-b, --batch_size`: Defines the number of samples per batch during training (default: `2`).
  
- `-e, --epochs`: The number of epochs (full training cycles) to run during training (default: `50`).
  
- `-w, --num_workers`: Number of worker threads to use for loading data during training (default: `16`).
  
- `--lr`: Learning rate for the optimizer. This controls the step size at each iteration while moving towards the optimal solution (default: `1e-4`).
  
- `-ag, --accum_grad_batches`: Specifies the number of batches to accumulate for gradient accumulation, which can help reduce memory usage by simulating larger batches (default: `1`).

- `-ws, --warmup_steps`: The number of warmup steps used to gradually increase the learning rate at the start of training (default: `0`).

- `-wd, --weight_decay`: Weight decay applied to the optimizer to prevent overfitting by penalizing large weights (default: `0.05`).
  
- `-b1, --beta1`: The first beta value for the AdamW optimizer, which controls the momentum for the moving average of the gradient (default: `0.9`).
  
- `-b2, --beta2`: The second beta value for the AdamW optimizer, controlling the momentum for the moving average of the squared gradient (default: `0.999`).
  
- `--losses`: Specifies the list of loss functions to be used during training. Options include `"focal"` and `"dice"` (default: `["focal", "dice"]`).
  
- `--save_dir`: Specifies the directory where logs and other outputs will be saved (default: `/scratch/salonso/sparse-nns/faser/deep_learning/faserDL`).
  
- `--name`: Defines the name of the model version being trained or tested (default: `"v1"`).
  
- `--log_every_n_steps`: Number of steps between logging training information (default: `50`).
  
- `--save_top_k`: Number of top checkpoints to save during training (default: `1`).
  
- `--checkpoint_path`: Directory path where model checkpoints will be saved (default: `/scratch/salonso/sparse-nns/faser/deep_learning/faserDL/checkpoints`).
  
- `--checkpoint_name`: Defines the name for the checkpoint file to be saved (default: `"v1"`).
  
- `--load_checkpoint`: The name of a specific checkpoint file to load for resuming training or testing (default: `None`).
  
- `--gpus`: List of GPU device indices to use for training. If more than one GPU is specified, the script will use parallel processing (default: `[0]`).


