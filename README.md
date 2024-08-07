Minkowski Engine UNet Training Script

This script trains a Minkowski Engine UNet model using a dataset of sparse data. The script supports various configurations via command-line arguments.

## Requirements

- Python 3.7+
- PyTorch
- MinkowskiEngine
- tensorboardX
- torchvision
- tqdm
- numpy
- argparse

## Usage

```
python script.py [options]
```

# Options

- `--dataset_path` (str): Path to the dataset. Default is `"/scratch/salonso/sparse-nns/faser/events"`.
- `--log_dir` (str): Directory for TensorBoard logs. Default is `"logs/unet_v1"`.
- `--model_save_path` (str): Path to save the model. Default is `"/scratch2/salonso/faser/models/model_"`.
- `--batch_size` (int): Batch size for DataLoader. Default is `8`.
- `--num_workers` (int): Number of worker threads for DataLoader. Default is `8`.
- `--learning_rate` (float): Learning rate for the optimizer. Default is `0.001`.
- `--betas` (float, nargs=2): Betas for Adam optimizer. Default is `(0.9, 0.999)`.
- `--eps` (float): Epsilon for Adam optimizer. Default is `1e-9`.
- `--weight_decay` (float): Weight decay for Adam optimizer. Default is `0.0001`.
- `--num_cycles` (int): Number of cycles for the cosine annealing scheduler. Default is `49`.
- `--num_warmup_steps` (int): Number of warmup steps. Default is `1`.
- `--device` (str): Device to use for training (`cuda` or `cpu`). Default is `"cuda"`.

# Example

```
python script.py --dataset_path /path/to/dataset --batch_size 16 --learning_rate 0.0005 --betas 0.9 0.999 --eps 1e-8 --weight_decay 0.0002 --num_cycles 50 --num_warmup_steps 5 --device cuda
```

