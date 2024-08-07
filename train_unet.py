import torch
import numpy as np
import MinkowskiEngine as ME
import os
import time
import math
import tqdm
from glob import glob
from torch.utils.data import Subset, Dataset, DataLoader, random_split
from tensorboardX import SummaryWriter
from modules import *
from warmup_scheduler_pytorch import WarmUpScheduler
import torchvision
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Train a Minkowski Engine UNet model.')
parser.add_argument('--dataset_path', type=str, default="/scratch/salonso/sparse-nns/faser/events", help='Path to the dataset.')
parser.add_argument('--log_dir', type=str, default='logs/unet_v1', help='Directory for TensorBoard logs.')
parser.add_argument('--model_save_path', type=str, default="/scratch2/salonso/faser/models/model_", help='Path to save the model.')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for DataLoader.')
parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads for DataLoader.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='Betas for Adam optimizer.')
parser.add_argument('--eps', type=float, default=1e-9, help='Epsilon for Adam optimizer.')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for Adam optimizer.')
parser.add_argument('--num_cycles', type=int, default=49, help='Number of cycles for the cosine annealing scheduler.')
parser.add_argument('--num_warmup_steps', type=int, default=1, help='Number of warmup steps.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (cuda or cpu).')

args = parser.parse_args()

# manually specify the GPUs to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Set device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Load dataset
dataset = SparseFASERCALDataset(args.dataset_path)

# DataLoader parameters
batch_size = args.batch_size
num_workers = args.num_workers

# Split dataset into training, validation, and test sets
fulllen = len(dataset)
train_len = int(fulllen * 0.6)
val_len = int(fulllen * 0.1)
test_len = fulllen - train_len - val_len
train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(7))

# Create DataLoader instances for training, validation, and test sets
collate = dataset.collate_sparse_minkowski
train_loader = DataLoader(train_set, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
valid_loader = DataLoader(val_set, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)
test_loader = DataLoader(test_set, collate_fn=collate, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=args.log_dir)

num_classes = 3
disable = False
bce_loss_fn = torch.hub.load(
    'adeelh/pytorch-multi-class-focal-loss',
    model='focal_loss',
    alpha=None,
    gamma=1,
    reduction='mean',
    device=device,
    force_reload=False
)
lr = args.learning_rate
min_val_loss = np.inf

print("________________________________")
print("Initializing model...")
model = minkunet.MinkUNet14A(in_channels=1, out_channels=num_classes, D=3).to(device)

# Set up optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=tuple(args.betas), eps=args.eps, weight_decay=args.weight_decay)
num_cycles = args.num_cycles
num_warmup_steps = args.num_warmup_steps

# Calculate arguments for scheduler
warmup_steps = len(train_loader) * num_warmup_steps
cosine_annealing_steps = len(train_loader) * num_cycles

# Create a warm-up scheduler that gradually increases the learning rate
warmup_scheduler = CustomLambdaLR(optimizer, warmup_steps)

# Create a cosine annealing scheduler that reduces learning rate in a cosine-like manner
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_annealing_steps, T_mult=1, eta_min=0.)

# Combine the warm-up and cosine annealing schedulers
scheduler = CombinedScheduler(optimizer=optimizer, 
                              scheduler1=warmup_scheduler, 
                              scheduler2=cosine_scheduler, 
                              warmup_steps=warmup_steps, 
                              lr_decay=1.0)

count = 0
for epoch in range(0, num_cycles * 1 + num_warmup_steps):
    if count >= 30:
        print("Early stopping: finishing....")
        break

    # Train
    loss = train(model, train_loader, bce_loss_fn, dice_loss_fn, optimizer, scheduler, disable, epoch, writer, device)
    
    # Validation
    val_loss = test(model, valid_loader, bce_loss_fn, dice_loss_fn, disable, epoch, writer, device)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        print(f"Saving model with val loss {min_val_loss:.2f}")
        torch.save(model.state_dict(), f"{args.model_save_path}{epoch}")

writer.close()

# Clean up
del model, optimizer
gc.collect()
torch.cuda.empty_cache()
