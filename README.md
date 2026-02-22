# Sparse MAE-ViT for FASER neutrino events

A two-stage deep learning pipeline for neutrino interaction analysis in the FASERCal detector at CERN. The model is a Masked Autoencoder (MAE) Vision Transformer (ViT) with sparse 3D convolutional patch embedding, designed to process sparse detector hits from a multi-component detector system comprising 3DCal, ECAL, AHCAL, and a muon spectrometer.

**Stage 1 — Self-supervised pre-training** on multi-task hit-level objectives to learn rich representations of neutrino interactions.
**Stage 2 — Supervised fine-tuning** for event-level classification and kinematic regression.

This code was used in the following publication:

```bibtex
@techreport{CERN-FASER-NOTE-2026-004,
  author       = {Alonso-Monsalve, S. and Cavanagh, C. and Cufino, F. and Kose, U. and
                  Masciellani, A. and Rubbia, A. and Sgalaberna, D. and Villa, E. and
                  Zhao, X. and Axiotis, K. and others},
  title        = {FASERCal Conceptual Design Report},
  institution  = {CERN},
  number       = {CERN-FASER-NOTE-2026-004},
  year         = {2026},
  url          = {https://cds.cern.ch/record/2954673},
}
```

## Table of contents

- [Detector geometry](#detector-geometry)
- [Architecture](#architecture)
- [Pre-training tasks](#pre-training-tasks)
- [Fine-tuning tasks](#fine-tuning-tasks)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Command-line arguments](#command-line-arguments)
- [Licence](#licence)

## Detector geometry

The model processes data from four sequential sub-detectors placed along the neutrino beam direction (+Z):

| Sub-detector | Technology | Voxel size | Grid dimensions | Information |
|---|---|---|---|---|
| **FASERCal** | SuperFGD-like scintillator | 1×1×1 cm³ | 48×48×200 (10 modules of 48×48×20) | Sparse 3D hits with charge, particle-level truth |
| **ECAL** | Electromagnetic calorimeter | — | 5×5 energy matrix (48×48 cm² face) | Energy deposit matrix capturing EM shower profiles |
| **AHCAL** | Hadronic calorimeter | 4×4×4 cm³ | 18×18×40 | Sparse 3D hits with charge only |
| **Muon spectrometer** | Tracking + magnet | — | Per-muon tuples (charge, px, py, pz) | Fitted muon tracks |

The neutrino interaction vertex is always within FASERCal. The sub-detectors are sequentially placed but not aligned, so geometric augmentations are not applied across detector boundaries.

## Architecture

The model uses a dual-branch hierarchical encoder with a Perceiver-IO bottleneck:

1. **Sparse patch embedding** — 3D sparse convolutions (`spconv`) convert raw voxel hits into patch-level tokens: FASERCal patches of 12×12×10 voxels (yielding a 4×4×20 patch grid) and AHCAL patches of 6×6×5 voxels (yielding a 3×3×8 patch grid).
2. **Intra-module self-attention** — FASERCal tokens are grouped by module (each module spans 2 patches in depth) and self-attended independently, with module-level and AHCAL-level learned embeddings contributing positional context. AHCAL tokens are processed through a separate self-attention branch.
3. **Perceiver-IO cross-attention bottleneck** — A set of learned latent tokens cross-attends to the combined FASERCal and AHCAL tokens, followed by self-attention blocks, producing a compact representation that merges both calorimeter branches alongside global features (ECAL and muon spectrometer) injected via a dedicated encoder.
4. **Decoder (pre-training only)** — Cross-attention from mask tokens to latent tokens, with multi-rank separable DCT basis heads for voxel-level occupancy and charge reconstruction.

Two model sizes are provided:

| Variant | Embedding dim | Encoder depth (FASERCal / AHCAL) | Perceiver-IO depth | Decoder dim | Decoder heads |
|---|---|---|---|---|---|
| `tiny` | 384 | 2 / 2 | 6 | 256 | 8 |
| `base` | 384 | 4 / 4 | 4 | 256 | 8 |

All variants use 12 attention heads with an MLP ratio of 4.

## Pre-training tasks

Pre-training combines masked reconstruction and relational (hit-level) objectives, optimised jointly with Kendall uncertainty weighting:

**Reconstruction tasks (Pass A — on masked patches):**
- Occupancy and charge prediction for masked FASERCal patches
- Occupancy and charge prediction for masked AHCAL patches
- ECAL energy matrix prediction (when masked)
- Muon spectrometer summary prediction (when masked)

**Relational tasks (Pass B — on kept patches):**
- Ghost/primary/secondary/tertiary hit classification (hierarchy labels)
- EM shower / muon-MIP / hadronic particle type classification (PID labels)
- Ghost hit identification

Reconstruction losses support multiple modes (standard, hybrid, distance-aware, focal distance transform), with optional soft chamfer and distance-weighted regression components for spatial forgiveness near shower boundaries.

## Fine-tuning tasks

Fine-tuning uses the frozen or unfrozen pre-trained encoder (without the decoder) and adds task-specific heads with Kendall uncertainty weighting:

- **Flavour identification** — 6-class classification: CC νe, CC νμ, CC ντ (+ their NC counterparts)
- **Charm identification** — 4-class classification of charm production modes
- **Visible momentum** — Regression of (pT, φ, pz) in cylindrical coordinates
- **Jet momentum** — Regression of (pT, φ, pz) in cylindrical coordinates
- **Primary lepton momentum** — Derived from visible and jet momentum consistency
- **Vertex position** — 3D vertex regression

Fine-tuning employs layer-wise learning rate decay, exponential moving average (EMA), and label smoothing.

## Installation

### Requirements

- Python ≥ 3.11
- CUDA ≥ 12.1
- A CUDA-capable GPU

### Setup

```bash
git clone https://github.com/saulam/faserDLTrans.git
cd faserDLTrans
pip install -r requirements.txt
```

The main dependencies are:

- [PyTorch](https://pytorch.org/) (≥ 2.5)
- [spconv-cu121](https://github.com/traveller59/spconv) (≥ 2.3) — sparse 3D convolutions
- [PyTorch Lightning](https://lightning.ai/) (≥ 2.5) — training framework
- [timm](https://github.com/huggingface/pytorch-image-models) (≥ 1.0) — Vision Transformer building blocks
- [webdataset](https://github.com/webdataset/webdataset) (≥ 1.0) — sharded data loading
- [torch-ema](https://github.com/fadel/pytorch_ema) — exponential moving average

> **Note:** [ROOT](https://root.cern/) (PyROOT) is additionally required only for the data preparation scripts (`dataset/read_root*.py`) that convert raw simulation ROOT files to NumPy format. It is not needed for training or inference and should be installed separately (e.g., via conda: `conda install -c conda-forge root`).

## Data preparation

The training pipeline expects pre-processed events stored as compressed NumPy files (`.npz`). To convert raw FASER simulation ROOT files:

```bash
cd dataset
python read_root_v7.py --number <events_per_file>
```

A metadata statistics file is also required and can be generated with:

```bash
python -m dataset.metadata_stats
```

This produces a pickle file containing robust standardisation parameters (median, MAD) computed over the full dataset, which is passed to both training scripts via `--metadata_path`.

## Usage

### Pre-training

```bash
./pretrain.sh
```

This runs masked autoencoder pre-training (Stage 1). Edit `pretrain.sh` to adjust hyperparameters such as mask ratio, learning rate, batch size, or loss mode. The script calls:

```bash
python -m train.pretrain --train --stage1 [options]
```

### Fine-tuning

```bash
./finetune.sh
```

This runs supervised fine-tuning (Stage 2) starting from a pre-trained checkpoint. Edit `finetune.sh` to point `load_checkpoint` to your pre-trained model. The script calls:

```bash
python -m train.finetune --train --stage2 [options]
```

### Resuming training

Both pre-training and fine-tuning support resuming from a checkpoint via `--resume_checkpoint`. This restores the full training state (model weights, optimiser, learning rate scheduler, epoch counter, etc.):

```bash
python -m train.pretrain --train --stage1 --resume_checkpoint path/to/checkpoint.ckpt [options]
python -m train.finetune --train --stage2 --resume_checkpoint path/to/checkpoint.ckpt [options]
```

### Loading pre-trained weights

Use `--load_checkpoint` to load model weights from a checkpoint and start a fresh training run (no optimiser state or epoch restoration):

- **Pre-training**: all checkpoint keys must match the model exactly (strict loading).
- **Fine-tuning**: only encoder weights are transferred; task-specific heads are randomly initialised (flexible matching).

```bash
python -m train.pretrain --train --stage1 --load_checkpoint path/to/pretrain_checkpoint.ckpt [options]
python -m train.finetune --train --stage2 --load_checkpoint path/to/pretrain_checkpoint.ckpt [options]
```

### Training from scratch

```bash
./scratch.sh
```

This runs the fine-tuning tasks without loading any pre-trained weights, i.e. training the model end-to-end from a random initialisation. It is useful as a baseline to quantify the benefit of pre-training.

Both scripts support multi-GPU training via DDP. Set the `gpus` variable to a list of GPU IDs (e.g., `gpus=(0 1)`).

### Sharded datasets

For large-scale training, the pipeline supports [webdataset](https://github.com/webdataset/webdataset) shards. Use `--web_dataset_path` instead of `--dataset_path` and provide a directory containing `.tar` shards and a `metadata.json` file.

## Project structure

```
├── pretrain.sh                  # Pre-training launch script
├── finetune.sh                  # Fine-tuning launch script
├── scratch.sh                   # Training from scratch launch script
├── requirements.txt
├── dataset/
│   ├── dataset.py               # Map-style and iterable dataset classes
│   ├── metadata_stats.py        # Robust standardisation metadata computation
│   ├── metadata.py              # Legacy metadata script
│   ├── read_root.py             # ROOT-to-NumPy converter (v5.1)
│   ├── read_root_v6.py          # ROOT-to-NumPy converter (v6.0)
│   └── read_root_v7.py          # ROOT-to-NumPy converter (v7.0)
├── model/
│   ├── sparsemaevit.py          # Sparse MAE-ViT (pre-training architecture)
│   ├── sparsevit.py             # Sparse ViT (fine-tuning architecture)
│   ├── lightning_model_pretrain.py  # Lightning module for pre-training
│   ├── lightning_model_finetune.py  # Lightning module for fine-tuning
│   └── utils.py                 # Attention blocks, positional embeddings, heads
├── train/
│   ├── pretrain.py              # Pre-training entry point
│   └── finetune.py              # Fine-tuning entry point
└── utils/
    ├── args.py                  # CLI argument parser
    ├── augmentations.py         # Data augmentation pipeline
    ├── distance_losses.py       # Distance-aware reconstruction losses
    ├── funcs.py                 # Data loading, collation, scheduling utilities
    ├── logger.py                # Split TensorBoard logger (train/val)
    ├── losses.py                # Classification, regression, and focal losses
    ├── lr_decay.py              # Layer-wise learning rate decay
    ├── pdg.py                   # PDG code to particle cluster mapping
    ├── plot.py                  # Visualisation utilities
    └── rotation_conversions.py  # Rotation representation conversions
```

## Command-line arguments

Both training scripts share a common argument parser. The main options are listed below; see `utils/args.py` for the full set.

### General

| Argument | Default | Description |
|---|---|---|
| `--train` / `--test` | `--train` | Training or testing mode |
| `--stage1` / `--stage2` | `--stage1` | Pre-training (stage 1) or fine-tuning (stage 2) |
| `--model` | `base` | Model variant (`tiny` or `base`) |
| `--dataset_path` | — | Path to the dataset directory (supports glob patterns) |
| `--web_dataset_path` | — | Path to webdataset shards (alternative to `--dataset_path`) |
| `--metadata_path` | — | Path to the metadata statistics pickle file (required) |

### Training

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 2 | Batch size per GPU |
| `--epochs` | 50 | Number of training epochs |
| `--lr` | — | Learning rate (overrides `--blr` if set) |
| `--blr` | — | Base learning rate (linearly scaled by effective batch size / 256) |
| `--weight_decay` | 0.05 | AdamW weight decay |
| `--beta1` / `--beta2` | 0.9 / 0.999 | AdamW beta parameters |
| `--warmup_epochs` | 0 | Linear warmup epochs |
| `--cosine_annealing_epochs` | 0 | Cosine annealing epochs |
| `--accum_grad_batches` | 1 | Gradient accumulation steps |
| `--gpus` | `0` | GPU device IDs (space-separated for multi-GPU) |

### Checkpoints

| Argument | Default | Description |
|---|---|---|
| `--load_checkpoint` | — | Path to a checkpoint to load weights from (starts fresh training). For pre-training: strict key matching (all keys must match). For fine-tuning: loads encoder weights only with flexible matching |
| `--resume_checkpoint` | — | Path to a checkpoint to resume training from (restores optimiser state, epoch counter, etc.) |

### Pre-training specific

| Argument | Default | Description |
|---|---|---|
| `--mask_ratio` | 0.75 | Fraction of patches masked during pre-training |
| `--reconstruction_loss_mode` | `standard` | Reconstruction loss mode (`standard`, `hybrid`, `distance`, `focal_dt`) |
| `--relational_pass_prob` | 0.5 | Probability of running the relational pass |
| `--relational_mask_ratio` | 0.25 | Mask ratio for the relational pass |

### Fine-tuning specific

| Argument | Default | Description |
|---|---|---|
| `--layer_decay` | 0.9 | Layer-wise learning rate decay factor |
| `--ema_decay` | 0.9999 | Exponential moving average decay |
| `--head_init` | 0.001 | Task head weight initialisation scale |
| `--drop_path_rate` | 0.0 | Stochastic depth rate |
| `--mixup_alpha` | 0.0 | Mixup interpolation alpha |

### Regularisation

| Argument | Default | Description |
|---|---|---|
| `--dropout` | 0.0 | Dropout rate (encoder) |
| `--attn_dropout` | 0.0 | Attention dropout rate (encoder) |
| `--label_smoothing` | 0.0 | Label smoothing factor |
| `--preprocessing_input` | — | Input transform (`log` or `sqrt`) |
| `--preprocessing_output` | — | Output transform (`log` or `sqrt`) |

## Licence

This project is released under the [MIT Licence](LICENSE).

