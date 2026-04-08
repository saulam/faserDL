# Sparse MAE-ViT for FASERCal

This repository contains the sparse masked autoencoder / Vision Transformer
pipeline used for FASERCal neutrino-event studies, together with downstream
fine-tuning and transfer-learning studies.

The main workflow has two stages:

- Stage 1: self-supervised pretraining on FASERCal events
- Stage 2: supervised fine-tuning for event classification and kinematic
  regression

The repository also includes:

- transfer learning from a pretrained FASERCal encoder to a public
  scintillator PID dataset
- transfer learning to PILArNet
- a data-efficiency study for stage-2 fine-tuning

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

## Detector Inputs

The main FASERCal model consumes four detector branches along the beam
direction:

| Sub-detector | Technology | Grid | Information used |
|---|---|---|---|
| `FASERCal` | SuperFGD-like scintillator | `48 x 48 x 200` voxels (`1 x 1 x 1 cm^3`) | Sparse 3D hits with charge and particle-level truth |
| `ECAL` | Electromagnetic calorimeter | `5 x 5` image | Energy-deposit matrix |
| `AHCAL` | Hadronic calorimeter | `18 x 18 x 40` voxels (`4 x 4 x 4 cm^3`) | Sparse 3D hits with charge |
| `Muon spectrometer` | Tracking + magnet | Per-track tuples | Charge and fitted momentum |

The neutrino interaction vertex is always inside FASERCal. The detector
branches are sequential but not aligned to one shared voxel frame, so geometry
augmentations are applied within each branch rather than across the full
detector chain.

## Model Summary

- Sparse 3D convolutions (`spconv`) convert FASERCal and AHCAL hits into patch
  tokens.
- Local self-attention runs over FASERCal modules and AHCAL windows.
- A Perceiver-IO bottleneck merges calorimeter tokens together with ECAL and
  muon features.
- During pretraining, a decoder reconstructs masked detector content.

Two model sizes are provided:

| Variant | Encoder depth (FASERCal / AHCAL) | Perceiver depth | Decoder dim |
|---|---|---|---|
| `tiny` | `2 / 2` | `6` | `256` |
| `base` | `4 / 4` | `4` | `256` |

Both variants use embedding dimension `384`, `12` attention heads, and an MLP
ratio of `4`.

## Training Targets

### Pretraining

- masked occupancy and charge prediction for FASERCal patches
- masked occupancy and charge prediction for AHCAL patches
- masked ECAL energy-matrix prediction
- masked muon-summary prediction
- hit-level relational targets on unmasked patches

### Fine-Tuning

- flavour identification: `CC nue`, `CC numu`, `CC nutau` and the
  corresponding NC classes
- charm identification: 4 classes
- visible momentum regression
- jet momentum regression
- primary lepton momentum regression
- 3D vertex regression

## Setup

Requirements:

- Python `>= 3.11`
- CUDA `>= 12.1`
- a CUDA-capable GPU

Install from the repository root:

```bash
git clone https://github.com/saulam/faserDLTrans.git
cd faserDLTrans
pip install -r requirements.txt
```

`ROOT` / `PyROOT` is only needed for the raw ROOT-to-NumPy conversion scripts
in [`dataset/`](dataset). It is not required for training or evaluation once
the `.npz` files have been produced.

## Data Preparation

The main FASERCal pipeline expects compressed NumPy event files.

To convert raw FASERCal simulation ROOT files:

```bash
cd dataset
python read_root_v7.py --number <events_per_file>
cd ..
```

To build the metadata file used by training:

```bash
python -m dataset.metadata_stats
```

The resulting pickle file is passed to training with `METADATA_PATH` or
`--metadata_path`.

## Main FASERCal Pipeline

### Pretraining

```bash
export DATASET_PATH='path/to/events_v7.0*'
export METADATA_PATH='path/to/metadata_stats.pkl'

bash pretrain.sh
```

### Fine-tuning from a pretrained checkpoint

```bash
export DATASET_PATH='path/to/events_v7.0*'
export METADATA_PATH='path/to/metadata_stats.pkl'
export LOAD_CHECKPOINT='path/to/pretrain_checkpoint.ckpt'

bash finetune.sh
```

### Fine-tuning from scratch

```bash
export DATASET_PATH='path/to/events_v7.0*'
export METADATA_PATH='path/to/metadata_stats.pkl'

bash scratch.sh
```

`LOAD_CHECKPOINT` starts a fresh run with loaded weights. `RESUME_CHECKPOINT`
restores the full training state. Multi-GPU runs are supported by setting
`GPUS='0 1'` or similar.

## Transfer Learning

### Public Scintillator Dataset

The scintillator study adapts the pretrained FASERCal encoder to a 4-class
particle-ID task on a public scintillator dataset.

Build charge metadata:

```bash
export DATA_DIR='path/to/scintillator_dataset'

bash transfer_learning/transfer_scintillator/build_charge_metadata.sh
```

Train with transferred weights:

```bash
export DATA_DIR='path/to/scintillator_dataset'
export LOAD_CHECKPOINT='path/to/pretrain_checkpoint.ckpt'

bash transfer_learning/transfer_scintillator/train.sh
```

Evaluate a checkpoint:

```bash
export DATA_DIR='path/to/scintillator_dataset'
export CHECKPOINT='path/to/checkpoint_or_directory'

bash transfer_learning/transfer_scintillator/evaluate.sh
```

Scratch baselines are available through
`train_scratch.sh` and `evaluate_scratch.sh`.
See [transfer_learning/transfer_scintillator/README.md](transfer_learning/transfer_scintillator/README.md)
for dataset layout and extra options.

### PILArNet

The PILArNet study transfers the encoder to particle-level PID on the PILArNet
dataset, with both single-particle and multi-particle variants.

Build the manifest and charge metadata:

```bash
python -m transfer_learning.transfer_pilarnet.preprocess \
    --data_dir path/to/pilarnet/larcv3 \
    --out transfer_learning/transfer_pilarnet/manifest_768px.npz \
    --min_voxels 5

bash transfer_learning/transfer_pilarnet/build_charge_metadata.sh
```

Run training:

```bash
export LOAD_CHECKPOINT='path/to/pretrain_checkpoint.ckpt'

bash transfer_learning/transfer_pilarnet/train_single.sh
bash transfer_learning/transfer_pilarnet/train_multi.sh
```

See [transfer_learning/transfer_pilarnet/README.md](transfer_learning/transfer_pilarnet/README.md)
for the full workflow, including scratch baselines and evaluation commands.

### Data-Efficiency Sweep

The repository also includes a stage-2 data-efficiency study that compares
pretrained and scratch fine-tuning across fixed training budgets. See
[data_efficiency_study/README.md](data_efficiency_study/README.md).

## Repository Layout

- `dataset/`: ROOT conversion, metadata building, and dataset loaders for the
  main FASERCal pipeline
- `model/`: core FASERCal model definitions and Lightning modules
- `train/`: pretraining and fine-tuning entry points
- `transfer_learning/transfer_scintillator/`: public scintillator transfer
  study
- `transfer_learning/transfer_pilarnet/`: PILArNet transfer study
- `data_efficiency_study/`: fixed-budget fine-tuning study
- `utils/`: losses, augmentations, schedulers, logging, and shared helpers

For the full set of arguments, see [`utils/args.py`](utils/args.py) for the
main FASERCal pipeline and the corresponding `train.py` / `evaluate.py` files
inside each transfer-learning directory.

## License

This project is released under the [MIT License](LICENSE).
