# Scintillator Transfer-Learning Study

This directory extends the [main README](../../README.md) with the files used
for particle-level PID transfer from the FASERCal encoder to the public
scintillator dataset.

## Scope

- target classes: `proton`, `pion`, `muon`, `electron`.
- data layout: `training/` is used for training and validation; `testing/` is
  reserved for final evaluation.
- geometry: hits are stored on a 10 mm grid and cropped to fixed local windows
  inside a `200 x 200 x 200` detector volume.

## Files

- `scintillator_dataset.py` loads the `.pt` events and builds the local crop
  plus detector-context features.
- `build_charge_metadata.sh` computes the `q_log1p` metadata used by the
  transfer charge preprocessing.
- `train.py` and `evaluate.py` are the training and evaluation entry points.
- `train*.sh` and `evaluate*.sh` are the checked-in launchers.

## Data Preparation

From the repository root:

```bash
export DATA_DIR='path/to/scintillator_dataset'

bash transfer_learning/transfer_scintillator/build_charge_metadata.sh
```

Expected dataset layout:

```text
path/to/scintillator_dataset/
  training/
    proton/
    pion/
    muon/
    electron/
  testing/
    proton/
    pion/
    muon/
    electron/
```

This produces:

- `charge_metadata.pkl`: charge standardisation metadata.

## Training

Transfer runs require a pretrained FASERCal checkpoint:

```bash
export DATA_DIR='path/to/scintillator_dataset'
export LOAD_CHECKPOINT='path/to/pretrain_checkpoint.ckpt'
export GPUS='0'

bash transfer_learning/transfer_scintillator/train.sh
```

Scratch baselines:

```bash
export DATA_DIR='path/to/scintillator_dataset'
export GPUS='0'

bash transfer_learning/transfer_scintillator/train_scratch.sh
```

Useful overrides:

- `CHARGE_METADATA_PATH`, `SAVE_DIR`, `CHECKPOINT_PATH`, `NAME`,
  `CHECKPOINT_NAME`.
- `BATCH_SIZE`, `EPOCHS`, `BLR`, `SPATIAL_SHAPE`, `PATCH_SIZE`,
  `WINDOW_SIZE`.
- `NUM_WORKERS`, `WARMUP_EPOCHS`, `COSINE_ANNEALING_EPOCHS`,
  `WEIGHT_DECAY`, `LAYER_DECAY`, `LABEL_SMOOTHING`.

## Evaluation

Evaluation defaults to the `testing` split. Set the checkpoint to test:

```bash
export DATA_DIR='path/to/scintillator_dataset'
export CHECKPOINT='path/to/checkpoint.ckpt'
export GPU='0'

bash transfer_learning/transfer_scintillator/evaluate.sh
```

Scratch checkpoints can be evaluated with `evaluate_scratch.sh`.

## Notes

- Training uses only `training/` and holds out 5% for validation with seed 42.
- Matching encoder weights are transferred from the FASERCal checkpoint;
  detector-specific branches are skipped.
- Crops are clamped to detector boundaries, and detector-context features are
  injected before the transferred bottleneck.
- Evaluation on `testing/` matches the public benchmark split.
