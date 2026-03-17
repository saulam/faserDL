# PILArNet Transfer-Learning Study

This directory extends the [main README](../../README.md) with the files used
for particle-level PID transfer from the FASERCal encoder to PILArNet.

## Scope

- `single`: one particle crop per sample;
- `multi`: all selected particles from an event, with a context transformer on
  top of the shared encoder;
- label space: `photon`, `electron`, `muon`, `pion`, `proton`.

## Files

- `preprocess.py` builds the particle manifest and the fixed
  80k / 2k / 18k train/validation/test split.
- `build_charge_metadata.sh` computes the `q_log1p` metadata used by the
  transfer charge preprocessing.
- `train.py` and `evaluate.py` are the training and evaluation entry points.
- `train_*.sh` and `evaluate_*.sh` are the checked-in launchers.

## Data Preparation

From the repository root:

```bash
python -m transfer_learning.transfer_pilarnet.preprocess \
    --data_dir path/to/pilarnet/larcv3 \
    --out transfer_learning/transfer_pilarnet/manifest_768px.npz \
    --min_voxels 5

bash transfer_learning/transfer_pilarnet/build_charge_metadata.sh
```

This produces:

- `manifest_768px.npz`: full manifest;
- `manifest_768px_train.npz`: training split;
- `manifest_768px_val.npz`: validation split;
- `manifest_768px_test.npz`: test split;
- `charge_metadata.pkl`: charge standardisation metadata.

## Training

Transfer runs require a pretrained FASERCal checkpoint:

```bash
export LOAD_CHECKPOINT='path/to/pretrain_checkpoint.ckpt'
export GPUS='0'

bash transfer_learning/transfer_pilarnet/train_single.sh
bash transfer_learning/transfer_pilarnet/train_multi.sh
```

Scratch baselines:

```bash
export GPUS='0'

bash transfer_learning/transfer_pilarnet/train_single_scratch.sh
bash transfer_learning/transfer_pilarnet/train_multi_scratch.sh
```

Useful overrides:

- `MANIFEST`, `VAL_MANIFEST`, `CHARGE_METADATA_PATH`;
- `SAVE_DIR`, `CHECKPOINT_PATH`, `NAME`, `CHECKPOINT_NAME`;
- `BATCH_SIZE`, `EPOCHS`, `BLR`, `SPATIAL_SHAPE`, `PATCH_SIZE`,
  `WINDOW_SIZE`;
- `CONTEXT_LAYERS`, `CONTEXT_HEADS`, `CONTEXT_DROPOUT` for `multi`.

## Evaluation

Evaluation defaults to `manifest_768px_test.npz`. Set the checkpoint to test:

```bash
export CHECKPOINT='path/to/checkpoint.ckpt'

bash transfer_learning/transfer_pilarnet/evaluate_single.sh
bash transfer_learning/transfer_pilarnet/evaluate_multi.sh
```

Scratch checkpoints can be evaluated with the corresponding
`*_scratch.sh` launchers.

## Notes

- Matching encoder weights are transferred from the FASERCal checkpoint;
  detector-specific branches are skipped.
- `single` can use a reduced particle-metadata vector.
- `multi` augments each particle with event-context features before the context
  transformer.
