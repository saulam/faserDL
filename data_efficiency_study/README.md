# Data-Efficiency Study

This directory extends the [main project README](../README.md) with the files needed to run the fine-tuning data-efficiency sweep.

## Study design

- Training budgets: 100, 300, 1K, 3K, 10K, 30K and 100K events.
- Conditions: fine-tuning from a pre-trained encoder and training from scratch.
- Seeds: 3 per setting, for 42 runs in total.
- Split: the same canonical 85/5/10 train/validation/test split as the main fine-tuning pipeline, with seed 7.
- Hyperparameters: inherited from `finetune.sh` and `scratch.sh`, with budget-dependent changes to batch size, schedule length and early stopping.

## Files

- `subsample_dataset.py` writes the canonical validation and test manifests together with the budgeted training subsets.
- `train_with_manifest.py` runs stage-2 training from explicit train and validation manifests.
- `run_all.sh` launches the full sweep and resumes from `last.ckpt` when a run already exists.

## Running the study

From the repository root:

```bash
python -m data_efficiency_study.subsample_dataset --dataset-path '<dataset_glob>'

export DATASET_PATH='<dataset_glob>'
export METADATA_PATH='<path/to/metadata_stats.pkl>'
export PRETRAINED_CKPT='<path/to/pretrained_checkpoint.ckpt>'

bash data_efficiency_study/run_all.sh
```

`PRETRAINED_CKPT` is only required when the `pretrained` condition is enabled.

Optional overrides:

```bash
export CHECKPOINT_BASE='<path/to/checkpoints_data_efficiency>'
export SAVE_DIR='logs_data_efficiency'
export GPUS_OVERRIDE='0'
```

To run only part of the sweep:

```bash
CONDITIONS_OVERRIDE='scratch' \
BUDGETS_OVERRIDE='1000 3000 10000' \
SEEDS_OVERRIDE='1 2' \
bash data_efficiency_study/run_all.sh
```

## Outputs

- `data_efficiency_study/manifests/` contains the generated manifests.
- `logs_data_efficiency/` contains CSV logs and TensorBoard logs by default.
- `checkpoints_data_efficiency/` is the default checkpoint directory unless `CHECKPOINT_BASE` is set.
- These generated artefacts are ignored by Git.
