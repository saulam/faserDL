# JEPA pretraining baseline

This repository supports the published masked-autoencoder objective and a
controlled JEPA-style latent-prediction objective.  The JEPA baseline is
deliberately narrow: it changes the stage-1 objective while preserving the
event representation, sparse patching, hierarchical encoder, data pipeline,
optimiser/scheduler conventions, checkpointing, and downstream model.

## Existing MAE data flow

The map dataset loads one compressed NumPy event per file.  The principal
inputs are:

- FASERCal reconstructed hits: physical coordinates and charge are converted
  to integer `48 x 48 x 200` voxel coordinates and one preprocessed charge
  feature per occupied voxel;
- AHCAL hits: sparse coordinates and charge on an `18 x 18 x 40` grid;
- ECAL: a preprocessed dense `5 x 5` energy matrix;
- muon spectrometer: a variable-length sequence of five-feature track
  summaries plus the track count.

`utils.funcs.collate` constructs sparse-convolution tensors with indices
`[N_hit, 4]` (`batch, x, y, z`) and features `[N_hit, 1]`.  It also pads the
muon sequence and constructs its attention mask.  A one-slot, masked dummy is
now used when an entire batch contains no muon tracks; this fixes the
batch-size-one case without changing non-empty batches.

The existing sparse convolutions and tokenisation are unchanged:

| Branch | Voxel patch | Patch grid | Maximum patch tokens |
|---|---:|---:|---:|
| FASERCal | `12 x 12 x 10` | `4 x 4 x 20` | 320 |
| AHCAL | `6 x 6 x 5` | `3 x 3 x 8` | 72 |

Only occupied patches become active tokens.  FASERCal tokens are rearranged to
`[B, 10, 32, 384]`, corresponding to ten detector modules and 32 patch
positions per module.  The published occupancy-aware random mask is applied
independently within each module.  AHCAL uses the same masking helper on its
72 patch positions.  ECAL and muon tokens are independently dropped with
probability `mask_ratio / 2`.

Kept FASERCal tokens pass through module-local self-attention, and AHCAL tokens
through its parallel self-attention stack.  Two class tokens per FASERCal
module and two AHCAL class tokens form 22 fixed Perceiver latent slots.  The
Perceiver cross/self-attention loop produces a tensor `[B, 22, 384]` plus a
valid-slot mask `[B, 22]`.

For MAE, masked FASERCal and AHCAL patch positions become 256-dimensional
decoder queries.  These cross-attend to the encoder latents.  The learned
multi-rank separable bases, initialized from DCT modes, expand each query to
voxel predictions: 1,440 voxels per FASERCal patch and 180 per AHCAL patch.
Separate heads predict occupancy and charge; compact heads reconstruct dropped
ECAL and muon summaries.  The optional second relational pass predicts ghost,
hierarchy, and particle-category labels on kept FASERCal voxels.  The Lightning
module combines reconstruction, global, and optional relational losses with
bounded homoscedastic uncertainty weights and basis regularisation.

## JEPA design

`SparseJEPA` creates two instances of the existing encoder-only
`SparseMAEViT` backbone:

- the context encoder is trainable and receives the existing masked view;
- the target encoder is initialized as an exact copy, never receives
  gradients, remains deterministic, and receives the full event
  (`mask_ratio=0`);
- after each optimiser step, target parameters are updated as
  `target = decay * target + (1 - decay) * context`; the decay follows a cosine
  schedule from `JEPA_EMA_DECAY` to `JEPA_EMA_END_DECAY`;
- a discardable slot-wise predictor applies
  `LayerNorm -> Linear -> GELU -> Dropout -> Linear` to `[B, 22, 384]` context
  latents.

The prediction target is the final Perceiver latent representation, exactly
the encoder output pooled by the downstream model.  No voxel reconstruction,
DCT basis, reconstruction cross-attention decoder, or relational head is
constructed in JEPA mode.

By default, a latent slot contributes to the loss when its FASERCal module (or
the AHCAL group) contains at least one masked occupied patch.  This is the
smallest well-defined mapping from the current patch mask to the current fixed
latent structure.  `JEPA_TARGET_SELECTION=all` instead supervises every valid
slot.  Prediction and stop-gradient target tensors are layer-normalized per
slot and compared with Smooth L1 loss by default.  Normalization (`none`,
`layernorm`, `l2`) and loss (`smooth_l1`, `mse`, `cosine`) are configurable.

The target encoder is included in stage-1 checkpoints, but downstream loading
uses only one encoder.  `PRETRAINED_ENCODER_SOURCE=auto` prefers the EMA target
from JEPA checkpoints and preserves the single-encoder MAE behavior.  Explicit
`context` and `target` choices are also supported.  The EMA target is the
default because it is the temporally smoothed representation used to define
the training target; the context branch remains available for a direct
ablation.

In-domain fine-tuning requires complete encoder coverage.  The model variant
must therefore match across stages: `tiny` pretraining loads into `tiny`
fine-tuning, and `base` into `base`.  The loader reports parameter coverage and
the main fine-tuning entry point fails on a partial load.  The default is now
`tiny` consistently in `pretrain.sh`, `finetune.sh`, and `scratch.sh`.

## What is unchanged

- all raw inputs and preprocessing;
- FASERCal and AHCAL voxel grids and patch sizes;
- occupied-patch selection and module grouping;
- random occupancy-aware patch masking and global-token dropping;
- sparse patch embeddings, local attention, auxiliary detector embeddings,
  and the full Perceiver encoder;
- AdamW, effective-batch-size learning-rate scaling, warm-up, cosine schedule,
  gradient clipping, logging, DDP, and Lightning checkpoints;
- downstream `SparseViT`, task heads, losses, fine-tuning, and evaluation.

The objective-specific change is therefore MAE reconstruction machinery and
voxel/global targets versus an EMA encoder, latent predictor, and latent loss.

## Usage

Activate the requested environment and set the common data paths:

```bash
conda activate spconv
export DATASET_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*'
export METADATA_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz/metadata_stats.pkl'
```

Run the original MAE path (the default):

```bash
PRETRAINING_OBJECTIVE=mae \
NAME=pretrain_mae_control \
CHECKPOINT_NAME=pretrain_mae_control \
bash pretrain.sh
```

Run the JEPA baseline:

```bash
PRETRAINING_OBJECTIVE=jepa \
NAME=pretrain_jepa_latent \
CHECKPOINT_NAME=pretrain_jepa_latent \
JEPA_EMA_DECAY=0.996 \
JEPA_EMA_END_DECAY=1.0 \
JEPA_PREDICTOR_HIDDEN_DIM=384 \
JEPA_TARGET_NORMALIZATION=layernorm \
JEPA_LOSS_TYPE=smooth_l1 \
JEPA_TARGET_SELECTION=masked_modules \
bash pretrain.sh
```

The launcher defaults to 400 epochs with 40 warm-up epochs for MAE, and to a
hypothesis-driven 100 epochs with 10 warm-up epochs for JEPA.  The shorter JEPA
default follows the convergence-efficiency motivation in the
[I-JEPA paper](https://arxiv.org/abs/2301.08243), whose primary ImageNet
comparison trained I-JEPA for 300 epochs versus 1,600 for MAE.  This is evidence
for testing a shorter schedule, not evidence that 100 epochs is already optimal
for FASERCal.  Run the necessary epoch-matched control with:

```bash
PRETRAINING_OBJECTIVE=jepa \
EPOCHS=400 \
WARMUP_EPOCHS=40 \
COSINE_ANNEALING_EPOCHS=360 \
NAME=pretrain_jepa_epoch_matched \
CHECKPOINT_NAME=pretrain_jepa_epoch_matched \
bash pretrain.sh
```

`EPOCHS`, `WARMUP_EPOCHS`, and `COSINE_ANNEALING_EPOCHS` always take priority.
The objective-specific defaults can also be changed with `JEPA_EPOCHS`,
`JEPA_WARMUP_EPOCHS`, and `JEPA_COSINE_ANNEALING_EPOCHS` (or their `MAE_*`
counterparts).

Fine-tune the completed `tiny` JEPA run from its final EMA target encoder:

```bash
conda run -n spconv bash finetune_jepa.sh
```

The wrapper uses both local GPUs by default with 512 events per device (the
same global batch of 1,024 as the one-GPU fine-tuning recipe), loads
`checkpoints_final/pretrain_jepa_latent/loss_total_val/last.ckpt`, selects the
EMA target branch, and delegates every downstream task and optimisation choice
to `finetune.sh`.  Override `LOAD_CHECKPOINT`, `GPUS`, or
`PRETRAINED_ENCODER_SOURCE=context` for an ablation.  For a future `base` run,
set `MODEL=base` in both pretraining and fine-tuning.

For the local MAE-versus-JEPA downstream comparison, run the MAE control with
the same model, seed, global batch, and two-GPU topology:

```bash
LOAD_CHECKPOINT=/path/to/mae/last.ckpt \
PRETRAINED_ENCODER_SOURCE=auto MODEL=tiny SEED=7 \
GPUS='0 1' BATCH_SIZE=512 \
NAME=finetune_mae CHECKPOINT_NAME=finetune_mae \
conda run -n spconv bash finetune.sh
```

After the JEPA fine-tuning run has completed all 20 validation epochs, create
the test-set CSV with:

```bash
conda run --no-capture-output -n spconv \
  python -m evaluation.create_csv_v7_ft --gpu 0
```

The evaluator reads the architecture, preprocessing, dataset, metadata, seed,
and JEPA transfer provenance from the fine-tuning `hparams.yaml`.  It recreates
the exact deterministic 85/5/10 split used by training and evaluates only the
122,639 test files, without shuffling or dropping the final batch.  It chooses
the lowest-validation-loss checkpoint separately for flavour, charm, visible
momentum, jet momentum, and vertex, matching the existing evaluation method.
The CSV has the canonical 45 columns; a test-file manifest and JSON provenance
sidecar are written beside it under `results/`.

For a short post-training smoke output, use a distinct automatically named
partial CSV:

```bash
conda run --no-capture-output -n spconv \
  python -m evaluation.create_csv_v7_ft --gpu 0 --max_batches 2
```

Existing outputs are never overwritten unless `--overwrite` is supplied.

The final checkpoint is intentional.  Although the minimum JEPA validation
loss occurred at epoch 1, a 32-event real-data diagnostic found an almost
rank-one event representation there.  The epoch-99 EMA encoder had much
healthier event diversity (participation rank 19.39 out of a maximum 31 and
mean off-diagonal cosine 0.143), so a moving-target latent loss is not a safe
standalone downstream-checkpoint selector for this run.

Run unit/integration tests and the local real-data smoke test:

```bash
conda run -n spconv python -m unittest discover -s tests -p 'test_*.py'
conda run -n spconv python -m tests.real_data_smoke --device cuda
conda run -n spconv python -m tests.inspect_jepa_checkpoint \
  checkpoints_final/pretrain_jepa_latent/loss_total_val/last.ckpt \
  --samples 32 --device cuda
conda run -n spconv python -m tests.real_finetune_smoke \
  --checkpoint checkpoints_final/pretrain_jepa_latent/loss_total_val/last.ckpt \
  --model tiny --device cuda
```

Exercise the actual two-GPU Lightning/DDP fine-tuning entry point without
writing checkpoints or starting a full run:

```bash
FAST_DEV_RUN=1 EPOCHS=1 BATCH_SIZE=2 NUM_WORKERS=0 \
conda run -n spconv bash finetune_jepa.sh
```

## Lightweight measured snapshot

On the `tiny` configuration used by `pretrain.sh`, the local implementation
has the following directly counted parameters:

| Quantity | Parameters |
|---|---:|
| MAE trainable total | 34,259,399 |
| Shared encoder trainable | 31,925,952 |
| MAE reconstruction machinery (trainable difference) | 2,333,447 |
| JEPA predictor | 296,448 |
| JEPA trainable total (context + predictor) | 32,222,400 |
| Frozen JEPA target encoder (stored) | 31,965,888 |
| JEPA stored total | 64,228,224 |

Thus JEPA has about 2.04 million fewer trainable parameters than MAE in this
configuration, but stores a second full encoder and executes it once without
backpropagation.  Real-data batch-size-one tests ran independently on both an
H100 NVL and an A100 80GB with full sparse-convolution backward enabled.  Both
produced finite MAE and JEPA losses, context/predictor gradients only for JEPA,
an optimizer step, and a non-zero EMA update.  The JEPA checkpoint loaded 263
encoder tensors into `SparseViT`; the 20 missing tensors were all downstream
task-head parameters, as intended.

After one optimizer warm-up step, the standalone FP32 measurements were:

| GPU | MAE step | JEPA step | MAE peak allocated | JEPA peak allocated |
|---|---:|---:|---:|---:|
| H100 NVL | 83.2 ms | 60.1 ms | 735.7 MiB | 817.0 MiB |
| A100 80GB | 119.0 ms | 65.8 ms | 686.3 MiB | 773.0 MiB |

These are batch-size-one smoke measurements, with the MAE hybrid reconstruction
loss enabled and its optional relational pass disabled.  They show that this
JEPA implementation used roughly 11--13% more peak allocated memory, despite a
shorter step in these two runs.  Full-size batches, mixed precision, repeated
timing samples, and the production data-loader/DDP setup are still required for
a reliable throughput comparison.

## Current limitations and follow-up experiments

This baseline predicts final module-level Perceiver slots, not one latent per
masked patch.  This is intentional: the current downstream representation is
fixed-size, whereas exposing patch-level encoder targets would require a more
invasive encoder-interface change.  The existing random per-module mask is
also retained rather than replaced with contiguous JEPA blocks.  ECAL and muon
masking influence the latent prediction but do not have dedicated target
slots.  Finally, JEPA removes the decoder but evaluates a second full encoder,
so reduced wall time or memory must be measured rather than assumed.
On the first real-data smoke event, `masked_modules` selected all 15 valid
latent slots at a 75% mask ratio.  This is only one event, but it confirms that
module-level target selection can saturate; `jepa/target_coverage` is now logged
during training so this can be quantified over the full validation set.

The clean first experiment uses identical data, masking ratio, encoder,
optimizer family, and fine-tuning.  Compare both the shorter JEPA schedule
(testing convergence efficiency) and the 400-epoch JEPA control (isolating the
objective at matched epochs).  Useful follow-ups, each varied separately, are:

1. context versus EMA-target initialization for downstream fine-tuning;
2. masked-module versus all-valid-slot supervision;
3. EMA decay and latent normalization/loss sweeps;
4. a shallow attention predictor versus the default slot-wise MLP;
5. contiguous/module-spanning target masks;
6. only after the objective comparison, patch-level targets or alternative
   patching/tokenisation.

The default predictor is deliberately much smaller than the predictors in
canonical I-JEPA: it is a shared two-layer MLP over already-globalized latent
slots.  That is a reasonable minimum-control predictor for the current 22-slot
objective, because the encoder has already mixed information across slots, but
it cannot perform additional cross-slot reasoning.  The official
[I-JEPA configuration](https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml)
uses a narrow 12-block, width-384 Transformer predictor for patch-position
queries.  A two-block latent Transformer is therefore a useful isolated
follow-up for the module-level objective; a positional Transformer/cross-attention
predictor becomes substantially more important for a future patch-target JEPA.
