# JEPA pretraining baseline

This repository supports the published masked-autoencoder objective and a
controlled JEPA-style latent-prediction objective.  The JEPA baseline is
deliberately narrow: it changes the stage-1 objective while preserving the
event representation, sparse patching, hierarchical encoder, data pipeline,
optimiser/scheduler conventions, checkpointing, and downstream model.

A separate, spconv-free implementation that follows I-JEPA more closely
(plain global ViT with varlen flash attention, multi-block 3D masking, no
Perceiver) is described in [FLASH_JEPA.md](FLASH_JEPA.md).  It leaves
everything in this document unchanged.

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
  schedule from `JEPA_EMA_DECAY` to `JEPA_EMA_END_DECAY`.  The update runs in
  `JEPAPreTrainer.optimizer_step`, after `optimizer.step()` has returned.  (The
  completed `pretrain_jepa_latent` run used an earlier `on_before_zero_grad`
  hook, which Lightning calls inside the step closure before backward: each
  update used the previous step's context weights.  At decay >= 0.996 this
  one-step lag has a negligible effect on that run.);
- a discardable predictor maps the context view to the targets selected by
  `JEPA_TARGET_LEVEL` (below).

No voxel reconstruction, DCT basis, voxel heads, or relational head is
constructed in JEPA mode.

### Patch-level targets (`JEPA_TARGET_LEVEL=patch`, the default)

The targets are the target encoder's own per-token outputs on the full event:
the FASERCal patch tokens after the module transformer and its final norm
(before the module embedding is added), the AHCAL patch tokens after the AHCAL
norm, and the ECAL and muon tokens.  A target is every *masked occupied*
FASERCal or AHCAL patch of the context view, plus the ECAL/muon token of each
event in which the context dropped it.

The predictor mirrors the MAE Perceiver-IO decoder that it replaces: width
256, 8 heads, 2 cross-attention blocks (`JEPA_PREDICTOR_EMBED_DIM`,
`JEPA_PREDICTOR_NUM_HEADS`, `JEPA_PREDICTOR_DEPTH`).  Each masked FASERCal
patch becomes a query `fixed sin-cos(intra-module position) + learned module
embedding + learned query token`; AHCAL patches use their sin-cos position and
an AHCAL query token; dropped ECAL/muon tokens use two learned global queries.
Queries attend only to the 22 context latent slots (never to visible patch
tokens or to one another), so every prediction must be inferred from the
representation that the downstream model pools.  A norm and a linear layer map
to the encoder width.  It has 1.78 M trainable parameters for `base`, versus
about 2.3 M for the MAE decoder machinery.  Predictions and stop-gradient
targets are layer-normalized per token and compared with Smooth L1, averaged
over all target tokens; `jepa/loss_{fcal,ahcal,ecal,muon}` are logged
separately.  `JEPA_TARGET_SELECTION` must stay `masked_modules`.

### Slot-level targets (`JEPA_TARGET_LEVEL=slot`, the first completed run)

The original objective predicts the 22 final Perceiver latent slots of the
full view from the 22 context slots with a slot-wise
`LayerNorm -> Linear -> GELU -> Dropout -> Linear` MLP
(`JEPA_PREDICTOR_HIDDEN_DIM`).  It has no positional queries and never
specifies which patches were hidden; it is closer to masked-view to full-view
self-distillation of module summaries than to I-JEPA.  A latent slot
contributes to the loss when its FASERCal module (or the AHCAL group) contains
at least one masked occupied patch, or every valid slot with
`JEPA_TARGET_SELECTION=all`.  `pretrain_jepa_latent` used this objective, so
reload or resume it with `--jepa_target_level slot` (the library default of
`SparseJEPA`/`jepa_vit_*`).  The two objectives store different predictor
weights, so a checkpoint cannot be resumed under the other objective.

For both levels, normalization (`none`, `layernorm`, `l2`) and loss
(`smooth_l1`, `mse`, `cosine`) are configurable.  `jepa/target_event_std` and
`jepa/context_event_std` report the per-slot spread of the target and context
latents across events (the collapse monitor for the downstream
representation).

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
the main fine-tuning entry point fails on a partial load.  The historical
`pretrain.sh` default is `tiny`; the generic fine-tuning and scratch launchers
retain the published `base` default.  Always set `MODEL` explicitly for a
controlled comparison.  `finetune_jepa.sh` pins `tiny` because that is the
architecture of the completed local JEPA checkpoint.

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

## Independent audit of the completed run (2026-10-21)

The code paths above were verified on real GENIE events: perturbing every
masked FASERCal/AHCAL patch and every dropped ECAL/muon input leaves the
context latents bit-identical; the target encoder receives no gradients and is
absent from AdamW; the EMA update matches its formula; slot/module target
selection matches an explicit loop; and `vit_tiny` loaded from the EMA branch
reproduces the pooled target latents exactly.  The MAE forward is bit-identical
to commit `bdba522` with the published checkpoint.

The *completed run*, however, is not an objective-only control of the
published MAE model.  Its code is shared, but its configuration is not:

| | Published MAE (`pretrain_7.0_lat_norel_v2`) | Completed JEPA (`pretrain_jepa_latent`) |
|---|---|---|
| Encoder | `base` (depth 4/4/4) | `tiny` (depth 2/2/6) |
| Epochs | 400 (40 warm-up) | 100 (10 warm-up) |
| Effective batch | 4,096 (8 GPUs x 512) | 2,048 (2 GPUs x 512 x accumulation 2) |
| blr / lr | 1e-4 / 1.6e-3 | 1.5e-4 / 1.2e-3 |
| Optimizer steps / events seen | ~101,600 / ~417 M | 50,900 / ~104 M |
| Seed | not recorded | not recorded (pre-`--seed` code) |

Fine-tuning used the published recipe (global batch 1,024, 20 epochs,
blr 5e-4, layer decay 0.75) on `tiny`.  The only architecture-matched reference
is `finetune_v7.0_finerel_paper` in `logs_final/`, which is a **from-scratch
`tiny`** run with the fine-tuning recipe (no `load_checkpoint`), not an MAE
run.  There is no `tiny` MAE checkpoint.

Representation diagnostics (4,096 validation events, pooled frozen features):
the epoch-1 EMA encoder is almost rank one (top eigenvalue 93% of the
variance), whereas the final encoder is not collapsed (participation ratio 55
versus 12 for the published MAE `base`; the predictor explains 93% of the
event-to-event variance of the normalized targets).  Frozen linear probes of
the final JEPA encoder are nevertheless *below* a random-initialization
encoder for flavour (macro-AUROC 0.61 versus 0.71) and visible momentum, and
above it only for the vertex (R^2 0.95 versus 0.84); MAE `base` gives 0.78 and
0.99.  Use `jepa/target_event_std` (per-slot spread across events in the loss
space) to monitor collapse: it is 0.13 for the epoch-1 checkpoint and 0.77 for
the final one, whereas the pooled `jepa/target_std` does not separate them.

Best validation losses on the shared validation split:

| Run | Flavour | Charm | Vis. p | Jet p | Lepton p | Vertex |
|---|---:|---:|---:|---:|---:|---:|
| Scratch `tiny`, fine-tuning recipe | 0.638 | 0.251 | 1.463 | 0.374 | 0.313 | 0.0093 |
| JEPA `tiny` (EMA target) | 0.587 | 0.242 | 1.349 | 0.322 | 0.281 | 0.0074 |
| Published scratch `base` (40 epochs) | 0.672 | 0.259 | 1.541 | 0.404 | 0.333 | 0.0096 |
| Published MAE `base` | 0.513 | 0.217 | 1.151 | 0.264 | 0.238 | 0.0061 |
| Published MAE+Rel `base` | 0.501 | 0.218 | 1.127 | 0.260 | 0.236 | 0.0057 |

Test-set metrics on the identical 122,639 events (unweighted, no nu_tau
reweighting; per-task minimum-validation-loss checkpoints; sigma_MAD of the
fractional residual).  The evaluator reproduces the published MAE CSV to 1e-6
in probability, and these vertex medians match the paper's 22.1/16.9/16.1 mm:

| Metric | Scratch `base` (pub.) | Scratch `tiny` | JEPA `tiny` | MAE `base` (pub.) | MAE+Rel `base` (pub.) |
|---|---:|---:|---:|---:|---:|
| Flavour accuracy | 0.744 | 0.759 | 0.781 | 0.815 | 0.821 |
| AUC nu_mu CC | 0.905 | 0.916 | 0.928 | 0.947 | 0.951 |
| AUC NC | 0.876 | 0.894 | 0.911 | 0.939 | 0.943 |
| AUC nu_tau CC -> had | 0.899 | 0.910 | 0.924 | 0.939 | 0.942 |
| AUC charm -> had | 0.783 | 0.804 | 0.830 | 0.869 | 0.870 |
| AUC charm -> mu | 0.829 | 0.848 | 0.832 | 0.884 | 0.884 |
| sigma_MAD E_vis | 0.353 | 0.331 | 0.289 | 0.214 | 0.212 |
| sigma_MAD jet p | 0.236 | 0.229 | 0.200 | 0.164 | 0.161 |
| Median vertex error (mm) | 22.1 | 24.4 | 17.5 | 16.9 | 16.1 |

JEPA `tiny` is better than scratch `tiny` on every listed metric except
charm -> mu (paired event bootstrap; one seed per run, so seed variance is not
included).  It recovers roughly 35-40% of the scratch-to-MAE gap on
classification and momentum and most of it on the vertex.  Because the MAE
references are `base` models trained four times longer, these numbers do not
isolate the objective; the controlled comparison still requires the runs
listed in the follow-up section.

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

The `pretrain.sh` defaults (`MODEL=tiny`, `BLR=1.5e-4`,
`ACCUM_GRAD_BATCHES=2`, `GPUS='0 1'`) do not reproduce the published MAE
run.  Its recorded hparams (`pretrain_7.0_lat_norel_v2`) are `mae_vit_base`,
512 events per GPU on 2 nodes x 4 GPUs with no accumulation (effective batch
4,096), `blr=1e-4` (lr 1.6e-3), betas (0.9, 0.95), weight decay 0.05, mask
ratio 0.75, hybrid reconstruction loss, relational pass probability 0, and 400
epochs (40 warm-up, 360 cosine).  On the two local GPUs the same effective
batch, learning rate and step count are obtained with
`MODEL=base BLR=1e-4 BATCH_SIZE=1024 ACCUM_GRAD_BATCHES=2 GPUS='0 1'` (or
`BATCH_SIZE=512 ACCUM_GRAD_BATCHES=4`); use the same overrides for a JEPA run
intended as an objective-only control.

Run the controlled patch-level JEPA counterpart of the published MAE model
(same `base` encoder, effective batch 4,096, `blr=1e-4`, and 400 epochs with
40 warm-up epochs; two local GPUs, 1,024 events per GPU, accumulation 2).
Use a new `NAME`:
reusing `pretrain_jepa_latent` would overwrite the completed run's
`last.ckpt`.

```bash
PRETRAINING_OBJECTIVE=jepa \
JEPA_TARGET_LEVEL=patch \
MODEL=base \
SEED=7 \
GPUS='0 1' \
BATCH_SIZE=1024 \
ACCUM_GRAD_BATCHES=2 \
NUM_WORKERS=32 \
BLR=1e-4 \
EPOCHS=400 \
WARMUP_EPOCHS=40 \
COSINE_ANNEALING_EPOCHS=360 \
NAME=pretrain_jepa_patch_base \
CHECKPOINT_NAME=pretrain_jepa_patch_base \
bash pretrain.sh
```

Throughput on the local A100 + H100 pair, measured with the real launcher at
the same effective batch: 512 x 4 with 16 workers per GPU gave ~1,800
events/s while JEPA still built the MAE relational voxel labels; skipping
those labels (now automatic for JEPA, since it reads only detector inputs and
the augmented inputs are bit-identical) gave 3,060 events/s; 32 workers gave
3,170; and 1,024 x 2 with 32 workers gave 3,650 events/s (~4.8 min per
training epoch).  2,048 events per GPU is not possible: spconv's
sparse-convolution backward requires fewer than 11,184,810 FASERCal voxels per
micro-batch (`voxels x 96 channels x 2 bytes < 2^31`).  A 2,048-event batch
averages ~11.6 M voxels, whereas 1,024-event batches average 5.8 M and the
largest of 400,000 simulated ones reached 61% of the limit.

Prefix the same command with `FAST_DEV_RUN=1 SAVE_DIR=/tmp/jepa_dry
CHECKPOINT_PATH=/tmp/jepa_dry` to run one training and one validation batch
through the full DDP configuration without writing to `logs_final` or
`checkpoints_final`.

Fine-tune the completed `pretrain_jepa_patch_base` run with the published
recipe (`finetune.sh` defaults: 20 epochs, `blr=5e-4`, layer decay 0.75,
drop path 0.2, global batch 1,024) from its EMA target encoder.  Two GPUs with
512 events each keep the paper's global batch, learning rate, and step count;
measured throughput was 2,850 events/s, versus 2,400 events/s on one H100 with
1,024.  Write the checkpoints to `/scratch4`: up to 7 monitored losses x
(10 best + `last`) x ~0.5 GB is ~39 GB, more than is free on `/scratch`.

```bash
LOAD_CHECKPOINT=/scratch/salonso/sparse-nns/faser_develop/checkpoints_final/pretrain_jepa_patch_base/loss_total_val/last.ckpt \
PRETRAINED_ENCODER_SOURCE=target \
MODEL=base \
SEED=7 \
GPUS='0 1' \
BATCH_SIZE=512 \
NUM_WORKERS=32 \
NAME=finetune_jepa_patch_base \
CHECKPOINT_NAME=finetune_jepa_patch_base \
CHECKPOINT_PATH=/scratch4/salonso/faser/checkpoints_final \
bash finetune.sh
```

Then create its test CSV
(`results/results_v7_test_finetune_jepa_patch_base.csv`):

```bash
python -m evaluation.create_csv_v7_ft \
  --checkpoint_dir /scratch4/salonso/faser/checkpoints_final/finetune_jepa_patch_base --gpu 0
```

Result of that controlled pair (2026-10-23; one fine-tuning seed each; same
122,639 test events; unweighted; σ_MAD of fractional residuals).  JEPA-patch
`base` was pretrained and fine-tuned with the published MAE configuration and
recipe; only the stage-1 objective differs:

| Metric | Scratch `base` (pub.) | JEPA-patch `base` | MAE `base` (pub.) | MAE+Rel `base` (pub.) |
|---|---:|---:|---:|---:|
| Flavour accuracy | 0.744 | 0.803 | 0.815 | 0.821 |
| AUC nu_mu CC | 0.905 | 0.942 | 0.947 | 0.951 |
| AUC NC | 0.876 | 0.930 | 0.939 | 0.943 |
| AUC nu_tau CC -> had | 0.899 | 0.932 | 0.939 | 0.942 |
| AUC nu_tau CC -> mu | 0.806 | 0.840 | 0.836 | 0.841 |
| AUC charm -> had | 0.783 | 0.853 | 0.869 | 0.870 |
| AUC charm -> mu | 0.829 | 0.886 | 0.884 | 0.884 |
| σ_MAD E_vis | 0.353 | 0.226 | 0.214 | 0.212 |
| σ_MAD pT_miss | 0.823 | 0.611 | 0.557 | 0.559 |
| σ_MAD jet p | 0.236 | 0.175 | 0.164 | 0.161 |
| Median vertex error (mm) | 22.1 | 16.6 | 16.9 | 16.1 |

JEPA recovers about 80-90% of the scratch-to-MAE gap on classification and
momentum.  It is slightly but significantly behind MAE on most metrics (paired
event bootstrap), indistinguishable on nu_tau -> mu, charm -> e and charm -> mu,
and better on the median vertex error.  Several JEPA validation losses were
still decreasing at epoch 19, whereas the MAE fine-tune had plateaued by about
epoch 11.  Seed-to-seed variation is not included in these comparisons.

A longer fine-tune tests whether JEPA is limited by the fine-tuning budget (its
training loss stays at or above its validation loss at epoch 19, whereas MAE's
is 19-23% lower).  It changes the recipe, so it is a separate experiment from
the controlled comparison above; comparing it with MAE requires the same
40-epoch recipe for MAE.  `SAVE_TOP_K=1 SAVE_LAST=0
CHECKPOINT_MONITORS=evaluated` keeps only the single best checkpoint of each
of the five task losses that the test evaluation reads (no `last.ckpt`, and
no total or lepton checkpoints), so such a run cannot be resumed after a crash.

```bash
LOAD_CHECKPOINT=/scratch/salonso/sparse-nns/faser_develop/checkpoints_final/pretrain_jepa_patch_base/loss_total_val/last.ckpt \
PRETRAINED_ENCODER_SOURCE=target \
MODEL=base \
SEED=7 \
GPUS='0 1' \
BATCH_SIZE=512 \
NUM_WORKERS=32 \
EPOCHS=40 \
WARMUP_EPOCHS=5 \
COSINE_ANNEALING_EPOCHS=35 \
SAVE_TOP_K=1 \
SAVE_LAST=0 \
CHECKPOINT_MONITORS=evaluated \
NAME=finetune_jepa_patch_base_ep40 \
CHECKPOINT_NAME=finetune_jepa_patch_base_ep40 \
CHECKPOINT_PATH=/scratch4/salonso/faser/checkpoints_final \
bash finetune.sh
```

The matching MAE control fine-tunes the published MAE-only checkpoint
(`pretrain_v7.0_lat_norel_v2`, the encoder behind the published MAE `base`
fine-tune, 101,600 pretraining steps like the JEPA run) with the identical
40-epoch recipe, topology, seed, and checkpoint retention.  Its 255 encoder
tensors load with 100% coverage (the 91 decoder tensors are discarded), and
`SparseViT` reproduces the pooled MAE latents exactly:

```bash
LOAD_CHECKPOINT=/scratch2/salonso/faser/checkpoints_final/pretrain_v7.0_lat_norel_v2/loss_total_val/last.ckpt \
PRETRAINED_ENCODER_SOURCE=auto \
MODEL=base \
SEED=7 \
GPUS='0 1' \
BATCH_SIZE=512 \
NUM_WORKERS=32 \
EPOCHS=40 \
WARMUP_EPOCHS=5 \
COSINE_ANNEALING_EPOCHS=35 \
SAVE_TOP_K=1 \
SAVE_LAST=0 \
CHECKPOINT_MONITORS=evaluated \
NAME=finetune_mae_base_ep40 \
CHECKPOINT_NAME=finetune_mae_base_ep40 \
CHECKPOINT_PATH=/scratch4/salonso/faser/checkpoints_final \
bash finetune.sh
```

`evaluation.create_csv_v7_ft` accepts JEPA (`target` or `context` branch) and
MAE fine-tunes alike, refusing scratch runs, partial encoder coverage, and an
encoder source that does not match the pretraining checkpoint.  Evaluate each
run with `--checkpoint_dir /scratch4/salonso/faser/checkpoints_final/<NAME>`.

Results at 20 and 40 fine-tuning epochs (2026-10-25; one seed per run; same
122,639 test events; unweighted).  The two 40-epoch runs differ only in the
loaded pretraining checkpoint.  The 20-epoch MAE run is the published one,
which also differs in code version, topology (1 GPU), and seed (unseeded):

| Metric | JEPA 20 ep | MAE 20 ep (pub.) | JEPA 40 ep | MAE 40 ep |
|---|---:|---:|---:|---:|
| Flavour accuracy | 0.803 | 0.815 | 0.797 | 0.813 |
| AUC NC | 0.930 | 0.939 | 0.927 | 0.938 |
| AUC charm -> had | 0.853 | 0.869 | 0.847 | 0.864 |
| σ_MAD E_vis | 0.226 | 0.214 | 0.239 | 0.233 |
| σ_MAD jet p | 0.175 | 0.164 | 0.179 | 0.182 |
| σ_MAD lepton p (CC) | 0.397 | 0.386 | 0.438 | 0.419 |
| Median vertex error (mm) | 16.6 | 16.9 | 16.1 | 18.6 |

The 40-epoch recipe helped neither objective.  Its validation minima occurred
mid-schedule (MAE epochs 13-24, JEPA epochs 19-35) while the learning rate was
still high, after which both overfit (JEPA's flavour training loss fell below
its validation loss at about epoch 25).  JEPA-40 is worse than JEPA-20 on every
metric except the vertex (paired event bootstrap), so the improvement at the
end of the 20-epoch run came from learning-rate annealing, not from an
unexhausted fine-tuning budget.  Both recipes give the same ranking: MAE leads
on flavour and charm classification (by 1.2 and 1.7 points of flavour accuracy
at 20 and 40 epochs) and JEPA leads on the median vertex error (by 0.3 and
2.5 mm).  The published 20-epoch recipe remains the best for both objectives
and is the headline controlled comparison.

`base` is required for a direct comparison with the published headline MAE
model.  Omitting `MODEL=base` selects the launcher's historical `tiny` default,
which is a valid separate architecture but not an objective-only paper control.

The launcher defaults to 400 epochs with 40 warm-up epochs for MAE, and to a
hypothesis-driven 100 epochs with 10 warm-up epochs for JEPA.  The shorter JEPA
default follows the convergence-efficiency motivation in the
[I-JEPA paper](https://arxiv.org/abs/2301.08243), whose primary ImageNet
comparison trained I-JEPA for 300 epochs versus 1,600 for MAE.  This is evidence
for testing a shorter schedule, not evidence that 100 epochs is already optimal
for FASERCal.  For that shorter variant, use the command above with
`EPOCHS=100 WARMUP_EPOCHS=10 COSINE_ANNEALING_EPOCHS=90` and a different `NAME`.

`EPOCHS`, `WARMUP_EPOCHS`, and `COSINE_ANNEALING_EPOCHS` always take priority.
The objective-specific defaults can also be changed with `JEPA_EPOCHS`,
`JEPA_WARMUP_EPOCHS`, and `JEPA_COSINE_ANNEALING_EPOCHS` (or their `MAE_*`
counterparts).

JEPA defaults to `SAVE_TOP_K=0` and `EARLY_STOP_PATIENCE=0`: the EMA teacher
changes across epochs, so its validation loss is not a valid cross-epoch model
selection criterion.  `last.ckpt` is always saved.  Override these settings
only when using an independently justified selection rule.

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

The default objective now predicts one target-encoder token per masked patch
(and per dropped ECAL/muon token) through positional queries, while the
downstream representation remains the fixed-size pooled latent set.  Patch
targets are the outputs of the module-local (FASERCal) and AHCAL transformers:
the encoder has no deeper per-patch representation, because the Perceiver
updates only its latent slots.  The predictor therefore reads only the context
latents, as the MAE decoder does; giving it the visible patch tokens would let
it interpolate locally without passing information through the downstream
representation.  The existing random per-module mask is retained rather than
replaced with contiguous JEPA blocks, and JEPA evaluates a second full encoder,
so wall time and memory must be measured rather than assumed.

Patch tokens contain their input position, so a content-blind predictor can
explain the positional part of every target.  Check `jepa/target_event_std`
together with the loss: a falling loss with a vanishing cross-event spread
indicates a positional or collapsed solution rather than learned content.

The clean first experiment uses identical data, masking ratio, encoder,
optimizer configuration, and fine-tuning.  Useful follow-ups, each varied
separately, are:

1. context versus EMA-target initialization for downstream fine-tuning;
2. EMA decay and latent normalization/loss sweeps;
3. predictor depth/width (the MAE-decoder size is the controlled default);
4. contiguous/module-spanning target masks;
5. only after the objective comparison, alternative patching/tokenisation.
