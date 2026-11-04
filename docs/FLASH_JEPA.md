# Flash-attention JEPA (spconv-free)

A second JEPA implementation that follows I-JEPA more closely than the
patch-level JEPA in [JEPA.md](JEPA.md).  It keeps none of the published
architecture's special structure: no spconv, no module-local transformers, no
Perceiver bottleneck.  The published MAE, the slot-level JEPA and the
patch-level JEPA are unchanged and still run from `pretrain.sh` / `finetune.sh`
in the `spconv` environment, so all paper comparisons stay reproducible.

Everything here runs in the `neutrino-jepa` environment (PyTorch 2.14, CUDA
13).  Its flash attention is PyTorch's built-in
`torch.nn.attention.varlen.varlen_attn` (FlashAttention-2 kernels).  The
environment has no spconv, timm, torch_ema or webdataset, and the new code
needs none of them.

## Recommended recipe (as of 2026-11-03)

The rest of this file is a chronological log.  This is the best recipe it
found; the evidence is in "Stabilisation", under "Predictor QK-norm, three
seeds: test metrics".

**Pretraining:** K = 16 data2vec targets with predictor QK-norm.  Both are
opt-in, so set them explicitly; the launcher defaults (`TARGET_LAYERS=1`,
`PREDICTOR_QK_NORM=0`) keep the earlier runs reproducible.
- **Targets:** `TARGET_LAYERS=16`, as in data2vec.  The MLP outputs of
  all 16 target-encoder blocks, before the residual, are each
  layer-normalised, averaged, and normalised again per token
  (`TARGET_NORM=token`, the default; see "data2vec targets").
- **Predictor QK-norm:** `PREDICTOR_QK_NORM=1`.  Without it, every seed of
  this recipe had a collapse episode (2 seeds at BLR 1e-4, 3 at 5e-5).
  With it, 3 of 3 seeds had none.
- **Everything else is the launcher's default:**
  - `base` encoder;
  - effective batch 4,096 and BLR 1e-4 (peak LR 1.6e-3), with 40 warm-up
    epochs then cosine, for 400 epochs;
  - weight decay 0.05;
  - teacher EMA 0.996 -> 1.0;
  - the default masks.
- **Monitoring (optional):** `LOG_GRAD_NORM=1` logs the gradient norm per
  block and of the predictor.
- **Use `loss_total_val/last.ckpt`.**  The fine-tune loads its target (EMA)
  encoder.

**Fine-tuning:** the published 20-epoch recipe (`finetune_flash.sh`
defaults) with `LAYER_DECAY=0.85`.
- Defaults: global batch 1,024, BLR 5e-4, 5 warm-up and 15 cosine epochs,
  drop path 0.2, label smoothing 0.02, weight decay 0.05, model EMA 0.9999.
- 0.85 beat 0.75 and 0.80; 0.90 was no better (see the two layer-decay
  sections).
- Fine-tuning seed 7 throughout.

**Results:** test set, the same 122,639 events.
- Pretraining seeds 7, 8 and 9 give flavour accuracy 0.8256, 0.8243 and
  0.8250 (published MAE base: 0.8153).
- Their mean is better than the published MAE base on all 15 metrics
  (paired event bootstrap).
- It recovers 102-141% of the scratch -> MAE gap per metric.
- **With fine-tuning noise counted** (each encoder fine-tuned twice): the
  QK-norm encoders are better than the MAE by more than 4 standard errors
  on 13 of 15 metrics.  pT_miss is a tie (-0.003, 0.7 SE), and the vertex
  median is 1.6 mm lower (1.9 SE).  See "300 epochs and repeat
  fine-tunes: results".

**Cost on Clariden:** about 5 h of pretraining on 2 nodes (8 GH200),
then about 40 min of fine-tune plus test CSV on 1 node.
- **A cheaper option, `EPOCHS=300` (25% less compute), tested in one
  run.**  It anneals the LR and the teacher EMA fully by epoch 300.
  - Seed 7 on dlnu falls within the 400-epoch seeds' spread on all 15
    metrics: |z| <= 2.3 against their mean and SD.  The largest
    differences have opposite signs: AUC nutau->had -0.0018, nutau->e
    +0.0016.
  - Flavour accuracy 0.8238, against 0.8243-0.8256 at 400 epochs.
  - One seed can't resolve differences below ~0.001 in flavour accuracy,
    so 400 epochs stays the default here.
  - See "300 epochs and repeat fine-tunes: results".  The epoch-299
    snapshots of the 400-epoch runs had already fine-tuned as well as
    `last.ckpt` ("Snapshot fine-tunes: results",
    `results/flash_audit_20261026/qkn_epochs.png`).
- **Cost on dlnu** (A100 + H100, under another user's CPU load): 4.7 min
  per epoch, so about 23 h for 300 epochs.  Each fine-tune plus test CSV
  takes 1.5-2 h.

**Caveats:**
- The vertex median does not measurably improve on K = 1 or on the
  healthy encoders without QK-norm: the differences (0.2-0.6 mm) are within
  1 SE.
  - A repeat fine-tune of one encoder moves it with an SD of 0.73 mm.
    The spread across pretraining seeds is about as large.
  - The event bootstrap's 95% half-width is only ~0.1 mm, so it can't be
    used alone for single-run vertex comparisons.
  - The regressions are understated too, by less: for |p_lep| CC the SD is
    0.0085, against a half-width of ~0.003.
- The QK-norm's learned gain and bias still let the last predictor blocks'
  attention logits grow in the second half of training.  No episode
  followed in 400 epochs, but longer runs are untested.

**Commands on Clariden,** from `/users/salonso/faserDL`.  One job pretrains,
fine-tunes and writes the test CSV:

```bash
NAME=pretrain_flash_jepa_base_d2v16_qkn SEED=7 TARGET_LAYERS=16 PREDICTOR_QK_NORM=1 LOG_GRAD_NORM=1 FT_NAME=finetune_flash_jepa_base_d2v16_qkn_ld085 sbatch clariden/pretrain_finetune_flash.sbatch
```

The wrapper fine-tunes with `FT_SEED=7` and `FT_LAYER_DECAY=0.85` by default.
To fine-tune and evaluate an existing pretraining separately (1 node, fits
the debug partition):

```bash
LOAD_CHECKPOINT=/capstor/scratch/cscs/salonso/faser/checkpoint/pretrain_flash_jepa_base_d2v16_qkn/loss_total_val/last.ckpt NAME=finetune_flash_jepa_base_d2v16_qkn_ld085 SEED=7 LAYER_DECAY=0.85 sbatch clariden/finetune_evaluate_flash.sbatch
```

**Commands on dlnu (2 GPUs):** the "Commands" section below, with the
recipe's settings added:

```bash
NAME=pretrain_flash_jepa_base_d2v16_qkn TARGET_LAYERS=16 PREDICTOR_QK_NORM=1 LOG_GRAD_NORM=1 DATASET_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*' METADATA_PATH=/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz/metadata_stats.pkl bash pretrain_flash.sh
NAME=finetune_flash_jepa_base_d2v16_qkn_ld085 LAYER_DECAY=0.85 GPUS='0 1' BATCH_SIZE=512 NUM_WORKERS=24 CHECKPOINT_PATH=/scratch4/salonso/faser/checkpoints_final SAVE_TOP_K=1 SAVE_LAST=0 CHECKPOINT_MONITORS=evaluated LOAD_CHECKPOINT=checkpoints_final/pretrain_flash_jepa_base_d2v16_qkn/loss_total_val/last.ckpt DATASET_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*' METADATA_PATH=/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz/metadata_stats.pkl bash finetune_flash.sh
python -m evaluation.create_csv_v7_ft --checkpoint_dir /scratch4/salonso/faser/checkpoints_final/finetune_flash_jepa_base_d2v16_qkn_ld085
```

The pretraining curves of the three seeds, with seed 8 without QK-norm for
reference: `results/flash_audit_20261026/pretraining_curves_qkn.png`
(`plot_pretraining_curves_qkn.py`).

## Design

**Tokens.**  Every occupied FASERCal patch (`12 x 12 x 10` voxels, `4 x 4 x 20`
grid) and every occupied AHCAL patch (`6 x 6 x 5`, `3 x 3 x 8` grid) is one
token, plus one ECAL token (`Linear(25)`) and one muon-spectrometer token per
event.  The muon token is the mean of per-track MLP embeddings, plus a
track-count embedding and a has-tracks embedding.  Each token gets a fixed 3D
sin-cos position (FASERCal on its global grid, so the module index is implicit)
and a learned detector-type embedding.  Events average 80 FASERCal and 34 AHCAL
tokens (p99 250 and 72), so a 1,024-event batch is about 120k tokens.

**Patch embedding without spconv.**  `SparsePatchEmbed` computes the same
function as the spconv models' sparse convolution with kernel = stride, using
an occupancy table, `index_add_` and one dense matrix product per step.  That
includes the two-step embedding of 1,440-voxel FASERCal patches (`4 x 4 x 2`
cells, then LayerNorm + GELU, then `3 x 3 x 5`).  Outputs exist only at
occupied cells, coincident voxels are summed, and out-of-grid voxels are
dropped, all as in spconv.  A test copies spconv weights into it and matches
spconv's output to float32 precision.  On 32 real augmented events the
relative difference is about 1e-7.

How it is computed (2026-11-04; the parameters, and therefore all
checkpoints, are unchanged):
- **Index geometry.**  Every voxel's cell gets a key ordered by (event,
  patch, cell within the patch).  An occupancy table over all possible keys
  and its cumulative sum rank the occupied cells in the order `torch.unique`
  gave, without sorting the ~11M voxels of a 2,048-event batch.  Out-of-grid
  voxels go to an extra row that is dropped.  The geometry depends only on
  the coordinates: `FlashJEPA.forward` computes it once
  (`FlashEncoder.patch_geometry`) for both encoders.
- **Compiled dense steps.**  On GPUs the voxel keys and everything after the
  geometry (`_embed_one_step`, `_embed_two_step`) run through
  `torch.compile(dynamic=True)`.  Eager PyTorch's LayerNorm is slow at width
  96 over ~2.1M cells, and the fp32 round trips of autocast added copies; the
  compiled code fuses LayerNorm, GELU and the casts.  The weights are
  arguments, so all encoders share the compiled code.  Each function compiles
  at most four times (training, target encoder, validation, fp32 inference);
  batch-size changes do not recompile.  The first step takes 11-21 s longer.
  `FLASH_COMPILE_PATCH_EMBED=0` runs the same code eagerly.  CPU tensors
  always run eagerly.

Two AHCAL details differ from the old pipeline:
- About 1.6% of events have AHCAL hits at x or y = 18, outside the
  18-wide grid.  spconv silently drops them, and so does this embedding.
- The spconv collate inserts a dummy origin voxel for events without AHCAL
  hits, and the spconv models then mask the AHCAL tokens of any event with
  fewer than 2 AHCAL voxels.  The new collate has no dummy and keeps every real
  AHCAL patch.  There is no voxel-count limit: spconv's
backward caps a micro-batch at ~11.2M FASERCal voxels, which is why
2,048-event batches failed before.

**Encoder.**  A plain pre-norm ViT over all tokens of an event: width 384,
6 heads of 64, MLP ratio 4.  `base` has depth 16 (30.3M parameters, against
31.9M for the published `vit_base`); `tiny` has depth 6.  The events of a
batch are packed into one sequence with no padding, and attention is
restricted to each event through `cu_seqlens`.  Stochastic depth drops whole
events.  Weights use the spconv models' initialisation (xavier-uniform
linear layers); see the collapse check below for why not I-JEPA's.  In fp32, or without `varlen_attn`, an exact padded
`scaled_dot_product_attention` is used instead; evaluation runs this path in
fp32.

**Masking (I-JEPA multi-block, adapted to sparse 3D events).**  Per event and
per detector (FASERCal, AHCAL), each of 4 target blocks takes the
`ceil(s * n)` occupied patches nearest to a random occupied patch, with
`s ~ U(0.15, 0.20)` of that event's `n` occupied patches.  Distances use a
random per-axis stretch in `[0.5, 2]`, so blocks range from long segments
along the beam axis to compact transverse blobs.  The union of the blocks is
the target set, about 52% of FASERCal and 57% of AHCAL tokens.  Two
safeguards: no block takes all patches of a detector, and an event whose
blocks cover everything gets its least-covered patch back, so at least one
patch per detector stays visible.  ECAL and the muon token are targets with
probability 0.25 each.  Unlike module-wise random masking, the targets are
contiguous pieces of the actual tracks and showers, so they cannot be
interpolated from immediately adjacent visible patches.

**JEPA objective.**
- The context encoder sees only the visible tokens; they are dropped, not
  masked.
- The EMA target encoder (0.996 to 1.0 on a cosine schedule, updated after
  each optimiser step) sees all tokens, in eval mode.
- The targets are its final token representations, layer-normalised as in
  I-JEPA.  Opt-in alternative: `TARGET_LAYERS=K` (`--flash_target_layers`,
  default 1) uses data2vec's targets instead.  Each of the last K blocks'
  MLP outputs, taken before the residual addition, is layer-normalised; the
  K are averaged and the average is layer-normalised again.  See "data2vec
  targets" below.
- The predictor is a 6-block, 256-wide transformer over the context tokens
  plus one mask token per target. Each mask token carries that target's
  position and type, in the predictor's own sin-cos and type embeddings.
- The predictor regresses every target token with a Smooth-L1 loss averaged
  over tokens.

The predictor sees only context content, so predictions are invariant to the
deposits inside masked patches (tested).

**Fine-tuning.**  `FlashViT` is the encoder followed by a per-event mean pool
over all output tokens and the same task heads as `SparseViT`: the same
LayerNorm and Dropout, cylindrical momentum heads, and head initialisation.
`FlashViTFineTuner` subclasses the published `ViTFineTuner` and changes only
the batch format.  Losses, Kendall weighting, layer-wise LR decay, EMA, the
schedule and the checkpoint monitors are the same code.  The encoder is loaded
from the JEPA target branch with `load_mae_encoder(require_full_encoder=True)`,
which gives 100% coverage.  Pretraining and fine-tuning checkpoints store
`flash_encoder_config`; fine-tuning refuses a mismatched architecture, and
evaluation rebuilds the exact encoder from it.

## Differences from the patch-level JEPA

| | Patch-level JEPA (`pretrain.sh`) | Flash JEPA (`pretrain_flash.sh`) |
|---|---|---|
| Patching | spconv sparse conv | identical function, scatter + GEMM |
| Encoder | module-local ViT, AHCAL ViT, 22-latent Perceiver | one global ViT over all tokens |
| Attention | padded, masked SDPA | packed varlen FlashAttention-2 |
| Downstream representation | mean of 22 latents | mean of all tokens |
| Masking | 75% random patches within each module; ECAL/muon dropped with p = 0.375 | 4 nearest-neighbour 3D blocks per detector (~52%); ECAL/muon p = 0.25 |
| Predictor queries | attend only to the 22 context latents | self-attention with all context tokens |
| Targets | layer-normed stage-2 tokens | layer-normed final tokens |
| Loss normalisation | LayerNorm on both sides | LayerNorm on targets only (I-JEPA) |
| Encoder parameters (`base`) | 31.9M | 30.3M |

## Changes to shared code (behaviour-preserving)

- `model/common.py` holds the timm-free helpers moved out of `model/utils.py`
  (sin-cos embeddings, `choose_k1_k2`, `CylindricalHeadNormalized`), which
  `model/utils.py` re-exports.
- `model/__init__.py`, `utils/funcs.py`, `dataset/dataset.py` and
  `model/lightning_model_finetune.py` import spconv, timm, webdataset and
  torch_ema only when available.  `utils/ema.py` is a drop-in copy of
  torch_ema 0.3's update rule and state format, used only without torch_ema.
- `utils.funcs.collate_targets` is the target half of `collate`, shared with
  `collate_tokens`.  `split_dataset` takes an optional `collate_fn`.
- `checkpoint_monitors` moved from `train/finetune.py` to `utils/funcs.py`.
- `utils/funcs.py` also gains `split_fingerprint`, `require_split_fingerprint`
  and `share_file_lists`.
- `dataset/slim_shards.py` and `train/split_fingerprint.py` are new tools.
- The flash pretrainer logs per-step training metrics without a cross-rank
  all-reduce; the epoch means are still reduced.
- `dataset/dataset.py` decompresses the truth arrays only for stage-1 labels
  (see Throughput).  Outputs are bit-identical.
- `utils/lr_decay.py` puts `token_type_embed` in layer 0.
- `evaluation/` reads `backbone: flash` from the fine-tune hparams, checks it
  against the pretraining checkpoint, and uses `collate_tokens` and
  `FlashViT`.  The per-task best-validation-loss selection is unchanged.

## Throughput

Measured with the real launcher on the local A100 + H100 pair, in bf16, at
the same effective batch of 4,096:

| Per GPU x accumulation, loader workers/GPU | epoch time | events/s | peak GPU memory |
|---|---:|---:|---:|
| 1,024 x 2, 32, before the loader fix | 3:31-3:50 | 4,500-4,900 | 22 GB allocated |
| 1,024 x 2, 24 | 3:15-3:22 | ~5,200 | 33-36 GB reserved |
| **2,048 x 1, 24 (default)** | **3:06-3:11** | **~5,500** | 67-68 GB reserved |

400 epochs take about 22 h, against about 37 h for the patch-level JEPA.
Both batch layouts gave the same validation metrics after 2 epochs.

**Loader fix.**  Training was data-bound.  Decompressing the `.npz` truth
arrays (`true_hits`, `true_index`, `link_weight`, `indptr`) took about half of
the 10.7 ms per event, although only the MAE's stage-1 relational labels use
them.  The dataset now reads them only when `stage1` is set, which also speeds
up fine-tuning.  That gives 5.5 ms per event, with outputs bit-identical to
the committed code on 300 events in train and test mode.  Loader-only
throughput is now 7,300 events/s with 32 workers in total, 8,500 with 48, and
5,700 with 64, so 24 workers per GPU is the optimum on this 128-thread
machine.

- The isolated 1,024-event step takes 195 ms on the H100 and 325 ms on the A100.  DDP
  waits for the A100.
- Fine-tuning (2 GPUs x 512, 32 workers, before the loader fix) runs at 3.7 to 3.8 it/s, i.e. about
  3,800 events/s or 4.5 min per epoch, against 2,850 events/s for the spconv
  fine-tune.

**Faster patch embedding (2026-11-04).**  On one real 2,048-event batch (K =
16 targets, bf16, forward and backward, no optimiser), the patch embeddings
took 11% of the step.  Most of that was memory traffic, not matrix products:
the LayerNorm over 2.1M cells x 96 channels, autocast's fp32 copies, two
`torch.unique` sorts of 11M voxels, and the target encoder repeating all
index work.  After the change (see "Patch embedding without spconv"):

| | A100 | H100 |
|---|---:|---:|
| Patch embeddings per JEPA step | 66 → 23 ms | 38 → 14 ms |
| Training step | 624 → 577 ms | 377 → 356 ms (± 5 ms) |
| Peak memory | 40.5 → 39.4 GiB | 40.5 → 39.5 GiB |

DDP waits for the A100, so GPU-bound epochs are ~7% faster.  The committed and
new code give the same outputs from the published checkpoints: a fine-tuned
`FlashViT` (`finetune_flash_jepa_base_d2v16_qkn_ld085`, 256 events) agrees to
6e-7 relative in fp32 (3e-7 between two runs of the old code).  In bf16 each
output is as close to fp32 as before.  The pretrained target encoder's final
tokens agree to 4e-6.

### Against the published MAE (2026-10-27)

**dlnu, back to back, same data files** (`results/flash_audit_20261026/throughput_mae_vs_flash.sh`;
logs, GPU use and load average in `/scratch4/salonso/faser/throughput_20261027`):

```bash
bash results/flash_audit_20261026/throughput_mae_vs_flash.sh
```

- The published MAE recipe: `mae_vit_base`, effective batch 4,096 as
  2 x 1,024 x accumulation 2, relational pass probability 0, 32 workers per
  GPU.
- The flash JEPA with K = 16 targets, at its defaults.
- Both use a shortened LR schedule, which doesn't change the cost per step.
- The MAE was stopped after one full epoch: at 44 min per epoch, 3 epochs
  would have taken 2.2 h, and its step time was constant from step 225 to 500.

| | train / epoch | val | events/s | GPUs busy | 400 epochs |
|---|---:|---:|---:|---:|---:|
| MAE (published recipe) | 42.9 min (5.06 s/step) | 56 s | 405 | 95-96% | ~290 h (12 days) |
| flash JEPA, K = 16 | 3:51-3:53 | 11 s | 4,500 | 68-73% | ~27 h |

That is **11x less time per epoch**, for the same number of epochs and
steps.
- **The MAE is limited by GPU compute, not by loading.**  Its GPUs were
  nearly always busy.  The patch-level JEPA, with the same spconv encoder,
  ran at 3,650 events/s on this pair.  So most of the MAE's cost is its
  objective (per-voxel decoder and distance-transform losses, not
  profiled), not spconv.
- **The flash run was partly limited by loading.**  Another user's jobs kept
  the load average at 74 (MAE) and 99 (flash) on the 128 threads, and that
  slows the flash run more.  Unloaded K = 1 epochs took 3:06-3:11.  11x is
  therefore a lower bound.

**Clariden, from the run logs** (`results/flash_audit_20261026/published_mae_walltime.{py,out}`
reads the published MAE's TensorBoard files):

```bash
python results/flash_audit_20261026/published_mae_walltime.py /scratch/salonso/sparse-nns/faser/deep_learning/faserDLTrans/logs_final/tb_logs/pretrain_7.0_lat_norel_v2_train > results/flash_audit_20261026/published_mae_walltime.out
```

| | nodes | 400 epochs | per epoch |
|---|---:|---:|---:|
| MAE `pretrain_7.0_lat_norel_v2` (Feb 2026, 8 job segments) | 2 | 73.0 h | ~11 min |
| flash JEPA K = 16 (job 3760078) | 2 | 4.3 h | ~39 s |

That is 17x, but it also includes the data pipeline.  The flash run reads
slim shards staged into each node's RAM; the MAE read the full `.npz` files
from Lustre, with a software stack eight months older.  The dlnu
comparison above is the like-for-like one.

## Collapse check

The collapse monitors are logged every step:
- `jepa/target_std`: per-dimension spread of the layer-normed targets across
  tokens;
- `jepa/target_event_std` and `jepa/context_event_std`: spread across events
  of the mean-pooled representation;
- `jepa/cosine`.

A constant representation drives the spreads to 0 and the cosine to 1.  The
reference values are the patch-level JEPA run's: target spread 0.66 at
initialisation, then 0.83 and cross-event spread 0.54 after 2,500 steps.

| 3-epoch test, 1-epoch warm-up to the full LR 1.6e-3 | target spread | target cross-event | context cross-event | cosine |
|---|---:|---:|---:|---:|
| xavier init (used), validation after epoch 3 | 0.52 | 0.44 | 0.58 | 0.984 |
| I-JEPA init (trunc-normal 0.02, output projections / sqrt(2 l)) | 0.27 | 0.15 | 0.17 | 0.991 |

**Xavier init.**  The spread dips from 0.28 to 0.13 during the fast warm-up,
then grows steadily once the representations specialise.

**I-JEPA init.**  It starts more diverse (0.63, because it avoids the depth-wise
token similarity of xavier at initialisation) but stalls at a much less diverse
representation.

**No warm-up.**  Starting at the full LR collapses within 50 steps: spread 0.06,
cosine 0.997.  Keep the warm-up.  The production schedule's 40 warm-up epochs
(10,160 steps) are 40 times gentler than the test above.

Watch `jepa/target_event_std/val` in the real run.  It should stay well above
0.1 and grow during the warm-up.

With varying batch shapes, `varlen_attn` on an H100 picks cuDNN attention by
default.  cuDNN rebuilds its graph for every new shape: about 125 ms per
attention call, against 3.3 ms for FlashAttention-2.  The encoder passes
`window_size=(max_seqlen, max_seqlen)`, which is full attention (FlashAttention
treats windows at least as long as the sequence as unbounded) and selects the
FlashAttention-2 kernels for the forward and backward passes on every GPU.
Outputs and gradients match exact fp32 attention (tested).

## Commands

Activate the environment first:

```bash
source /scratch/salonso/anaconda3/etc/profile.d/conda.sh && conda activate neutrino-jepa
```

Dry run (one training and one validation batch through 2-GPU DDP; writes
nothing to `logs_final` or `checkpoints_final`):

```bash
FAST_DEV_RUN=1 SAVE_DIR=/scratch4/salonso/faser/flash_dryrun CHECKPOINT_PATH=/scratch4/salonso/faser/flash_dryrun/ckpt DATASET_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*' METADATA_PATH=/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz/metadata_stats.pkl bash pretrain_flash.sh
```

Pretraining, controlled against the published MAE and the patch-level JEPA:
`base`, effective batch 4,096 (2 GPUs x 2,048, no accumulation), `blr=1e-4`
(lr 1.6e-3), betas (0.9, 0.95), weight decay 0.05, 400 epochs (40 warm-up),
last.ckpt only:

```bash
NAME=pretrain_flash_jepa_base DATASET_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*' METADATA_PATH=/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz/metadata_stats.pkl bash pretrain_flash.sh
```

Fine-tuning with the published recipe (20 epochs, `blr=5e-4`, layer decay
0.75, drop path 0.2, global batch 1,024), from the EMA target encoder:

```bash
NAME=finetune_flash_jepa_base GPUS='0 1' BATCH_SIZE=512 NUM_WORKERS=24 CHECKPOINT_PATH=/scratch4/salonso/faser/checkpoints_final SAVE_TOP_K=1 SAVE_LAST=0 CHECKPOINT_MONITORS=evaluated LOAD_CHECKPOINT=checkpoints_final/pretrain_flash_jepa_base/loss_total_val/last.ckpt DATASET_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*' METADATA_PATH=/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz/metadata_stats.pkl bash finetune_flash.sh
```

The depth-16 encoder has 18 layer-decay levels, against 14 in `vit_base`.  At
0.75 its patch embedding therefore gets 0.75^18 = 0.6% of the head LR, versus
1.8% in the published model.  `LAYER_DECAY=0.8` (0.8^18 = 1.8%) matches the
published bottom-layer LR if the identical recipe turns out to under-train the
early layers.

Test-set CSV (same split, per-task best-validation-loss checkpoints, same
schema as the paper results):

```bash
python -m evaluation.create_csv_v7_ft --checkpoint_dir /scratch4/salonso/faser/checkpoints_final/finetune_flash_jepa_base
```

## Clariden (CSCS)

**Environment.**
- Job scripts are in `clariden/` and run from `/users/salonso/faserDL`.
- Each training step runs in the `pytorch/v2.9.1:v2` uenv (default view)
  with the venv `/users/salonso/venvs/neutrino-jepa-pt2.9`.
- The uenv's NCCL/libfabric settings tune multi-node communication for
  Slingshot.
- The uenv is loaded per job step (`srun --uenv=...`).  Loading its view in
  the batch shell breaks `srun` itself, because its OpenSSL shadows the
  system library.

**Flash attention with PyTorch 2.9.**  PyTorch 2.9.1 has no
`torch.nn.attention.varlen`.  `model/flash_vit.py` then calls the same
FlashAttention-2 kernels directly (`FlashVarlenAttention`, around
`aten::_flash_attention_forward` and `aten::_flash_attention_backward`).
- Accuracy and speed equal `varlen_attn`'s: identical errors against exact
  attention, forward and backward, and 6.6 against 6.7-7.2 ms per call.
- Jagged nested tensors reach the same kernels but took 3-5x longer per call,
  so they are not used.
- With `varlen_attn` available (dlnu), the code is unchanged.

**Identical splits on both machines.**
- Seed 7 gives bit-identical `random_split` indices under PyTorch 2.5.1
  (x86), 2.14 (x86) and 2.9.1 (aarch64).
- The split therefore depends only on the ordered list of event files, and
  the metadata pickles are byte-identical (SHA-256 `cc6437c3...`).
- `python -m train.split_fingerprint --dataset_path '<path>/events_v7.0*'`
  prints the per-directory counts and a fingerprint of the
  (train, val, test) event names.  On dlnu it is `02078f72b5ec...` for
  1,226,375 events: 11,281 in `500`, 1,106,777 in `501` and 108,317 in `tau`.
- The Clariden jobs pass it as `EXPECTED_SPLIT_SHA256`, so every rank refuses
  to start on an incomplete or different copy.  Mid-copy, two ranks listed
  different file sets, which this check catches.
- A Clariden evaluation reproduces dlnu's test-set and full-inventory
  fingerprints exactly.

**Data: slim shards staged to node memory.**
- **Problem.** Reading the 546 GB of `.npz` files from Lustre limited 2
  nodes to ~11k events/s.  Only 10% of each file is needed without the MAE's
  relational labels: 54 of 541 KB; `true_hits` alone is 348 KB.
- **Shards.** `python -m dataset.slim_shards` writes those arrays
  byte-for-byte, under the same `<directory>/<file>` names, into 64 tar shards
  (63 GB).  Built on dlnu and copied, identical by SHA-256, to
  `/iopsstor/scratch/cscs/salonso/faser_slim_v7.0` (with `manifest.json`).
  Loading is bit-identical to the full files (tested), and the split
  fingerprint is unchanged.
- **Staging.** Each job extracts the shards into `/dev/shm` on every node
  (`clariden/stage_slim_dataset.sh`): 107 GB with 64 KB pages, in 12 s.
  Training then reads RAM.  `SLIM_SHARDS=""` reads `DATASET_PATH` instead.
- **Scope.** The shards cannot serve the MAE (stage 1 needs the truth
  arrays), and the full copy in `faser_new` is not needed for the flash jobs.

**Memory on GH200 (64 KB pages).**  Forked loader workers copied ~1 GB each
of the parent's memory on first touch, which ran nodes out of memory.  Two
fixes:
- the event paths are numpy arrays (`utils.funcs.share_file_lists`);
- `gc.freeze()` runs before `trainer.fit`, so Python's collector does not
  write to inherited objects.

With both, 64 workers per GPU run without OOM; 48 (the default) leaves ample
headroom.  More workers did not speed anything up.

**Scripts.**
- `clariden/pretrain_flash.sbatch`:
  - default 2 nodes x 4 GPUs, effective batch `EFFECTIVE_BATCH=4096` (as on
    dlnu; per-GPU batch = 4096 / (4 x nodes), LR by linear scaling);
  - 48 loader workers per GPU;
  - a snapshot every 25 epochs in `<checkpoint>/<name>/snapshots`;
  - automatic resume from `last.ckpt`, e.g. after the 12 h limit.  The first
    test resumed a run that had finished; `last.ckpt` is written after every
    epoch only since 2026-10-27 (before, with `SAVE_TOP_K=0`, Lightning 2.6
    wrote it only when training ended, so a killed run restarted from
    scratch).  Kill-and-resume test under Tests.
- `clariden/finetune_flash.sbatch`:
  - 1 node, global batch 1,024 (256 per GPU);
  - keeps only the best checkpoint of each of the five evaluated losses.
- `clariden/evaluate_flash.sbatch`: test-set CSV on one GPU.

Logs go to `/capstor/scratch/cscs/salonso/faser/logs/{logs,tb_logs}/<name>`,
checkpoints to `/capstor/scratch/cscs/salonso/faser/checkpoint/<name>`, and
Slurm output to `/capstor/scratch/cscs/salonso/faser/slurm/`.

**Measured throughput** (full data, staged, effective batch 4,096):

| Nodes x GPUs | batch/GPU | train time per epoch | events/s | 400 epochs (with validation) | GPU-hours |
|---|---:|---:|---:|---:|---:|
| 1 x 4 | 1,024 | 55-59 s | ~18k | ~7 h | ~28 |
| **2 x 4 (default)** | 512 | 35-39 s | ~27k | ~5 h | ~40 |
| 4 x 4 | 256 | 28-30 s | ~35k | ~4 h | ~65 |

- Data loading is not the limit (more workers did not help).  Neither is
  per-step metric syncing: training metrics are no longer all-reduced every
  step, with no change in speed.
- The remaining gap to the single-GPU benchmark (5,170 events/s per GPU at
  512) is synchronisation inside the step and across GPUs.  Larger
  effective batches (`EFFECTIVE_BATCH=8192`) would run faster but change the
  recipe used for the dlnu and paper comparisons.

Fine-tuning on 1 node runs at ~10.6 it/s (~11k events/s), about 1:40 per
epoch, so ~40 min for the 20-epoch recipe.

**Commands** (from `/users/salonso/faserDL` on the Clariden login node):

```bash
NAME=pretrain_flash_jepa_base sbatch clariden/pretrain_flash.sbatch
```

```bash
LOAD_CHECKPOINT=/capstor/scratch/cscs/salonso/faser/checkpoint/pretrain_flash_jepa_base/loss_total_val/last.ckpt NAME=finetune_flash_jepa_base sbatch clariden/finetune_flash.sbatch
```

```bash
CHECKPOINT_DIR=/capstor/scratch/cscs/salonso/faser/checkpoint/finetune_flash_jepa_base sbatch clariden/evaluate_flash.sbatch
```

Add `--nodes=4` to the pretraining `sbatch` for ~4 h instead of ~5 h.  For a
shorter schedule, e.g. 100 epochs, set `EPOCHS=100 WARMUP_EPOCHS=10` and a new
`NAME`.  For the data2vec targets, set `TARGET_LAYERS=16` and a new `NAME`
(see "data2vec targets" below).

## Tests

`tests/test_flash_jepa.py` (`python -m unittest tests.test_flash_jepa`).  The
spconv-equivalence and target-collation tests run in the `spconv`
environment; the flash-kernel tests need a GPU.  The full suite
(`python -m unittest discover -s tests -t .`) passes in both environments:
- `spconv`: 69 tests, none skipped;
- `neutrino-jepa`: 51 tests, with the 7 spconv-only tests skipped.

Two CPU tests (2026-10-27) fit a toy module with the pretraining checkpoint
callbacks (`train.pretrain_flash.build_callbacks`).  They check that
`last.ckpt` holds the previous epoch at the start of every epoch, that
resuming from it continues with the next epoch, and that top-k files,
snapshots and `last.ckpt` coexist.  The first test fails with the previous
callbacks.  Run with the GPUs hidden (dlnu's were busy):
- `neutrino-jepa` (Lightning 2.6.6): 53 tests OK, 9 skipped;
- `spconv` (Lightning 2.5.0): the new tests pass; its 11 CUDA tests were
  not rerun (no GPU), and the change does not touch them.

On Clariden (2026-10-27), `tests.test_flash_jepa` on a GH200 debug node
passes: 28 tests, 5 spconv-only skipped.  Kill-and-resume with the real
script, 2 debug nodes and the full staged data, from `/users/salonso/faserDL`
(the test directory was removed afterwards):

```bash
T=/iopsstor/scratch/cscs/salonso/faser_flash_tests NAME=test_resume_lastckpt SAVE_DIR=$T/logs CHECKPOINT_PATH=$T/ckpt EPOCHS=4 WARMUP_EPOCHS=1 SAVE_EVERY_N_EPOCHS=0 sbatch -p debug --time=00:15:00 --output=$T/%x_%j.out clariden/pretrain_flash.sbatch
```

- Job 3763782 wrote `last.ckpt` after epochs 0 and 1, 38 s apart (the epoch
  time before the change).  It was cancelled (`scancel`) during epoch 2.
- Job 3763860, the same command, resumed automatically from `last.ckpt`,
  ran epochs 2 and 3 only, and stopped at `max_epochs=4`.  It left a single
  `last.ckpt` and logged no warnings.

The faster patch embedding (2026-11-04) adds four tests: the new code
against the original `torch.unique` implementation (kept in the test file
as the reference) on synthetic voxels (CPU and GPU, one and two input
channels, an empty event, coincident and out-of-grid voxels, fp32 and bf16,
outputs and gradients) and on 32 real events, a detector without voxels, and
one shared geometry per JEPA step.  With both dlnu GPUs:
- `neutrino-jepa`: 55 passed, 7 spconv-only skipped (pytest);
- `spconv`: 80 tests OK (`unittest discover`), including the spconv
  equivalence tests through the compiled code (torch 2.5.1).

Two pre-existing test-isolation problems were fixed:
- `tests/test_jepa_resume.py` left `torch.use_deterministic_algorithms(True)`
  switched on, which broke later cuBLAS tests once CUDA had been initialised
  earlier in the run;
- the spconv-only test modules now skip instead of failing to import.

End-to-end check, all in `/scratch4/salonso/faser/flash_dryrun`, nothing in
`logs_final` or `checkpoints_final`:
1. A 3-epoch pretraining.
2. One fine-tuning epoch from its last.ckpt.  The target encoder loaded with
   100% coverage.
3. `evaluation.create_csv_v7_ft --max_batches 4`, which validated the
   provenance, selected the per-task best checkpoints and wrote a 512-row CSV
   in the standard 45-column schema.

## Completed pretraining (`pretrain_flash_jepa_base`, 2026-10-25/26)

The run used the defaults: 2 GPUs x 2,048, 24 workers, 400 epochs.  It took
23.4 h in one uninterrupted session.  The audit script and its output are in
`results/flash_audit_20261026/`.

**Checkpoint.**
- `last.ckpt` is at epoch 400, step 101,600 (254 x 400), with all tensors
  finite.
- The final LR is 1.6e-5 (1% of peak) and the EMA decay reached 1.0.
- The context and target encoders differ by 5.5% (relative norm).

**Curves (validation).**
- The loss rose from 0.012 (epoch 2) to 0.11 at the end of warm-up as the
  targets became informative, then fell to 0.061.  Cosine ended at 0.93.
- Token-level target spread peaked at 0.98 and ended at 0.63.
- The spread of the event-averaged representation peaked at 0.85 (epoch 9)
  and ended at 0.22.  The probes below show that this decline is not collapse.
- Mask fractions were as designed: FASERCal 54%, AHCAL 56%, ECAL and muon 25%.

**Predictor (fixed validation masks).**
- R^2 against the targets: FASERCal 0.80, AHCAL 0.85, ECAL 0.82, muon 0.63.
- Patch position alone explains 41% of the FASERCal target variance.  The
  predictor explains 67% of the remainder, against 22% for the patch-level
  JEPA.

**Frozen linear probes.**  Same protocol and events as the patch-JEPA audit:
mean-pooled features, 8,192 training-split / 4,096 validation-split events,
logistic (C = 0.1) or ridge (alpha = 10).

| Encoder | flavour AUC | charm AUC | vis R^2 | vertex R^2 | mean cosine | participation ratio |
|---|---:|---:|---:|---:|---:|---:|
| random-init flash base | 0.681 | 0.566 | 0.100 | 0.854 | 0.97 | 6.9 |
| **flash JEPA target** | **0.748** | **0.615** | **0.233** | **0.987** | 0.52 | 8.9 |
| flash JEPA context | 0.749 | 0.613 | 0.234 | 0.987 | 0.55 | 8.4 |
| patch JEPA target | 0.708 | 0.589 | 0.150 | 0.978 | 0.55 | 12.1 |
| published MAE | 0.783 | 0.583 | 0.247 | 0.987 | 0.70 | 11.7 |

**The falling event-level spread is not a collapse.**  `jepa/target_event_std`
peaked at 0.85 early on, but early, high values come from one dominant
direction, not information.  The 3-epoch stress-test encoder (a complete
short schedule) has:
- pooled features of participation ratio 1.4, with 82% of the variance in one
  direction;
- probes at the random-init level: flavour AUC 0.678, charm 0.575, vis R^2
  0.131, vertex R^2 0.879;
- tokens of an event that are nearly identical: only 16% of the token variance
  is within events, with effective rank 1.7.

After 400 epochs:
- 83% of the token variance is within events;
- the effective rank is 19.7 for tokens and 9.3 for event means.

The tokens became specific to their local content, which is what a patch
prediction objective rewards.  Mean pooling averages that within-event
variance out, so the absolute spread of the pooled vector falls while its
rank and probe accuracy rise (`results/flash_audit_20261026/variance_split.out`,
`audit_flash_3epoch.out`).

**Transfer.**  `FlashViT` loads the target encoder with 100% coverage, all
values equal, and pooled features matching to 8e-6.

**Fine-tuning speed.**  A one-epoch fine-tune from this checkpoint
(2 x 512, 24 workers) ran at 3.06 it/s, or 5.5 min per epoch.  At the time,
another user's CPU jobs held the load average at ~83.  The isolated GPU step
(182 ms per 512 events on the A100) caps the pair at ~5,600 events/s, which
the fixed loader exceeds on an idle machine (~8,500 events/s).

## Fine-tuning results (20 epochs, published recipe)

Run `finetune_flash_jepa_base`:
- fine-tuned from the EMA target encoder (coverage 1.0), 2 GPUs x 512;
- best epochs: flavour 19, charm 17, vis 19, jet 17, vertex 18;
- test CSV `results/results_v7_test_finetune_flash_jepa_base.csv`.

The comparison uses the same 122,639 test events and the metrics of the
patch-JEPA audit.  The script and its output are
`results/flash_audit_20261026/metrics_flash_ft.{py,out}`.  Intervals are paired
event-bootstrap 95% intervals (200 resamples) and cover test-set sampling
only: every run is a single seed.

| Metric | MAE (pub) | patch JEPA | flash JEPA | flash - MAE [95%] | flash - patch [95%] |
|---|---:|---:|---:|---:|---:|
| AUC nue | 0.9756 | 0.9714 | 0.9764 | +0.0007 [+0.0001, +0.0013] | +0.0049 [+0.0043, +0.0055] |
| AUC numu | 0.9473 | 0.9421 | 0.9451 | -0.0022 [-0.0029, -0.0016] | +0.0031 [+0.0025, +0.0038] |
| AUC nutau->e | 0.9103 | 0.9051 | 0.9154 | +0.0051 [+0.0020, +0.0090] | +0.0103 [+0.0066, +0.0138] |
| AUC nutau->mu | 0.8364 | 0.8403 | 0.8404 | +0.0040 [-0.0013, +0.0088] | +0.0000 [-0.0051, +0.0055] |
| AUC nutau->had | 0.9386 | 0.9321 | 0.9392 | +0.0005 [-0.0009, +0.0020] | +0.0070 [+0.0056, +0.0087] |
| AUC NC | 0.9388 | 0.9304 | 0.9361 | -0.0027 [-0.0036, -0.0017] | +0.0058 [+0.0049, +0.0068] |
| Flavour accuracy | 0.8153 | 0.8034 | 0.8115 | -0.0038 [-0.0053, -0.0023] | +0.0082 [+0.0065, +0.0096] |
| AUC charm->e | 0.8042 | 0.7925 | 0.8088 | +0.0046 [-0.0105, +0.0176] | +0.0163 [+0.0031, +0.0268] |
| AUC charm->mu | 0.8844 | 0.8855 | 0.8832 | -0.0013 [-0.0120, +0.0101] | -0.0023 [-0.0140, +0.0112] |
| AUC charm->had | 0.8691 | 0.8531 | 0.8673 | -0.0019 [-0.0044, +0.0011] | +0.0141 [+0.0115, +0.0171] |
| sigma_MAD E_vis (lower better) | 0.2144 | 0.2263 | 0.2291 | +0.0148 [+0.0129, +0.0166] | +0.0028 [+0.0008, +0.0047] |
| sigma_MAD pT_miss | 0.5573 | 0.6107 | 0.6205 | +0.0632 [+0.0594, +0.0680] | +0.0098 [+0.0057, +0.0140] |
| sigma_MAD \|p_jet\| | 0.1637 | 0.1750 | 0.1610 | -0.0027 [-0.0042, -0.0010] | -0.0140 [-0.0156, -0.0125] |
| sigma_MAD \|p_lep\| (CC) | 0.3857 | 0.3968 | 0.3893 | +0.0037 [-0.0002, +0.0076] | -0.0074 [-0.0112, -0.0038] |
| Vertex median (mm) | 16.88 | 16.58 | 14.86 | -2.02 [-2.14, -1.91] | -1.72 [-1.84, -1.62] |

The flash JEPA improves on the patch JEPA in 12 of 15 metrics; the exceptions
are E_vis and pT_miss (worse) and charm->mu (equal).  Against the published
MAE it is:
- statistically equal on 6 metrics: nutau->mu, nutau->had, all three charm
  AUCs and the lepton momentum;
- better on 4: nue, nutau->e, jet momentum and vertex (-2.0 mm, 12%);
- slightly worse on 5: numu, NC, flavour accuracy (-0.4 pp), and clearly
  worse on E_vis and pT_miss.

It recovers 95-116% of the scratch-to-MAE gap on the classification
metrics, 76-104% on the momentum resolutions and 139% on the vertex.

## data2vec targets (opt-in, `TARGET_LAYERS`)

**Why.**  Both JEPAs trail the published MAE on E_vis and pT_miss, which
depend on how much energy each patch holds.  The MAE reconstructs the
deposits.  The I-JEPA target is the final output alone, which can discard
that low-level content.  data2vec's targets average several blocks instead.

**Definition.**  `TARGET_LAYERS=K` (`--flash_target_layers K`).  K = 1, the
default, is the I-JEPA target used so far: same code path, bit-identical
targets (tested).  For K > 1:
1. take the MLP outputs of the last K target-encoder blocks, before the
   residual addition (`Block.forward_with_mlp`,
   `FlashEncoder.forward_mlp_outputs`);
2. layer-normalise each;
3. average them and layer-normalise the average.

Masking, predictor, loss, EMA, schedule and checkpoints are unchanged, and
the fine-tuning and evaluation code does not depend on K.

**Probe of the existing encoder.**  The audit's frozen probes (same events
and protocol) on the mean-pooled targets that the 400-epoch target encoder
gives for each K (`results/flash_audit_20261026/target_layers_check.{py,out}`):

| Pooled target | flavour AUC | charm AUC | vis R^2 | vertex R^2 |
|---|---:|---:|---:|---:|
| K = 1 (used so far) | 0.748 | 0.614 | 0.233 | 0.987 |
| K = 4 | 0.711 | 0.607 | 0.220 | 0.978 |
| K = 8 | 0.757 | 0.626 | 0.249 | 0.981 |
| K = 16 (all blocks) | **0.808** | **0.626** | **0.295** | **0.988** |
| for reference: published MAE encoder output | 0.783 | 0.583 | 0.247 | 0.987 |

The extra information is in the earlier blocks.  The last 4 alone carry less
than the final output; all 16 carry the most, more than the MAE encoder's
features on flavour and vis.  This measures what each target definition
contains in an encoder trained with K = 1.  It does not show what a
K = 16 pretraining will learn, but it makes K = 16 the setting to try.

**Cost.**  One training step on 2,048 real events (H100 NVL, bf16):

| K | ms per step | peak memory |
|---:|---:|---:|
| 1 | 392 | 46.1 GB |
| 8 | 397 | 46.1 GB |
| 16 | 401 | 46.1 GB |

So about 2% slower: a 400-epoch run stays ~5 h on 2 Clariden nodes.

**Checks.**
- Four unit tests: K = 1 bit-identical to the final-output target; K > 1
  against a hand-computed average, with gradients only in the context
  encoder and predictor; invalid K rejected; flag passed by the script.
- `pretrain_flash.sh` dry run with `TARGET_LAYERS=8` on dlnu: the split
  fingerprint matched, and one training and one validation step ran.  The
  initial loss is 0.66, against ~0.01 for K = 1 at initialisation: the
  averaged targets differ between tokens from the start.
- The flash tests pass on Clariden.

**Run** (Clariden), then fine-tune with the depth-matched layer decay:

```bash
NAME=pretrain_flash_jepa_base_d2v16 TARGET_LAYERS=16 sbatch clariden/pretrain_flash.sbatch
```

```bash
LOAD_CHECKPOINT=/capstor/scratch/cscs/salonso/faser/checkpoint/pretrain_flash_jepa_base_d2v16/loss_total_val/last.ckpt NAME=finetune_flash_jepa_base_d2v16_ld080 LAYER_DECAY=0.80 sbatch clariden/finetune_flash.sbatch
```

On dlnu (both GPUs, ~1 day, like the K = 1 run), with the checkpoint on
`/scratch4`: a full checkpoint (~0.54 GB) is written every 25 epochs, ~9 GB
in all, which does not fit on `/scratch`.  If the run is interrupted, add
`RESUME_CHECKPOINT=/scratch4/salonso/faser/checkpoints_final/pretrain_flash_jepa_base_d2v16/loss_total_val/last.ckpt`
to the same command.  (The dlnu run started on 2026-10-26 predates the
per-epoch `last.ckpt`: it would resume from its newest
`snapshots/epoch=<NNN>.ckpt`.)

```bash
NAME=pretrain_flash_jepa_base_d2v16 TARGET_LAYERS=16 CHECKPOINT_PATH=/scratch4/salonso/faser/checkpoints_final EXPECTED_SPLIT_SHA256=02078f72b5ec271f81e14f3a24b769e665a9a928908b141bd1fb2134c730fd7c DATASET_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*' METADATA_PATH=/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz/metadata_stats.pkl nohup bash pretrain_flash.sh > /scratch4/salonso/faser/pretrain_flash_jepa_base_d2v16.log 2>&1 &
```

Watch `jepa/target_event_std/val` and the loss through the warm-up as for
K = 1.

## Layer-decay fine-tunes (Clariden, 2026-10-27)

The 400-epoch K = 1 checkpoint (`pretrain_flash_jepa_base_dlnu`, identical
to dlnu's) fine-tuned with the published recipe except `LAYER_DECAY` 0.80 and
0.85 (jobs 3760074/3760076, evaluated by 3760075/3760077).
`results/flash_audit_20261026/metrics_flash_ld.{py,out}` scores them on the
same 122,639 test events.

**Validation (chosen without the test set).**  Best per-task validation
losses fall with the layer decay, except vertex:

| layer decay | flavour | charm | vis | jet | vertex |
|---|---:|---:|---:|---:|---:|
| 0.75 (dlnu) | 0.5146 | 0.2208 | 1.1818 | 0.2672 | 0.005517 |
| 0.80 | 0.5117 | 0.2184 | 1.1691 | 0.2647 | 0.005706 |
| 0.85 | 0.5085 | 0.2166 | 1.1608 | 0.2631 | 0.005640 |

**Test, 0.85 minus published MAE** (paired event bootstrap, 95%):
- Better: AUC nue +0.0019, nutau->e +0.0079, nutau->had +0.0017,
  charm->e +0.0175, charm->mu +0.0087; sMAD |p_jet| -0.0094; vertex
  -2.19 mm.
- Tied (interval includes 0): AUC numu -0.0003, nutau->mu, NC -0.0007,
  charm->had; flavour accuracy 0.8152 against 0.8153.
- Worse: sMAD E_vis +0.0079 (+0.0148 at 0.75), pT_miss +0.0209 (+0.0632 at
  0.75), |p_lep| CC +0.0073.

0.80 lies between 0.75 and 0.85 on almost every metric.  Caveats: one
fine-tuning seed per setting; the layer decay was tuned for the JEPA and not
for the MAE (its published 0.75 recipe); 0.75 was fine-tuned on dlnu, 0.80
and 0.85 on Clariden.

## data2vec K = 16 pretraining (Clariden, 2026-10-27)

Job 3760078, 2 nodes, 400 epochs in 4 h 18 min; checkpoint
`/capstor/scratch/cscs/salonso/faser/checkpoint/pretrain_flash_jepa_base_d2v16`
(`loss_total_val/last.ckpt` and `snapshots/epoch=024..399.ckpt`).  Stable up
to epoch ~310 (validation loss 0.085, `jepa/context_event_std` 0.40,
`jepa/target_std` 0.93).  Between epochs 316 and 340 the validation loss
spiked to 0.132 at epoch 328, `context_event_std` fell to 0.08 and
`target_std` to 0.53.  The loss then settled lower (0.035-0.053), which is
the signature of easier, less informative targets: a partial collapse.  The
last snapshot before it is `epoch=299.ckpt`.  The K = 1 run showed no such
event.  Fine-tuned from epochs 299 and 400 below ("K = 16 and layer-decay
0.90 fine-tunes").

**Block-by-block check** (`results/flash_audit_20261026/collapse_check.{py,out}`,
Clariden debug job 3764034).  The same 2,048 validation events go through
snapshots 199-399 and the K = 1 checkpoint.  For each block, the
layer-normalised MLP output (what the K = 16 target averages) is split into:
- `same`: the part identical for every token;
- position-only: the part fixed by the patch position alone;
- `event`: the rest, which depends on the event.

Teacher (target encoder), `event` fraction:

| | K=1 | e299 | e324 | e349 | e399 |
|---|---:|---:|---:|---:|---:|
| block 4 | 0.60 | 0.67 | 0.65 | 0.49 | 0.49 |
| block 8 | 0.61 | 0.88 | 0.87 | 0.47 | 0.52 |
| block 12 | 0.91 | 0.92 | 0.89 | 0.29 | 0.38 |
| block 16 | 0.26 | 0.60 | 0.59 | 0.05 | 0.01 |
| K = 16 target | 0.56 | 0.66 | 0.62 | 0.22 | 0.23 |
| final output | 0.42 | 0.56 | 0.45 | 0.05 | 0.04 |

- **Not a few quiet blocks, and not the biases.**  The hypothesis was that a
  few blocks would collapse to their constant bias and layer norm would
  amplify them.  But the bias is at most 4% of any block's output and did
  not grow.  Instead, the whole deep half (blocks ~5-16) lost most of its
  event content between epochs 324 and 349.
- **The lost part became one global constant**, not a position code.  In
  blocks 10-15 the `same` fraction rose from below 7% to 52-69%; the
  position-only part stayed at a few percent.
- **Block 16 turned into a loud constant.**  Its MLP output went from 0.20 to
  0.88 of the residual stream's RMS and became 99% the same for every token.
  So the final output, the predictor's input, became ~95% identical
  everywhere.  Event spread of the final output: 0.405 (e299) to 0.12.
- **The student led.**  At epoch 324 the context encoder's final output was
  already down to 0.22 event content against the teacher's 0.45.  By epoch
  399 the student was more collapsed than the teacher (blocks 7-15 at
  0.07-0.21).
- **Why it stayed partial.**  Blocks 1-4, closest to the input, kept 0.28-0.49
  event content, which is the 22% left in the target.
- **Epoch 299 is healthy.**  It has more event content than the K = 1 run in
  most blocks (final output 0.56 against 0.42, spread 0.405 against 0.224),
  so it is the K = 16 checkpoint to fine-tune.  (Fine-tuning later showed
  the collapsed epoch-400 checkpoint to be about as good; see below.)

Because the collapse goes to a single constant, removing each target
channel's mean over the tokens (data2vec's instance normalisation, which it
uses for speech) would take away exactly this shortcut.  Stopping at ~300
epochs or a slower teacher are the alternatives.  The batch-standardised
targets below implement the first.

## K = 16 and layer-decay 0.90 fine-tunes (Clariden, 2026-10-27)

Published recipe, 20 epochs, one seed each:
- K = 16, epoch 299 (`snapshots/epoch=299.ckpt`), `LAYER_DECAY` 0.85: job
  3763674, evaluated by 3763675.
- K = 16, epoch 400 (`loss_total_val/last.ckpt`, after the partial
  collapse), 0.85: job 3765548, evaluated by 3765549.
- K = 1 (`pretrain_flash_jepa_base_dlnu`), 0.90: job 3765551, evaluated by
  3765553.

The first attempts at the last two (3763672, 3763676) were killed by a node
failure at epoch 7.  Their partial outputs are kept as `<name>_nodefail` in
the checkpoint and log directories.
`results/flash_audit_20261026/metrics_flash_d2v.{py,out}` scores all of them
on the same 122,639 test events (paired event bootstrap, 200 resamples):

```bash
python results/flash_audit_20261026/metrics_flash_d2v.py > results/flash_audit_20261026/metrics_flash_d2v.out 2>&1
```

**Validation** (best per-task loss):

| run | flavour | charm | vis | jet | vertex | total |
|---|---:|---:|---:|---:|---:|---:|
| K = 1, 0.85 | 0.5085 | 0.2166 | 1.1608 | 0.2631 | 0.005640 | 1.2524 |
| K = 1, 0.90 | 0.5122 | 0.2163 | 1.1693 | 0.2649 | 0.005613 | 1.2529 |
| K = 16 e299, 0.85 | 0.4909 | 0.2127 | 1.1168 | 0.2528 | 0.005307 | 1.1761 |
| K = 16 e400, 0.85 | 0.4896 | 0.2136 | 1.1121 | 0.2529 | 0.005010 | 1.1674 |

**Test:**

| metric | MAE (pub) | MAE+Rel (pub) | K = 1, 0.85 | K = 16 e299 | K = 16 e400 |
|---|---:|---:|---:|---:|---:|
| AUC nue | 0.9756 | 0.9772 | 0.9775 | 0.9787 | 0.9788 |
| AUC numu | 0.9473 | 0.9510 | 0.9470 | 0.9520 | 0.9518 |
| AUC nutau->e | 0.9103 | 0.9141 | 0.9181 | 0.9225 | 0.9221 |
| AUC nutau->mu | 0.8364 | 0.8408 | 0.8413 | 0.8488 | 0.8483 |
| AUC nutau->had | 0.9386 | 0.9417 | 0.9404 | 0.9444 | 0.9437 |
| AUC NC | 0.9388 | 0.9426 | 0.9382 | 0.9443 | 0.9443 |
| flavour acc | 0.8153 | 0.8212 | 0.8152 | 0.8238 | 0.8231 |
| AUC charm->e | 0.8042 | 0.8043 | 0.8217 | 0.8262 | 0.8234 |
| AUC charm->mu | 0.8844 | 0.8839 | 0.8932 | 0.8996 | 0.9003 |
| AUC charm->had | 0.8691 | 0.8702 | 0.8710 | 0.8785 | 0.8768 |
| sMAD E_vis | 0.2144 | 0.2124 | 0.2222 | 0.1981 | 0.1963 |
| sMAD pT_miss | 0.5573 | 0.5589 | 0.5782 | 0.5488 | 0.5552 |
| sMAD \|p_jet\| | 0.1637 | 0.1612 | 0.1543 | 0.1540 | 0.1516 |
| sMAD \|p_lep\| CC | 0.3857 | 0.3838 | 0.3930 | 0.3506 | 0.3429 |
| vertex median mm | 16.88 | 16.07 | 14.68 | 14.67 | 15.16 |

- **K = 16 e299 against the published MAE: better on all 15 metrics**, every
  interval excluding 0.  For example, flavour accuracy +0.0084
  [0.0069, 0.0100], sMAD E_vis -0.0162, pT_miss -0.0085, |p_lep| CC -0.0351,
  vertex -2.21 mm.  E_vis and pT_miss were the two metrics where K = 1 still
  trailed the MAE.
- **Against MAE+Rel** (point estimates only, no interval computed): higher on
  all 15 too.  The smallest margins are AUC numu +0.0010, nue +0.0015 and NC
  +0.0017.
- **Against K = 1 at the same layer decay: better on 11 of 15**, with
  intervals excluding 0.  The biggest gains are flavour accuracy +0.0086,
  E_vis -0.0241, pT_miss -0.0295 and |p_lep| CC -0.0424.  charm->e, charm->mu,
  |p_jet| and vertex are tied.
- **The partial collapse cost little after fine-tuning.**  e400 against
  e299 is tied on all 10 classification metrics.  On the regressions it is
  slightly better for E_vis, |p_jet| and |p_lep| but worse for pT_miss
  (+0.0065) and vertex (+0.49 mm), and its validation total is slightly
  lower.  Full fine-tuning seems to rebuild what the deep blocks lost, from
  the early blocks that kept their event content.  This is one fine-tune
  each, with no uncollapsed epoch-400 model to compare against.  Frozen use
  is not clearly worse either: the external review's frozen linear probe
  scores epoch 399 above epoch 299 (flavour macro-AUC 0.763 against 0.732).
- **Layer decay 0.90 is no better than 0.85** (K = 1).  Validation total is
  tied (1.2529 against 1.2524), with flavour and vis worse.  On test the
  classification is tied, E_vis and |p_lep| are better, and pT_miss, |p_jet|
  and vertex (+0.82 mm) are worse.  0.85 stays the choice.

Caveats: one fine-tuning seed per run.  The layer decay was chosen on K = 1
validation and not re-tuned for K = 16.  The MAE numbers use their published
0.75 recipe.  Epoch 299 was picked from the pretraining diagnostic, before
any fine-tune.  All K = 16 results in this section come from one
pretraining seed (7).  The second seed's fine-tune does not reproduce them:
it is mixed against the MAE (worse on flavour accuracy, E_vis and pT_miss)
and below K = 1 (see "Second pretraining seed"
below).

## Second pretraining seed, K = 16 `token` (Clariden, 2026-10-27)

Every K = 16 result above comes from one pretraining run (seed 7).  The same
setup with `SEED=8` (initialisation, data order and masks; the split stays
seed 7, as `split_dataset` fixes it) tests whether the gain over the MAE
holds across pretraining runs.  Its `last.ckpt` is fine-tuned as for the
others (published recipe, `LAYER_DECAY=0.85`, fine-tuning seed 7), then
evaluated.  The Clariden code matched dlnu's (46 files, sha256) at
submission.

```bash
c=/capstor/scratch/cscs/salonso/faser/checkpoint; p=pretrain_flash_jepa_base_d2v16_s8; f=finetune_flash_jepa_base_d2v16_s8_ld085; pid=$(NAME=$p TARGET_LAYERS=16 SEED=8 sbatch --parsable clariden/pretrain_flash.sbatch) && fid=$(LOAD_CHECKPOINT=$c/$p/loss_total_val/last.ckpt NAME=$f LAYER_DECAY=0.85 sbatch --parsable --dependency=afterok:$pid clariden/finetune_flash.sbatch) && eid=$(CHECKPOINT_DIR=$c/$f sbatch --parsable --dependency=afterok:$fid clariden/evaluate_flash.sbatch)
```

Jobs: pretraining 3769470 (2 nodes), fine-tune 3769471, evaluation 3769472,
each starting only if the previous one succeeds.

**Pretraining result** (3769470: 400 epochs in 4 h 21 min, no NaN; run
settings confirmed as `seed: 8`, `flash_target_layers: 16`,
`flash_target_norm: token`).  Validation diagnostics:

| epoch | 130 | 138 | 142 | 144 | 146 | 148 | 150 | 152 | 160 | 200 | 250 | 300 | 399 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val loss | 0.077 | 0.101 | 0.144 | 0.000 | 0.002 | 0.254 | 0.002 | 0.041 | 0.091 | 0.054 | 0.040 | 0.037 | 0.034 |
| `context_event_std` | 0.52 | 0.44 | 0.02 | 0.01 | 0.60 | 0.63 | 0.20 | 0.42 | 0.60 | 0.50 | 0.42 | 0.41 | 0.41 |
| `target_event_std` | 0.57 | 0.52 | 0.39 | 0.01 | 0.02 | 0.62 | 0.06 | 0.34 | 0.48 | 0.55 | 0.46 | 0.47 | 0.48 |
| `target_std` | 0.94 | 0.72 | 0.61 | 0.01 | 0.06 | 0.63 | 0.06 | 0.31 | 0.62 | 0.79 | 0.94 | 0.94 | 0.94 |

- **A complete collapse at epochs ~138-152, then recovery.**  At epoch 144
  the targets were nearly constant (`target_std` 0.01) and the loss ~0.
  The run then oscillated for a few epochs and rebuilt the targets over
  epochs 152-220.
- **Stable from epoch ~250 to the end.**  No late collapse: the minimum
  `context_event_std` after epoch 200 is 0.40, against 0.07 for seed 7.  At
  epoch 399, `target_std` (0.94) and the event spreads (0.41, 0.48) are
  close to seed 7's values before its collapse (epoch 300: 0.93, 0.40,
  0.54).
- **Both K = 16 `token` seeds collapsed, at different times.**  Seed 7
  collapsed partially at epochs 316-346 and did not recover.  Seed 8
  collapsed completely at epochs 138-152 and did.  The instability is a
  property of the setup, not of one late stretch of the schedule.
- **The validation loss settles lower than seed 7's** (0.037-0.040 against
  0.085-0.090 at epochs 250-310; 0.034 at epoch 399), with similar target
  scale.  Its cause is
  not established.  Fine-tuning of `last.ckpt` (the pre-planned choice) will
  show whether either episode matters.
- **K = 1 for comparison** (`pretrain_flash_jepa_base`, dlnu): no abrupt
  event.  Its spreads only narrow slowly over training (`target_std` 0.99 to
  0.63, `context_event_std` 0.65 to 0.23), with no epoch after 25 below 0.5
  in `target_std`.

**Block-by-block check** (`results/flash_audit_20261026/collapse_check_s8.{sbatch,out}`,
Clariden debug job 3780035, 77 s; same script and the same 2,048 validation
events as for seed 7).  Seed 7's epoch 299 is repeated and reproduces
`collapse_check.out` exactly.

```bash
sbatch results/flash_audit_20261026/collapse_check_s8.sbatch   # from /users/salonso/faserDL
```

Teacher (target encoder) `event` fraction, with `same` (identical for every
token) in brackets for selected rows:

| | s7 e299 | s8 e124 | s8 e149 | s8 e174 | s8 e199 | s8 e249 | s8 e399 |
|---|---:|---:|---:|---:|---:|---:|---:|
| block 1 | 0.32 | 0.25 | 0.47 | 0.57 | 0.42 | 0.20 | 0.23 |
| block 4 | 0.67 | 0.71 | 0.35 (0.63) | 0.42 (0.53) | 0.47 | 0.73 | 0.58 |
| block 8 | 0.88 | 0.83 | 0.22 (0.77) | 0.54 | 0.73 | 0.76 | 0.73 |
| block 12 | 0.92 | 0.97 | 0.22 (0.76) | 0.58 | 0.78 | 0.79 | 0.83 |
| block 16 | 0.60 | 0.55 | 0.24 (0.75) | 0.39 | 0.79 | 0.87 | 0.76 |
| K = 16 target | 0.66 | 0.70 | 0.29 (0.69) | 0.45 (0.50) | 0.52 (0.36) | 0.57 | 0.55 (0.06) |
| final output | 0.56 | 0.68 | 0.41 | 0.47 | 0.35 | 0.65 | 0.59 |
| block 1 loud | 0.84 | 1.36 | 56.7 | 89.4 | 15.1 | 1.44 | 0.72 |

- **A different collapse from seed 7's.**  At epoch 149 the student (context
  encoder) had no event content left in any block from 4 on (`event`
  0.000, pooled spread 0.009): it was constant.  Block 1's MLP output was
  57x the size of the patch embedding it adds to (`loud`), and 89x at epoch
  174, so it swamped the input.  The teacher followed halfway (`same`
  0.50-0.78).  Seed 7 instead lost its deep half while blocks 1-4 kept the
  event content.  Both end in the same shortcut, a representation shared by
  every token, reached from opposite ends of the encoder.
- **The recovery is real.**  By epoch 249 block 1's `loud` is back to 1.4,
  and at epoch 399 the deep blocks carry 0.73-0.90 event content.  That is
  close to before the collapse (epoch 124) and to seed 7's healthy epoch
  299, and nothing like seed 7's epoch 399 (block 16: 0.76 against 0.01).
  Final output: event 0.59, spread 0.40, against seed 7's 0.04 and 0.12 at
  epoch 399.
- **Why the loss is lower than seed 7's.**  The K = 16 target at epoch 399
  is 45% predictable from the token's place alone (`layout` = 1 - `event`),
  against 30-34% at seed 8 e124 and seed 7 e299.  The extra part is
  position-specific, not a shared constant (`same` 0.06).  It is spread over
  blocks 4-12, whose event content dropped by 0.1-0.2 from epoch 124
  (block 6: 0.82 to 0.59; block 11: 0.96 to 0.83), while block 16 rose (0.55
  to 0.76).  The predictor is given the positions, so this part is easy to
  predict.
- **Both collapses started with a loss spike** (seed 7: 0.087 to 0.132 at
  epochs 316-328; seed 8: 0.085 to 0.223 at epochs 136-140), despite
  gradient clipping at norm 1.0.  The LR was very different: 1.3e-3 for
  seed 8, 2.2e-4 to 1.7e-4 for seed 7 (logged `lr`).  An optimisation spike followed by a fall into
  the constant-target shortcut fits both; the trigger is not established.
- MLP biases in blocks 1-3 have grown slowly since recovery (6-13% of the
  output at e399, from below 1% at e124).  Small, and not part of either
  collapse.
- This healthier representation did not give a better fine-tune: see the
  fine-tune result below.

**Fine-tune on dlnu instead.**  The Clariden fine-tune and evaluation
(3769471/3769472) were cancelled before starting: no outputs.  `last.ckpt`
was copied to dlnu (sha256 5a689619... on both sides) and fine-tuned there
(2 x 512, same global batch) from 2026-10-28 09:03 to 11:08, evaluation
included.  In the replicate pair below, the machine difference is small
on classification and up to ~0.004 sMAD on the regressions.  Hyperparameters are identical to the batch-standardised
dlnu fine-tune's except the name and the checkpoint (`hparams.yaml` diff).

```bash
cd /scratch/salonso/sparse-nns/faser_develop && source /scratch/salonso/anaconda3/etc/profile.d/conda.sh && conda activate neutrino-jepa && n=finetune_flash_jepa_base_d2v16_s8_ld085_dlnu && c=/scratch4/salonso/faser/checkpoints_final && nohup bash -c "NAME=$n LAYER_DECAY=0.85 GPUS='0 1' BATCH_SIZE=512 NUM_WORKERS=24 CHECKPOINT_PATH=$c SAVE_TOP_K=1 SAVE_LAST=0 CHECKPOINT_MONITORS=evaluated LOAD_CHECKPOINT=$c/pretrain_flash_jepa_base_d2v16_s8/loss_total_val/last.ckpt EXPECTED_SPLIT_SHA256=02078f72b5ec271f81e14f3a24b769e665a9a928908b141bd1fb2134c730fd7c DATASET_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*' METADATA_PATH=/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz/metadata_stats.pkl bash finetune_flash.sh && python -m evaluation.create_csv_v7_ft --checkpoint_dir $c/$n" > /scratch4/salonso/faser/$n.log 2>&1 &
```

**Fine-tune result.**  The log has no errors.  It loaded 100% of the
encoder from the target branch of epoch 399, and the task checkpoints were
picked at epochs 17-19.
`results/flash_audit_20261026/metrics_flash_s8.{py,out}` scores it on the
same 122,639 test events (paired event bootstrap, 200 resamples):

```bash
python results/flash_audit_20261026/metrics_flash_s8.py > results/flash_audit_20261026/metrics_flash_s8.out 2>&1
```

Validation (best per-task loss):

| run | flavour | charm | vis | jet | vertex | total |
|---|---:|---:|---:|---:|---:|---:|
| K = 1, 0.85 | 0.5085 | 0.2166 | 1.1608 | 0.2631 | 0.005640 | 1.2524 |
| K = 16 seed 7 e299 | 0.4909 | 0.2127 | 1.1168 | 0.2528 | 0.005307 | 1.1761 |
| K = 16 seed 7 e400 | 0.4896 | 0.2136 | 1.1121 | 0.2529 | 0.005010 | 1.1674 |
| K = 16 batch (dlnu) | 0.5081 | 0.2153 | 1.1559 | 0.2654 | 0.006010 | 1.2520 |
| K = 16 seed 8 e400 (dlnu) | 0.5268 | 0.2185 | 1.2161 | 0.2765 | 0.005953 | 1.3473 |

Seed 8's total validation loss is the highest of the five at every one of
the 20 epochs.

Test:

| metric | MAE (pub) | K = 1 | seed 7 e299 | seed 7 e400 | seed 8 e400 | seed 8 - seed 7 e400 [95%] |
|---|---:|---:|---:|---:|---:|---:|
| AUC nue | 0.9756 | 0.9775 | 0.9787 | 0.9788 | 0.9756 | -0.0032 [-0.0037, -0.0027] |
| AUC numu | 0.9473 | 0.9470 | 0.9520 | 0.9518 | 0.9424 | -0.0093 [-0.0100, -0.0087] |
| AUC nutau->e | 0.9103 | 0.9181 | 0.9225 | 0.9221 | 0.9150 | -0.0072 [-0.0107, -0.0041] |
| AUC nutau->mu | 0.8364 | 0.8413 | 0.8488 | 0.8483 | 0.8418 | -0.0065 [-0.0105, -0.0017] |
| AUC nutau->had | 0.9386 | 0.9404 | 0.9444 | 0.9437 | 0.9376 | -0.0061 [-0.0076, -0.0046] |
| AUC NC | 0.9388 | 0.9382 | 0.9443 | 0.9443 | 0.9316 | -0.0126 [-0.0136, -0.0118] |
| flavour acc | 0.8153 | 0.8152 | 0.8238 | 0.8231 | 0.8062 | -0.0169 [-0.0185, -0.0153] |
| AUC charm->e | 0.8042 | 0.8217 | 0.8262 | 0.8234 | 0.8148 | -0.0086 [-0.0195, +0.0028] |
| AUC charm->mu | 0.8844 | 0.8932 | 0.8996 | 0.9003 | 0.8897 | -0.0106 [-0.0208, -0.0009] |
| AUC charm->had | 0.8691 | 0.8710 | 0.8785 | 0.8768 | 0.8677 | -0.0090 [-0.0117, -0.0069] |
| sMAD E_vis | 0.2144 | 0.2222 | 0.1981 | 0.1963 | 0.2332 | +0.0368 [+0.0351, +0.0392] |
| sMAD pT_miss | 0.5573 | 0.5782 | 0.5488 | 0.5552 | 0.6049 | +0.0497 [+0.0460, +0.0543] |
| sMAD \|p_jet\| | 0.1637 | 0.1543 | 0.1540 | 0.1516 | 0.1623 | +0.0108 [+0.0095, +0.0120] |
| sMAD \|p_lep\| CC | 0.3857 | 0.3930 | 0.3506 | 0.3429 | 0.3851 | +0.0422 [+0.0386, +0.0457] |
| vertex median mm | 16.88 | 14.68 | 14.67 | 15.16 | 14.28 | -0.88 [-0.99, -0.77] |

- **Seed 7's K = 16 gain does not replicate.**  Seed 8 is worse than seed
  7's e400 fine-tune on 13 of 15 metrics (intervals excluding 0), tied on
  charm->e, and better only on vertex.  The same holds against seed 7's
  e299 (vertex -0.39 mm).  The gaps are about 10x the difference between
  the two fine-tunes of the `batch` checkpoint below: flavour accuracy
  -0.0169 against +0.0001, E_vis +0.0368 against -0.0034.  That is a single
  cross-machine pair, not a variance estimate.
- **Below K = 1.**  Against K = 1 at the same layer decay: worse on 10
  (including flavour accuracy -0.0089, E_vis +0.0110, pT_miss +0.0267),
  better on 2 (|p_lep| CC -0.0079, vertex -0.40 mm), tied on 3
  (nutau->mu, charm->e, charm->mu).  It is also below the
  batch-standardised fine-tune on 13 of 15.
- **Mixed against the published MAE, with clear losses on central
  metrics.**  Better on 2 (nutau->e +0.0047, vertex -2.60 mm), worse on 5
  (numu, NC, flavour accuracy -0.0091, E_vis +0.0188, pT_miss +0.0476), tied
  on 8.
- **The pretraining diagnostics did not predict this.**  Seed 8 ended with
  the lower pretraining loss (0.037-0.040 against 0.085-0.090 at epochs
  250-310) and the healthier representation by every block-level measure
  above.  Yet it fine-tunes
  worst, while seed 7's partly collapsed e400 fine-tunes best.  Neither the
  pretraining loss nor the `event`/spread measures are a proxy for
  fine-tuned quality here.
- **Collapse or seed?  Not separable from these runs.**  One candidate:
  seed 8's collapse came early, near the peak LR, and its recovery used
  about 100 epochs (138-250).  It also left a target that is more
  position-predictable (45% against 30-34%).  Seed 7's collapse came after
  epoch 316, at a low LR, when its early blocks kept their event content.
  The other candidate is that K = 16 pretraining seeds simply differ this
  much.  Either way, the spread between the two seeds (flavour accuracy
  0.0169) is larger than seed 7's gain over K = 1 (0.0079).
- Vertex is the one metric where seed 8 is the best of all runs (14.28 mm).
  Not explained.  Its two Clariden fine-tunes later gave 15.33 and
  16.35 mm, so this is within the spread between repeat fine-tunes (see
  the Clariden controls results).

**Consequence.**  K = 16 `token` is not established as the final
configuration.  The flash JEPA has one fine-tune clearly above the MAE
(K = 16 seed 7) and one mixed against it, worse on flavour accuracy, E_vis
and pT_miss (K = 16 seed 8).  K = 1 and
`batch`, one pretraining seed each, are about the MAE's level, with mixed
signs.  None of these final checkpoints shows a gain over the MAE that has
been reproduced across pretraining seeds.  The epoch-124 snapshots of both
K = 16 seeds do, on all metrics but pT_miss.  That snapshot was chosen with
hindsight; see the epoch-124 results below.

**Epoch-124 fine-tunes (dlnu, started 2026-10-28).**  These locate when
the gap between the seeds appeared, by fine-tuning both seeds' epoch-124
snapshots, from before seed 8's collapse, with the same recipe as above.
They time the gap but do not test its cause: after epoch 124 the two runs
also follow different trajectories (external review, G1).
- If the two come out close, the gap opened after epoch 124.  That fits
  the collapse causing it but does not prove it.
- If seed 8 is already worse at epoch 124, the seeds differed before the
  collapse.  Later damage on top is still possible.

A causal test would continue seed 8 from a pre-collapse snapshot, with the
optimizer and teacher state kept (the checkpoints hold no RNG state), with
and without a stabilising change.  The run without a change is the seed-8
reproducer below.

**Prediction, recorded 2026-10-28 13:35, before either result.**  The
external review's frozen linear probe (`codex_review2_report.md` section 5)
scores the two epoch-124 teachers close: flavour macro-AUC 0.749 (seed 7)
and 0.738 (seed 8), against 0.662 for seed 8's final snapshot.  If the
probe tracks fine-tuning, the two epoch-124 fine-tunes come out close to
each other and both above seed 8's final fine-tune.  The probe is not
validated:
- On the five finished fine-tunes it gives the same order as flavour
  accuracy, except seed 7's e299/e400, which are tied (Spearman 0.9).  The
  sample is tiny, and two of the five share seed 7.
- It scores `batch` and seed 8's final snapshot alike (0.665 and 0.662),
  but their fine-tunes differ by 0.0088 in flavour accuracy.

The review's programs, logs and frozen features are kept, without its
checkpoint copies, in `/scratch4/salonso/faser/codex_review2_evidence/`.
The report's `/tmp/faser-codex-review2.ZfUx6P` links point to the original
temporary copy.

Seed 7 is the Clariden run (3760078) that the seed-7 fine-tunes above came
from.  The dlnu run of the same name
(`checkpoints_final/pretrain_flash_jepa_base_d2v16`, 2 x 2,048, stopped at
epoch 135) is a different realisation and is not used.  The script waits
for a valid CSCS SSH certificate (checked locally; no login attempts).  It
then copies both snapshots from Clariden with sha256 checked: seed 7 to
`checkpoints_final/pretrain_flash_jepa_base_d2v16_clariden/snapshots/`,
seed 8 to `checkpoints_final/pretrain_flash_jepa_base_d2v16_s8/snapshots/`.
Finally it fine-tunes and evaluates `finetune_flash_jepa_base_d2v16_ep124_ld085_dlnu`
and `finetune_flash_jepa_base_d2v16_s8_ep124_ld085_dlnu` one after the other
(~2 h each).  It refuses to overwrite any existing output.

```bash
cd /scratch/salonso/sparse-nns/faser_develop && nohup bash results/flash_audit_20261026/finetune_ep124_pair.sh > /scratch4/salonso/faser/finetune_ep124_pair.log 2>&1 &
```

**Epoch-124 results, both seeds (2026-10-28).**  Scored with
`python results/flash_audit_20261026/metrics_flash_ep124.py`; the output,
with every paired interval, is `metrics_flash_ep124.out` next to it.
- **Seed 7's results came first**, while seed 8's fine-tune was still
  running.  The bootstrap draws do not depend on which pairs are scored, so
  its intervals are the same in the full output.
- **Seed 8's run checks out.**  `finetune_flash_jepa_base_d2v16_s8_ep124_ld085_dlnu`
  (16:16) loaded `epoch=124.ckpt`:
  - global step 31,750;
  - sha256 d8cc63eb..., the same on Clariden and dlnu (seed 7's is 7fccb385...);
  - full encoder coverage, `target` source.

  It ran all 20 epochs.  Its log has no NaN or error.
- **Platforms.**  Every e124 fine-tune and seed 8's final fine-tune ran on
  dlnu.  Seed 7's e299 and e400 fine-tunes ran on Clariden.

| Metric | MAE | K = 1 | s7 e124 | s8 e124 | s7 e299 | s7 e400 | s8 e400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation total | | 1.2524 | 1.2168 | 1.2073 | 1.1761 | 1.1674 | 1.3473 |
| flavour acc | 0.8153 | 0.8152 | 0.8198 | 0.8191 | 0.8238 | 0.8231 | 0.8062 |
| AUC NC | 0.9388 | 0.9382 | 0.9413 | 0.9406 | 0.9443 | 0.9443 | 0.9316 |
| sMAD E_vis | 0.2144 | 0.2222 | 0.2115 | 0.2048 | 0.1981 | 0.1963 | 0.2332 |
| sMAD pT_miss | 0.5573 | 0.5782 | 0.5710 | 0.5691 | 0.5488 | 0.5552 | 0.6049 |
| sMAD \|p_lep\| CC | 0.3857 | 0.3930 | 0.3714 | 0.3507 | 0.3506 | 0.3429 | 0.3851 |
| vertex median mm | 16.88 | 14.68 | 15.40 | 14.63 | 14.67 | 15.16 | 14.28 |

- **The two seeds were level at epoch 124 (same platform).**  Seed 8
  against seed 7:
  - tied on 11 of 15 metrics: all six flavour AUCs, flavour accuracy
    (seed 7 +0.0007 [-0.0006, +0.0021]), pT_miss, |p_jet|, charm->e and
    charm->mu;
  - better on the other 4: E_vis (-0.0066), |p_lep| CC (-0.0207), vertex
    (-0.78 mm) and charm->had (+0.0028).

  So the gap between the seeds' final fine-tunes opened after epoch 124.
  That fits seed 8's collapse causing it but does not prove it (G1).
- **Seed 8 lost ground after epoch 124 (same platform).**  Its e124
  fine-tune is better than its final one on 11 of 15:
  - flavour accuracy +0.0129 [+0.0114, +0.0141];
  - E_vis -0.0284, pT_miss -0.0358, |p_jet| -0.0095 and |p_lep| CC -0.0344.

  It is tied on nutau->mu, charm->e and charm->mu, and worse only on vertex
  (+0.34 mm).  Seed 7 instead gained after epoch 124 (below).
- **Both seeds beat the MAE at epoch 124.**  This is the first gain over the
  MAE seen in both pretraining seeds.
  - Seed 8 is better on 14 of 15 metrics and worse only on pT_miss
    (+0.0118).
  - Seed 7 is better on 12, tied on 2 and worse on pT_miss (+0.0137).

  Three limits:
  - **Epoch 124 was chosen with hindsight.**  It is the last snapshot before
    seed 8's collapse, picked from its pretraining diagnostics, not a
    stopping rule fixed in advance.  A run without that hindsight would have
    used `last.ckpt`.
  - **One fine-tuning seed.**  Both runs used fine-tuning seed 7.
  - **Part of the gain may be architecture.**  The Clariden scratch control
    below measures how much.
- **The vertex differences in this section (0.3-0.8 mm) are within the
  ~1 mm spread between repeat fine-tunes** measured in the Clariden
  controls below.  The event bootstrap does not cover that spread.
- **Both seeds beat K = 1 at epoch 124.**  Seed 8 is better on 9 and tied
  on 6, including vertex.  Seed 7 is better on 11, tied on 3 and worse on
  vertex (+0.72 mm).
- **Seed 7 was already above the MAE and K = 1 at epoch 124.**
  - Against the MAE it is better on 12 of 15 metrics (intervals excluding 0),
    tied on charm->e and charm->mu, and worse only on pT_miss (+0.0137).
  - Against K = 1 it is better on 11, tied on 3 and worse on vertex
    (+0.72 mm).
- **Pretraining past epoch 124 still helped seed 7.**  Against e299 it is
  worse on 12 of 15:
  - flavour accuracy -0.0040 [-0.0053, -0.0026];
  - E_vis +0.0133, pT_miss +0.0223 and |p_lep| CC +0.0208;
  - vertex +0.73 mm.

  It is better only on |p_jet| (-0.0023), and tied on nutau->e and
  nutau->mu.  The regression gaps are several times the ~0.004 sMAD
  cross-machine pair.  This e124 fine-tune ran on dlnu, while e299/e400 ran
  on Clariden.
- **The recorded prediction held; the probe still partly misses.**
  - Both halves of the prediction held.  The two epoch-124 fine-tunes are
    tied on flavour accuracy, and both are above seed 8's final fine-tune
    (0.8198 and 0.8191 against 0.8062).
  - The miss stands: the probe put seed 7's epoch 124 above epoch 299
    (0.749 against 0.732), and the fine-tunes put it below.
  - One right and one wrong ordering on new runs do not validate the probe.

**Clariden controls (submitted 2026-10-28 13:56).**  These are the two
cheapest steps the external review recommends (`codex_review2_report.md`
section 6, items 2 and 3).
- **A FlashViT trained from scratch.**  This measures how much of the gain
  over the MAE comes from the architecture rather than from pretraining.
- **A second fine-tuning seed for both final K = 16 checkpoints.**  This
  shows whether the gap between pretraining seeds 7 and 8 survives a
  different fine-tuning seed.

Together with the existing Clariden seed-7 fine-tune
(`finetune_flash_jepa_base_d2v16_last_ld085`), seeds 7 and 8 are then
crossed with fine-tuning seeds 7 and 8 on one platform.  All four runs use
the published recipe with layer decay 0.85: 20 epochs, 1 x 4 GH200, global
batch 1,024.

| Run | Encoder | Fine-tuning seed | Fine-tune / evaluation job |
|---|---|---:|---|
| `finetune_flash_scratch_base_ld085` | random init (flash base) | 7 | 3782686 / 3782687 |
| `finetune_flash_jepa_base_d2v16_last_ld085_ft8` | K16 seed 7 `last.ckpt` (epoch 400) | 8 | 3782688 / 3782689 |
| `finetune_flash_jepa_base_d2v16_s8_ld085` | K16 seed 8 `last.ckpt` (sha256 5a689619...) | 7 | 3782690 / 3782691 |
| `finetune_flash_jepa_base_d2v16_s8_ld085_ft8` | K16 seed 8 `last.ckpt` | 8 | 3782692 / 3782693 |

- **The scratch run is a control, not a tuned scratch baseline.**  It keeps
  the pretrained runs' recipe, including layer decay 0.85, which lowers the
  learning rate of the early blocks even though they start from random
  weights.  The published scratch baseline used its own recipe on the
  SparseViT.
- **Two small code changes made the scratch run possible.**
  - `clariden/finetune_flash.sbatch` now accepts an empty `LOAD_CHECKPOINT`,
    as its message already said it should.
  - `evaluation.create_csv_v7_ft --allow_scratch` (`ALLOW_SCRATCH=1` in
    `clariden/evaluate_flash.sbatch`) accepts a run that loaded no
    checkpoint.  It records the pretraining objective as `none` and the
    encoder coverage as 0.0.  Without the flag, the script refuses such a
    run as before.
  - Both files and the test were copied to `/users/salonso/faserDL` before
    submission; the code there matched dlnu's.

From `/users/salonso/faserDL` on Clariden:

```bash
c=/capstor/scratch/cscs/salonso/faser/checkpoint; s7=$c/pretrain_flash_jepa_base_d2v16/loss_total_val/last.ckpt; s8=$c/pretrain_flash_jepa_base_d2v16_s8/loss_total_val/last.ckpt
sub() { fid=$(LOAD_CHECKPOINT=$2 NAME=$1 SEED=$3 LAYER_DECAY=0.85 sbatch --parsable clariden/finetune_flash.sbatch) && eid=$(ALLOW_SCRATCH=${4:-} CHECKPOINT_DIR=$c/$1 sbatch --parsable --dependency=afterok:$fid clariden/evaluate_flash.sbatch) && echo "$1 finetune=$fid evaluate=$eid"; }
sub finetune_flash_scratch_base_ld085 "" 7 1
sub finetune_flash_jepa_base_d2v16_last_ld085_ft8 $s7 8
sub finetune_flash_jepa_base_d2v16_s8_ld085 $s8 7
sub finetune_flash_jepa_base_d2v16_s8_ld085_ft8 $s8 8
```

**Clariden controls: results (2026-10-28).**
- **All eight jobs completed.**  The fine-tunes ran 17:45-18:20, about 34
  minutes each.  The evaluations ran 21:54-22:00.
- **The CSVs were copied** to `results/`, with sha256 checked against
  Clariden.
- **Provenance checks out.**  Every CSV covers the same 122,639 test events
  (inventory 57c3a1b1...) after 20 epochs.  The three pretrained runs
  loaded their encoder fully.  The scratch run records objective `none`
  and coverage 0.0.
- **Scripts.**  Scored with
  `python results/flash_audit_20261026/metrics_flash_controls.py`, whose
  output with every interval is `metrics_flash_controls.out`.  A contrast
  there can average over both fine-tuning seeds.  Vertex offsets come from
  `vertex_offset_check.{py,out}`.

The flash runs ran on Clariden except the last column.  s7/s8 is the
pretraining seed, FT7/FT8 the fine-tuning seed.

| Metric | scratch SparseViT (pub) | MAE (pub) | flash scratch | s7 FT7 | s7 FT8 | s8 FT7 | s8 FT8 | s8 FT7 (dlnu) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation total | | | 1.5931 | 1.1674 | 1.1749 | 1.3342 | 1.3421 | 1.3473 |
| flavour acc | 0.7437 | 0.8153 | 0.7846 | 0.8231 | 0.8230 | 0.8062 | 0.8063 | 0.8062 |
| AUC NC | 0.8762 | 0.9388 | 0.9159 | 0.9443 | 0.9442 | 0.9318 | 0.9317 | 0.9316 |
| AUC charm->had | 0.7829 | 0.8691 | 0.8381 | 0.8768 | 0.8801 | 0.8689 | 0.8669 | 0.8677 |
| sMAD E_vis | 0.3528 | 0.2144 | 0.2704 | 0.1963 | 0.1993 | 0.2278 | 0.2294 | 0.2332 |
| sMAD pT_miss | 0.8233 | 0.5573 | 0.6910 | 0.5552 | 0.5581 | 0.6104 | 0.6161 | 0.6049 |
| sMAD \|p_lep\| CC | 0.5130 | 0.3857 | 0.4286 | 0.3429 | 0.3549 | 0.3740 | 0.3747 | 0.3851 |
| vertex median mm | 22.06 | 16.88 | 19.07 | 15.16 | 15.94 | 15.33 | 16.35 | 14.28 |

- **The flash architecture alone recovers 50-85% of the paper's
  scratch-to-MAE gap.**
  - Flash scratch is better than the SparseViT scratch on all 15 metrics:
    flavour accuracy +0.0409 [+0.0391, +0.0427], E_vis -0.082, vertex
    -2.99 mm.
  - Its share of the gap is 57% on flavour accuracy, 60% on E_vis, 50% on
    pT_miss and 58% on vertex.
  - It is still below the MAE, worse on 12 of 15: flavour accuracy -0.0307,
    E_vis +0.056, pT_miss +0.134, vertex +2.19 mm.  It is tied on
    nutau->e, charm->e and charm->mu.
  - **These shares are lower bounds.**  The run had 20 epochs, against 40
    for the published SparseViT scratch, and layer decay 0.85.  Every
    per-task best checkpoint is its last epoch.  The validation total was
    still falling (1.607 to 1.593 over the last epoch).  The comparison
    with the SparseViT scratch also mixes recipes.
- **Pretraining adds about as much again on the flash encoder.**
  - Seed 7 (FT7) is better than flash scratch on all 15: flavour accuracy
    +0.0385, E_vis -0.074, pT_miss -0.136, vertex -3.91 mm.
  - Seed 8 (FT7) is too: flavour accuracy +0.0217, vertex -3.73 mm.
  - Flash scratch's under-training makes these upper bounds.
  - So the flash vertex result needs both the encoder and the pretraining.
    The encoder alone gives 19.1 mm.
- **The gap between pretraining seeds survives a change of fine-tuning
  seed.**
  - Seed 7 minus seed 8 in flavour accuracy is +0.0169 with fine-tuning
    seed 7 and +0.0167 with seed 8.
  - Averaged over both fine-tuning seeds, seed 7 is better on all 15:
    flavour accuracy +0.0168 [+0.0153, +0.0181], E_vis -0.031, pT_miss
    -0.057, |p_lep| CC -0.025, vertex -0.29 mm.
- **The fine-tuning seed barely moves classification.**
  - Flavour accuracy changes by 0.0001 for both encoders, and every flavour
    AUC is tied.
  - Regressions move by up to 0.003 (E_vis), 0.006 (pT_miss) and 0.012
    (|p_lep| CC, seed 7).
  - Seed 8's Clariden and dlnu fine-tunes (same seed, other platform) agree
    as closely on classification.
  - On these final checkpoints, flavour accuracy varies about 100x more
    between pretraining seeds than between fine-tuning seeds.
- **Vertex medians differ by up to ~1 mm between repeat fine-tunes.**
  - With the same encoder, another fine-tuning seed gives 0.78 mm (seed 7)
    and 1.01 mm (seed 8).
  - With the same encoder and seed, another platform gives 1.05 mm (seed 8)
    and 0.26 mm (the `batch` pair below).
  - The event bootstrap gives about +-0.1 mm, so it understates this
    spread about 10x.
  - **It is not an offset.**  Removing each run's median signed residual
    changes the 3D median by at most 0.4 mm.  The z spread itself differs
    between seed 8's three fine-tunes: 19.6, 20.4 and 22.5 mm.
  - **Single-run vertex differences below ~1 mm are not established.**
    This covers seed 8 being the best on vertex (14.28 mm), the epoch-124
    vertex differences, and the SparseViT MAE / JEPA-patch / MAE+Rel order
    (16.88 / 16.58 / 16.07 mm).
- **Against the MAE, averaged over both fine-tuning seeds:**
  - **Seed 7 is better on 14 of 15**, including flavour accuracy +0.0078
    [+0.0065, +0.0094] and vertex -1.33 mm.  It is tied on pT_miss
    (-0.0006 [-0.0040, +0.0031]).
  - **Seed 8 is mixed.**  It is better on 5 (nutau->e, nutau->mu, |p_jet|,
    |p_lep| CC, vertex -1.04 mm), worse on 5 (nu_mu, NC, flavour accuracy
    -0.0090, E_vis +0.0143, pT_miss +0.056) and tied on 5.
  - The MAE is a single fine-tune, so the vertex gaps are about one
    replicate spread.

Commands (dlnu, repository root, `neutrino-jepa` environment):

```bash
rsync -a --ignore-existing clariden:/users/salonso/faserDL/results/results_v7_test_finetune_{flash_scratch_base_ld085,flash_jepa_base_d2v16_last_ld085_ft8,flash_jepa_base_d2v16_s8_ld085,flash_jepa_base_d2v16_s8_ld085_ft8}.{csv,manifest.txt,metadata.json} results/
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 python -u results/flash_audit_20261026/metrics_flash_controls.py > results/flash_audit_20261026/metrics_flash_controls.out
python results/flash_audit_20261026/vertex_offset_check.py | tee results/flash_audit_20261026/vertex_offset_check.out
```

**Seed-8 reproducer (Clariden job 3785161, submitted 2026-10-28 19:40).**
This tests whether seed 8's collapse at epochs ~138-152 follows from its
epoch-124 state and the schedule, or was a chance event.
- **If it collapses again,** a candidate fix can be tested in ~25 minutes
  from the same snapshot, instead of a 4.3 h pretraining plus a 2 h
  fine-tune.
- **If it does not,** the collapse was chance, and a guard that rolls back
  to the last snapshot is the cheaper answer.

The run, `pretrain_flash_jepa_base_d2v16_s8_r124`:
- **Resumes** `pretrain_flash_jepa_base_d2v16_s8/snapshots/epoch=124.ckpt`
  (global step 31,750, sha256 d8cc63eb..., as copied to dlnu above).  It
  keeps the student, teacher, optimizer and LR-scheduler state.
- **Keeps the original settings:** K = 16 `token`, seed 8, 2 nodes x 4
  GH200, 512 per GPU, and `EPOCHS=400`, so the LR and teacher-momentum
  schedules are the original ones.
- **Changes only the name and the snapshots,** every 5 epochs
  (`epoch=129`, `134`, ..., `164`) instead of every 25; `last.ckpt` is still
  written every epoch.
- **Is not a replay.**  The checkpoint holds no RNG state, so masks,
  augmentation draws and the non-deterministic GPU kernels differ from the
  original run.
- **Stops by time limit.**  There is no stop-at-epoch option, and changing
  `EPOCHS` would change the schedules.  At the original ~38.5 s per epoch,
  31 minutes reach epoch ~165-170.  The epoch cut by the time limit is lost.
- **Uses unchanged code.**  On Clariden, only `evaluation/create_csv_v7_ft.py`
  and its test changed after seed 8's job (3769470) started, for the
  controls above.  No code was copied for this run.

Compare `jepa/target_std`, `jepa/context_event_std` and the loss over epochs
125-165 with the original run's log.  From `/users/salonso/faserDL` on
Clariden:

```bash
NAME=pretrain_flash_jepa_base_d2v16_s8_r124 TARGET_LAYERS=16 SEED=8 SAVE_EVERY_N_EPOCHS=5 RESUME_CHECKPOINT=/capstor/scratch/cscs/salonso/faser/checkpoint/pretrain_flash_jepa_base_d2v16_s8/snapshots/epoch=124.ckpt sbatch --time=00:31:00 clariden/pretrain_flash.sbatch
```

**Reproducer result (2026-10-28).**  The job ran 21:54-22:25 and stopped
on its time limit during epoch 170.  It wrote snapshots `epoch=129` to
`epoch=169` and `last.ckpt` (epoch 169), with no error or NaN.
- **Resume confirmed.**  The log shows the restore from `epoch=124.ckpt`.
  The first logged step is 31,799 (31,750 + 49), and the teacher momentum
  is 0.99689, on the original schedule.
- **Comparison script.**  `results/flash_audit_20261026/reproducer_s8_r124.{py,out}`
  compares the two runs epoch by epoch from the two `metrics.csv` files,
  copied to `/scratch4/salonso/faser/reproducer_s8_r124/`.

Validation values (`ctx_ev` = `context_event_std`):

| epoch | 136 | 137 | 140 | 142 | 144 | 145 | 146 | 148 | 150 | 155 | 160 | 169 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original val loss | 0.085 | 0.131 | 0.223 | 0.144 | 0.000 | 0.000 | 0.002 | 0.254 | 0.002 | 0.096 | 0.091 | 0.081 |
| rerun val loss | 0.082 | 0.082 | 0.143 | 0.107 | 0.128 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| original `target_std` | 0.94 | 0.78 | 0.74 | 0.61 | 0.01 | 0.01 | 0.06 | 0.63 | 0.06 | 0.55 | 0.62 | 0.69 |
| rerun `target_std` | 0.94 | 0.94 | 0.83 | 0.71 | 0.51 | 0.06 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| original `ctx_ev` | 0.47 | 0.29 | 0.38 | 0.02 | 0.01 | 0.01 | 0.60 | 0.63 | 0.20 | 0.58 | 0.60 | 0.64 |
| rerun `ctx_ev` | 0.47 | 0.46 | 0.27 | 0.37 | 0.01 | 0.01 | 0.03 | 0.01 | 0.01 | 0.01 | 0.00 | 0.01 |

- **The collapse is reproducible.**
  - The rerun followed the original within ~0.01 in loss and spreads to
    epoch 136.
  - It then had the same loss spike, 3 epochs later (training mean 0.135 at
    epoch 140, against 0.168 at epoch 137).
  - It collapsed completely at epochs 145-146, against 144.

  Two realisations from the same state, with different masks, augmentations
  and kernels, both collapsed within 21 epochs.  The collapse follows from
  the epoch-124 state and the schedule (LR ~1.3e-3), not from one unlucky
  draw.
- **The recovery was chance.**  The original run recovered within a few
  epochs, after oscillating until epoch ~152.  The rerun stayed constant through epoch 169 (`target_std`
  0.01-0.02, `ctx_ev` <= 0.01, loss 0.000), 24 epochs after collapsing.
  Whether it would recover later is not known.  So seed 8's final
  checkpoint rests on a recovery that the rerun did not repeat.
- **A rollback guard alone would probably not help.**  Rolling back to
  epoch 124 with a new data order is what this rerun did, and it collapsed
  again.  A guard needs a change at the rollback, such as a lower LR or a
  stabilising term.
- **A cheap test bed.**  From `epoch=124.ckpt`, the collapse came by epoch
  ~145 in both runs: ~14 minutes on 2 nodes, plus ~2 minutes of start-up.
  A candidate fix can be judged on whether it passes epoch ~150 intact.
  Two runs are a small sample, and the trigger of the spike is still
  unidentified (no gradient-norm or per-block logging).

## Stabilisation (2026-10-29, last two days of the Clariden allocation)

**Recipe against the published configurations.**  Checked in the released
code and configs, not from memory (I-JEPA `configs/in1k_vith14_ep300.yaml`,
`src/train.py`, `src/helper.py`; V-JEPA 2 `configs/train/vitl16/pretrain-256px-16f.yaml`;
fairseq data2vec 2.0 `config/v2/base_images_only_task.yaml`; DINOv2
`ssl_default_config.yaml`; papers 2301.08243, 2506.09985, 2202.03555,
2212.07525, 2302.05442).

| | ours (K = 16 `token`) | I-JEPA | V-JEPA 2 | data2vec 2.0 (ViT-B) |
|---|---|---|---|---|
| teacher momentum | 0.996 -> 1.0 cosine (0.9970 at epoch 140) | 0.996 -> 1.0 linear | 0.99925 constant | 0.9998 -> 0.99999 |
| targets | mean of LN(MLP out) of all 16 blocks, then LN | LN(last block) | LN(last block) | IN per layer, mean of top 10 of 12, LN |
| loss | smooth L1 | smooth L1 | L1 | L2 |
| AdamW betas / weight decay | 0.9, 0.95 / 0.05 | 0.9, 0.999 / 0.04 -> 0.4 cosine | 0.9, 0.999 / 0.04 | 0.9, 0.95 / 0.05 |
| gradient clipping | norm 1.0 | none | none | norm 4 |
| LR schedule | warm-up 40 epochs, cosine | warm-up 40, cosine | warm-up, constant, cool-down | warm-up, cosine |
| init | xavier | trunc. normal 0.02 (its sqrt(2l) rescale is overwritten by `init_model`) | trunc. normal 0.02, rescaled | |

- **The clearest difference is the teacher's speed when seed 8 collapsed.**
  At epoch 140 ours averages over ~340 steps (1.3 epochs); V-JEPA 2's over
  ~1,300 steps and data2vec's over >= 5,000.  data2vec 1.0 (section 6)
  names a too-low teacher momentum, a too-large LR and a too-short warm-up
  as causes of collapse.
- data2vec 1.0 normalises vision targets per token (parameter-free LN, as
  here) and averages the top 6 of 12 blocks; instance normalisation is for
  speech.  data2vec 2.0 uses instance normalisation per layer.
- Not tried here: QK-norm (ViT-22B; not in I-JEPA or V-JEPA 2, and no
  attention-logit growth has been seen), I-JEPA's initialisation (an
  earlier short test was negative; needs a full run), per-event target
  normalisation (the `batch` variant lost the gain), VICReg / SIGReg terms
  (objective changes with weights to tune).

**New options** (all off by default; the recipe of every earlier run is
unchanged):
- `WEIGHT_DECAY_END` (`--flash_weight_decay_end`): I-JEPA's cosine schedule
  from `WEIGHT_DECAY` to this value over training.  Set before every step
  from the global step, so a resumed run follows it.
- `LR_SCALE_ON_RESUME` (`--flash_lr_scale_on_resume`): multiplies the
  restored LR, the warm-up and cosine base rates and the cosine floor of a
  `RESUME_CHECKPOINT` run, which then continues on that multiple of the
  original schedule.  Refused without a resume.
- `LOG_GRAD_NORM=1` (`--flash_log_grad_norm`): logs the gradient norms
  before clipping (`grad_norm/total`, `/predictor`, `/encoder_other`,
  `/block01`..`/block16`) every logged step.
- `clariden/pretrain_finetune_flash.sbatch`: pretraining on 2 nodes, then
  the 20-epoch fine-tune of `last.ckpt` on the first node (1 x 4 GH200,
  global batch 1,024, `FT_SEED` 7, `FT_LAYER_DECAY` 0.85) and the test CSV,
  in one job.  None of the pretraining variables reach the fine-tune.
  `clariden/finetune_flash.sbatch` (`FT_NODES`) and
  `clariden/evaluate_flash.sbatch` now pin their job steps to explicit node
  and task counts; standalone they run as before.
- Tests: `tests/test_flash_jepa.py`
  (`test_weight_decay_schedule_lr_scale_on_resume_and_grad_norms`: the
  decay follows the cosine formula, a resumed run with scale 0.5 has half
  the LR of the uninterrupted run at every later step), plus a one-batch
  DDP run on dlnu's two GPUs with `LOG_GRAD_NORM=1 WEIGHT_DECAY_END=0.4`.
- Wrapper, end to end on Clariden (2-epoch pretraining, `FT_EPOCHS=1`,
  `MAX_BATCHES=5`, output under `/capstor/scratch/cscs/salonso/faser/test_wrapper/`).
  The first test (job 3789878) failed in the fine-tune with an empty
  dataset: its staging and training steps had each asked for one node and
  were placed on different nodes.  Every step of `finetune_flash.sbatch`
  and `evaluate_flash.sbatch` now names its host(s) (`--nodelist`, the first
  `FT_NODES` hosts of the job).  The second test (job 3789953, 7.5 min)
  pretrained on 2 nodes, fine-tuned on 1 node x 4 GPUs (256 per GPU) and
  wrote the CSV.  Its fine-tune `hparams.yaml` equals that of
  `finetune_flash_jepa_base_d2v16_s8_ld085` apart from the name, the
  checkpoint, the test's epochs and the three new (unused) options.

**Resume tests from seed 8's `epoch=124.ckpt`** (debug partition: at most
90 node-minutes per job, so 2 nodes x 45 min, reaching epoch ~185; one job
at a time).  Pass: from epoch 150 to the last complete epoch, validation
`target_std` >= 0.5 and `context_event_std` >= 0.3.  Decision rule, fixed
before the results: the first test that passes, in the order S1, S4, S3,
S5, alone becomes the recipe of three full seeds (7, 8, 9); if none passes,
S1 and S3 together.

| test | change | job | result |
|---|---|---|---|
| S1 | teacher momentum constant 0.99925 (V-JEPA 2) | 3789641 | **fail**: collapsed at epoch 138-140 |
| S4 | K = 8 (top half of the blocks, data2vec 1.0) | 3789642 | **fail**: collapsed at epoch 142-143 |
| S3 | LR x 0.5 from the resume, gradient norms logged | 3789771 | **fail**: unstable from epoch 140, `target_std` 0.27-0.58 |
| S5 | weight decay 0.04 -> 0.4 cosine (I-JEPA), gradient norms logged | 3789943 | **fail**: collapsed at epoch 140-143 |
| S3b | LR x 0.25 from the resume, gradient norms logged (added after S3) | 3789952 | **fail**: recovered from epochs 135-144, unstable again at 160-187 |

**S1: a slower teacher does not prevent the collapse.**  Validation values:

| epoch | 135 | 136 | 137 | 138 | 139 | 140 | 145 | 150 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val loss | 0.078 | 0.222 | 0.171 | 0.369 | 0.035 | 0.002 | 0.000 | 0.000 |
| `target_std` | 0.942 | 0.939 | 0.924 | 0.890 | 0.272 | 0.063 | 0.009 | 0.005 |
| `context_event_std` | 0.484 | 0.321 | 0.337 | 0.003 | 0.001 | 0.001 | 0.002 | 0.002 |

- The loss spike came at epoch 136, one epoch earlier than in the original
  run, and the collapse was complete by epoch 140, earlier than both
  unchanged runs (144-146).
- **The student collapsed first**: its event spread was 0.003 at epoch 138
  while the teacher's target spread was still 0.89.  The teacher followed
  within two epochs.  The unchanged rerun shows the same order (epoch 144:
  `context_event_std` 0.012, `target_std` 0.51).  So the collapse starts
  in the context encoder and predictor, not in a teacher that moves too
  fast.
- Cancelled after epoch 150 (19.6 min), once the verdict was fixed, so
  that S4 could start.

**S4: K = 8 does not prevent it either.**

| epoch | 135 | 139 | 140 | 141 | 142 | 143 | 145 | 150 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val loss | 0.098 | 0.108 | 0.157 | 0.235 | 0.241 | 0.007 | 0.000 | 0.000 |
| `target_std` | 0.981 | 0.979 | 0.978 | 0.928 | 0.714 | 0.010 | 0.001 | 0.000 |
| `context_event_std` | 0.540 | 0.497 | 0.390 | 0.386 | 0.004 | 0.003 | 0.001 | 0.000 |

- The switch to 8 target blocks raised the loss (0.079 to ~0.10, other
  targets) and the spreads, but the spike came at epoch 140 and the
  collapse was complete by 143, again student first (epoch 142).
- Four realisations from this state (original, rerun, S1, S4), with
  different data order, teacher speed and targets, all collapsed between
  epochs 138 and 146.  What they share is the state and the LR schedule
  (~1.3e-3 there).  Cancelled after epoch 153.

**S3: half the LR weakens and delays it, but does not prevent it.**  LR
6.96e-4 at epoch 125 (1.39e-3 originally), so the scaling took effect.

| epoch | 130 | 134 | 138 | 139 | 140 | 141 | 143 | 146 | 150 | 153 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val loss | 0.072 | 0.072 | 0.081 | 0.125 | 0.215 | 0.062 | 0.019 | 0.143 | 0.053 | 0.028 |
| `target_std` | 0.940 | 0.932 | 0.942 | 0.938 | 0.918 | 0.535 | 0.348 | 0.576 | 0.290 | 0.304 |
| `context_event_std` | 0.484 | 0.501 | 0.447 | 0.439 | 0.253 | 0.248 | 0.184 | 0.075 | 0.078 | 0.192 |
| grad norm, median | 0.026 | 0.195 | 0.224 | 0.408 | 1.22 | 0.81 | 0.19 | 0.44 | 38.2 | 1.07 |

- Not a complete collapse (the loss never reached 0.000 and the spreads
  oscillate), but below the pass line from epoch 141.  Cancelled after
  epoch 153.
- **Clipping at 1.0 is inactive in normal training**: the gradient norm
  before clipping is ~0.02-0.025 up to epoch 130, 40-50x below it.
- **The predictor goes first.**  Median gradient norms per module:

  | epoch | 125 | 131 | 132 | 134 | 137 | 138 | 139 | 140 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | predictor | 0.016 | 0.036 | 0.069 | 0.194 | 0.176 | 0.209 | 0.393 | 1.17 |
  | encoder block 1 | 0.0054 | 0.0064 | 0.0074 | 0.0076 | 0.0126 | 0.0271 | 0.0605 | 0.144 |
  | encoder block 8 | 0.0025 | 0.0031 | 0.0034 | 0.0050 | 0.0047 | 0.0049 | 0.0189 | 0.036 |
  | encoder block 16 | 0.0021 | 0.0027 | 0.0029 | 0.0054 | 0.0050 | 0.0051 | 0.0070 | 0.020 |

  The predictor's gradient grows ~10x over epochs 131-134 while the
  encoder's stays flat; the encoder follows from epoch 137, early blocks
  first (block 1 x5 by epoch 138).  The validation loss moves only from
  epoch 139.  So the gradient norm warns ~8 epochs ahead, and the
  instability starts in the predictor.

**S5: the I-JEPA weight-decay schedule does not prevent it** (decay 0.121
at epoch 125, 0.139 at 140).

| epoch | 130 | 134 | 137 | 138 | 140 | 141 | 143 | 150 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val loss | 0.081 | 0.083 | 0.116 | 0.123 | 0.312 | 0.006 | 0.000 | 0.000 |
| `target_std` | 0.951 | 0.950 | 0.951 | 0.874 | 0.827 | 0.139 | 0.012 | 0.001 |
| `context_event_std` | 0.504 | 0.520 | 0.422 | 0.387 | 0.023 | 0.010 | 0.005 | 0.001 |
| grad norm, predictor | 0.028 | 0.155 | 0.088 | 3.36 | 4.83 | 0.28 | 0.022 | 0.006 |

Same sequence: the predictor's gradient grows from epoch 131, the loss
spikes at 137-140, the student collapses at 140 and the teacher at 141.
Cancelled after epoch 152.

**S3b: a quarter of the LR survives the first episode but not the next.**
Ran to its time limit (epoch 187).

| epoch | 134 | 138 | 140 | 144 | 145 | 150 | 155 | 160 | 165 | 170 | 175 | 180 | 185 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val loss | 0.063 | 0.081 | 0.076 | 0.053 | 0.094 | 0.080 | 0.083 | 0.331 | 0.002 | 0.171 | 0.037 | 0.028 | 0.133 |
| `target_std` | 0.932 | 0.892 | 0.764 | 0.533 | 0.828 | 0.908 | 0.900 | 0.894 | 0.077 | 0.566 | 0.387 | 0.225 | 0.520 |
| `context_event_std` | 0.452 | 0.281 | 0.138 | 0.163 | 0.506 | 0.529 | 0.510 | 0.210 | 0.067 | 0.013 | 0.233 | 0.025 | 0.304 |
| grad norm, predictor | 0.041 | 1.31 | 6.75 | 1.75 | 0.25 | 0.23 | 0.21 | | | | | | |

- The predictor's gradient grew from epoch 132, as in every run, but
  the spreads recovered by epoch 145 and held to 159.  The predictor's
  gradient norm stayed ~15x above its level before the episode (0.2
  against 0.013).
- A second episode began at epoch 160 and was still oscillating at 187.
- **Conclusion of the resume tests.**  From this state none of the five
  changes keeps the run healthy: the teacher's speed (S1), the target
  layers (S4) and weight decay (S5) change nothing; lowering the LR delays
  and softens the episodes (S3) and makes the first recoverable (S3b) but
  not the later ones.  The instability starts in the predictor ~8 epochs
  before the loss shows it.  These tests start from a state trained 124
  epochs at the full LR; whether a run that never has the full LR reaches
  such a state is what a full run answers.

All five tests and the two earlier runs: `results/flash_audit_20261026/screens_s8_r124.{py,out}`.

**Full runs: K = 16 `token` at half the LR (`BLR=5e-5`), seeds 7, 8, 9
(submitted 2026-10-29 11:30).**  None of the tests passed, so the
fall-back applied.  The agreed fall-back was S1 + S3; S1 had made the
collapse slightly earlier, so on the user's go-ahead ("do whatever you
think is best") the runs use the lower LR alone, which is also the external
review's first stabilisation trial (`codex_review2_report.md`, section 6,
item 5).  Peak LR 8e-4; everything else as the K = 16 `token` runs (400
epochs, 40 warm-up, teacher 0.996 -> 1.0, weight decay 0.05), plus
gradient-norm logging.  Each job pretrains, fine-tunes `last.ckpt` (seed 7,
layer decay 0.85) and writes the test CSV.  Seeds 7 and 8 start from the
same initialisation as the earlier `token` seeds 7 (job 3760078) and 8
(job 3769470).

| seed | pretraining | fine-tune | job |
|---:|---|---|---|
| 7 | `pretrain_flash_jepa_base_d2v16_blr5e5` | `finetune_flash_jepa_base_d2v16_blr5e5_ld085` | 3790448 |
| 8 | `pretrain_flash_jepa_base_d2v16_blr5e5_s8` | `finetune_flash_jepa_base_d2v16_blr5e5_s8_ld085` | 3790449 |
| 9 | `pretrain_flash_jepa_base_d2v16_blr5e5_s9` | `finetune_flash_jepa_base_d2v16_blr5e5_s9_ld085` | 3790451 |

What to read when they finish: `jepa/target_std`, `jepa/context_event_std`
and `grad_norm/predictor` over training (a predictor gradient rising
~10x is the early warning seen in every resume test), then the test
metrics against the `token` seeds and the published MAE.  From
`/users/salonso/faserDL` on Clariden:

```bash
for s in 7 8 9; do suf=$([[ $s == 7 ]] && echo "" || echo "_s$s"); NAME=pretrain_flash_jepa_base_d2v16_blr5e5$suf SEED=$s TARGET_LAYERS=16 BLR=5e-5 LOG_GRAD_NORM=1 FT_NAME=finetune_flash_jepa_base_d2v16_blr5e5${suf}_ld085 sbatch --job-name=pt_ft_blr5e5_s$s clariden/pretrain_finetune_flash.sbatch; done
```

**Full runs at half the LR: results (2026-10-29).**  Half the LR does not
remove the episodes, and on average it fine-tunes worse.
- **All three jobs completed** (5 h 15 min to 5 h 28 min: pretraining 4 h
  34 min to 4 h 49 min, fine-tune 33-34 min, CSV ~5.5 min; every Slurm step
  COMPLETED).  Each fine-tune ran its 20 epochs from the epoch-399
  `last.ckpt` (target encoder, coverage 1.0, fine-tuning seed 7).  Each
  CSV has the 122,639 test events, `partial: false`.  The fetcher
  (`results/flash_audit_20261026/fetch_blr5e5.sh`) copied the CSVs, with
  sha256 checked, and the `metrics.csv` files to
  `/scratch4/salonso/faser/blr5e5/`.
- **Every run had an episode**, and the predictor's gradient warned first
  again.  An episode here means a validation epoch from epoch 10 with
  `target_std` < 0.5 or `context_event_std` < 0.3.  The warning is the
  first epoch where the predictor's largest gradient norm exceeds twice
  its median over epochs 40-149; in no run did this happen between
  epochs 40 and 200.

  At the first episode, the table gives the LR and the teacher momentum,
  with in brackets the epochs the teacher averages over, 1 / (1 -
  momentum) steps of 254 per epoch.  The last column is the end of
  training: val loss / `target_std` / `context_event_std`.

  | run | warning | first episode | episode epochs | LR | teacher momentum | end of training |
  |---|---:|---:|---:|---:|---:|---|
  | seed 7, BLR 1e-4 (earlier) | not logged | 324 | 76 | 1.8e-4 | 0.99966 (11.7) | 0.053 / 0.54 / 0.11 |
  | seed 8, BLR 1e-4 (earlier) | not logged | 137 | 11 | 1.3e-3 | 0.99706 (1.3) | 0.034 / 0.94 / 0.41 |
  | seed 7, BLR 5e-5 | 314 | 315 | 71 | 1.1e-4 | 0.99958 (9.4) | 0.108 / 0.79 / 0.04 |
  | seed 8, BLR 5e-5 | 215 | 223 | 43 | 3.9e-4 | 0.99837 (2.4) | 0.066 / 0.81 / 0.46 |
  | seed 9, BLR 5e-5 | 224 | 232 | 58 | 3.6e-4 | 0.99851 (2.6) | 0.005 / 0.53 / 0.47 |

  - Half the LR moved seed 8's first episode from epoch 137 to 223, but
    did not prevent it; seed 7's still came at ~315-324.  No single LR
    triggers it: the episodes began at LRs from 1.1e-4 to 1.3e-3.
  - The sequence is the same as in the resume tests.  For ~10 epochs the
    predictor's gradient creeps up (x2-6) and `context_event_std` slides
    (0.47 to 0.41).  Then the predictor's gradient jumps to 3-12, the
    loss spikes and the context encoder's event spread drops.  The encoder
    blocks follow, and in the worst epochs the gradient norm before
    clipping reaches 100-850 (clipped to 1.0).
  - Seed 9 ended in a near-trivial state: validation loss 0.005 from
    epoch ~245 to the end.
- **Test metrics** (fine-tuning seed 7; the BLR 1e-4 columns are the
  earlier `finetune_flash_jepa_base_d2v16_last_ld085` and
  `..._s8_ld085`):

  | | flash scratch | MAE base (pub) | s7 1e-4 | s8 1e-4 | s7 5e-5 | s8 5e-5 | s9 5e-5 | mean 1e-4 | mean 5e-5 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | flavour acc | 0.7846 | 0.8153 | 0.8231 | 0.8062 | 0.8202 | 0.7942 | 0.7584 | 0.8147 | 0.7909 |
  | AUC NC | 0.9159 | 0.9388 | 0.9443 | 0.9318 | 0.9427 | 0.9242 | 0.8932 | 0.9380 | 0.9200 |
  | AUC charm->had | 0.8381 | 0.8691 | 0.8768 | 0.8689 | 0.8766 | 0.8517 | 0.7999 | 0.8728 | 0.8427 |
  | sMAD E_vis | 0.2704 | 0.2144 | 0.1963 | 0.2278 | 0.1994 | 0.2498 | 0.3232 | 0.2121 | 0.2574 |
  | sMAD pT_miss | 0.6910 | 0.5573 | 0.5552 | 0.6104 | 0.5709 | 0.6525 | 0.7599 | 0.5828 | 0.6611 |
  | vertex median mm | 19.07 | 16.88 | 15.16 | 15.34 | 15.75 | 17.30 | 22.45 | 15.25 | 18.50 |

  Share of the published scratch -> MAE flavour-accuracy gap recovered:
  seed 7 107%, seed 8 70%, seed 9 21% (flash scratch 57%; the BLR 1e-4
  seeds 111% and 87%).  Seed 9 is worse than training from scratch on
  every metric.  All 15 metrics with paired bootstrap intervals:
  `results/flash_audit_20261026/metrics_flash_blr5e5.out`.
- **Paired event-bootstrap differences** (95% intervals cover test-set
  noise only; the pretraining seed matters far more, flavour accuracy
  spreading 0.062 across the three BLR 5e-5 seeds against 0.017 across
  the two BLR 1e-4 seeds):
  - mean(BLR 5e-5) - mean(BLR 1e-4): worse on all 15 metrics, flavour
    accuracy -0.0238 [-0.0247, -0.0227], `pT_miss` sMAD +0.078.
  - Same seed and initialisation: seed 7 -0.0030 [-0.0043, -0.0016] in
    flavour accuracy, seed 8 -0.0121 [-0.0137, -0.0105].
  - mean(BLR 5e-5) - flash scratch: flavour accuracy +0.0063 [0.0052,
    0.0076], so the pretraining keeps only ~20% of the gain over scratch
    that the BLR 1e-4 seeds give.
  - Seed 7 at BLR 5e-5 against the published MAE: better on 14 of 15
    metrics (flavour accuracy +0.0048 [0.0032, 0.0066]), worse on
    `pT_miss` (+0.014).  Seeds 8 and 9 are worse than MAE on most
    metrics (flavour accuracy -0.021 and -0.057).
- **When the first episode comes predicts the fine-tune.**
  - Both runs whose first episode came late (seed 7 at either LR, epochs
    315-324) fine-tune well, and seed 7's earlier pretraining fine-tuned
    the same from its epoch-299 snapshot as from `last.ckpt` (flavour
    0.8238 against 0.8231, `metrics_flash_ep124.out`).
  - The three whose first episode came mid-training (seed 8 at either LR,
    epochs 137 and 223, and seed 9 at 232) fine-tune clearly worse.
  - The fine-tune starts from the teacher (EMA) encoder, which at the late
    episodes averages over 9-12 epochs and at the mid-training ones over
    1.3-2.6.  So a likely explanation, not tested, is that a late episode
    stays mostly in the student, while earlier the teacher follows it.
    `target_std` at the end does not measure the damage: the earlier seed
    7 ends at 0.54 and fine-tunes well.
- **Conclusion.**  The LR sets when the instability arrives, not whether it
  does.  The instability starts in the predictor, and the fine-tune
  depends mainly on whether the teacher still has a pre-episode state.
  BLR 5e-5 is not a better recipe: mean flavour accuracy 0.7909 against
  0.8147 at BLR 1e-4.  Of the five K = 16 `token` pretrainings, two are
  clean at their end (seed 7 at both LRs) and three are damaged.

Curves: `results/flash_audit_20261026/curves_blr5e5.{py,out}`.  Test
metrics: `results/flash_audit_20261026/metrics_flash_blr5e5.{py,out}`.
Figure of all seven full pretrainings:
`results/flash_audit_20261026/pretraining_curves.png`
(`plot_pretraining_curves.py`).  It has one row per run (K = 1, the K = 16
seeds at both LRs, batch-standardised targets) and four columns: validation
loss, `target_std`, `context_event_std`, and the predictor gradient norm.
- Every run is disturbed at least once.  K = 1 has a small spike at
  epoch ~182, then its `context_event_std` drifts below 0.3 without an
  abrupt event.  The batch-standardised run has short drops of
  `context_event_std` at epochs ~183 and ~372; its `target_std` is 1 by
  construction.
- In every run `context_event_std` falls from ~0.87 to ~0.5 during the
  warm-up and then keeps sliding slowly until the episode.
- Training and validation loss agree within ~3% at every epoch of every
  run (augmentations on for training, off for validation), so the model
  is not memorising the training events.
On dlnu (the BLR 1e-4 curves copied first, see the docstring of
`curves_blr5e5.py`):

```bash
bash results/flash_audit_20261026/fetch_blr5e5.sh
python results/flash_audit_20261026/curves_blr5e5.py > results/flash_audit_20261026/curves_blr5e5.out
python results/flash_audit_20261026/metrics_flash_blr5e5.py > results/flash_audit_20261026/metrics_flash_blr5e5.out
```

**Fine-tunes of the pre-warning snapshots (submitted 2026-10-29 17:57).**
- **Rule, fixed before any of these results:** for each BLR 5e-5 run,
  take the last saved snapshot (every 25 epochs) before the first epoch
  whose largest predictor gradient norm exceeds twice its median over
  epochs 40-149.  That gives epoch 299 for seed 7 (warning at epoch 314),
  and epoch 199 for seeds 8 and 9 (warnings at 215 and 224).
- **Recipe:** the published one, as for `last.ckpt` (20 epochs, layer
  decay 0.85, fine-tuning seed 7, 1 x 4 GH200), then the test CSV.
- **What they test:** whether the episodes are what costs the fine-tune,
  and what the three seeds give when healthy.
- **Comparison:** the earlier runs' healthy snapshots gave flavour
  accuracy 0.8238 (seed 7, epoch 299) and 0.8191 (seed 8, epoch 124).

| seed | snapshot | fine-tune | first jobs, cancelled (fine-tune / evaluate) | debug job |
|---:|---|---|---|---:|
| 7 | `pretrain_flash_jepa_base_d2v16_blr5e5/snapshots/epoch=299.ckpt` | `finetune_flash_jepa_base_d2v16_blr5e5_e299_ld085` | 3794337 / 3794338 | 3801599 |
| 8 | `pretrain_flash_jepa_base_d2v16_blr5e5_s8/snapshots/epoch=199.ckpt` | `finetune_flash_jepa_base_d2v16_blr5e5_s8_e199_ld085` | 3794340 / 3794341 | 3801600 |
| 9 | `pretrain_flash_jepa_base_d2v16_blr5e5_s9/snapshots/epoch=199.ckpt` | `finetune_flash_jepa_base_d2v16_blr5e5_s9_e199_ld085` | 3794342 / 3794343 | 3802005 |

The code on Clariden matched dlnu's (sha256 of the sbatch files,
`finetune_flash.sh`, `evaluation/create_csv_v7_ft.py`,
`model/lightning_model_flash.py`, `utils/args.py`).  From
`/users/salonso/faserDL` on Clariden:

```bash
c=/capstor/scratch/cscs/salonso/faser/checkpoint; sub() { fid=$(LOAD_CHECKPOINT=$2 NAME=$1 SEED=7 LAYER_DECAY=0.85 sbatch --parsable clariden/finetune_flash.sbatch) && eid=$(CHECKPOINT_DIR=$c/$1 sbatch --parsable --dependency=afterok:$fid clariden/evaluate_flash.sbatch) && echo "$1 finetune=$fid evaluate=$eid"; }
sub finetune_flash_jepa_base_d2v16_blr5e5_e299_ld085 $c/pretrain_flash_jepa_base_d2v16_blr5e5/snapshots/epoch=299.ckpt
sub finetune_flash_jepa_base_d2v16_blr5e5_s8_e199_ld085 $c/pretrain_flash_jepa_base_d2v16_blr5e5_s8/snapshots/epoch=199.ckpt
sub finetune_flash_jepa_base_d2v16_blr5e5_s9_e199_ld085 $c/pretrain_flash_jepa_base_d2v16_blr5e5_s9/snapshots/epoch=199.ckpt
```

The first start estimate was 08:00 the next morning.  The fine-tunes take
~35 min and the evaluations ~6, so their limits were lowered from 4 h and
1 h so that they can backfill:

```bash
for j in 3794337 3794340 3794342; do scontrol update job=$j TimeLimit=01:00:00; done; for j in 3794338 3794341 3794343; do scontrol update job=$j TimeLimit=00:20:00; done
```

**Moved to the debug partition (2026-10-30 11:08).**
- **Why:** at 11:05 the next day none had started.  The reason was
  `QOSGrpNodeLimit`: the normal partition's QoS (`normalp`, 400 nodes
  shared by all users) was full; account a0178 itself had nothing
  running.  `debug-qos` has no group cap (4 nodes and 90 node-minutes per
  job, 1 running and 2 submitted per user).
- **Change:** the six pending jobs were cancelled, before any had started.
  A new one-job script, `clariden/finetune_evaluate_flash.sbatch`, runs
  `finetune_flash.sbatch` and then `evaluate_flash.sbatch` on the same
  node (the sequence of the second half of
  `pretrain_finetune_flash.sbatch`).  It refuses to start if the run or
  its CSV exists.
- **Same recipe, names and outputs** as the cancelled jobs.  Seed 7 is job
  3801599, seed 8 job 3801600.  Seed 9 is submitted when seed 7 ends (one
  running and two submitted per user in debug).

From `/users/salonso/faserDL` on Clariden (the script copied there first,
sha256 checked):

```bash
c=/capstor/scratch/cscs/salonso/faser/checkpoint
LOAD_CHECKPOINT=$c/pretrain_flash_jepa_base_d2v16_blr5e5/snapshots/epoch=299.ckpt NAME=finetune_flash_jepa_base_d2v16_blr5e5_e299_ld085 SEED=7 LAYER_DECAY=0.85 sbatch --partition=debug --job-name=ftev_s7_e299 clariden/finetune_evaluate_flash.sbatch
LOAD_CHECKPOINT=$c/pretrain_flash_jepa_base_d2v16_blr5e5_s8/snapshots/epoch=199.ckpt NAME=finetune_flash_jepa_base_d2v16_blr5e5_s8_e199_ld085 SEED=7 LAYER_DECAY=0.85 sbatch --partition=debug --job-name=ftev_s8_e199 clariden/finetune_evaluate_flash.sbatch
LOAD_CHECKPOINT=$c/pretrain_flash_jepa_base_d2v16_blr5e5_s9/snapshots/epoch=199.ckpt NAME=finetune_flash_jepa_base_d2v16_blr5e5_s9_e199_ld085 SEED=7 LAYER_DECAY=0.85 sbatch --partition=debug --job-name=ftev_s9_e199 clariden/finetune_evaluate_flash.sbatch
```

Results are copied by `bash results/flash_audit_20261026/fetch_blr5e5_snapshots.sh`
(it waits on the three job names) and scored by
`python results/flash_audit_20261026/metrics_flash_blr5e5_snapshots.py > results/flash_audit_20261026/metrics_flash_blr5e5_snapshots.out`.

**Pre-warning snapshots: results (2026-10-30).**  The episodes are what cost
the fine-tune.  Before them, every seed gives the same, good encoder.
- **All three jobs completed** (39 min 52 s to 40 min 15 s each, fine-tune
  plus CSV).  Each fine-tune loaded the intended snapshot (epochs 299, 199,
  199; target encoder, coverage 1.0), ran 20 epochs with seed 7 and
  evaluated the 122,639 test events (`partial: false`).  The CSVs were
  copied with sha256 checked.
- **Flavour accuracy:**

  | | seed 7 | seed 8 | seed 9 | mean | max - min |
  |---|---:|---:|---:|---:|---:|
  | BLR 5e-5, `last.ckpt` (epoch 399) | 0.8202 | 0.7942 | 0.7584 | 0.7909 | 0.0618 |
  | BLR 5e-5, pre-warning snapshot (299 / 199 / 199) | 0.8200 | 0.8195 | 0.8208 | 0.8201 | 0.0013 |
  | BLR 1e-4, `last.ckpt` | 0.8231 | 0.8062 | | 0.8147 | |
  | BLR 1e-4, healthy snapshot (299 / 124, dlnu) | 0.8238 | 0.8191 | | 0.8214 | |

  Published MAE base: 0.8153.  Flash scratch: 0.7846.
- **Paired event-bootstrap differences** (95% intervals; test-set noise
  only):
  - Snapshot minus `last.ckpt`, flavour accuracy:
    - seed 7: -0.0001 [-0.0013, 0.0009], the same (its episode came late);
    - seed 8: +0.0253 [0.0236, 0.0269];
    - seed 9: +0.0624 [0.0606, 0.0647].
    Seeds 8 and 9 also improve on every other metric, e.g. seed 9's
    `pT_miss` sMAD -0.199 and vertex resolution -7.1 mm.
  - Mean of the three BLR 5e-5 snapshots against:

    | | flavour accuracy | AUC NC | AUC charm->had | sMAD E_vis | sMAD pT_miss | vertex (mm) |
    |---|---|---|---|---|---|---|
    | published MAE base | +0.0048 [0.0034, 0.0064] | +0.0033 | +0.0081 | -0.0131 | +0.0058 [0.0026, 0.0093] | -1.52 |
    | flash scratch | +0.0355 [0.0341, 0.0370] | +0.0263 | +0.0391 | -0.069 | -0.128 | -3.72 |
    | K = 1 | +0.0050 [0.0039, 0.0061] | +0.0040 | +0.0062 | -0.021 | -0.015 | +0.67 |
    | mean of the BLR 1e-4 `last.ckpt` | +0.0054 [0.0046, 0.0063] | +0.0041 | +0.0044 | -0.011 | -0.020 | +0.11 |
    | mean of the BLR 1e-4 healthy snapshots | -0.0013 [-0.0022, -0.0005] | -0.0004 | -0.0016 | -0.0002 | +0.0042 | +0.71 |

  - **Against the published MAE**, the mean of the three snapshots is
    better on 14 of 15 metrics and worse only on `pT_miss` (+0.006).
- **Conclusions:**
  - A healthy K = 16 `token` encoder is reproducible across seeds
    (flavour accuracy within 0.0013).  It beats the published MAE base on
    14 of 15 metrics and recovers 106-108% of the scratch -> MAE gap in
    flavour accuracy.
  - At BLR 5e-5 the healthy encoder is about as good as at 1e-4 (-0.0013),
    so halving the LR neither helps nor costs anything when the run stays
    healthy.
  - What costs the fine-tune is a mid-training episode.  The aim is now a
    run that never has one; the predictor QK-norm runs below test that.
    Failing that, the fall-back is the last snapshot before the predictor
    gradient warning, a rule fixed before these results.

**Predictor diagnostic (2026-10-30, dlnu): attention-logit growth in the
predictor's last block.**  `results/flash_audit_20261026/predictor_diagnostic.py`
(method in its docstring; same 2,048 validation events as
`collapse_check.py`, the same masks for every checkpoint, bf16 as in
training).  Its loss on the first batch matches `FlashJEPA.forward`'s.
Snapshots:
- seed 9 at BLR 5e-5: epochs 199, 224 (its warning epoch) and 249 (in
  the episode);
- the BLR 1e-4 seeds at epoch 124: seed 8, 13 epochs before its episode,
  and seed 7, 200 epochs before;
- the K = 1 run at epoch 399 (no abrupt episode).

- **No positional shortcut before the episode.**  Before every episode,
  the predictor relies on the event's own context (smooth-L1 loss on the
  target tokens):

  | | s9 e199 | s9 e224 | s9 e249 | s8 e124 | s7 e124 | K = 1 e399 |
  |---|---:|---:|---:|---:|---:|---:|
  | own context (training loss) | 0.087 | 0.086 | 0.325 | 0.080 | 0.082 | 0.060 |
  | another event's context | 0.474 | 0.463 | 0.351 | 0.460 | 0.472 | 0.230 |
  | no context | 0.509 | 0.498 | 0.349 | 0.416 | 0.466 | 0.448 |
  | best guess from position alone | 0.321 | 0.311 | 0.279 | 0.315 | 0.323 | 0.167 |

  Only inside the episode (s9 e249) does the predictor stop using the
  event: another event's context then does as well, and all of them do
  worse than the position-only guess.  That is a consequence of the
  episode, not its start.
- **The last predictor block's queries and keys grow large.**  RMS of q
  and k (mean over heads):

  | block | s9 e199 | s9 e224 | s9 e249 | s8 e124 | s7 e124 | K = 1 e399 |
  |---|---:|---:|---:|---:|---:|---:|
  | predictor 1-4, q (range) | 0.8-1.1 | 0.8-1.2 | 1.2-1.6 | 0.7-1.0 | 0.8-1.0 | 0.8-182 |
  | predictor 5, q / k | 1.2 / 3.1 | 2.3 / 4.3 | 9.9 / 10 | 1.2 / 2.8 | 1.2 / 2.8 | 325 / 20 |
  | predictor 6, q / k | 13 / 16 | 209 / 43 | 1444 / 1048 | 8.2 / 17 | 2.6 / 15 | 98 / 27 |
  | context encoder, every block, q and k | ~0.8-1.2 | ~0.8-1.2 | 1.0-2.2 | | | |

  - **Seed 9, block 6.**  Between epochs 199 and 224 its q RMS grows 16x
    and k 3x.  The logit spread softmax sees (per query, largest logit
    minus the row mean) jumps from 9 to 7,500, and to 1.5M at 249.
  - **What block 6 attends to.**  Its mask-token queries put 97-100% of
    their attention on the context tokens.  Their large logits mainly
    separate context keys from mask-token keys.
  - **The encoder is normal** (spread 3-6) until the episode.
  - **Same ordering in every run.**  The block-6 q RMS ranks the runs by
    how soon the episode came: seed 7 at epoch 124 has 2.6 (episode 200
    epochs later), seed 8 at 124 has 8.2 (13 epochs later), seed 9 has 13
    at 199 (33 epochs before) and 209 at its warning epoch.
  - **K = 1 has the same growth by epoch 399, in blocks 4-6, without an
    abrupt episode.**  So the growth comes from this predictor, whatever
    the target, and how it ends varies.
- **bf16 precision.**  In block 6 the attention computed from bf16 q and k
  (as in training) differs from float32 by a total-variation distance of
  0.13-0.15 for seed 9 and seed 8 (0.08 for seed 7 at 124).  In their
  other blocks it is below 0.02.  For K = 1 it is 0.17 and 0.09 in blocks
  4 and 5, where its q grew most.  Part of that attention is rounding
  noise.
- **Gradient at the warning epoch** (seed 9, e199 -> e224; training loss,
  batches of 128): block 6 attention 0.018 -> 0.126, block 5 MLP 0.006 ->
  0.093.  The rest of the predictor and the context encoder are unchanged
  (encoder 0.111 -> 0.111).  At e249 the encoder's gradient is 27 and the
  predictor's embedding 2.5.
- **Interpretation.**  This is the attention-logit growth instability:
  query and key norms grow until the attention saturates and the
  training diverges.  ViT-22B (Dehghani et al. 2023, arXiv:2302.05442)
  and Wortsman et al. 2023 (arXiv:2309.14322) report it.  Both fix it
  with QK-layernorm, which in Wortsman et al. also removes the
  instability's dependence on the LR.
  - It fits every observation here: the predictor's gradient warns first,
    the encoder only follows, and a lower LR only delays it.
  - The resume tests also fit: from epoch 124, when block 6 had already
    grown, changing the teacher, the target layers or the weight decay
    could not undo it.
  - A plausible driver, not tested: separating context tokens from mask
    tokens is easy (the mask token is one shared vector).  Adam's
    normalised steps keep enlarging that separation at a rate set by the
    LR, and a weight decay of 0.05 does not hold it back.

Outputs: `results/flash_audit_20261026/predictor_diagnostic_{s9,blr1e4_e124,k1}.out`.
Seed 9's snapshots were copied from Clariden to
`/scratch4/salonso/faser/checkpoints_final/pretrain_flash_jepa_base_d2v16_blr5e5_s9/snapshots/`
(sha256 checked), with the run's `hparams.yaml` next to them.  On dlnu:

```bash
D='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*'; M=events_v7.0_500_npz/metadata_stats.pkl; c=/scratch4/salonso/faser/checkpoints_final; s=$c/pretrain_flash_jepa_base_d2v16_blr5e5_s9/snapshots
CUDA_VISIBLE_DEVICES=1 python results/flash_audit_20261026/predictor_diagnostic.py "$D" $M e199=$s/epoch=199.ckpt e224=$s/epoch=224.ckpt e249=$s/epoch=249.ckpt > results/flash_audit_20261026/predictor_diagnostic_s9.out
CUDA_VISIBLE_DEVICES=0 python results/flash_audit_20261026/predictor_diagnostic.py "$D" $M s8_e124=$c/pretrain_flash_jepa_base_d2v16_s8/snapshots/epoch=124.ckpt s7_e124=$c/pretrain_flash_jepa_base_d2v16_clariden/snapshots/epoch=124.ckpt > results/flash_audit_20261026/predictor_diagnostic_blr1e4_e124.out
CUDA_VISIBLE_DEVICES=0 TARGET_LAYERS=1 python results/flash_audit_20261026/predictor_diagnostic.py "$D" $M k1_e399=checkpoints_final/pretrain_flash_jepa_base/loss_total_val/last.ckpt > results/flash_audit_20261026/predictor_diagnostic_k1.out
```

**Predictor QK-norm (opt-in, `PREDICTOR_QK_NORM=1` /
`--flash_predictor_qk_norm`; 2026-10-30).**
- **What:** each predictor attention layer normalises every head's queries
  and keys (LayerNorm over the head dimension, eps 1e-6, learned gain and
  bias) before the dot product, as `timm`'s `qk_norm` (ViT-22B).  This
  bounds the logits whatever the scale of the query/key weights.
- **Where:** `model/flash_vit.py` (`Attention` / `Block`, `qk_norm=False`),
  `model/flash_jepa.py` (`FlashPredictor`, `FlashJEPA(predictor_qk_norm)`),
  `utils/args.py`, `train/pretrain_flash.py`, `pretrain_flash.sh`.
- **What stays the same:**
  - Off by default, with no new parameters, so every earlier run and
    checkpoint is unchanged.
  - The encoder never has it, so fine-tuning, evaluation and the
    checkpoints they load are unaffected.
  - The new norm parameters are 1-D, so the optimiser exempts them from
    weight decay, like the other norms.
- **Tests** (`tests/test_flash_jepa.py`):
  - `test_predictor_qk_norm_bounds_the_logits_and_leaves_the_encoder_unchanged`:
    - off adds no keys, and on adds exactly the predictor's q/k norms;
    - the encoder keys are the same with and without;
    - multiplying every predictor query and key by 100 leaves the
      predictions unchanged with QK-norm and changes them without;
    - the norms receive finite gradients;
    - the option reaches `build_flash_jepa`.
  - `test_predictor_qk_norm_runs_under_bf16_flash_attention` (GPU): the
    bf16 FlashAttention loss is within 2% of float32, and all gradients
    are finite.
  - The launcher test checks the new line and default.
  - All pass: `test_flash_jepa.py` 30 passed, 3 skipped (spconv); every
    other test file importing the changed modules 51 passed, 7 skipped.
- **Real-data smoke test on dlnu:** one DDP training and validation step
  on both GPUs with real events, K = 16, gradient logging and QK-norm.
  Exit 0; `flash_predictor_qk_norm: True` in the saved settings.

  ```bash
  d=/scratch4/salonso/faser/flash_dryrun/qkn; NAME=smoke_qk_norm FAST_DEV_RUN=1 TARGET_LAYERS=16 LOG_GRAD_NORM=1 PREDICTOR_QK_NORM=1 BATCH_SIZE=256 NUM_WORKERS=16 SAVE_DIR=$d CHECKPOINT_PATH=$d/ckpt DATASET_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*' METADATA_PATH=/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz/metadata_stats.pkl bash pretrain_flash.sh
  ```

**Full runs with predictor QK-norm: K = 16 `token`, BLR 1e-4, seeds 7, 8,
9.**  The recipe is the original K = 16 `token` one (BLR 1e-4, 400 epochs,
teacher 0.996 -> 1.0, weight decay 0.05) plus `PREDICTOR_QK_NORM=1` and
`LOG_GRAD_NORM=1`.  Each job pretrains, fine-tunes `last.ckpt` (seed 7,
layer decay 0.85) and writes the test CSV
(`clariden/pretrain_finetune_flash.sbatch`).  Seeds 7 and 8 start from the
same initialisation as the earlier `token` seeds, apart from the new norms.

Submission: `results/flash_audit_20261026/sync_and_submit_qkn.sh`, log
`/scratch4/salonso/faser/sync_and_submit_qkn.log`.
- It waits until the queued debug job `ftev_s9_e199` has started (its
  fine-tune has already imported the model code).
- It copies the six changed files under temporary names and moves them
  into place, then checks that all 22 code files on Clariden match
  dlnu's sha256.
- It refuses if any new run name already exists (an existing pretraining
  name would be resumed), then submits.  The code is synced before the
  submission, so no job can start on the old code.  From
  `/users/salonso/faserDL` on Clariden:

```bash
for s in 7 8 9; do suf=$([[ $s == 7 ]] && echo '' || echo _s$s); NAME=pretrain_flash_jepa_base_d2v16_qkn$suf SEED=$s TARGET_LAYERS=16 BLR=1e-4 PREDICTOR_QK_NORM=1 LOG_GRAD_NORM=1 FT_NAME=finetune_flash_jepa_base_d2v16_qkn${suf}_ld085 sbatch --job-name=pt_ft_qkn_s$s clariden/pretrain_finetune_flash.sbatch; done
```

| seed | pretraining | fine-tune | job |
|---:|---|---|---|
| 7 | `pretrain_flash_jepa_base_d2v16_qkn` | `finetune_flash_jepa_base_d2v16_qkn_ld085` | 3802469 |
| 8 | `pretrain_flash_jepa_base_d2v16_qkn_s8` | `finetune_flash_jepa_base_d2v16_qkn_s8_ld085` | 3802470 (cancelled; debug chain, see below) |
| 9 | `pretrain_flash_jepa_base_d2v16_qkn_s9` | `finetune_flash_jepa_base_d2v16_qkn_s9_ld085` | 3802471 |

- **The first attempt (12:34) changed nothing.**  The home file system
  `/users` (VAST) was at its 50 GB quota, and the first copy failed
  before anything was moved into place.  The evaluations also write their
  CSV there: seed 8's snapshot CSV, written at 12:28, had filled it.
- **Fix:** on the user's go-ahead, `pip cache purge` on Clariden removed
  `~/.cache/pip` (1,219 files, 3.4 GB).  What remains in `/users` is
  mostly three 2025 container images (`~/spconv`, `~/mink-engine`,
  `~/podman`, ~41 GB together), left as they are.
- **Second attempt, submitted 12:47:** the code was verified first (22
  files equal to dlnu).  Logs:
  `/scratch4/salonso/faser/sync_and_submit_qkn{.attempt1,}.log`.  The
  first start estimate was 13:19.
- **Results:** `bash results/flash_audit_20261026/fetch_qkn.sh` copies
  them to `/scratch4/salonso/faser/qkn/` and `results/`.

**Faster start (2026-10-30 15:10).**  The three jobs kept waiting on the
normal partition's shared 400-node cap (`QOSGrpNodeLimit` on QoS
`normalp`; the day before, normal jobs had not started at all).
- **Seeds 7 and 9 (jobs 3802469, 3802471)** are now also eligible for the
  `preemptable` partition.  It accepts QoS `normal`, has normal's priority
  tier (20) and no 400-node cap.  Only `highprio` jobs (CSCS staff) can
  preempt it, after a 240 s grace period.  The jobs are set to requeue on
  preemption (the cluster default is no requeue).
  - A requeued job restarts `pretrain_finetune_flash.sbatch`, whose
    pretraining resumes from `last.ckpt`.  Lightning writes that file
    atomically through a temporary file, every epoch.
  - A preemption during the fine-tune would make the wrapper refuse to
    restart (the fine-tuning run exists).  The fine-tune and evaluation
    would then be rerun alone with `finetune_evaluate_flash.sbatch`.
- **Seed 8:** its normal job (3802470) was cancelled before it started.
  It runs instead as a chain of debug jobs.
  - Debug allows 90 node-minutes per job and one running job per user,
    so the pretraining runs as 2-node, 45-min segments of
    `clariden/pretrain_flash.sbatch` (about 60 epochs each), each resuming
    from `last.ckpt`.
  - Then `clariden/finetune_evaluate_flash.sbatch` runs in debug.
  - Expected: about 7 segments plus 40 min, roughly 6 h.  Seed 8 collapsed
    at epoch 137 at this LR without QK-norm, so its window 137-200 is
    passed after 3-4 segments.
- **Driver:** `results/flash_audit_20261026/debug_chain_qkn_s8.sh`, run
  detached on dlnu (log `/scratch4/salonso/faser/debug_chain_qkn_s8.log`).
  - It submits one segment at a time.  After each one it copies every log
    version's `metrics.csv` to `/scratch4/salonso/faser/qkn/progress/` and
    prints the last epochs' spreads, the episode epochs so far and the
    largest predictor gradient norm.
  - It stops if a segment ends other than by its time limit or makes no
    progress.  At the end it fetches the results (CSVs sha256 checked).
- **Segment 1** (job 3804768) started at 15:15 on 2 nodes (512 events per
  GPU, effective batch 4,096).  Its `hparams.yaml` shows
  `flash_predictor_qk_norm: true`, `flash_target_layers: 16`, `blr: 1e-4`
  and seed 8.
- **Fetcher:** `fetch_qkn.sh` now handles seeds 7 and 9 and copies every
  log version (a requeued run resumes into a new one).  It runs detached
  (log `/scratch4/salonso/faser/fetch_qkn.log`).

From `/users/salonso/faserDL` on Clariden:

```bash
for j in 3802469 3802471; do scontrol update jobid=$j Partition=normal,preemptable Requeue=1; done; scancel 3802470
```

On dlnu:

```bash
setsid nohup bash results/flash_audit_20261026/debug_chain_qkn_s8.sh > /scratch4/salonso/faser/debug_chain_qkn_s8.log 2>&1 < /dev/null &
setsid nohup bash results/flash_audit_20261026/fetch_qkn.sh > /scratch4/salonso/faser/fetch_qkn.log 2>&1 < /dev/null &
```

What to read when they finish:
- `grad_norm/predictor` over training, against the warnings at epochs
  215-314 of the BLR 5e-5 runs;
- validation `target_std` and `context_event_std`, using the same episode
  definition as before;
- `predictor_diagnostic.py` (`PREDICTOR_QK_NORM=1`) on the snapshots;
- the test metrics against the `token` seeds and the published MAE.

**Seed 8 with predictor QK-norm: results (2026-10-31).**
- **Run:** the debug chain finished the 400 epochs in 7 segments, jobs
  3804768, 3805294, 3805777, 3808412, 3808915, 3809175 and 3809471
  (2026-10-30 15:15 to 2026-10-31 00:26).  Segments 1-6 ended at their
  45-min limit with 57-60 new epochs each; segment 7 completed (48).  The
  3.5 h queue wait came before segment 3 (debug was busy in the
  afternoon).
  - The fine-tune and evaluation (job 3809639, debug, 40 min) loaded the
    target encoder of `last.ckpt` (epoch 399, step 101,600).
  - Accounting: `/scratch4/salonso/faser/qkn/sacct_qkn_s8_chain.txt`.
  - Pretraining metrics: one file per segment,
    `/scratch4/salonso/faser/qkn/pretrain_flash_jepa_base_d2v16_qkn_s8_version_{0..6}_metrics.csv`.
    Together they give epochs 0-399 once each.
- **No episode, no warning.**
  - After epoch 10, the lowest validation `target_std` is 0.909 and the
    lowest `context_event_std` 0.396, both in the last epochs.
  - The predictor's gradient norm (largest per epoch) never exceeded twice
    its median over epochs 40-149 (0.035).  Its largest value after
    epoch 40 is 0.056 (epoch 95).  The median per 50 epochs falls
    steadily, from 0.058 (epochs 0-49) to 0.017 (350-399).
  - The same seed without QK-norm, same initialisation apart from the new
    norms:
    - its `context_event_std` rose to 0.53 at epoch 131;
    - its episode started at epoch 137 (`target_std` 0.78,
      `context_event_std` 0.29) and reached `target_std` 0.058 at 146;
    - it recovered to 0.94 by epoch 250.
    With QK-norm, epochs 131-150 have `target_std` 0.940-0.943 and
    `context_event_std` 0.48-0.49.
  - The validation loss is higher than without QK-norm before the latter's
    episode: 0.088 against 0.077 at epoch 131.  It ends at 0.0724.
    Without QK-norm the loss after the episode (0.034 at epoch 399) is not
    comparable, because the targets are much narrower.
- **Test metrics:** fine-tuning seed 7, layer decay 0.85, the same 122,639
  events.  Paired event-bootstrap 95% intervals (200 draws).
  "Significant" means the interval excludes 0.

| | flavour acc | better (sig.) | not sig. | worse (sig.) |
|---|---:|---:|---:|---:|
| QK-norm seed 8 (`last.ckpt`) | **0.8243** | | | |
| same seed without QK-norm, `last.ckpt` (episode at 137-153) | 0.8062 | 14 | 1 | 0 |
| same seed without QK-norm, healthy epoch-124 snapshot | 0.8191 | 9 | 5 | 1 (\|p_lep\| CC) |
| mean of the 5 pre-warning snapshots | 0.8206 | 10 | 4 | 1 (\|p_lep\| CC) |
| seed 7 without QK-norm, `last.ckpt` (best earlier full run) | 0.8231 | 7 | 6 | 2 (E_vis, \|p_lep\| CC) |
| published MAE base | 0.8153 | 13 | 2 | 0 |
| K = 1 (layer decay 0.85) | 0.8152 | 14 | 1 | 0 |
| flash scratch | 0.7846 | 15 | 0 | 0 |

  - The comparison columns count the 15 metrics where QK-norm seed 8 is
    better, not significantly different, or worse.
  - The 5 pre-warning snapshots are: seed 7 epoch 299 and seed 8 epoch 124
    at BLR 1e-4; seed 7 epoch 299 and seeds 8 and 9 epoch 199 at BLR 5e-5.
  - Against the published MAE base, the two non-significant metrics are
    charm->e (+0.0155, [-0.0013, +0.0270]) and the pT_miss sMAD (-0.0009,
    [-0.0048, +0.0033]).  Both point in QK-norm's favour.
  - It recovers 100-149% of the published scratch -> MAE gap per metric
    (flavour accuracy 113%).
  - Over the 5 pre-warning snapshots, the gain is +0.0037 in flavour
    accuracy ([+0.0028, +0.0047]).  The 3 BLR 5e-5 snapshots spread by
    0.0013 (0.8195-0.8208).
  - **Caveat:** this is one pretraining seed and one fine-tuning seed.
    Seeds 7 and 9 (jobs 3802469 and 3802471) decide whether the fix holds.

Output: `results/flash_audit_20261026/metrics_flash_qkn_s8.out`.  On dlnu:

```bash
python results/flash_audit_20261026/metrics_flash_qkn_s8.py > results/flash_audit_20261026/metrics_flash_qkn_s8.out
```

**Predictor diagnostic, seed 8 with QK-norm (epochs 124, 149, 399).**
QK-norm removed the runaway, but not all growth of the last blocks'
attention logits.
- **The predictor still uses the event.**  Target-token loss with the
  event's own context, then with another event's context, then with none:
  - epoch 124: 0.086, 0.462, 0.507;
  - epoch 149: 0.088, 0.451, 0.501;
  - epoch 399: 0.072, 0.408, 0.515.
  The position-only reference is 0.28-0.32.
- **The logit spread of blocks 5 and 6 grows late in training.**  Mean
  over queries of (largest logit minus the row mean), mean over heads:

| | block 5 | block 6 | bf16 TV, block 5 / 6 |
|---|---:|---:|---|
| QK-norm seed 8, epoch 124 | 9.5 | 13.3 | 0.022 / 0.221 |
| QK-norm seed 8, epoch 149 | 10.7 | 17.6 | 0.040 / 0.329 |
| QK-norm seed 8, epoch 399 | 117 | 294 | 0.205 / 0.150 |
| seed 8 without QK-norm, epoch 124 (episode at 137) | 9.3 | 8.7 | 0.006 / 0.132 |
| seed 9 BLR 5e-5, epoch 224 (warning) | 12.4 | 7,498 | 0.019 / 0.134 |
| seed 9 BLR 5e-5, epoch 249 (episode at 232) | 145 | 1,480,428 | 0.140 / 0.082 |

- **Where the growth comes from:** the norms' learned gains and biases.
  These are 1-D, so they have no weight decay.  Mean absolute values in
  blocks 5 and 6:
  - query gain: 1.5 and 4.0 at epoch 124, 15 and 20 at epoch 399;
  - query bias: 0.2 and 2.1 at epoch 124, 11 and 17 at epoch 399;
  - key gain: 1.4 at epoch 124, 5.2 and 3.8 at epoch 399.

  The key bias grows most (29 -> 62 in block 6), but it adds the same
  amount to every logit of a query's row, so it does not change the
  attention.
- **Reading:**
  - Without QK-norm, the query and key weights and their inputs grew
    together.  In seed 9 the block-6 spread grew from 9 to 7,498 within
    25 epochs, ahead of its episode.
  - With QK-norm, only these gains and biases can grow.  Adam moves them
    by at most about the learning rate per step, so they grow steadily
    rather than explosively.  By epoch 399 the spread is 294: still 25 to
    5,000 times smaller than seed 9's around its episode.
  - The bf16 rounding error of blocks 5 and 6 at epoch 399 (TV 0.15-0.21)
    is as large as before the earlier episodes.
  - So QK-norm appears to have cut the feedback loop, and seed 8 had no
    episode in 400 epochs.  The last two blocks still drift towards hard attention in
    the second half of training.  That is a residual risk for longer
    runs, and seeds 7 and 9 will show whether it matters.
  - Not tested: QK-norm without the learned affine, or with weight decay
    on its gain, would remove this path.
- The "What" line above holds for the query/key weights only.  The
  learned gain and bias of the norm still scale the logits.

Output: `results/flash_audit_20261026/predictor_diagnostic_qkn_s8.out`.
The three snapshots were copied from Clariden to
`/scratch4/salonso/faser/checkpoints_final/pretrain_flash_jepa_base_d2v16_qkn_s8/snapshots/`
(sha256 checked), with the run's `hparams.yaml`.  On dlnu:

```bash
D='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*'; M=events_v7.0_500_npz/metadata_stats.pkl; s=/scratch4/salonso/faser/checkpoints_final/pretrain_flash_jepa_base_d2v16_qkn_s8/snapshots
CUDA_VISIBLE_DEVICES=0 PREDICTOR_QK_NORM=1 python results/flash_audit_20261026/predictor_diagnostic.py "$D" $M e124=$s/epoch=124.ckpt e149=$s/epoch=149.ckpt e399=$s/epoch=399.ckpt > results/flash_audit_20261026/predictor_diagnostic_qkn_s8.out
```

**Seeds 7 and 9 with QK-norm: pretraining (2026-10-31).**
- **Scheduling:** both ran on `preemptable` and were preempted four times
  (`/scratch4/salonso/faser/qkn/sacct.txt`), each time requeued and
  resumed from `last.ckpt`:
  - 2026-10-30 16:26-16:30;
  - 2026-10-31 11:30-13:36;
  - 13:44, after 0.5 min (seed 9) and 4.6 min (seed 7);
  - seed 9 13:50-17:10, seed 7 13:56-17:10.

  The pretraining logs have three versions (epochs 0-2, 3-171, 172-399).
- **Both finished the 400 epochs** (`last.ckpt` at 16:55 and 16:48), with
  no episode and no predictor-gradient warning:

| seed | lowest `target_std` | lowest `context_event_std` | predictor grad: median 40-149, largest after 40 |
|---:|---:|---:|---|
| 7 | 0.914 | 0.438 | 0.033, 0.058 (epoch 55) |
| 8 | 0.909 | 0.396 | 0.035, 0.056 (epoch 95) |
| 9 | 0.894 | 0.419 | 0.032, 0.052 (epoch 54) |

  The lowest values are taken from epoch 10 on.  Without QK-norm, at the
  same LR both earlier seeds (7 and 8) went through an episode.  At half
  the LR, all three seeds did.
- **The fine-tunes were cut:** the 17:10 preemption hit both jobs during
  the fine-tune, at epochs 7 (seed 7) and 8 (seed 9) of 20.
  - On requeue at 17:27, the wrapper refused to restart, because the
    fine-tuning runs existed (exit 2, as expected).
  - The Slurm output files were overwritten by that requeue (Slurm
    truncates on restart).
  - The partial runs `finetune_flash_jepa_base_d2v16_qkn{,_s9}_ld085` are
    left as they are.
- **Reruns (17:36):** same settings, new names
  `finetune_flash_jepa_base_d2v16_qkn{,_s9}_ld085_rerun`.
  - Driver: `results/flash_audit_20261026/rerun_ft_qkn.sh` (log
    `/scratch4/salonso/faser/rerun_ft_qkn.log`).
  - Seed 7 is job 3815152 (debug).  Seed 9 is job 3815153 (normal), moved
    to debug if still pending when seed 7 ends.

  On dlnu:

  ```bash
  setsid nohup bash results/flash_audit_20261026/rerun_ft_qkn.sh > /scratch4/salonso/faser/rerun_ft_qkn.log 2>&1 < /dev/null &
  ```

  - Seed 7: job 3815152, debug, 17:36-18:15.
  - Seed 9: its normal job (3815153) was still pending when seed 7 ended.
    It was cancelled and rerun as job 3815749, debug, 18:16-18:57.
  - Both loaded `last.ckpt` (epoch 399, target encoder) and completed the
    20 epochs.  The CSVs were copied to `results/` (sha256 checked).
  - Accounting: `/scratch4/salonso/faser/qkn/sacct_rerun_ft.txt`.

**Predictor QK-norm, three seeds: test metrics (2026-10-31).**
Fine-tuning seed 7, layer decay 0.85, the same 122,639 events.  Seeds 7 and
9 are the `_rerun` fine-tunes.  Paired event-bootstrap 95% intervals (200
draws).

| | flavour acc | vertex median (mm) |
|---|---:|---:|
| QK-norm seed 7 | 0.8256 | 15.98 |
| QK-norm seed 8 | 0.8243 | 14.31 |
| QK-norm seed 9 | 0.8250 | 16.03 |
| **mean of the 3 QK-norm seeds** | **0.8250** | 15.44 |
| without QK-norm, BLR 1e-4, seeds 7 and 8 (`last.ckpt`) | 0.8147 | 15.25 |
| without QK-norm, BLR 5e-5, seeds 7-9 (`last.ckpt`) | 0.7909 | 18.50 |
| mean of the 5 pre-warning snapshots | 0.8206 | 15.07 |
| published MAE base | 0.8153 | 16.88 |
| K = 1 (layer decay 0.85) | 0.8152 | 14.68 |
| flash scratch | 0.7846 | 19.07 |

The mean of the three QK-norm seeds against each reference, over the 15
metrics:

| mean of the 3 QK-norm seeds vs | better (sig.) | not sig. | worse (sig.) |
|---|---:|---:|---:|
| published MAE base | 15 | 0 | 0 |
| flash scratch | 15 | 0 | 0 |
| without QK-norm, BLR 5e-5, 3 seeds | 15 | 0 | 0 |
| without QK-norm, BLR 1e-4, 2 seeds | 13 | 1 | 1 (vertex, +0.19 mm) |
| K = 1 | 13 | 1 | 1 (vertex, +0.76 mm) |
| the 5 pre-warning snapshots | 9 | 5 | 1 (vertex, +0.37 mm) |

- Against the published MAE base: flavour accuracy +0.0097 ([+0.0082,
  +0.0112]), vertex median -1.44 mm.  It recovers 102-141% of the
  published scratch -> MAE gap per metric (flavour accuracy 113%).
- Against the 5 healthy pre-warning snapshots: flavour accuracy +0.0043
  ([+0.0036, +0.0051]).  The 5 non-significant metrics are the ντ->e and
  ντ->μ AUCs and the three charm AUCs.
- **Same seed, same LR, with and without QK-norm:**
  - seed 8: 14 better, 1 not significant, 0 worse.  This seed collapsed
    at epochs 137-153 without QK-norm.
  - seed 7: 8 better, 6 not significant, 1 worse (vertex, +0.82 mm).
    Without QK-norm, this seed's episode began at epoch 324 and lasted to
    the end, yet its `last.ckpt` fine-tune still reached 0.8231.
- **Reproducibility across pretraining seeds** (max - min):
  - flavour accuracy: 0.0013 with QK-norm, against 0.0047 over the 5
    snapshots and 0.0618 over the 3 BLR 5e-5 runs;
  - the vertex median varies most: seed 8 is 1.7 mm better than seeds 7
    and 9.
- **The vertex "worse" entries are within fine-tuning noise.**  Repeat
  fine-tunes of one encoder move the vertex median by up to ~1 mm (see
  "Clariden controls": 0.78 and 1.01 mm with another fine-tuning seed).
  The event bootstrap understates that about 10x.  So the +0.19 to
  +0.82 mm differences above are not established.  The mean's -1.44 mm
  against the MAE exceeds that range.  Seeds 7 and 9 alone are only 0.85
  and 0.90 mm better than the MAE, within it; seed 8 is 2.57 mm better.
- **Conclusion:**
  - With predictor QK-norm, the K = 16 `token` recipe at BLR 1e-4 had no
    episode in 3 of 3 seeds.  Without it, every seed tried had one: at
    BLR 1e-4, seed 7 (epochs 324-399) and seed 8 (137-153); at BLR 5e-5,
    seed 7 (315-399), seed 8 (223-276) and seed 9 (232-294).
  - Its fine-tunes are the best flash-JEPA results so far, and the most
    consistent across seeds.  They beat the published MAE base on all 15
    metrics.
  - The vertex median does not measurably improve on the healthy encoders
    without QK-norm or on K = 1.  Its differences there are below the
    fine-tuning noise.
  - The residual growth of the last predictor blocks' logits (above)
    remains for longer runs.

Output: `results/flash_audit_20261026/metrics_flash_qkn.out`.  On dlnu:

```bash
python results/flash_audit_20261026/metrics_flash_qkn.py > results/flash_audit_20261026/metrics_flash_qkn.out
```

**Snapshot fine-tunes of the QK-norm runs (submitted 2026-10-31 21:02, last
evening of the allocation).**
- **Questions:**
  - Does the recipe need the last 100 epochs?
  - Did the late growth of the predictor's attention logits hurt the
    encoder?
  - How does the downstream score grow with pretraining length?
- **Runs:**
  - epoch 299 of seeds 7, 8 and 9;
  - epochs 99 and 199 of seed 8.

  Same settings as the `last.ckpt` fine-tunes (fine-tuning seed 7, layer
  decay 0.85).  Names: `finetune_flash_jepa_base_d2v16_qkn{,_s8,_s9}_e299_ld085`
  and `finetune_flash_jepa_base_d2v16_qkn_s8_e{199,099}_ld085`.
- **Driver:** `results/flash_audit_20261026/snapshot_ft_qkn.sh` (log
  `/scratch4/salonso/faser/snapshot_ft_qkn.log`).
  - All five went to `preemptable` without requeue, as jobs 3819022,
    3819023, 3819026, 3819028 and 3819030.  All started at once.
  - Any that does not complete is rerun in debug with the suffix `_r2`.
  - All five completed at 21:42-21:43, none preempted.  Each loaded the
    target encoder of the intended epoch.
- **Second round (22:08):** epochs 99 and 199 of seeds 7 and 9 (jobs
  3820318, 3820319, 3820325, 3820329, `preemptable`).  The
  length curve then has all three seeds.
  - The driver now takes the run list as arguments, and `TAG` names its
    accounting file.
  - Fixed: its accounting query listed only every other job.  `printf`
    with two `%s` consumed two names per cycle.  Round 1's
    `/scratch4/salonso/faser/qkn/sacct_snapshot_ft.txt` was rewritten
    with all five jobs.

  On dlnu:

  ```bash
  setsid nohup bash results/flash_audit_20261026/snapshot_ft_qkn.sh > /scratch4/salonso/faser/snapshot_ft_qkn.log 2>&1 < /dev/null &
  TAG=_2 setsid nohup bash results/flash_audit_20261026/snapshot_ft_qkn.sh qkn_e199 qkn_s9_e199 qkn_e099 qkn_s9_e099 > /scratch4/salonso/faser/snapshot_ft_qkn_2.log 2>&1 < /dev/null &
  python results/flash_audit_20261026/metrics_flash_qkn_snapshots.py > results/flash_audit_20261026/metrics_flash_qkn_snapshots.out
  python results/flash_audit_20261026/plot_qkn_epochs.py
  ```

- **All four round-2 jobs completed** at 22:47-22:49, none preempted.
  Accounting: `/scratch4/salonso/faser/qkn/sacct_snapshot_ft_2.txt`.

**Snapshot fine-tunes: results.**  12 fine-tunes: 3 seeds x epochs 99, 199,
299 and 399 (`last.ckpt`).  Fine-tuning seed 7, layer decay 0.85, the
same 122,639 events, paired event-bootstrap 95% intervals (200 draws).
Plot: `results/flash_audit_20261026/qkn_epochs.png`.

Mean of the three seeds:

| pretraining epochs | flavour acc | AUC NC | E_vis sMAD | pT_miss sMAD | \|p_lep\| CC sMAD | vertex (mm) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.8160 | 0.9390 | 0.2090 | 0.5643 | 0.3642 | 15.15 |
| 200 | 0.8224 | 0.9437 | 0.1980 | 0.5549 | 0.3511 | 14.72 |
| 300 | 0.8248 | 0.9454 | 0.1981 | 0.5546 | 0.3523 | 15.30 |
| 400 | 0.8250 | 0.9451 | 0.1979 | 0.5532 | 0.3498 | 15.44 |
| published MAE base | 0.8153 | 0.9388 | 0.2144 | 0.5573 | 0.3857 | 16.88 |

Each snapshot epoch against epoch 399 (mean of the 3 seeds; counts over
the 15 metrics):

| mean of the 3 seeds | better (sig.) | not sig. | worse (sig.) | flavour acc |
|---|---:|---:|---:|---|
| epoch 299 - epoch 399 | 2 (ντ->e AUC; vertex -0.14 mm) | 12 | 1 (\|p_lep\| CC +0.0024) | -0.0001 [-0.0007, +0.0004] |
| epoch 199 - epoch 399 | 1 (vertex -0.72 mm) | 9 | 5 (νe, νμ, ντ->had, NC AUCs; flavour) | -0.0026 [-0.0034, -0.0018] |
| epoch 99 - epoch 399 | 1 (vertex -0.29 mm) | 0 | 14 | -0.0090 [-0.0099, -0.0081] |

Each snapshot epoch against the published MAE base:

| mean of the 3 seeds vs MAE | better (sig.) | not sig. | worse (sig.) |
|---|---:|---:|---:|
| epoch 99 | 10 | 4 (νμ, NC, charm->e AUCs; flavour acc) | 1 (pT_miss +0.0070) |
| epoch 199 | 14 | 1 (pT_miss) | 0 |
| epoch 299 | 14 | 1 (pT_miss) | 0 |
| epoch 399 | 15 | 0 | 0 |

- **The last 100 epochs add nothing measurable.**  Epoch 299 equals
  epoch 399 in flavour accuracy and on 12 of 15 metrics.  The vertex
  differences are below the ~1 mm fine-tuning noise (see "Clariden
  controls").
  - Per seed, epoch 299 minus 399 in flavour accuracy is 0.0000, +0.0007
    and -0.0012.  The regressions move in opposite directions for
    different seeds: seed 7's E_vis and |p_lep| are worse at 299, seed
    8's better.
- **The late growth of the predictor's attention logits** (diagnostic
  above, epochs 149-399) **did not hurt the encoder:** epoch 399 is no
  worse than 299.
- **Classification keeps improving until epoch ~300.**  At 200 epochs it
  is 0.0026 lower in flavour accuracy.  The energy regressions (E_vis,
  pT_miss, |p_jet|, |p_lep|) are already at their final level by 200.
- **Against the MAE:**
  - after 100 epochs the run is already better on 10 metrics, but worse on
    pT_miss;
  - pT_miss is the last metric to cross: significant only at 400 epochs
    (-0.0041 [-0.0074, -0.0008]).
- **Pretraining seeds agree at every length:** flavour accuracy spreads by
  0.0003-0.0017.
- **Caveat: these are snapshots of 400-epoch runs.**  At epoch 299 the
  learning rate has not finished its cosine decay, and the teacher EMA has
  not reached 1.0.  A run scheduled for 300 epochs anneals both fully.  It
  may match these snapshots or do slightly better, but that is not tested.
  The recipe stays at 400 epochs.  300 epochs (25% cheaper) is a
  well-supported option to test first when compute is short.

Output: `results/flash_audit_20261026/metrics_flash_qkn_snapshots.{out,csv}`.

**After the allocation (2026-11-02).**  Clariden now lists account `a0178`
at QoS `stop`, so no more jobs run there.
- **Checkpoints copied to dlnu,** into
  `/scratch4/salonso/faser/checkpoints_final/`.  All 66 files match
  Clariden's sha256 (list: `qkn_checkpoints.sha256` there).
  - The three QK-norm pretrainings: `last.ckpt` and the 16 snapshots of
    each.
  - Their completed fine-tunes: `finetune_flash_jepa_base_d2v16_qkn_s8_ld085`
    and the `_rerun` ones of seeds 7 and 9.  The originals cut by
    preemption were not copied.
  - Everything else was mirrored later that day (see "Clariden mirror"
    below).
  - A first attempt listed the cut fine-tunes.  It was stopped during seed
    8's pretraining and restarted with the list fixed
    (`/scratch4/salonso/faser/fetch_qkn_checkpoints.attempt1.log`).
- **300-epoch run on dlnu (started 10:25).**  The recipe with `EPOCHS=300`,
  seed 7: warm-up 40 epochs, cosine over 260, teacher EMA 0.996 -> 1.0 by
  epoch 300 (76,200 steps).  Then the same fine-tune (layer decay 0.85,
  fine-tuning seed 7, 2 x 512) and the test CSV.
  - Names: `pretrain_flash_jepa_base_d2v16_qkn_ep300_dlnu` and
    `finetune_flash_jepa_base_d2v16_qkn_ep300_ld085_dlnu`.
  - 4-6 min per epoch: another user's jobs keep the load average at 60-100
    of the 128 threads, and the data loader is CPU-bound.  So 21-28 h of
    pretraining, then about 2 h of fine-tune and CSV.
  - `score_qkn_ep300.sh` waits for the run to end, then writes
    `results/flash_audit_20261026/metrics_flash_qkn_ep300.out`
    (`metrics_flash_qkn_ep300.py`; tested with seed 7's epoch-299 CSV in
    place of the new one).
  - The comparison is with the 400-epoch runs' mean and seed spread, and
    with their epoch-299 snapshots.
  - It is not a replicate of seed 7's Clariden run.  Two ranks instead of
    eight give other data orders and masks, at the same effective batch
    (2 x 2,048).  The earlier dlnu/Clariden fine-tune replicate differed
    little on classification and by up to ~0.004 sMAD on the regressions
    (see "Fine-tune on dlnu instead").

  On dlnu:

  ```bash
  setsid nohup bash results/flash_audit_20261026/fetch_qkn_checkpoints.sh > /scratch4/salonso/faser/fetch_qkn_checkpoints.log 2>&1 < /dev/null &
  setsid nohup bash results/flash_audit_20261026/dlnu_qkn_ep300.sh > /scratch4/salonso/faser/dlnu_qkn_ep300.log 2>&1 < /dev/null &
  setsid nohup bash results/flash_audit_20261026/score_qkn_ep300.sh > /scratch4/salonso/faser/score_qkn_ep300.log 2>&1 < /dev/null &
  ```

- **Fine-tuning seed 8 of the QK-norm encoders (queued 2026-11-02 10:48).**
  `dlnu_qkn_ft8.sh` waits for `dlnu_qkn_ep300.sh` to end.  Then it
  fine-tunes the three QK-norm `last.ckpt` encoders with fine-tuning seed 8
  (otherwise their Clariden seed-7 recipe, as 2 x 512), writes each test CSV
  and runs `metrics_flash_qkn_ft8.py`.
  - Names: `finetune_flash_jepa_base_d2v16_qkn{,_s8,_s9}_ld085_ft8_dlnu`.
    About 2 h each, so the scores come about 6-7 h after the 300-epoch run
    ends.  Output: `results/flash_audit_20261026/metrics_flash_qkn_ft8.out`.
  - **Questions:** how far does a repeat fine-tune move the vertex median
    on these encoders?  Are the QK-norm encoders' vertex differences from
    the MAE, K = 1, the pre-warning snapshots and the runs without QK-norm
    larger than that?
  - **Each new pair differs in both seed and platform.**  That is still a
    repeat fine-tune of one encoder.  The platform alone, at the same
    fine-tuning seed, gave vertex medians 1.05 mm (seed 8 without QK-norm)
    and 0.26 mm (batch-standardised targets) lower on dlnu than on
    Clariden.
  - **The scoring:**
    1. each repeat minus the encoder's Clariden seed-7 fine-tune, with
       event-bootstrap intervals;
    2. the pooled repeat-fine-tune SD per metric, over the three QK-norm
       encoders and the two without QK-norm that already have repeats
       (6 degrees of freedom);
    3. the QK-norm mean over its 6 fine-tunes against the MAE, K = 1, the
       5 pre-warning snapshots and the two BLR 1e-4 `last.ckpt` encoders
       without QK-norm.  Each has two standard errors: from fine-tuning
       noise, and from the spread of the encoder means.

    Tested with the QK-norm epoch-299 CSVs in place of the new ones.  From
    the existing repeats alone, the vertex median's repeat-fine-tune SD
    without QK-norm is 0.90 mm.

  On dlnu:

  ```bash
  setsid nohup bash results/flash_audit_20261026/dlnu_qkn_ft8.sh > /scratch4/salonso/faser/dlnu_qkn_ft8.log 2>&1 < /dev/null &
  ```

- **Clariden mirror (2026-11-02 14:23).**  Capstor scratch can be purged,
  so `fetch_clariden_checkpoints.sh` copies all of Clariden's
  `/capstor/scratch/cscs/salonso/faser/checkpoint/` to
  `/scratch4/salonso/faser/checkpoints_clariden/`, in the same layout.
  That is 53 runs, 354 files and 171 GB, including the BLR 5e-5 runs, the
  snapshot fine-tunes, the `s8_r124` screens and the fine-tunes cut by
  preemption.
  - It is a tree of its own, not merged into `checkpoints_final`.  Some
    names there are other runs: dlnu's own
    `pretrain_flash_jepa_base_d2v16` is the realisation that stopped at
    epoch 135, not Clariden's 400-epoch seed 7.
  - Files identical to the QK-norm copies in `checkpoints_final` (same
    size and modification time) are hard links (`rsync --link-dest`), so
    they take no extra space.
  - Every file's sha256 is checked against Clariden's (list:
    `/scratch4/salonso/faser/checkpoints_clariden.sha256`).  Done at
    15:02: all 354 match.
  - 63 files are hard links.  Seed 8's QK-norm epochs 124, 149 and 399 were
    in `checkpoints_final` before the copy above, with other modification
    times, so they were copied again (1.6 GB).  The mirror takes 140 GB of
    new space.
  - Clariden's training logs and Slurm output (`logs/` and `slurm/`, 381
    files, 0.5 GB) are in `/scratch4/salonso/faser/clariden_scratch/`.  A
    checksum dry run (`rsync -c -n`) against Clariden listed no
    differences.

  On dlnu:

  ```bash
  setsid nohup bash results/flash_audit_20261026/fetch_clariden_checkpoints.sh > /scratch4/salonso/faser/fetch_clariden_checkpoints.log 2>&1 < /dev/null &
  mkdir -p /scratch4/salonso/faser/clariden_scratch && cd /scratch4/salonso/faser/clariden_scratch && rsync -a --ignore-existing clariden:/capstor/scratch/cscs/salonso/faser/logs clariden:/capstor/scratch/cscs/salonso/faser/slurm ./
  rsync -a -c -n --itemize-changes clariden:/capstor/scratch/cscs/salonso/faser/logs clariden:/capstor/scratch/cscs/salonso/faser/slurm ./
  ```

**300 epochs and repeat fine-tunes: results (2026-11-03).**  All four new
CSVs have the 122,639 test events, the expected split (1,042,418 / 61,318 /
122,639, same test inventory sha256), `partial: false` and
`completed_epoch` 19.  Their metadata names the intended encoder and
fine-tuning seed.
- **Timing:** the 300-epoch pretraining ran 2026-11-02 10:25 to 2026-11-03
  09:46 (4.7 min per epoch).  Each fine-tune plus CSV then took 1 h 32 min to
  2 h 05 min.
- **Health:** it had no episode.
  - The gradient norm's largest step after warm-up was 0.068 (epoch 52),
    against 0.22 during epochs 10-39.  The per-epoch maxima had a median of
    0.034 and fell to ~0.02 by the end.
  - The teacher EMA reached 0.99996 by epoch 280 and 1.0 at epoch 299.

*300 epochs against 400* (fine-tuning seed 7; the 400-epoch fine-tunes ran
on Clariden).  z = (ep300 - mean) / (SD x sqrt(1 + 1/3)), where SD is the
three 400-epoch seeds' SD.  That SD includes fine-tuning noise but has
only 2 degrees of freedom.

| metric | ep300 (dlnu) | e399 mean | e399 min-max | diff | z |
|---|---|---|---|---|---|
| AUC nue | 0.9788 | 0.9790 | 0.9787-0.9793 | -0.0002 | -0.7 |
| AUC numu | 0.9524 | 0.9525 | 0.9523-0.9528 | -0.0001 | -0.3 |
| AUC nutau->e | 0.9238 | 0.9222 | 0.9216-0.9229 | +0.0016 | +2.1 |
| AUC nutau->mu | 0.8475 | 0.8488 | 0.8462-0.8516 | -0.0013 | -0.4 |
| AUC nutau->had | 0.9432 | 0.9450 | 0.9442-0.9454 | -0.0018 | -2.3 |
| AUC NC | 0.9453 | 0.9452 | 0.9449-0.9454 | +0.0001 | +0.5 |
| flavour acc | 0.8238 | 0.8250 | 0.8243-0.8256 | -0.0012 | -1.6 |
| AUC charm->e | 0.8221 | 0.8237 | 0.8197-0.8307 | -0.0016 | -0.2 |
| AUC charm->mu | 0.9044 | 0.9031 | 0.8981-0.9095 | +0.0013 | +0.2 |
| AUC charm->had | 0.8812 | 0.8789 | 0.8768-0.8806 | +0.0023 | +1.0 |
| sMAD E_vis | 0.1939 | 0.1979 | 0.1915-0.2013 | -0.0040 | -0.6 |
| sMAD pT_miss | 0.5584 | 0.5532 | 0.5503-0.5564 | +0.0052 | +1.5 |
| sMAD \|p_jet\| | 0.1504 | 0.1503 | 0.1479-0.1520 | +0.0001 | +0.1 |
| sMAD \|p_lep\| CC | 0.3423 | 0.3499 | 0.3363-0.3574 | -0.0076 | -0.6 |
| vertex median mm | 14.69 | 15.44 | 14.31-16.03 | -0.75 | -0.7 |

- **The 300-epoch run is indistinguishable from the 400-epoch runs.**  No
  metric has |z| above 2.3, and the two near 2 have opposite signs.  It is
  outside the 400-epoch min-max on 5 metrics, by at most 0.0020.
- **Against the published MAE:** better on 14 of 15 (event-bootstrap
  intervals excluding 0).  That includes the vertex median, 2.19 mm lower,
  but see the fine-tuning noise below.  pT_miss is a tie (+0.0011
  [-0.0030, +0.0048]).
- **It is one seed, on another platform:** two ranks give other data
  orders and masks than Clariden's eight.  dlnu fine-tunes have given
  0.26-1.05 mm lower vertex medians at the same seed.

Output: `results/flash_audit_20261026/metrics_flash_qkn_ep300.out`.

*Repeat fine-tunes.*  Each QK-norm encoder now has its Clariden fine-tune
(fine-tuning seed 7) and a dlnu one (seed 8).

| encoder | vertex median, repeat - FT7 (mm) | flavour acc, repeat - FT7 |
|---|---|---|
| QK-norm s7 (FT8 dlnu) | -0.23 | +0.0003 |
| QK-norm s8 (FT8 dlnu) | +0.34 | -0.0013 |
| QK-norm s9 (FT8 dlnu) | -1.19 | -0.0023 |
| 1e-4 s7 without QK-norm (FT8 Clariden) | +0.78 | -0.0001 |
| 1e-4 s8 without QK-norm (FT8 Clariden) | +1.01 | +0.0001 |
| 1e-4 s8 without QK-norm (FT7 dlnu) | -1.05 | +0.0000 |

- **Pooled repeat-fine-tune SD** over the 5 encoders (6 degrees of
  freedom):

  | metric | SD |
  |---|---|
  | vertex median | 0.73 mm (0.52 with QK-norm, 0.90 without) |
  | flavour accuracy | 0.0008 |
  | AUC charm->e / charm->mu / charm->had | 0.0029 / 0.0036 / 0.0017 |
  | sMAD E_vis / pT_miss / \|p_jet\| / \|p_lep\| CC | 0.0030 / 0.0039 / 0.0012 / 0.0085 |
  | flavour AUCs | 0.0002-0.0013 |

- **The event bootstrap roughly captures the classification noise, but not
  the vertex and regression noise.**
  - For classification, the 95% half-width of a repeat difference is
    within a factor of 2 of what this SD implies.  Flavour accuracy:
    ~0.0012 against 1.96 x sqrt(2) x 0.0008 = 0.0022.
  - For the vertex median it is ~0.1 mm against 2.0 mm, and for
    |p_lep| CC ~0.003 against 0.024.
  - So single-run vertex and |p_lep| differences of up to ~2 mm and ~0.02
    are not established, even when their event-bootstrap intervals
    exclude 0.
- **The QK-norm group (3 encoders x 2 fine-tunes)** against the other
  groups.  Each cell is the difference, with the difference / SE in
  brackets; the SE includes the spread of the encoder means.

  | QK-norm minus | flavour acc | sMAD pT_miss | sMAD \|p_lep\| CC | vertex median mm |
  |---|---|---|---|---|
  | MAE base (pub) | +0.0091 (8.9) | -0.0028 (-0.7) | -0.0368 (-4.2) | -1.62 (-1.9) |
  | K = 1 | +0.0093 (9.1) | -0.0238 (-5.9) | -0.0441 (-5.0) | +0.58 (+0.7) |
  | 5 pre-warning snapshots | +0.0038 (3.6) | -0.0070 (-1.7) | -0.0054 (-1.9) | +0.19 (+0.4) |
  | 1e-4 `last.ckpt` without QK-norm (2 encoders) | +0.0098 (1.2) | -0.0291 (-1.1) | -0.0146 (-1.0) | -0.18 (-0.4) |

  - **Against the MAE:** better by more than 4 SE on 13 of 15 metrics.
    pT_miss is a tie, and the vertex is better by 1.9 SE.
  - **Against K = 1:** better on 13 of 15.  AUC charm->e (+0.2 SE) and the
    vertex (+0.7 SE) are ties.
  - **Against the 5 pre-warning snapshots:** better beyond 2 SE only on
    AUC nue, numu, nutau->had and NC, flavour accuracy and |p_jet|.  The
    rest are within 2 SE.
  - **Against the 2 encoders without QK-norm:** the encoder spread is too
    large to resolve (all about 1 SE).  Seed 8 had an episode and seed 7
    did not.
- **Vertex conclusion:** the QK-norm encoders' vertex median (15.26 mm,
  mean of 6 fine-tunes) is not measurably different from K = 1, the
  pre-warning snapshots or the encoders without QK-norm.  It is 1.6 mm
  better than the MAE, at about 2 SE.
- **Platform:** the mean of FT8 dlnu - FT7 over the QK-norm encoders is
  -0.36 mm [-0.42, -0.29] on the vertex median.  It mixes the seed and the
  platform.
  - The same-seed dlnu - Clariden shifts were -0.26 and -1.05 mm.  So
    dlnu may give vertex medians a few tenths of a mm lower.  The
    300-epoch run's -0.75 mm fits that.
  - That is within the repeat SD, and is not established.  If real, it
    inflates the pooled SD slightly, since 4 of its 6 degrees of freedom
    compare platforms.

Output: `results/flash_audit_20261026/metrics_flash_qkn_ft8.out`.

**What the encoder knows before it sees a label (2026-11-04).**  Frozen
probes and 2D latent maps of the best encoder: K = 16 `token` targets with
predictor QK-norm, pretraining seed 7.  It is shown untrained (torch seed 7),
at its epoch-24 and epoch-99 snapshots, and at `last.ckpt` (epoch 399, the
encoder its fine-tunes used).  The same analysis of the published MAE
follows ("The same for the published MAE").
- **Features:** the target (EMA) encoder's tokens, mean-pooled per event,
  with no masking.  Two kinds, from one pass:
  - `last`: the final output, which is what FlashViT pools;
  - `k16`: the data2vec target.
- **Events:** 100,000 training-split events train the probes, 10,000
  validation-split events choose the regularisation, and all 122,639 test
  events are scored with the fine-tune metrics.  The fine-tuned column
  reproduces the seed-7 fine-tune's CSV scores exactly (0.8256, 15.98 mm),
  which checks the metric code.
- **Linear probes:** logistic regression for flavour and charm.  Ridge
  regression for the momenta and vertex, with log(1 + |p|) and the unit
  direction as targets, and lepton = visible - jet as in the fine-tune.
- **kNN:** the 20 nearest training events by cosine similarity.

Test metrics, `last` features (sMAD and vertex: lower is better).  The last
column is the supervised flash fine-tune from scratch (no pretraining).

| metric | linear, untrained | linear, e24 | linear, e99 | linear, e399 | kNN, e399 | fine-tuned e399 | scratch fine-tune |
|---|---|---|---|---|---|---|---|
| flavour acc | 0.594 | 0.667 | 0.748 | **0.767** | 0.728 | 0.826 | 0.785 |
| AUC nue | 0.777 | 0.908 | 0.951 | 0.953 | 0.912 | 0.979 | 0.969 |
| AUC numu | 0.741 | 0.830 | 0.907 | 0.921 | 0.892 | 0.953 | 0.931 |
| AUC NC | 0.697 | 0.789 | 0.887 | 0.907 | 0.859 | 0.945 | 0.916 |
| AUC nutau->had | 0.755 | 0.827 | 0.880 | 0.890 | 0.806 | 0.945 | 0.925 |
| AUC nutau->e | 0.705 | 0.808 | 0.860 | 0.860 | 0.688 | 0.922 | 0.906 |
| AUC nutau->mu | 0.641 | 0.682 | 0.741 | 0.761 | 0.604 | 0.852 | 0.826 |
| AUC charm->e | 0.574 | 0.611 | 0.663 | 0.649 | 0.513 | 0.831 | 0.791 |
| AUC charm->mu | 0.591 | 0.663 | 0.705 | 0.751 | 0.530 | 0.898 | 0.872 |
| AUC charm->had | 0.648 | 0.702 | 0.728 | 0.724 | 0.624 | 0.879 | 0.838 |
| sMAD E_vis | 0.768 | 0.530 | 0.442 | 0.425 | 0.527 | 0.192 | 0.270 |
| sMAD \|p_jet\| | 0.649 | 0.420 | 0.356 | 0.373 | 0.440 | 0.148 | 0.185 |
| sMAD \|p_lep\| CC | 0.880 | 0.820 | 0.750 | 0.733 | 0.811 | 0.336 | 0.429 |
| vertex median mm | 137.7 | 77.1 | 55.2 | 62.2 | 131.3 | 16.0 | 19.1 |

- **Pretraining alone captures most of the flavour information.**
  - A linear classifier on the frozen features reaches flavour accuracy
    0.767.  Untrained gives 0.594, and always guessing numu gives 0.560.
  - That closes 75% of the gap from untrained to the fine-tune (0.826).
    It comes close to a fully supervised fine-tune from scratch (0.785),
    although the encoder never saw a label.
  - ν_e CC, ν_μ CC and NC are the most linearly separable (AUC 0.91-0.95).
- **Most of it is learned by epoch 99** (0.748).  Epochs 100-399 add 0.019
  in flavour accuracy.  The vertex and |p_jet| probes are best at epoch 99.
- **Charm, the vertex position and the lepton momentum are mostly not in the
  pooled features.**
  - Charm: linear AUCs 0.65-0.75 against 0.83-0.90 fine-tuned.  kNN is near
    chance for charm->e and ->mu (0.51-0.53).
  - Vertex: 62 mm against 16 mm.  |p_lep| CC: sMAD 0.73 against 0.34.
  - These need local detail that mean pooling discards, and fine-tuning
    recovers it.
- **pT_miss** (left out of the table): the linear probe gets worse with
  training (sMAD 5.8 untrained, 15.1 at epoch 399).  A linear map to
  direction cosines can't resolve the small transverse component.  kNN
  improves from 3.5 to 1.5, against 0.55 fine-tuned.
- **Neighbourhoods are label-consistent for the common classes:** kNN flavour
  accuracy is 0.728, but it is weak for ν_τ and charm.
- **`k16` and `last` probe alike** for this encoder (flavour 0.761 against
  0.767).  In the K = 1 encoder, the K = 16 average had probed far better
  ("Probe of the existing encoder").  K = 16 training has moved that
  information into the final output.

*Maps* (`latent_maps_{pca,tsne}.png`, `latent_maps_tsne_flavour.png`): 11,400
test events, 1,900 per flavour (fixed seed).
- Each row is an encoder, and the columns colour the same points by
  outgoing lepton (electron: ν_e CC and ν_τ->e; muon: ν_μ CC and ν_τ->μ;
  none: NC and ν_τ->had), by charm, and by E_vis.  The coordinates are in
  `latent_maps.npz`.
- **PCA shows little for the trained encoders.**  With the `last`
  features, the first two components hold about 3% of the variance each
  (11% and 7% for `k16`): the information is spread over many dimensions.  The untrained encoder's PCA is one curve ordered by energy
  (PC1 30%).
- **The t-SNE of epoch 399 separates the flavours partly.**  It has ν_μ CC
  regions, a ν_e CC core and NC-rich regions, while the ν_τ classes spread
  across them.
  - The detached island is not flavour: 314 events of all flavours, with low
    energy (median E_vis 142 GeV, against 360) and upstream, off-centre
    vertices (median z -693 mm, against -39).
  - Charm shows no structure in any map.
- **Caveats:**
  - Mean pooling with a linear or kNN probe is a lower bound on what the
    tokens hold; an attentive probe could read more.
  - One encoder and one seed.
  - t-SNE distances and cluster sizes are not quantitative.
  - The maps' flavour-balanced sample overweights ν_τ relative to the test
    set.

On dlnu (~12 min of feature extraction on one H100, then ~3 min):

```bash
CUDA_VISIBLE_DEVICES=1 python results/flash_audit_20261026/latent_features.py /scratch4/salonso/faser/latents_qkn_s7 24 > /scratch4/salonso/faser/latent_features.log
CUDA_VISIBLE_DEVICES=0 python results/flash_audit_20261026/latent_probes.py /scratch4/salonso/faser/latents_qkn_s7 > results/flash_audit_20261026/latent_probes.out
```

**The same for the published MAE (2026-11-04).**  This is the MAE base the
paper fine-tuned (`pretrain_v7.0_lat_norel_v2`).  It is shown untrained
(torch seed 7), at epoch 215 (its best-validation checkpoint) and at
`last.ckpt` (epoch 399, the encoder the published fine-tune used).  It has no
earlier snapshots.
- **Feature:** its Perceiver latents, mean-pooled over the valid latents, as
  SparseViT pools them for the fine-tune.  Its stacked encoder has no
  counterpart of the K = 16 target, so there is only one feature kind.
- **Same events, labels, probes and metrics** as the JEPA.  The features
  cover the same events in the same order.  The fine-tuned column
  reproduces the published CSV (0.8153, 16.88 mm).
- **The two encoders' pooled features are not the same object:** each is
  pooled the way its own fine-tune pools it.

The two final encoders (epoch 399) side by side:

| metric | JEPA linear | MAE linear | JEPA kNN | MAE kNN | JEPA fine-tuned | MAE fine-tuned |
|---|---|---|---|---|---|---|
| flavour acc | **0.767** | 0.696 | 0.728 | 0.632 | 0.826 | 0.815 |
| AUC nue | 0.953 | 0.928 | 0.912 | 0.869 | 0.979 | 0.976 |
| AUC numu | 0.921 | 0.846 | 0.892 | 0.744 | 0.953 | 0.947 |
| AUC NC | 0.907 | 0.818 | 0.859 | 0.694 | 0.945 | 0.939 |
| AUC nutau->had | 0.890 | 0.831 | 0.806 | 0.720 | 0.945 | 0.939 |
| AUC nutau->e | 0.860 | 0.820 | 0.688 | 0.646 | 0.922 | 0.910 |
| AUC nutau->mu | 0.761 | 0.696 | 0.604 | 0.553 | 0.852 | 0.836 |
| AUC charm->e | 0.649 | 0.644 | 0.513 | 0.512 | 0.831 | 0.804 |
| AUC charm->mu | 0.751 | 0.652 | 0.530 | 0.509 | 0.898 | 0.884 |
| AUC charm->had | 0.724 | 0.713 | 0.624 | 0.617 | 0.879 | 0.869 |
| sMAD E_vis | 0.425 | 0.597 | 0.527 | 0.771 | 0.192 | 0.214 |
| sMAD \|p_jet\| | 0.373 | 0.434 | 0.440 | 0.570 | 0.148 | 0.164 |
| sMAD \|p_lep\| CC | 0.733 | 0.881 | 0.811 | 0.910 | 0.336 | 0.386 |
| vertex median mm | 62.2 | **50.8** | 131.3 | **97.3** | 16.0 | 16.9 |

- **Untrained, the two start level:** linear flavour accuracy 0.594 (JEPA)
  against 0.601 (MAE).
- **Pretraining puts much more label information into the JEPA's frozen
  features.**
  - Linear flavour accuracy rises to 0.767 for the JEPA against 0.696 for
    the MAE.  That is 75% of the gap to its fine-tune against 44%.
  - The JEPA is ahead on every flavour AUC, by 0.025 (ν_e) to 0.089
    (NC).  It is also ahead on E_vis, |p_jet|, |p_lep| and charm->mu.
  - kNN shows the same picture (0.728 against 0.632).
  - So the MAE depends more on fine-tuning to make its features usable.
    Its fine-tune still comes within 0.011 of the JEPA's.
- **The MAE keeps more position information:** the vertex is 50.8 mm
  against 62.2 (linear) and 97 against 131 (kNN).  Reconstructing the
  inputs rewards it; predicting latents does not.
- **Epoch 215 and epoch 399 of the MAE probe alike** (flavour 0.700 against
  0.696).
- **Charm is weak in both** (linear AUCs 0.64-0.75, kNN near chance for
  charm->e and charm->mu).
- **Maps** (`latent_mae_maps_{pca,tsne}.png`,
  `latent_mae_maps_tsne_flavour.png`, coordinates in `latent_mae_maps.npz`):
  - Its epoch-399 t-SNE is mostly mixed, with one compact ν_μ CC patch.
    It has none of the JEPA's broad ν_μ and ν_e regions.
  - The untrained MAE's t-SNE splits into a few islands, each mixed in
    flavour.
  - PCA: 6.5-8% of the variance per component after training, 49% in PC1
    untrained.
- **Caveats:** as for the JEPA (pooled features, one encoder, t-SNE not
  quantitative).  The backbones also differ (SparseViT with Perceiver
  latents against the flash ViT), so this compares the two pretrained
  models as published and used, not the two objectives on one
  architecture.

Commands on dlnu (MAE extraction in the `spconv` environment, ~13 min; probes
~2 min in `neutrino-jepa`):

```bash
CUDA_VISIBLE_DEVICES=1 python results/flash_audit_20261026/latent_features_mae.py /scratch4/salonso/faser/latents_mae 24 > /scratch4/salonso/faser/latent_features_mae.log
CUDA_VISIBLE_DEVICES=0 python results/flash_audit_20261026/latent_probes.py /scratch4/salonso/faser/latents_mae mae > results/flash_audit_20261026/latent_probes_mae.out
```

- A second extraction was launched by mistake while the first was still
  running.  It was stopped before it wrote anything
  (`/scratch4/salonso/faser/latent_features_mae.duplicate_stopped.log`).
- A rerun of the probes reproduced every number and map coordinate exactly.

From `/users/salonso/faserDL` on Clariden:

```bash
c=/capstor/scratch/cscs/salonso/faser/checkpoint; r=$c/pretrain_flash_jepa_base_d2v16_s8/snapshots/epoch=124.ckpt
NAME=pretrain_flash_jepa_base_d2v16_s8_r124_ema99925 TARGET_LAYERS=16 SEED=8 SAVE_EVERY_N_EPOCHS=10 JEPA_EMA_DECAY=0.99925 JEPA_EMA_END_DECAY=0.99925 RESUME_CHECKPOINT=$r sbatch --partition=debug --time=00:45:00 --job-name=s8r124_ema clariden/pretrain_flash.sbatch
NAME=pretrain_flash_jepa_base_d2v16_s8_r124_k8 TARGET_LAYERS=8 SEED=8 SAVE_EVERY_N_EPOCHS=10 RESUME_CHECKPOINT=$r sbatch --partition=debug --time=00:45:00 --job-name=s8r124_k8 clariden/pretrain_flash.sbatch
NAME=pretrain_flash_jepa_base_d2v16_s8_r124_lr05 TARGET_LAYERS=16 SEED=8 SAVE_EVERY_N_EPOCHS=10 LR_SCALE_ON_RESUME=0.5 LOG_GRAD_NORM=1 RESUME_CHECKPOINT=$r sbatch --partition=debug --time=00:45:00 --job-name=s8r124_lr05 clariden/pretrain_flash.sbatch
```

S5 and S3b:

```bash
NAME=pretrain_flash_jepa_base_d2v16_s8_r124_wd TARGET_LAYERS=16 SEED=8 SAVE_EVERY_N_EPOCHS=10 WEIGHT_DECAY=0.04 WEIGHT_DECAY_END=0.4 LOG_GRAD_NORM=1 RESUME_CHECKPOINT=$r sbatch --partition=debug --time=00:45:00 --job-name=s8r124_wd clariden/pretrain_flash.sbatch
NAME=pretrain_flash_jepa_base_d2v16_s8_r124_lr025 TARGET_LAYERS=16 SEED=8 SAVE_EVERY_N_EPOCHS=10 LR_SCALE_ON_RESUME=0.25 LOG_GRAD_NORM=1 RESUME_CHECKPOINT=$r sbatch --partition=debug --time=00:45:00 --job-name=s8r124_lr025 clariden/pretrain_flash.sbatch
```

(S1 and S4 were submitted with `--time=01:00:00` and `00:50:00` and cut to
45 min with `scontrol update`; S1 ran before the new options were copied to
Clariden, S4 started before too, S3 with them.)

Scored with `results/flash_audit_20261026/screens_s8_r124.py` (commands
in its docstring).

## Batch-standardised targets (opt-in, `TARGET_NORM=batch`)

**Why.**  The K = 16 collapse moved the targets onto one vector shared by
every token.  Each token's target is layer-normalised by itself, so such a
vector is a valid, full-size target, and predicting it needs no context.

**Definition.**  `TARGET_NORM=batch` (`--flash_target_norm batch`,
`FlashJEPA(target_norm="batch")`, `standardise_per_kind` in
`model/flash_jepa.py`).  The usual targets (any `TARGET_LAYERS`) are then
standardised per channel over the GPU's batch, separately for each detector
type: subtract the mean over those tokens, divide by their standard
deviation (eps 1e-5).  Details:
- Statistics use all tokens of the full view, before the masked targets are
  selected.
- A token alone of its type in the batch gets a zero target.
- The default, `token`, is unchanged (bit-identical code path).

Effects:
- Anything shared by every token of a type is zeroed, so collapsing onto it
  no longer lowers the loss.
- Dividing by the spread keeps the remaining differences at unit scale.
- Differences between events, such as an event's overall energy, are kept;
  they would be lost by normalising within each event (data2vec's speech
  variant).

Limits:
- A component fixed by the patch position (identical across events) is not
  removed.  The block-by-block check found it small.
- The statistics are per GPU: 512 events per GPU on 2 Clariden nodes, 2,048
  on dlnu, so the per-type token counts are large.
- `jepa/target_std` is ~1 by construction and no longer a collapse monitor.
  Watch `jepa/context_event_std` and `jepa/target_event_std` instead.

**Checks.**
- Three unit tests:
  - `standardise_per_kind` gives zero mean and unit variance per channel and
    type, matches a hand-computed reference, gives zero for a lone token,
    and is invariant to a vector added to every token of a type;
  - with K = 1 and K = 2, the batch targets are exactly the standardised
    `token` targets, and gradients reach only the context encoder and
    predictor;
  - an invalid value is rejected, and the flag and launcher variable are
    passed through.
- dlnu, GPUs hidden: `neutrino-jepa` 55 tests OK, 9 skipped.  On a Clariden
  GH200, the flash tests pass: 30 tests, 5 skipped.
- Clariden debug job 3764131 ran a 3-epoch pretraining (2 nodes, full staged
  data, `TARGET_LAYERS=16 TARGET_NORM=batch`, test directory since removed).
  The split fingerprint matched, and the train loss fell from 0.38 to 0.049
  with no NaN.  `target_std` was 1.00, `context_event_std` 0.58-0.62 and
  `target_event_std` 0.81-0.83.

**Run** (Clariden), then fine-tune as for the other runs:

```bash
NAME=pretrain_flash_jepa_base_d2v16_bn TARGET_LAYERS=16 TARGET_NORM=batch sbatch clariden/pretrain_flash.sbatch
```

```bash
c=/capstor/scratch/cscs/salonso/faser/checkpoint; name=finetune_flash_jepa_base_d2v16_bn_ld085; jid=$(LOAD_CHECKPOINT=$c/pretrain_flash_jepa_base_d2v16_bn/loss_total_val/last.ckpt NAME=$name LAYER_DECAY=0.85 sbatch --parsable clariden/finetune_flash.sbatch); CHECKPOINT_DIR=$c/$name sbatch --dependency=afterok:$jid clariden/evaluate_flash.sbatch
```

**Pretraining result** (job 3764165, 2 nodes, 400 epochs in 4 h 28 min, no
NaN; `last.ckpt` and `snapshots/epoch=024..399.ckpt`).  Validation event
spread, with the `token` K = 16 run for comparison:

| epoch | 300 | 320 | 328 | 340 | 350 | 365 | 370 | 380 | 399 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `context_event_std`, token | 0.40 | 0.35 | 0.16 | 0.11 | 0.13 | | | | 0.11 |
| `context_event_std`, batch | 0.48 | 0.49 | 0.50 | 0.51 | 0.51 | 0.47 | 0.23 | 0.44 | 0.35 |
| `target_event_std`, token | 0.54 | 0.53 | 0.51 | 0.33 | 0.34 | | | | 0.35 |
| `target_event_std`, batch | 0.57 | 0.58 | 0.58 | 0.58 | 0.58 | 0.59 | 0.59 | 0.59 | 0.59 |

- **No collapse at epochs 316-346**, where the `token` run collapsed.
- **The student wobbled at epochs ~365-371.**  Its event spread fell from
  0.49 to 0.23 and partly recovered (0.35-0.44).  The validation loss rose
  from 0.049 to 0.086, then settled at 0.07-0.08.  A collapse onto an easy
  target would have lowered the loss instead.
- **The teacher did not follow the student.**  Its event spread stayed at
  0.586-0.588, and the EMA decay is ~1 this late.  The teacher is what
  fine-tuning loads.
- **Transient at epoch ~185**: the loss rose to 0.156, then fell to 0.042,
  and both spreads rose (target 0.57 to 0.88) before settling at ~0.58 by
  epoch 210.
- **Losses are not comparable with the `token` run**, because its targets
  have a different scale.

### Fine-tune of `last.ckpt` (2026-10-27)

Published recipe, `LAYER_DECAY=0.85`, one seed, run on two machines:
- **dlnu** (2 x 512, same global batch of 1,024): 20 epochs in 2 h,
  evaluation 8 min.  Results below.
- **Clariden** (command above, 1 node, 4 x 256): job 3768156 (33 min),
  evaluated by 3768157.  Same checkpoint, as a replicate across machines
  (below).  Results copied to `results/` (sha256 checked).

```bash
cd /scratch/salonso/sparse-nns/faser_develop && source /scratch/salonso/anaconda3/etc/profile.d/conda.sh && conda activate neutrino-jepa && n=finetune_flash_jepa_base_d2v16_bn_ld085_dlnu && c=/scratch4/salonso/faser/checkpoints_final && nohup bash -c "NAME=$n LAYER_DECAY=0.85 GPUS='0 1' BATCH_SIZE=512 NUM_WORKERS=24 CHECKPOINT_PATH=$c SAVE_TOP_K=1 SAVE_LAST=0 CHECKPOINT_MONITORS=evaluated LOAD_CHECKPOINT=$c/pretrain_flash_jepa_base_d2v16_bn/loss_total_val/last.ckpt EXPECTED_SPLIT_SHA256=02078f72b5ec271f81e14f3a24b769e665a9a928908b141bd1fb2134c730fd7c DATASET_PATH='/scratch/salonso/sparse-nns/faser_develop/events_v7.0*' METADATA_PATH=/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz/metadata_stats.pkl bash finetune_flash.sh && python -m evaluation.create_csv_v7_ft --checkpoint_dir $c/$n" > /scratch4/salonso/faser/$n.log 2>&1 &
```

The checkpoint was copied from Clariden (sha256 a4201449... on both sides).
`results/flash_audit_20261026/metrics_flash_bn.{py,out}` scores both runs
on the same 122,639 test events (paired event bootstrap, 200 resamples):

```bash
python results/flash_audit_20261026/metrics_flash_bn.py > results/flash_audit_20261026/metrics_flash_bn.out 2>&1
```

**Validation** (best per-task loss):

| run | flavour | charm | vis | jet | vertex | total |
|---|---:|---:|---:|---:|---:|---:|
| K = 1, 0.85 | 0.5085 | 0.2166 | 1.1608 | 0.2631 | 0.005640 | 1.2524 |
| K = 16 e400, 0.85 | 0.4896 | 0.2136 | 1.1121 | 0.2529 | 0.005010 | 1.1674 |
| K = 16 batch, 0.85 (dlnu) | 0.5081 | 0.2153 | 1.1559 | 0.2654 | 0.006010 | 1.2520 |
| K = 16 batch, 0.85 (Clariden) | 0.5066 | 0.2157 | 1.1556 | 0.2651 | 0.006219 | 1.2528 |

Both batch runs' total validation losses are above the e400 run's at all 20
epochs.

**Test:**

| metric | MAE (pub) | K = 1 | K = 16 e400 | K = 16 batch (dlnu) | batch - e400 [95%] |
|---|---:|---:|---:|---:|---:|
| AUC nue | 0.9756 | 0.9775 | 0.9788 | 0.9770 | -0.0018 [-0.0024, -0.0013] |
| AUC numu | 0.9473 | 0.9470 | 0.9518 | 0.9480 | -0.0037 [-0.0045, -0.0032] |
| AUC nutau->e | 0.9103 | 0.9181 | 0.9221 | 0.9204 | -0.0017 [-0.0048, +0.0008] |
| AUC nutau->mu | 0.8364 | 0.8413 | 0.8483 | 0.8466 | -0.0017 [-0.0061, +0.0028] |
| AUC nutau->had | 0.9386 | 0.9404 | 0.9437 | 0.9404 | -0.0032 [-0.0043, -0.0021] |
| AUC NC | 0.9388 | 0.9382 | 0.9443 | 0.9386 | -0.0057 [-0.0065, -0.0050] |
| flavour acc | 0.8153 | 0.8152 | 0.8231 | 0.8150 | -0.0081 [-0.0095, -0.0067] |
| AUC charm->e | 0.8042 | 0.8217 | 0.8234 | 0.8133 | -0.0101 [-0.0196, +0.0009] |
| AUC charm->mu | 0.8844 | 0.8932 | 0.9003 | 0.9020 | +0.0017 [-0.0076, +0.0104] |
| AUC charm->had | 0.8691 | 0.8710 | 0.8768 | 0.8744 | -0.0024 [-0.0048, +0.0001] |
| sMAD E_vis | 0.2144 | 0.2222 | 0.1963 | 0.2166 | +0.0202 [+0.0186, +0.0220] |
| sMAD pT_miss | 0.5573 | 0.5782 | 0.5552 | 0.5801 | +0.0248 [+0.0211, +0.0284] |
| sMAD \|p_jet\| | 0.1637 | 0.1543 | 0.1516 | 0.1561 | +0.0045 [+0.0034, +0.0057] |
| sMAD \|p_lep\| CC | 0.3857 | 0.3930 | 0.3429 | 0.3738 | +0.0308 [+0.0266, +0.0341] |
| vertex median mm | 16.88 | 14.68 | 15.16 | 15.27 | +0.12 [+0.02, +0.23] |

- **Worse than K = 16 without it on 10 of 15 metrics** (intervals excluding
  0) and better on none.  nutau->e, nutau->mu and the three charm AUCs are
  tied.  For example, flavour accuracy -0.0081, E_vis +0.0202, pT_miss
  +0.0248, |p_lep| CC +0.0308.
- **About K = 1's level.**  Against K = 1 at the same layer decay: better on
  5 (numu, nutau->mu, charm->had, E_vis -0.0057, |p_lep| CC -0.0192), worse
  on 2 (|p_jet| +0.0017, vertex +0.59 mm), tied on 8 including flavour
  accuracy.  The validation totals are tied (1.2520 against 1.2524).
- **Against the published MAE**: better on 9, worse on E_vis (+0.0022) and
  pT_miss (+0.0228), tied on numu, NC, flavour accuracy and charm->e.
- **So the batch standardisation prevented the pretraining collapse but gave
  up most of K = 16's fine-tuning gain.**  The collapse itself barely
  affected fine-tuning (e400 against e299, above), so `token` stays the
  choice for fine-tuned results.  Whether `batch` helps a frozen encoder is
  untested.  This compares against `token` seed 7 only.  `token` seed 8
  fine-tunes below `batch` on 13 of 15 metrics (see "Second pretraining
  seed"), so with two `token` seeds the order of the two options is not
  established.
- Why is not established.  One untested candidate: dividing by each
  channel's spread gives low-variance channels the same weight in the loss
  as the informative ones.

**Replicate across machines.**  The same checkpoint, recipe and seed,
fine-tuned on Clariden (4 GPUs x 256) and on dlnu (2 x 512).  Hardware, GPU
count and therefore data order differ:

| metric | dlnu | Clariden | Clariden - dlnu [95%] |
|---|---:|---:|---:|
| AUC nue | 0.9770 | 0.9768 | -0.0002 [-0.0006, +0.0002] |
| AUC numu | 0.9480 | 0.9478 | -0.0003 [-0.0006, +0.0002] |
| AUC nutau->e | 0.9204 | 0.9198 | -0.0006 [-0.0026, +0.0011] |
| AUC nutau->mu | 0.8466 | 0.8477 | +0.0011 [-0.0023, +0.0037] |
| AUC nutau->had | 0.9404 | 0.9403 | -0.0002 [-0.0010, +0.0008] |
| AUC NC | 0.9386 | 0.9385 | 0.0000 [-0.0005, +0.0005] |
| flavour acc | 0.8150 | 0.8151 | +0.0001 [-0.0009, +0.0013] |
| AUC charm->e | 0.8133 | 0.8130 | -0.0003 [-0.0080, +0.0092] |
| AUC charm->mu | 0.9020 | 0.9005 | -0.0015 [-0.0077, +0.0045] |
| AUC charm->had | 0.8744 | 0.8735 | -0.0009 [-0.0025, +0.0006] |
| sMAD E_vis | 0.2166 | 0.2131 | -0.0034 [-0.0049, -0.0021] |
| sMAD pT_miss | 0.5801 | 0.5805 | +0.0004 [-0.0022, +0.0034] |
| sMAD \|p_jet\| | 0.1561 | 0.1580 | +0.0020 [+0.0008, +0.0030] |
| sMAD \|p_lep\| CC | 0.3738 | 0.3769 | +0.0031 [+0.0001, +0.0062] |
| vertex median mm | 15.27 | 15.53 | +0.26 [+0.17, +0.37] |

- **Classification agrees within the bootstrap intervals on all 10
  metrics.**  Flavour accuracy differs by 0.0001.
- **Four regressions differ by more than the bootstrap allows:** E_vis,
  |p_jet|, |p_lep| CC and vertex.  The bootstrap covers test sampling only,
  so for regressions it understates the spread between fine-tuning runs.
  The regressions differ by up to ~0.004 sMAD and ~0.3 mm.  This is one
  observed pair, mixing machine, GPU count and data order.  It is not a
  noise threshold, and it may not carry over to other checkpoints.
- **Same machine, the conclusion holds.**  Clariden batch minus e400 is worse
  on 12 of 15 metrics and better on none: flavour accuracy -0.0080, E_vis
  +0.0168, pT_miss +0.0253, |p_lep| CC +0.0340, vertex +0.37 mm.  nutau->e,
  nutau->mu and charm->mu are tied.
- **Earlier comparisons against this spread.**  K = 16 against the MAE and
  against K = 1 exceed it on every metric that was called different.  Inside
  it: e400 against e299 on E_vis (-0.0018) and |p_jet| (-0.0024), and layer
  decay 0.90 against 0.85 on |p_jet| (+0.0019).

Caveats: one fine-tuning seed per machine.  This pair says nothing about
variation between pretraining seeds; the second pretraining seed (section
above) tests that.

**Layer-wise LR decay is not depth-matched.**  `param_groups_lrd` gives
layer id `i` the scale `decay^(num_layers - i)`.  The published SparseViT base
has 14 layer ids; the 16-block flash encoder has 18.  With the published
`LAYER_DECAY=0.75`, the flash encoder's lower layers get about 3x less of the
peak LR than the published model's:

| LR scale | patch embed | first block | middle | last block |
|---|---:|---:|---:|---:|
| published SparseViT, 0.75 | 0.018 | 0.024 | ~0.10 | 0.75 |
| flash, 0.75 (run above) | 0.006 | 0.008 | 0.056 | 0.56 |
| flash, 0.80 | 0.018 | 0.023 | 0.107 | 0.64 |

`LAYER_DECAY=0.80` reproduces the published profile.  E_vis and pT_miss,
the two metrics where the flash JEPA trails the patch JEPA, are the ones
that most need the low-level energy features to adapt.  0.80, 0.85 and 0.90
have now been run (above); 0.85 is best on validation and 0.90 is no
better.  Choose on the validation losses, not on the test set.

Masking scales, predictor size and depth are the I-JEPA-inspired defaults
above, untuned for this detector.  Every result is a single fine-tuning seed
(`SEED` changes initialisation and data order; the split stays seed 7).
