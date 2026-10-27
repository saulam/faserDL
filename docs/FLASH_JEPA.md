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
`torch.unique`, `index_add_` and one dense matrix product per step.  That
includes the two-step embedding of 1,440-voxel FASERCal patches (`4 x 4 x 2`
cells, then LayerNorm + GELU, then `3 x 3 x 5`).  Outputs exist only at
occupied cells, coincident voxels are summed, and out-of-grid voxels are
dropped, all as in spconv.  A test copies spconv weights into it and matches
spconv's output to float32 precision.  On 32 real augmented events the
relative difference is about 1e-7.

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
- **The partial collapse barely matters after fine-tuning.**  e400 against
  e299 is tied on all 10 classification metrics.  On the regressions it is
  slightly better for E_vis, |p_jet| and |p_lep| but worse for pT_miss
  (+0.0065) and vertex (+0.49 mm), and its validation total is slightly
  lower.  Full fine-tuning seems to rebuild what the deep blocks lost, from
  the early blocks that kept their event content.  The collapse is therefore
  a problem for the representation (and for frozen-encoder uses), not for
  these fine-tuned results.
- **Layer decay 0.90 is no better than 0.85** (K = 1).  Validation total is
  tied (1.2529 against 1.2524), with flavour and vis worse.  On test the
  classification is tied, E_vis and |p_lep| are better, and pT_miss, |p_jet|
  and vertex (+0.82 mm) are worse.  0.85 stays the choice.

Caveats: one fine-tuning seed per run.  The layer decay was chosen on K = 1
validation and not re-tuned for K = 16.  The MAE numbers use their published
0.75 recipe.  Epoch 299 was picked from the pretraining diagnostic, before
any fine-tune.

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

## Not yet done

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
