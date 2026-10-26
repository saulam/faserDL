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
  I-JEPA.
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

## Tests

`tests/test_flash_jepa.py` (`python -m unittest tests.test_flash_jepa`).  The
spconv-equivalence and target-collation tests run in the `spconv`
environment; the flash-kernel tests need a GPU.  The full suite
(`python -m unittest discover -s tests -t .`) passes in both environments:
- `spconv`: 63 tests, with the 2 flash-kernel tests skipped;
- `neutrino-jepa`: 45 tests, with the spconv-only modules skipped.

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

## Not yet done

The fine-tuning has not been run yet.  Masking
scales, predictor size and depth are the I-JEPA-inspired defaults above,
untuned for this detector.
