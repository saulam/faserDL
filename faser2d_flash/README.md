# FASERCal 2D projection transformer

This folder is a self-contained alternative to the repository's `spconv`
pipeline. It trains a transformer encoder from scratch on sparse XZ, YZ, and
optional XY projections. It neither imports `spconv` nor changes any Conda
environment.

Production runs require the existing `platon-flashattn` environment and use
FlashAttention 2's variable-length packed kernel. There is no silent attention
fallback. The explicitly configured `torch_sdpa` backend exists only for CPU
tests and small diagnostics.

Training uses PyTorch Lightning 2.4. Lightning owns DDP process launch,
distributed sampling, gradient synchronization and accumulation, validation
metric reduction, checkpointing, and full-state resume. The production
configuration uses both visible GPUs.

## Existing pipeline correspondence

The current fine-tuned model has five direct outputs:

- `out_flavour`: six logits for CC nue, CC numu, three CC nutau decay groups,
  and NC;
- `out_charm`: four logits for no charm, charm-to-electron,
  charm-to-muon, and charm-to-hadron;
- `out_vis`: cylindrical visible-momentum prediction with Cartesian `p_cart`;
- `out_jet`: the same representation for jet momentum;
- `out_vertex`: standardized three-dimensional primary vertex.

Lepton momentum is `out_vis["p_cart"] - out_jet["p_cart"]`, as in the
repository's current evaluation scripts. The cylindrical heads,
classification losses, consistency losses, bounded Kendall task weights,
label smoothing, AdamW settings, gradient clipping, EMA, 40-epoch scratch
schedule, and 85/5/10 split convention are retained.

The base 2D encoder uses dimension 384, 12 heads, MLP ratio 4, and 20
pre-norm transformer blocks. With the local image encoders it has 36,804,373
parameters, close to the 39.1-million-parameter current base model. Both
input modes instantiate exactly the same parameters.

## Input representation

See [docs/DATA_INSPECTION.md](docs/DATA_INSPECTION.md) for measured data
properties. Pixel-level attention is too long, while analytical patch
summaries discard local track and shower structure. The selected tokenizer
therefore preserves every pixel within each non-empty patch:

- XZ and YZ are treated as ten active `48×20` modules separated by nine
  four-pixel Z gaps. Each active module is divided into `12×5` patches, giving
  the same `4×4` patch grid per module.
- Each `48×48` XY view is divided into `12×12` patches, also giving a `4×4`
  grid per module.
- Every patch contains two dense local channels: training-standardized
  `log1p(charge)` and binary occupancy. Empty cells remain zero and empty
  patches are not materialized.
- A two-layer local `3×3` CNN processes each patch before flattening and
  projecting it to the transformer dimension. XZ and YZ share one encoder;
  XY uses a shape-specific encoder. No centroid, variance, or charge-summary
  compression is used.

Position is encoded in a common normalized 3D frame with Fourier features.
The physical four-pixel Z gaps remain present in token positions even though
they are not tokenized as detector pixels.
Learned view-type embeddings distinguish XZ, YZ, and XY. Each enabled view
also receives an anchor token, including empty XY layers. Five task tokens are
prepended to every event. All event sequences are concatenated without
padding and described by cumulative sequence lengths for FlashAttention.

The two configurations differ only in whether the ten XY views are included.
The split manifests, targets, model capacity, optimizer, schedule, loss
weights, seed, and evaluation code are shared.

The default augmentation is a projection-safe event-wide log-normal gain
jitter. Hit-correspondence-dependent 3D translations and dropout are not
copied because the saved projections do not retain a mapping that would apply
the same dropped hit consistently to every view. This is an intentional
architectural difference from the 3D loader and is identical in both study
modes.

### Primary-origin artifact

The raw projections contain a reconstruction artifact at `(0,0)`. Across
1,500-event audits it appeared in XZ and YZ for roughly 97–99% of events and
the same charge was usually repeated in XZ, YZ, and XY layer 0. Values above
`1e5` were observed in one audit.

The default `data.remove_primary_origin_pixel: true` removes `(0,0)` from XZ,
YZ, and XY layer 0 before patching or statistics calculation. `(0,0)` pixels
in XY layers 1–9 are retained because their frequency and charge look
consistent with ordinary edge occupancy. Changing this option requires
regenerating metadata; training checks that the policy matches.

## Environment check

From the repository root:

```bash
conda run --no-capture-output -n platon-flashattn \
  python -m faser2d_flash.runtime --require-flash --precision bf16
```

The command reports the active PyTorch, CUDA, GPU, and FlashAttention runtime
and exits with an actionable error if the production backend cannot run.

## Inspect data

```bash
conda run --no-capture-output -n platon-flashattn \
  python -m faser2d_flash.inspect_dataset \
  --glob '/scratch4/salonso/faser/events_v8.0_100*_2d' \
  --samples 1500 --patch-size 12 \
  --output faser2d_flash/docs/data_inspection.json
```

## Build manifests and metadata

This step freezes the growing dataset, validates every selected file, creates
the shared deterministic splits, and calculates input and target statistics
from the training split only:

```bash
conda run --no-capture-output -n platon-flashattn \
  python -m faser2d_flash.build_metadata \
  --config faser2d_flash/configs/two_view.yaml \
  --workers 16
```

Generated files:

```text
faser2d_flash/metadata/v8_2d/
├── metadata.json
└── manifests/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── rejected.jsonl
```

Use `--max-files N` only for diagnostics. Metadata produced with that option
is marked `diagnostic_subset: true` and must not be used for the final study.

The metadata command uses `two_view.yaml` only for the shared paths and data
settings. It always calculates XZ, YZ, and XY input statistics plus two-view
and three-view token statistics. Both experiments use this one metadata file
and the same manifests.

## Train with two GPUs

Both production configs specify:

```yaml
distributed:
  devices: 2
  num_nodes: 1
```

Lightning launches NCCL DDP internally, so do not wrap these commands in
`torchrun`. Select the two physical GPUs through `CUDA_VISIBLE_DEVICES`.

Run the two-view experiment:

```bash
CONDA_EXE=/scratch/salonso/anaconda3/bin/conda \
GPU_IDS=0,1 \
bash faser2d_flash/scripts/run_two_view.sh
```

After it finishes, run the three-view experiment:

```bash
CONDA_EXE=/scratch/salonso/anaconda3/bin/conda \
GPU_IDS=0,1 \
bash faser2d_flash/scripts/run_three_view.sh
```

`run_two_view.sh` uses the configured per-GPU batch and accumulation settings.
`run_three_view.sh` halves the configured microbatch and doubles accumulation.
With the current configuration this means:

| Mode | Per-GPU batch | Accumulation | Global effective batch |
|---|---:|---:|---:|
| XZ/YZ | 1024 | 1 | 2048 |
| XZ/YZ/XY | 512 | 2 | 2048 |

This reduces three-view activation memory while keeping optimizer steps and
linear learning-rate scaling directly comparable.

Resume full state:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
conda run --no-capture-output -n platon-flashattn \
  python -m faser2d_flash.train \
  --config faser2d_flash/configs/two_view.yaml \
  --resume faser2d_flash/artifacts/xz_yz/checkpoints/last.ckpt
```

CLI overrides use dotted YAML keys:

```bash
python -m faser2d_flash.train \
  --config faser2d_flash/configs/two_view.yaml \
  --set training.batch_size=32 \
  --set training.accumulation_steps=32
```

Each run records the active attention backend, package/runtime report,
parameter counts, effective batch size, epoch metrics, resumable
`last.ckpt`, and best checkpoints for total, flavour,
charm, visible, jet, lepton-consistency, and vertex losses.

### Batch-size guidance

The full three-view model passed BF16 DDP optimizer steps at 1024 and 1152
events per GPU. At 1024, the limiting A100 used 67.42 GiB allocated and
68.63 GiB reserved. At 1152 it used 75.67/76.99 GiB, which is too close to
the 80 GiB limit for a long run. Use 1024 as the upper practical per-GPU
batch.

Batch size changes the optimisation regime:

```text
effective batch = per-GPU batch × 2 GPUs × accumulation steps
```

To retain effective batch 1024, use:

```bash
--set training.batch_size=512 \
--set training.accumulation_steps=1
```

To use the tested maximum-throughput setting with effective batch 2048, use:

```bash
--set training.batch_size=1024 \
--set training.accumulation_steps=1
```

See [docs/BATCH_SIZE_BENCHMARK.md](docs/BATCH_SIZE_BENCHMARK.md) for the
measurement details and memory values.

### TensorBoard

Training writes both CSV logs and standard TensorBoard event files:

```text
faser2d_flash/artifacts/<experiment>/tensorboard/version_<n>/
```

The existing `tensorboardX` package supplies the event writer; no environment
was modified. From any environment providing the TensorBoard viewer:

```bash
tensorboard --logdir faser2d_flash/artifacts --port 6006
```

The dashboard includes step and epoch losses, validation losses for every
task, Kendall weights, and learning rate.

## Evaluate

The default task-wise selection mirrors the existing repository: each output
uses the checkpoint selected by its own validation loss.

```bash
conda run --no-capture-output -n platon-flashattn \
  python -m faser2d_flash.evaluate \
  --config faser2d_flash/configs/two_view.yaml \
  --selection taskwise --split both --write-predictions
```

Use `--selection total` for one common best-total-loss checkpoint or
`--selection last` for the latest checkpoint. Add `--use-ema` to evaluate EMA
weights explicitly.

Evaluation JSON includes all training losses, confusion matrices, accuracy,
macro precision/recall/F1, per-class scores, component MAE/RMSE, vector and
magnitude errors, relative errors, angular errors, vertex displacement,
visible-energy MAE, and missing-transverse-momentum MAE.

## Export legacy-compatible CSV files

`create_csv_v8.py` reproduces the column names, ordering, class labels, and
derived quantities from `notebooks_mae/create_csv_v8.py`. It uses the best
validation checkpoint independently for flavour, charm, visible momentum,
jet momentum, and vertex. As in the legacy exporter, reconstructed lepton
momentum is visible momentum minus jet momentum.

The 2D files do not contain muon-spectrometer information, so
`nb_muspec_tracks` is the only legacy column intentionally omitted. No fields
are joined from the 3D dataset. By default, raw checkpoint weights are used,
matching the legacy exporter; `--use-ema` is available as an explicit
alternative.

Generate both test-set CSV files sequentially on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
MPLCONFIGDIR=/tmp/faser2d-matplotlib-$USER \
conda run --no-capture-output -n platon-flashattn \
  python -m faser2d_flash.create_csv_v8 --mode both
```

The default outputs are:

```text
faser2d_flash/results/results_v8.0_xz_yz.csv
faser2d_flash/results/results_v8.0_xz_yz_xy.csv
```

The exporter reads each experiment's `run_config.json`, including the actual
training-time batch settings. Override them when needed with
`--two-batch-size`, `--three-batch-size`, or `--num-workers`. Use
`--mode two` or `--mode three` to export only one experiment, and
`--split val` to export the validation split instead of the default test
split. Output is written atomically and sorted numerically by run and event.

## Run and compare both modes

The launcher builds metadata if necessary, checks FlashAttention, trains
two-view on both GPUs, then trains three-view on both GPUs, evaluates both,
and writes the comparison:

```bash
CONDA_EXE=/scratch/salonso/anaconda3/bin/conda \
GPU_IDS=0,1 \
bash faser2d_flash/scripts/run_both.sh
```

The experiments run sequentially; each experiment uses both GPUs
simultaneously. Override `LIGHTNING_DEVICES=1` only for a single-GPU run.
The combined launcher delegates training to `run_two_view.sh` and
`run_three_view.sh`, so it uses the same batch settings as the separate
launchers.

Or compare completed runs directly:

```bash
python -m faser2d_flash.compare \
  --two-run faser2d_flash/artifacts/xz_yz \
  --three-run faser2d_flash/artifacts/xz_yz_xy \
  --selection taskwise \
  --output-dir faser2d_flash/artifacts/comparison
```

The comparison utility writes JSON and a Markdown table with validation and
test results for every scalar output metric.

## Tests

```bash
bash faser2d_flash/scripts/smoke_test.sh
```

The tests load NPZ samples, collate packed batches in both modes, verify the
output contract, and complete a forward/backward pass with the explicitly
selected CPU diagnostic backend. On a CUDA allocation, the runtime check and
production configs additionally confirm `flash_attn_varlen`.

Run a full-size real-data GPU smoke check for both modes:

```bash
CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n platon-flashattn \
  python -m faser2d_flash.gpu_smoke \
  --config faser2d_flash/configs/two_view.yaml \
  --output faser2d_flash/docs/gpu_smoke.json
```

## Artifact layout

```text
faser2d_flash/artifacts/<experiment>/
├── run_config.json
├── logs/
│   └── version_<n>/
│       ├── hparams.yaml
│       └── metrics.csv
├── tensorboard/
│   └── version_<n>/
│       ├── events.out.tfevents.*
│       └── hparams.yaml
├── checkpoints/
│   ├── last.ckpt
│   ├── best_total.ckpt
│   ├── best_flavour.ckpt
│   ├── best_charm.ckpt
│   ├── best_vis.ckpt
│   ├── best_jet.ckpt
│   ├── best_lepton.ckpt
│   └── best_vertex.ckpt
└── evaluation/<selection>/
    ├── val_metrics.json
    ├── test_metrics.json
    └── optional prediction CSV files
```

## Assumptions and limitations

- The first ten `xy_projs` entries are the physical XY views. Non-empty
  entries beyond index 9 are treated as format drift and rejected by default.
- Projection coordinates are histogram-bin coordinates. The common 3D
  positional mapping follows the observed XZ/YZ/XY axis ranges.
- The generator does not store ECAL, AHCAL, or muon-spectrometer inputs in
  2D-only mode, so this study compares projection-only models rather than all
  detector branches of the 3D model.
- Full FlashAttention execution requires a compatible allocated GPU. Package
  presence alone is not considered sufficient.
- The full metadata scan is intentionally not run automatically because the
  dataset is large and still growing; running the documented snapshot command
  is part of starting the final experiment.

## Files in this folder

- `config.py`: YAML inheritance and validation.
- `schema.py`: NPZ validation and target definitions.
- `tokenization.py`: detector-aware sparse patch extraction and positions.
- `dataset.py`: map dataset and packed collation.
- `attention.py`: FlashAttention varlen and explicit diagnostic SDPA paths.
- `model.py`: transformer encoder and output heads.
- `losses.py`: current stage-2 losses and Kendall weighting.
- `statistics.py`: training-only normalization and target statistics.
- `build_metadata.py`: validated split/manifests and metadata generation.
- `inspect_dataset.py`: bounded format/sparsity inspection.
- `runtime.py`: environment and backend checks.
- `gpu_smoke.py`: full-model real-data FlashAttention forward/backward check.
- `checkpoint.py`: atomic checkpoints, RNG state, and EMA.
- `train.py`: Lightning scratch training, NCCL DDP, validation, logging, EMA,
  checkpointing, and resume.
- `metrics.py`: evaluation metrics.
- `evaluate.py`: validation/test inference and optional prediction CSVs.
- `compare.py`: two-mode metric comparison.
- `configs/`: common, two-view, and three-view YAML configurations.
- `scripts/`: complete study and smoke-test launchers.
- `tests/`: lightweight data, collation, output, and backward tests.
- `docs/`: inspected-format and design record.

The complete verification record is in
[docs/VERIFICATION.md](docs/VERIFICATION.md), and the exact created-file
manifest is in [docs/FILES_CREATED.md](docs/FILES_CREATED.md).
