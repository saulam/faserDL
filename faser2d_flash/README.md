# FASERCal 2D projection transformer

This folder is a self-contained alternative to the repository's `spconv`
pipeline. It trains a transformer encoder from scratch on sparse XZ, YZ, and
optional XY projections. It neither imports `spconv` nor changes any Conda
environment.

Production runs require the existing `platon-flashattn` environment and use
FlashAttention 2's variable-length packed kernel. There is no silent attention
fallback. The explicitly configured `torch_sdpa` backend exists only for CPU
tests and small diagnostics.

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

## Train

```bash
conda run --no-capture-output -n platon-flashattn \
  python -m faser2d_flash.train \
  --config faser2d_flash/configs/two_view.yaml

conda run --no-capture-output -n platon-flashattn \
  python -m faser2d_flash.train \
  --config faser2d_flash/configs/three_view.yaml
```

Resume full state:

```bash
conda run --no-capture-output -n platon-flashattn \
  python -m faser2d_flash.train \
  --config faser2d_flash/configs/two_view.yaml \
  --resume faser2d_flash/artifacts/xz_yz/checkpoints/last.pt
```

CLI overrides use dotted YAML keys:

```bash
python -m faser2d_flash.train \
  --config faser2d_flash/configs/two_view.yaml \
  --set training.batch_size=32 \
  --set training.accumulation_steps=32
```

Each run records the active attention backend, package/runtime report,
parameter counts, effective batch size, throughput, peak CUDA memory,
epoch metrics, resumable `last.pt`, and best checkpoints for total, flavour,
charm, visible, jet, lepton-consistency, and vertex losses.

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

## Run and compare both modes

```bash
bash faser2d_flash/scripts/run_both.sh
```

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
├── metrics.csv
├── checkpoints/
│   ├── last.pt
│   ├── best_total.pt
│   ├── best_flavour.pt
│   ├── best_charm.pt
│   ├── best_vis.pt
│   ├── best_jet.pt
│   ├── best_lepton.pt
│   └── best_vertex.pt
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
- `train.py`: scratch training, validation, logging, and resume.
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
