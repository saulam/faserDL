"""Evaluation entry point for the PILArNet transfer study."""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from spconv.pytorch import SparseConvTensor
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from torchmetrics.functional.classification import binary_auroc

from .adapted_model import pilarnet_encoder_tiny, pilarnet_encoder_base
from .multiparticle_model import pilarnet_multiparticle_tiny, pilarnet_multiparticle_base
from .pilarnet_dataset import (
    PILArNetParticleDataset,
    PILArNetMultiParticleDataset,
    PILARNET_SINGLE_PARTICLE_META_DIM,
    pilarnet_collate_fn,
    pilarnet_multiparticle_collate_fn,
)


SINGLE_MODEL_FACTORIES = {
    "tiny": pilarnet_encoder_tiny,
    "base": pilarnet_encoder_base,
}

MULTI_MODEL_FACTORIES = {
    "tiny": pilarnet_multiparticle_tiny,
    "base": pilarnet_multiparticle_base,
}

TYPE_NAMES = ["photon", "electron", "muon", "pion", "proton"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PILArNet particle model")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, choices=list(SINGLE_MODEL_FACTORIES.keys()), default="base")
    p.add_argument("--task_mode", type=str, choices=["single", "multi"], default="single")
    p.add_argument("--spatial_shape", type=int, nargs=3, default=[168, 168, 180])
    p.add_argument("--patch_size", type=int, nargs=3, default=[12, 12, 10])
    p.add_argument("--window_size", type=int, nargs=3, default=[2, 2, 3])
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--charge_metadata_path", type=str, default=None,
                   help="Transfer charge metadata pickle containing q_log1p stats.")
    p.add_argument("--num_cls", type=int, default=2)
    p.add_argument("--global_pool", action="store_true", default=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", type=str, default=None, help="Optional CSV output path")
    p.add_argument("--val_split", action="store_true", default=False,
                   help="Evaluate only on the held-out validation split "
                        "(10%%%%, seed=42) matching the training script.")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--grid_n", type=int, default=768,
                   help="Detector voxel grid size.")
    p.add_argument("--min_voxels", type=int, default=5)
    p.add_argument("--min_particles_per_event", type=int, default=2)
    p.add_argument("--max_particles_per_event", type=int, default=None)
    p.add_argument("--context_layers", type=int, default=2)
    p.add_argument("--context_heads", type=int, default=12)
    p.add_argument("--context_dropout", type=float, default=0.1)
    p.add_argument("--meta_dropout", type=float, default=0.1)
    p.add_argument("--single_particle_meta", action="store_true", default=False)
    return p.parse_args()


def resolve_checkpoint_path(checkpoint):
    path = Path(checkpoint)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")

    pattern = re.compile(r"acc=([0-9]+(?:\.[0-9]+)?)\.ckpt$")
    best_path = None
    best_acc = float("-inf")
    for candidate in path.glob("epoch=*-acc=*.ckpt"):
        match = pattern.search(candidate.name)
        if match is None:
            continue
        acc = float(match.group(1))
        if acc > best_acc:
            best_acc = acc
            best_path = candidate

    if best_path is not None:
        print(f"Resolved best validation-accuracy checkpoint: {best_path}")
        return str(best_path)

    last_path = path / "last.ckpt"
    if last_path.exists():
        print(f"Falling back to last checkpoint: {last_path}")
        return str(last_path)

    raise FileNotFoundError(f"No checkpoint files found under: {checkpoint}")


@torch.no_grad()
def run_inference(model, loader, device):
    """Run model over the full dataset and collect predictions + targets."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="Evaluating single", total=len(loader), ascii=True):
        # Move to device
        coords = batch["coords"].to(device)
        feats = batch["feats"].to(device)
        particle_meta = batch.get("particle_meta")
        if particle_meta is not None:
            particle_meta = particle_meta.to(device)
        B = batch["batch_size"]

        if coords.dtype != torch.int32:
            coords = coords.int()

        x_sp = SparseConvTensor(
            features=feats,
            indices=coords,
            spatial_shape=list(model.spatial_shape),
            batch_size=B,
        )

        outputs = model(x_sp, particle_meta)
        probs = outputs["out_pid"].softmax(dim=1).cpu()
        preds = probs.argmax(dim=1)

        all_preds.append(preds)
        all_labels.append(batch["type_label"])
        all_probs.append(probs)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    return preds, labels, probs


@torch.no_grad()
def run_inference_multi(model, loader, device):
    """Run the multi-particle model over the full dataset and collect predictions."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_h5_idx, all_event_idx, all_particle_idx = [], [], []

    for batch in tqdm(loader, desc="Evaluating multi", total=len(loader), ascii=True):
        coords = batch["coords"].to(device)
        feats = batch["feats"].to(device)
        particle_meta = batch["particle_meta"].to(device)
        event_offsets = batch["event_offsets"].to(device)
        B = batch["batch_size"]

        if coords.dtype != torch.int32:
            coords = coords.int()

        x_sp = SparseConvTensor(
            features=feats,
            indices=coords,
            spatial_shape=list(model.spatial_shape),
            batch_size=B,
        )

        outputs = model(x_sp, particle_meta, event_offsets)
        probs = outputs["out_pid"].softmax(dim=1).cpu()
        preds = probs.argmax(dim=1)

        all_preds.append(preds)
        all_labels.append(batch["type_label"])
        all_probs.append(probs)
        all_h5_idx.append(batch["h5_idx"])
        all_event_idx.append(batch["event_idx"])
        all_particle_idx.append(batch["particle_idx"])

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    metadata = {
        "h5_idx": torch.cat(all_h5_idx).numpy(),
        "event_idx": torch.cat(all_event_idx).numpy(),
        "particle_idx": torch.cat(all_particle_idx).numpy(),
    }
    return preds, labels, probs, metadata


def compute_predictive_entropy(probs, eps=1e-12):
    """Compute predictive entropy from class probabilities."""
    clipped = np.clip(probs.astype(np.float64, copy=False), eps, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def compute_paper_auroc(probs, preds, labels):
    """Compute the paper-style AUROC from predictive entropy."""
    correct = preds == labels
    n_correct = int(correct.sum())
    n_incorrect = int(len(labels) - n_correct)
    entropy = compute_predictive_entropy(probs)

    if n_correct == 0 or n_incorrect == 0:
        return float("nan")

    # The paper uses predictive entropy to separate correct from incorrect
    # predictions. Lower entropy means higher confidence, so negate it to use
    # the standard "higher score = more positive" AUROC convention.
    certainty_scores = torch.from_numpy((-entropy).astype(np.float32))
    correctness_targets = torch.from_numpy(correct.astype(np.int64))
    return float(binary_auroc(certainty_scores, correctness_targets))


def print_pid_metrics(preds, labels, probs):
    """Print classification metrics."""
    n = len(labels)
    correct = (preds == labels).sum()
    acc = correct / n
    paper_auroc = compute_paper_auroc(probs, preds, labels)

    print(f"\n{'='*50}")
    print(f"PID Classification Results  (N={n})")
    print(f"{'='*50}")
    print(f"Overall accuracy: {acc:.4f}  ({correct}/{n})")
    print(
        "AUROC (paper style; predictive entropy for correct vs incorrect): "
        f"{paper_auroc:.4f}"
        if np.isfinite(paper_auroc)
        else "AUROC (paper style; predictive entropy for correct vs incorrect): —"
    )
    print(
        "Paper alignment note: this matches the paper's entropy-based AUROC definition."
    )
    print("W1 and ECE remain omitted.")

    # Per-class
    print(f"\n{'Class':<15} {'Count':>7} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 42)
    for c in range(5):
        mask = labels == c
        cnt = mask.sum()
        if cnt == 0:
            print(f"{TYPE_NAMES[c]:<15} {cnt:>7}     —         —")
            continue
        cor = (preds[mask] == c).sum()
        print(f"{TYPE_NAMES[c]:<15} {cnt:>7} {cor:>8} {cor/cnt:>10.4f}")

    # Confusion matrix
    print(f"\nConfusion matrix (rows=true, cols=predicted):")
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(labels, preds):
        if 0 <= t < 5 and 0 <= p < 5:
            cm[t, p] += 1
    header = "         " + "  ".join(f"{TYPE_NAMES[i][:6]:>7}" for i in range(5))
    print(header)
    for i in range(5):
        row = f"{TYPE_NAMES[i]:<9}" + "  ".join(f"{cm[i,j]:>7}" for j in range(5))
        print(row)

    return acc

def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset_cls = (
        PILArNetParticleDataset
        if args.task_mode == "single"
        else PILArNetMultiParticleDataset
    )
    dataset_kwargs = dict(
        manifest_path=args.manifest,
        spatial_shape=tuple(args.spatial_shape),
        min_voxels=args.min_voxels,
        augment=False,
        cache_dir=args.cache_dir,
        grid_n=args.grid_n,
        charge_metadata_path=args.charge_metadata_path,
    )
    if args.task_mode == "single":
        dataset_kwargs["include_particle_meta"] = args.single_particle_meta
    else:
        dataset_kwargs.update(
            min_particles_per_event=args.min_particles_per_event,
            max_particles_per_event=args.max_particles_per_event,
        )
    full_ds = dataset_cls(**dataset_kwargs)

    if args.val_split:
        # Reproduce the exact train/val split used during training
        n_val = int(len(full_ds) * args.val_fraction)
        n_train = len(full_ds) - n_val
        _, ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        unit = "particles" if args.task_mode == "single" else "events"
        print(
            f"Using val split: {len(ds)} {unit} "
            f"({args.val_fraction*100:.0f}% of {len(full_ds)})"
        )
    else:
        ds = full_ds

    collate_fn = (
        pilarnet_collate_fn
        if args.task_mode == "single"
        else pilarnet_multiparticle_collate_fn
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model_kwargs = dict(
        spatial_shape=tuple(args.spatial_shape),
        patch_size=tuple(args.patch_size),
        window_size=tuple(args.window_size),
        num_cls=args.num_cls,
        global_pool=args.global_pool,
    )
    if args.task_mode == "single":
        factory = SINGLE_MODEL_FACTORIES[args.model]
        if args.single_particle_meta:
            model_kwargs.update(
                meta_dim=PILARNET_SINGLE_PARTICLE_META_DIM,
                meta_dropout=args.meta_dropout,
            )
    else:
        factory = MULTI_MODEL_FACTORIES[args.model]
        model_kwargs.update(
            meta_dropout=args.meta_dropout,
            context_layers=args.context_layers,
            context_heads=args.context_heads,
            context_dropout=args.context_dropout,
        )
    model = factory(**model_kwargs)

    # Load checkpoint (full Lightning checkpoint — contains argparse.Namespace)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    msg = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint load: {msg}")

    ema_state = ckpt.get("ema_state_dict")
    if ema_state is not None:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.0)
        ema.load_state_dict(ema_state)
        ema.copy_to(model.parameters())
        print("Applied EMA weights for evaluation.")

    model = model.to(device)

    unit = "particles" if args.task_mode == "single" else "events"
    print(f"\nRunning evaluation on {len(ds)} {unit}...")
    metadata = None
    if args.task_mode == "single":
        preds, labels, probs = run_inference(model, loader, device)
    else:
        preds, labels, probs, metadata = run_inference_multi(model, loader, device)

    acc = print_pid_metrics(preds, labels, probs)

    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            header = ["idx", "true_label", "pred_label", "true_class", "pred_class"]
            if metadata is not None:
                header = ["h5_idx", "event_idx", "particle_idx"] + header
            w.writerow(header)
            for i in range(len(preds)):
                row = [i, labels[i], preds[i], TYPE_NAMES[labels[i]], TYPE_NAMES[preds[i]]]
                if metadata is not None:
                    row = [
                        int(metadata["h5_idx"][i]),
                        int(metadata["event_idx"][i]),
                        int(metadata["particle_idx"][i]),
                    ] + row
                w.writerow(row)
        print(f"\nPer-particle predictions saved to {args.output}")


if __name__ == "__main__":
    main()
