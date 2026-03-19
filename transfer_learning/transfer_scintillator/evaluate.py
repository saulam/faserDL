"""Evaluation entry point for scintillator PID transfer."""

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import torch
from spconv.pytorch import SparseConvTensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .adapted_model import scintillator_encoder_base, scintillator_encoder_tiny
from .scintillator_dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    ScintillatorPIDDataset,
    scintillator_collate_fn,
)


MODEL_FACTORIES = {
    "tiny": scintillator_encoder_tiny,
    "base": scintillator_encoder_base,
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate scintillator PID model")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, choices=list(MODEL_FACTORIES.keys()), default="base")
    p.add_argument("--spatial_shape", type=int, nargs=3, default=[120, 120, 120])
    p.add_argument(
        "--charge_metadata_path",
        type=str,
        default=None,
        help="Transfer charge metadata pickle containing q_log1p stats.",
    )
    p.add_argument("--patch_size", type=int, nargs=3, default=[12, 12, 10])
    p.add_argument("--window_size", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--num_cls", type=int, default=2)
    p.add_argument("--global_pool", action="store_true", default=True)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--split",
        type=str,
        default="testing",
        choices=["training", "testing"],
        help="Which split to evaluate on (default: testing).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV output path for per-sample predictions.",
    )
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
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="Evaluating", total=len(loader), ascii=True):
        coords = batch["coords"].to(device)
        feats = batch["feats"].to(device)
        global_features = batch["global_features"].to(device)
        B = batch["batch_size"]

        if coords.dtype != torch.int32:
            coords = coords.int()

        x_sp = SparseConvTensor(
            features=feats,
            indices=coords,
            spatial_shape=list(model.spatial_shape),
            batch_size=B,
        )

        outputs = model(x_sp, global_features)
        probs = outputs["logits"].softmax(dim=1).cpu()
        preds = probs.argmax(dim=1)

        all_preds.append(preds)
        all_labels.append(batch["label"])
        all_probs.append(probs)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    return preds, labels, probs


def print_metrics(preds, labels):
    n = len(labels)
    correct = (preds == labels).sum()
    acc = correct / n

    print(f"\n{'=' * 55}")
    print(f"  PID Classification Results  (N={n:,})")
    print(f"{'=' * 55}")
    print(f"  Overall accuracy: {acc:.4f}  ({correct:,}/{n:,})")

    print(f"\n  {'Class':<12} {'Count':>8} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-' * 40}")
    for c in range(NUM_CLASSES):
        mask = labels == c
        cnt = mask.sum()
        if cnt == 0:
            print(f"  {CLASS_NAMES[c]:<12} {cnt:>8}       -         -")
            continue
        cor = (preds[mask] == c).sum()
        print(f"  {CLASS_NAMES[c]:<12} {cnt:>8} {cor:>8} {cor / cnt:>10.4f}")

    cm_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        if 0 <= true_label < NUM_CLASSES and 0 <= pred_label < NUM_CLASSES:
            cm_counts[pred_label, true_label] += 1

    col_sums = cm_counts.sum(axis=0, keepdims=True)
    cm_norm = np.divide(
        cm_counts.astype(np.float64),
        col_sums,
        out=np.zeros_like(cm_counts, dtype=np.float64),
        where=col_sums > 0,
    )

    print("\n  Confusion matrix (rows=predicted, cols=true; each true column sums to 1):")
    header = "           " + "  ".join(f"{CLASS_NAMES[i][:8]:>8}" for i in range(NUM_CLASSES))
    print(f"  {header}")
    for i in range(NUM_CLASSES):
        row = f"  {CLASS_NAMES[i]:<10}" + "  ".join(f"{cm_norm[i, j]:>8.4f}" for j in range(NUM_CLASSES))
        print(row)

    print()
    return acc, cm_norm


def main():
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    ds = ScintillatorPIDDataset(
        data_dir=args.data_dir,
        split=args.split,
        spatial_shape=tuple(args.spatial_shape),
        augment=False,
        charge_metadata_path=args.charge_metadata_path,
    )
    print(f"Evaluating on {args.split} split: {len(ds):,} samples")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=scintillator_collate_fn,
        pin_memory=True,
    )

    factory = MODEL_FACTORIES[args.model]
    model = factory(
        spatial_shape=tuple(args.spatial_shape),
        patch_size=tuple(args.patch_size),
        window_size=tuple(args.window_size),
        num_cls=args.num_cls,
        global_pool=args.global_pool,
    )

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in sd):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    msg = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint load: {msg}")

    model = model.to(device)

    print(f"Running inference on {len(ds):,} samples...")
    preds, labels, probs = run_inference(model, loader, device)
    _, _ = print_metrics(preds, labels)

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "idx",
                    "true_label",
                    "pred_label",
                    "true_class",
                    "pred_class",
                    *[f"prob_{class_name}" for class_name in CLASS_NAMES],
                ]
            )
            for i in range(len(preds)):
                writer.writerow([
                    i,
                    labels[i],
                    preds[i],
                    CLASS_NAMES[labels[i]],
                    CLASS_NAMES[preds[i]],
                    *probs[i].tolist(),
                ])
        print(f"Per-sample predictions saved to {args.output}")


if __name__ == "__main__":
    main()
