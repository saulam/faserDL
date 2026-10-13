"""Inspect a completed JEPA checkpoint on real GENIE events.

This is an explicit diagnostic, not a pytest test: it loads a large checkpoint
and local data, checks EMA/checkpoint integrity, measures downstream transfer
coverage, and reports event-level representation diversity.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from dataset import SparseFASERCALMapDataset
from model import jepa_vit_base, jepa_vit_tiny, vit_base, vit_tiny
from utils import arrange_input, collate, load_mae_encoder, move_obj


JEPA_FACTORIES = {"tiny": jepa_vit_tiny, "base": jepa_vit_base}
FINETUNE_FACTORIES = {"tiny": vit_tiny, "base": vit_base}


def _dataset_args(dataset_path, metadata_path):
    return SimpleNamespace(
        train=True,
        stage1=False,
        augmentations_enabled=False,
        preprocessing_input="log",
        preprocessing_output="log",
        label_smoothing=0.02,
        mixup_alpha=0.0,
        dataset_path=str(dataset_path),
        metadata_path=str(metadata_path),
    )


def _representation_stats(vectors):
    vectors = vectors.detach().float().cpu()
    normalized_features = F.layer_norm(vectors, (vectors.shape[-1],))
    unit = F.normalize(normalized_features, dim=-1)
    similarities = unit @ unit.T
    if len(vectors) > 1:
        off_diagonal = similarities[~torch.eye(len(vectors), dtype=torch.bool)]
        centered = normalized_features - normalized_features.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered)
        power = singular_values.square()
        probabilities = power / power.sum().clamp_min(torch.finfo(power.dtype).eps)
        participation_rank = power.sum().square() / power.square().sum().clamp_min(
            torch.finfo(power.dtype).eps
        )
        entropy_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
        )
        cosine_mean = off_diagonal.mean()
        cosine_std = off_diagonal.std(unbiased=False)
    else:
        participation_rank = entropy_rank = vectors.new_tensor(1.0)
        cosine_mean = cosine_std = vectors.new_tensor(float("nan"))

    return {
        "events": len(vectors),
        "raw_norm_mean": vectors.norm(dim=-1).mean().item(),
        "raw_norm_std": vectors.norm(dim=-1).std(unbiased=False).item(),
        "cross_event_feature_std_mean": normalized_features.std(
            dim=0, unbiased=False
        ).mean().item(),
        "off_diagonal_cosine_mean": cosine_mean.item(),
        "off_diagonal_cosine_std": cosine_std.item(),
        "participation_rank": participation_rank.item(),
        "entropy_rank": entropy_rank.item(),
        "maximum_centered_rank": max(0, min(len(vectors) - 1, vectors.shape[-1])),
    }


def _checkpoint_integrity(checkpoint):
    state = checkpoint["state_dict"]
    context_prefix = "model.context_encoder."
    target_prefix = "model.target_encoder."
    context = {
        key[len(context_prefix):]: value
        for key, value in state.items()
        if key.startswith(context_prefix)
    }
    target = {
        key[len(target_prefix):]: value
        for key, value in state.items()
        if key.startswith(target_prefix)
    }
    if context.keys() != target.keys():
        raise RuntimeError("Context and target state dictionaries do not match.")

    finite = True
    difference_squared = 0.0
    context_squared = 0.0
    maximum_absolute_difference = 0.0
    for key, context_value in context.items():
        target_value = target[key]
        if torch.is_floating_point(context_value):
            finite &= bool(torch.isfinite(context_value).all())
            finite &= bool(torch.isfinite(target_value).all())
            difference = (target_value - context_value).double()
            difference_squared += difference.square().sum().item()
            context_squared += context_value.double().square().sum().item()
            maximum_absolute_difference = max(
                maximum_absolute_difference, difference.abs().max().item()
            )
    if not finite:
        raise RuntimeError("The checkpoint contains non-finite encoder tensors.")

    predictor_tensors = sum(
        key.startswith("model.predictor.") for key in state
    )
    return {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "all_encoder_tensors_finite": finite,
        "context_tensors": len(context),
        "target_tensors": len(target),
        "predictor_tensors": predictor_tensors,
        "optimizer_states": len(checkpoint.get("optimizer_states", [])),
        "scheduler_states": len(checkpoint.get("lr_schedulers", [])),
        "target_context_relative_l2": math.sqrt(
            difference_squared / max(context_squared, 1e-30)
        ),
        "target_context_max_absolute_difference": maximum_absolute_difference,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model", choices=JEPA_FACTORIES, default="tiny")
    parser.add_argument("--finetune-model", choices=FINETUNE_FACTORIES, default="tiny")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/scratch4/salonso/faser/events_v7.0_500_npz"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/scratch4/salonso/faser/events_v7.0_500_npz/metadata_stats.pkl"),
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    integrity = _checkpoint_integrity(checkpoint)
    dataset = SparseFASERCALMapDataset(_dataset_args(args.dataset, args.metadata))
    sample_count = min(args.samples, len(dataset))
    batch = move_obj(
        collate([dataset[index] for index in range(sample_count)], test=False),
        device,
    )
    sparse_input, *global_inputs = arrange_input(batch)

    state = {
        key[len("model."):]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    jepa = JEPA_FACTORIES[args.model](metadata=dataset.metadata)
    jepa.load_state_dict(state, strict=True)
    jepa = jepa.to(device).eval()
    with torch.no_grad():
        latents, latent_keep, *_ = jepa.target_encoder.forward_encoder(
            sparse_input, global_inputs, mask_ratio=0.0
        )
        denominator = latent_keep.sum(dim=1, keepdim=True).clamp_min(1).to(latents.dtype)
        pooled = (
            latents * latent_keep.unsqueeze(-1).to(latents.dtype)
        ).sum(dim=1) / denominator
    representation = _representation_stats(pooled)

    # Report the exact pretraining/fine-tuning transfer requested for the
    # controlled MAE-versus-JEPA comparison.
    del jepa
    downstream_reports = {}
    for model_name in dict.fromkeys((args.model, args.finetune_model)):
        downstream = FINETUNE_FACTORIES[model_name](
            metadata=dataset.metadata,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )
        report = load_mae_encoder(downstream, checkpoint, encoder_source="target")
        downstream_reports[model_name] = {
            "source": report["source"],
            "encoder_coverage": report["encoder_coverage"],
            "loaded_tensors": len(report["loaded"]),
            "missing_encoder_tensors": report["missing_encoder"],
            "unused_source_tensors": len(report["unused_source"]),
            "shape_mismatches": report["shape_mismatches"],
        }

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset_events": len(dataset),
        "sampled_events": sample_count,
        "integrity": integrity,
        "event_representation": representation,
        "downstream_transfer": downstream_reports,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
