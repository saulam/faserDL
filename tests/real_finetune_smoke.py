"""One real-data downstream optimization step from a JEPA checkpoint."""

import argparse
import gc
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from dataset import SparseFASERCALMapDataset
from model import ViTFineTuner, vit_base, vit_tiny
from utils import collate, load_mae_encoder, move_obj


MODEL_FACTORIES = {"tiny": vit_tiny, "base": vit_base}


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


def _finetuner_args(batch_size):
    return SimpleNamespace(
        warmup_epochs=5,
        cosine_annealing_epochs=15,
        lr=None,
        blr=5e-4,
        cls_lr_scale=1.0,
        batch_size=batch_size,
        layer_decay=0.75,
        ema_decay=0.9999,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.05,
        eps=1e-8,
        kendall_w_max=5.0,
    )


def _all_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "checkpoints_final/pretrain_jepa_latent/loss_total_val/last.ckpt"
        ),
    )
    parser.add_argument("--encoder-source", choices=["auto", "context", "target"], default="target")
    parser.add_argument("--model", choices=MODEL_FACTORIES, default="tiny")
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    dataset = SparseFASERCALMapDataset(_dataset_args(args.dataset, args.metadata))
    batch = collate(
        [dataset[index] for index in range(min(args.batch_size, len(dataset)))],
        test=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_epoch = checkpoint.get("epoch")
    checkpoint_global_step = checkpoint.get("global_step")
    model = MODEL_FACTORIES[args.model](
        metadata=dataset.metadata,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        head_init=2e-5,
        head_dropout_cls=0.0,
        head_dropout_reg=0.0,
    )
    load_report = load_mae_encoder(
        model,
        checkpoint,
        encoder_source=args.encoder_source,
        require_full_encoder=True,
    )
    del checkpoint
    model = model.to(device).train()
    fine_tuner = ViTFineTuner(model=model, args=_finetuner_args(args.batch_size)).to(device)
    batch = move_obj(batch, device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in fine_tuner.parameters() if parameter.requires_grad],
        lr=1e-5,
    )
    first_encoder_parameter = next(model.fcal_patch_embed.parameters())
    first_head_parameter = next(model.heads["flavour"].parameters())
    encoder_before = first_encoder_parameter.detach().clone()
    head_before = first_head_parameter.detach().clone()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    batch_input, *batch_global, target = fine_tuner._arrange_batch(batch)
    output = fine_tuner(batch_input, batch_global)
    loss, part_losses, _, _ = fine_tuner.compute_losses(output, target)
    if not torch.isfinite(loss) or not _all_finite(output):
        raise RuntimeError("The downstream forward pass produced non-finite values.")
    loss.backward()
    encoder_has_grad = any(
        parameter.grad is not None for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in {
            id(head_parameter)
            for head in model.heads.values()
            for head_parameter in head.parameters()
        }
    )
    heads_have_grad = all(
        any(parameter.grad is not None for parameter in head.parameters())
        for head in model.heads.values()
    )
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    step_seconds = time.perf_counter() - start
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    )

    encoder_updated = not torch.equal(encoder_before, first_encoder_parameter.detach())
    head_updated = not torch.equal(head_before, first_head_parameter.detach())
    expected_outputs = {"out_flavour", "out_charm", "out_vis", "out_jet", "out_vertex"}
    if set(output) != expected_outputs:
        raise RuntimeError(f"Unexpected downstream task outputs: {sorted(output)}")

    # Exercise the exact Lightning state layout used by stage-2 checkpoints.
    with tempfile.TemporaryDirectory(prefix="faser_jepa_finetune_smoke_") as temp_dir:
        checkpoint_path = Path(temp_dir) / "finetune.ckpt"
        torch.save({"state_dict": fine_tuner.state_dict()}, checkpoint_path)
        saved_bytes = checkpoint_path.stat().st_size
        restored_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )["state_dict"]
        restored_model = MODEL_FACTORIES[args.model](
            metadata=dataset.metadata,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            head_init=2e-5,
            head_dropout_cls=0.0,
            head_dropout_reg=0.0,
        )
        restored = ViTFineTuner(
            model=restored_model, args=_finetuner_args(args.batch_size)
        )
        restored.load_state_dict(restored_state, strict=True)

    summary = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "encoder_source": load_report["source"],
        "downstream_model": args.model,
        "encoder_parameter_coverage": load_report["encoder_coverage"],
        "loaded_encoder_tensors": len(load_report["loaded"]),
        "missing_encoder_tensors": load_report["missing_encoder"],
        "batch_size": batch_input.batch_size,
        "tasks": sorted(name.removeprefix("out_") for name in output),
        "output_shapes": {
            name: (
                {subname: list(tensor.shape) for subname, tensor in value.items()}
                if isinstance(value, dict)
                else list(value.shape)
            )
            for name, value in output.items()
        },
        "loss": float(loss.detach().cpu()),
        "part_losses": part_losses,
        "finite": True,
        "encoder_has_grad": encoder_has_grad,
        "all_heads_have_grad": heads_have_grad,
        "encoder_updated": encoder_updated,
        "head_updated": head_updated,
        "optimization_step_seconds": step_seconds,
        "peak_memory_bytes": peak_memory,
        "finetune_checkpoint_bytes": saved_bytes,
        "finetune_checkpoint_strict_roundtrip": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    del fine_tuner, model, restored, restored_model, optimizer, output, loss
    gc.collect()


if __name__ == "__main__":
    main()
