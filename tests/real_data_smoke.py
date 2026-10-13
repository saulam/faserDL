"""Small real-GENIE smoke test for the MAE and JEPA pretraining paths.

This is intentionally not collected by pytest because it depends on the local
FASERCal datasets and is much heavier than a unit test.
"""

import argparse
import gc
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from dataset import SparseFASERCALMapDataset
from model import jepa_vit_tiny, mae_vit_tiny, vit_tiny
from model.lightning_model_pretrain import MAEPreTrainer
from utils import arrange_input, collate, load_mae_encoder, move_obj


def _dataset_args(dataset_path, metadata_path):
    return SimpleNamespace(
        train=True,
        stage1=True,
        augmentations_enabled=False,
        preprocessing_input="log",
        preprocessing_output="log",
        label_smoothing=0.02,
        mixup_alpha=0.0,
        dataset_path=str(dataset_path),
        metadata_path=str(metadata_path),
    )


def _count(module, trainable_only=False):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _mae_pretrainer_args():
    """Published-control loss settings needed for a standalone training step."""
    return SimpleNamespace(
        mask_ratio=0.75,
        warmup_epochs=0,
        cosine_annealing_epochs=0,
        lr=1e-4,
        blr=None,
        batch_size=1,
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.05,
        eps=1e-8,
        preprocessing_input="log",
        label_smoothing=0.02,
        reconstruction_loss_mode="hybrid",
        reconstruction_chamfer_weight=0.3,
        reconstruction_distance_reg_weight=0.3,
        reconstruction_max_distance_fcal=5.0,
        reconstruction_max_distance_ahcal=3.0,
        reconstruction_gamma_distance=2.0,
        relational_pass_prob=0.0,
        relational_mask_ratio=0.25,
        relational_voxel_keep_prob=0.5,
        relational_pass_seed=42,
        relational_loss_mode="standard",
        relational_distance_weight=0.3,
        relational_max_distance=3.0,
        relational_gamma_distance=2.0,
    )


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed(device, function):
    _synchronize(device)
    start = time.perf_counter()
    result = function()
    _synchronize(device)
    return result, time.perf_counter() - start


def _reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device):
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _freeze_sparse_embeddings_for_cpu(model):
    """Work around generic spconv's lack of a CPU convolution backward kernel."""
    model.context_encoder.fcal_patch_embed.requires_grad_(False)
    model.context_encoder.ahcal_patch_embed.requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default="/scratch4/salonso/faser", type=Path
    )
    parser.add_argument(
        "--metadata",
        default="/scratch4/salonso/faser/events_v7.0_500_npz/metadata_stats.pkl",
        type=Path,
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    dataset_names = [
        "events_v7.0_500_npz",
        "events_v7.0_501_npz",
        "events_v7.0_tau_npz",
    ]
    dataset_summary = {}
    first_dataset = None
    first_sample = None
    for name in dataset_names:
        dataset = SparseFASERCALMapDataset(
            _dataset_args(args.data_root / name, args.metadata)
        )
        sample = dataset[0]
        dataset_summary[name] = {
            "files": len(dataset),
            "fcal_hits_first_event": int(sample["feats"].shape[0]),
            "ahcal_hits_first_event": int(sample["ahcal_hits_feats"].shape[0]),
        }
        if first_dataset is None:
            first_dataset = dataset
            first_sample = sample

    batch = move_obj(collate([first_sample], test=False), device)
    fcal, *global_inputs = arrange_input(batch)
    fcal = move_obj(fcal, device)
    global_inputs = move_obj(tuple(global_inputs), device)

    mae_model = mae_vit_tiny(metadata=first_dataset.metadata).to(device)
    mae_model.train()
    mae_params = _count(mae_model, trainable_only=True)
    def _mae_forward():
        with torch.no_grad():
            return mae_model(
                fcal,
                global_inputs,
                mask_ratio=0.75,
                do_relational=False,
            )

    mae_output, mae_forward_seconds = _elapsed(
        device,
        _mae_forward,
    )
    mae_predictions = mae_output[0]
    mae_shapes = {name: list(value.shape) for name, value in mae_predictions.items()}
    mae_finite = all(torch.isfinite(value).all().item() for value in mae_predictions.values())
    del mae_output, mae_predictions

    mae_training_step_seconds = None
    mae_peak_memory_bytes = None
    mae_training_loss = None
    if device.type == "cuda":
        mae = MAEPreTrainer(
            model=mae_model,
            dataset=first_dataset,
            args=_mae_pretrainer_args(),
        ).to(device)
        # Only the loss helper needs global_step; no Trainer/logging hooks are
        # invoked by this standalone benchmark.
        mae._trainer = SimpleNamespace(global_step=0)
        mae_optimizer = torch.optim.AdamW(
            [parameter for parameter in mae.parameters() if parameter.requires_grad],
            lr=1e-4,
        )

        def _mae_step():
            mae_optimizer.zero_grad(set_to_none=True)
            batch_input, *batch_global, labels = mae._arrange_batch(batch)
            ahcal_sparse = batch_global[0]
            outputs = mae.forward(
                batch_input,
                batch_global,
                mask_ratio=0.75,
                do_relational=False,
                relational_mask_ratio=0.25,
            )
            preds, idx_fcal, idx_fcal_rel, idx_ahcal, glob_tgts, glob_masks, aux = outputs
            loss, *_ = mae.compute_losses(
                preds=preds,
                targ_reg=batch_input.features,
                targ_reg_ahcal=ahcal_sparse.features,
                idx_targets_fas_masked=idx_fcal,
                idx_targets_fas_rel=idx_fcal_rel,
                idx_targets_ahcal=idx_ahcal,
                labels=labels,
                glob_targets=glob_tgts,
                glob_masks=glob_masks,
                did_relational=aux["did_relational"],
                relational_mask_ratio=aux["relational_mask_ratio"],
                do_muon=True,
            )
            loss.backward()
            mae_optimizer.step()
            return loss

        # Warm-up initializes CUDA kernels and AdamW state; report the next
        # steady-state step and its total peak allocated memory.
        _mae_step()
        mae_optimizer.zero_grad(set_to_none=True)
        _reset_peak_memory(device)
        mae_loss_tensor, mae_training_step_seconds = _elapsed(device, _mae_step)
        mae_peak_memory_bytes = _peak_memory(device)
        mae_training_loss = float(mae_loss_tensor.detach().cpu())
        if not torch.isfinite(mae_loss_tensor):
            raise RuntimeError("The MAE smoke training step produced a non-finite loss.")
        del mae_loss_tensor, mae_optimizer, mae

    del mae_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    jepa = jepa_vit_tiny(metadata=first_dataset.metadata).to(device)
    jepa_context_trainable = _count(jepa.context_encoder, trainable_only=True)
    jepa_predictor_trainable = _count(jepa.predictor, trainable_only=True)
    if device.type == "cpu":
        _freeze_sparse_embeddings_for_cpu(jepa)
    jepa.train()

    def _jepa_forward():
        with torch.no_grad():
            return jepa(fcal, global_inputs, mask_ratio=0.75)

    jepa_forward_output, jepa_forward_seconds = _elapsed(device, _jepa_forward)
    if not torch.isfinite(jepa_forward_output["loss"]):
        raise RuntimeError("The JEPA smoke forward produced a non-finite loss.")
    del jepa_forward_output

    optimizer = torch.optim.AdamW(
        [parameter for parameter in jepa.parameters() if parameter.requires_grad],
        lr=1e-4,
    )
    first_target_parameter = next(jepa.target_encoder.parameters())

    def _jepa_step():
        optimizer.zero_grad(set_to_none=True)
        output = jepa(fcal, global_inputs, mask_ratio=0.75)
        output["loss"].backward()
        optimizer.step()
        jepa.update_target_encoder(0.996)
        return output

    # Warm-up initializes kernels and AdamW state before the measured step.
    _jepa_step()
    optimizer.zero_grad(set_to_none=True)
    target_before = first_target_parameter.detach().clone()
    _reset_peak_memory(device)
    jepa_output, jepa_step_seconds = _elapsed(device, _jepa_step)
    jepa_peak_memory_bytes = _peak_memory(device)
    jepa_loss = float(jepa_output["loss"].detach().cpu())
    target_slots = int(jepa_output["prediction_mask"].sum())
    valid_target_slots = int(jepa_output["target_latent_keep"].sum())
    target_updated = not torch.equal(target_before, first_target_parameter.detach())
    context_has_grad = any(
        parameter.grad is not None
        for parameter in jepa.context_encoder.parameters()
        if parameter.requires_grad
    )
    predictor_has_grad = any(
        parameter.grad is not None for parameter in jepa.predictor.parameters()
    )
    target_has_grad = any(
        parameter.grad is not None for parameter in jepa.target_encoder.parameters()
    )

    parameter_summary = {
        "mae_trainable": mae_params,
        "mae_encoder_trainable": jepa_context_trainable,
        "mae_reconstruction_machinery": mae_params
        - jepa_context_trainable,
        "jepa_trainable": jepa_context_trainable + jepa_predictor_trainable,
        "jepa_predictor": jepa_predictor_trainable,
        "jepa_smoke_trainable_after_device_adjustments": _count(
            jepa, trainable_only=True
        ),
        "jepa_frozen_target": _count(jepa.target_encoder, trainable_only=False),
        "jepa_stored_total": _count(jepa, trainable_only=False),
    }

    with tempfile.TemporaryDirectory(prefix="faser_jepa_smoke_") as temp_dir:
        checkpoint_path = Path(temp_dir) / "jepa.ckpt"
        checkpoint = {
            "state_dict": {
                f"model.{name}": value.detach().cpu()
                for name, value in jepa.state_dict().items()
            }
        }
        torch.save(checkpoint, checkpoint_path)
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        downstream = vit_tiny(
            metadata=first_dataset.metadata,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
        )
        load_report = load_mae_encoder(
            downstream,
            loaded_checkpoint,
            encoder_source="auto",
        )
        non_head_missing = [
            name for name in load_report["missing"] if not name.startswith("heads.")
        ]
        if non_head_missing:
            raise RuntimeError(
                f"JEPA checkpoint left downstream encoder tensors missing: {non_head_missing}"
            )
        checkpoint_bytes = checkpoint_path.stat().st_size

    summary = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device_capability": (
            list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None
        ),
        "cpu_sparse_embeddings_frozen_for_backward": device.type == "cpu",
        "datasets": dataset_summary,
        "mae": {
            "forward_seconds": mae_forward_seconds,
            "finite": mae_finite,
            "training_loss": mae_training_loss,
            "training_step_seconds": mae_training_step_seconds,
            "peak_memory_bytes": mae_peak_memory_bytes,
            "prediction_shapes": mae_shapes,
        },
        "jepa": {
            "forward_seconds": jepa_forward_seconds,
            "training_step_seconds": jepa_step_seconds,
            "peak_memory_bytes": jepa_peak_memory_bytes,
            "loss": jepa_loss,
            "finite": bool(torch.isfinite(jepa_output["loss"]).item()),
            "prediction_shape": list(jepa_output["prediction"].shape),
            "target_slots": target_slots,
            "valid_target_slots": valid_target_slots,
            "target_coverage": target_slots / max(1, valid_target_slots),
            "context_has_grad": context_has_grad,
            "predictor_has_grad": predictor_has_grad,
            "target_has_grad": target_has_grad,
            "target_updated": target_updated,
        },
        "parameters": parameter_summary,
        "checkpoint": {
            "bytes": checkpoint_bytes,
            "downstream_encoder_source": load_report["source"],
            "downstream_tensors_loaded": len(load_report["loaded"]),
            "downstream_missing_task_head_tensors": len(load_report["missing"]),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    del jepa, optimizer, downstream, loaded_checkpoint, checkpoint
    gc.collect()


if __name__ == "__main__":
    main()
