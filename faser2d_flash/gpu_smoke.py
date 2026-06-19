from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from .config import load_config, pipeline_path
from .dataset import ProjectionDataset, collate_events, load_metadata
from .losses import MultiTaskObjective
from .model import ProjectionTransformer, parameter_counts
from .runtime import require_flash_runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-data bf16 FlashAttention forward/backward smoke checks"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if config["model"]["attention_backend"] != "flash":
        raise ValueError("gpu_smoke requires model.attention_backend=flash")
    runtime = require_flash_runtime(config["training"]["precision"])
    device = torch.device("cuda")
    metadata = load_metadata(pipeline_path(config["data"]["metadata_path"]))
    if bool(metadata.get("remove_primary_origin_pixel", False)) != bool(
        config["data"].get("remove_primary_origin_pixel", True)
    ):
        raise ValueError(
            "Config origin-pixel policy does not match metadata. Rebuild metadata "
            "after changing data.remove_primary_origin_pixel."
        )
    manifest = pipeline_path(config["data"]["manifests_dir"]) / "train.txt"
    results = {"runtime": runtime, "modes": {}}

    for mode in ("xz_yz", "xz_yz_xy"):
        data_config = dict(config["data"])
        data_config["input_mode"] = mode
        dataset = ProjectionDataset(
            manifest,
            metadata,
            data_config,
            training=False,
            seed=int(config["experiment"]["seed"]),
        )
        batch = collate_events(
            [dataset[index] for index in range(min(args.batch_size, len(dataset)))]
        ).to(device)
        model = ProjectionTransformer(config["model"], metadata).to(device)
        objective = MultiTaskObjective(
            metadata,
            label_smoothing=float(config["training"]["label_smoothing"]),
            kendall_weight_min=float(config["training"]["kendall_weight_min"]),
            kendall_weight_max=float(config["training"]["kendall_weight_max"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(objective.parameters()), lr=1e-4
        )

        # First pass pays CUDA/kernel initialization costs; the second is timed.
        for measured in (False, True):
            optimizer.zero_grad(set_to_none=True)
            if measured:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch)
                loss, _ = objective(outputs, batch.targets)
            loss.backward()
            optimizer.step()
            if measured:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started

        results["modes"][mode] = {
            "attention_backend": model.attention_backend,
            "parameters": parameter_counts(model),
            "events": batch.batch_size,
            "tokens": batch.num_tokens,
            "patch_counts": batch.patch_counts.tolist(),
            "loss": float(loss.detach()),
            "step_seconds": elapsed,
            "events_per_second": batch.batch_size / elapsed,
            "tokens_per_second": batch.num_tokens / elapsed,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "output_shapes": {
                "out_flavour": list(outputs["out_flavour"].shape),
                "out_charm": list(outputs["out_charm"].shape),
                "out_vis": list(outputs["out_vis"]["p_cart"].shape),
                "out_jet": list(outputs["out_jet"]["p_cart"].shape),
                "out_vertex": list(outputs["out_vertex"].shape),
            },
        }
        del batch, model, objective, optimizer, outputs, loss
        gc.collect()
        torch.cuda.empty_cache()

    text = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
