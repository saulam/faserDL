"""GPU throughput and memory of flash JEPA pretraining and flash fine-tuning
steps on cached real batches, through the real Lightning modules but without
data loading (the batches are collated once and cached).

Not collected by pytest: it needs the local FASERCal events and a GPU.
Examples (dlnu, inside the neutrino-jepa environment):

    python -m tests.bench_flash_step --mode pretrain --gpu 0
    python -m tests.bench_flash_step --mode finetune --gpu 1 --profile

The model settings are the recommended recipe (K = 16 targets, predictor
QK-norm; fine-tuning layer decay 0.85).  Optimiser hyperparameters do not
change the compute.  The first call per (mode, batch size, batches) builds the
cache from random training events with the training augmentations.
"""

import argparse
import os
import statistics
import sys
import time
from functools import partial

DATA = "/scratch/salonso/sparse-nns/faser_develop/events_v7.0_500_npz"
CACHE = "/scratch4/salonso/faser/bench_cache"


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=None, help="default 2048 (pretrain) / 512 (finetune)")
    p.add_argument("--batches", type=int, default=6, help="distinct cached batches, cycled")
    p.add_argument("--steps", type=int, default=36, help="timed steps (whole cycles of the batches are fairest)")
    p.add_argument("--warmup", type=int, default=6, help="untimed steps first (compilation)")
    p.add_argument("--profile", action="store_true", help="print the top ops of 3 more steps")
    p.add_argument("--profile_sort", default="self_cuda_time_total", help="e.g. self_cpu_time_total")
    p.add_argument("--no_log", action="store_true", help="skip self.log (measures Lightning's logging overhead)")
    p.add_argument("--cache", default=CACHE)
    return p.parse_args(argv)


def training_args(mode, batch_size):
    from utils import add_flash_args, ini_argparse
    from train.pretrain_flash import MODEL_PRESETS
    argv = ["--stage2", "--augmentations_enabled", "--dataset_path", DATA,
            "--metadata_path", f"{DATA}/metadata_stats.pkl", "--model", "base",
            "--batch_size", str(batch_size), "--preprocessing_input", "log", "--preprocessing_output", "log",
            "--epochs", "1", "--warmup_epochs", "0", "--cosine_annealing_epochs", "0", "--num_workers", "32"]
    if mode == "pretrain":
        argv += ["--blr", "1e-4", "--beta2", "0.95", "--flash_target_layers", "16", "--flash_predictor_qk_norm"]
    else:
        argv += ["--blr", "5e-4", "--layer_decay", "0.85", "--drop_path_rate", "0.2", "--label_smoothing", "0.02"]
    args = add_flash_args(ini_argparse(MODEL_PRESETS)).parse_args(argv)
    args.backbone = "flash"
    return args


def cached_batches(args, n, cache_dir):
    import torch
    from torch.utils.data import DataLoader
    from dataset import SparseFASERCALMapDataset
    from model.flash_vit import collate_tokens

    path = f"{cache_dir}/batches_{args.batch_size}x{n}.pt"
    dataset = SparseFASERCALMapDataset(args)
    dataset.stage1 = False
    if os.path.exists(path):
        return torch.load(path, weights_only=False), dataset.metadata
    os.makedirs(cache_dir, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    pick = torch.randperm(len(dataset.data_files), generator=g)[: n * args.batch_size].tolist()
    dataset.data_files = [dataset.data_files[i] for i in pick]
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        collate_fn=partial(collate_tokens, test=False), drop_last=True)
    batches = list(loader)
    torch.save(batches, path)
    return batches, dataset.metadata


def main(argv=None):
    cli = parse(argv)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # as the training scripts: nvidia-smi numbering
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu)
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset

    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    batch_size = cli.batch_size or (2048 if cli.mode == "pretrain" else 512)
    args = training_args(cli.mode, batch_size)
    batches, metadata = cached_batches(args, cli.batches, cli.cache)

    pl.seed_everything(7, workers=True)
    if cli.mode == "pretrain":
        from model.lightning_model_flash import FlashJEPAPreTrainer
        from train.pretrain_flash import build_flash_jepa
        module = FlashJEPAPreTrainer(model=build_flash_jepa(args), args=args)
    else:
        from model.lightning_model_flash import FlashViTFineTuner
        from train.finetune_flash import build_flash_vit
        module = FlashViTFineTuner(model=build_flash_vit(args, metadata), args=args)

    if cli.no_log:
        module.log = lambda *a, **k: None
    total = cli.warmup + cli.steps + (3 if cli.profile else 0)

    class Cached(Dataset):
        def __len__(self):
            return total

        def __getitem__(self, i):
            return batches[i % len(batches)]

    class Timer(pl.Callback):
        def __init__(self):
            self.ends, self.losses, self.prof = [], [], None

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if batch_idx == cli.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self.ends.append(time.perf_counter())
            if cli.profile and batch_idx == cli.warmup + cli.steps:
                self.prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                self.prof.__enter__()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            self.losses.append(float(loss))           # synchronises
            if cli.warmup <= batch_idx < cli.warmup + cli.steps:
                self.ends.append(time.perf_counter())
                if batch_idx == cli.warmup + cli.steps - 1:
                    self.peak = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
            if self.prof is not None and batch_idx == total - 1:
                torch.cuda.synchronize()
                self.prof.__exit__(None, None, None)   # before Lightning's teardown copies

    timer = Timer()
    trainer = pl.Trainer(
        max_epochs=1, limit_val_batches=0, num_sanity_val_steps=0, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        precision="bf16-mixed", accelerator="gpu", devices=1, gradient_clip_val=1.0,
        gradient_clip_algorithm="norm", callbacks=[timer],
    )
    loader = DataLoader(Cached(), batch_size=None, num_workers=0, pin_memory=True)
    trainer.fit(module, train_dataloaders=loader)

    steps = [b - a for a, b in zip(timer.ends, timer.ends[1:])]
    step = statistics.median(steps)
    name = torch.cuda.get_device_name(0)
    flags = {k: v for k, v in os.environ.items() if k.startswith("FLASH_") or k == "PYTORCH_CUDA_ALLOC_CONF"}
    print(f"\n{cli.mode} on {name}, batch {batch_size}, flags {flags or '{}'}")
    print(f"  step: median {step * 1e3:.1f} ms, mean {statistics.mean(steps) * 1e3:.1f} ms, "
          f"min {min(steps) * 1e3:.1f} ms over {len(steps)} steps; {batch_size / statistics.mean(steps):.0f} events/s")
    print(f"  peak memory: {timer.peak[0] / 2**30:.1f} GiB allocated, {timer.peak[1] / 2**30:.1f} GiB reserved")
    print(f"  losses: {' '.join(f'{v:.5f}' for v in timer.losses[:8])}")
    if timer.prof is not None:
        print(timer.prof.key_averages().table(sort_by=cli.profile_sort, row_limit=25, max_name_column_width=60))


if __name__ == "__main__":
    sys.exit(main())
