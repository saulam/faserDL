import argparse
import os
import random
from glob import glob
from itertools import chain
from pathlib import Path

BUDGETS = [100, 300, 1000, 3000, 10000, 30000, 100000]
SEEDS = [1, 2, 3]
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"


def parse_args():
    parser = argparse.ArgumentParser(description="Create manifests for the data-efficiency study.")
    parser.add_argument(
        "--dataset-path",
        default=os.environ.get("DATASET_PATH"),
        help="Dataset directory or glob pattern used by the main training code.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MANIFEST_DIR,
        help="Directory where manifest files are written.",
    )
    parser.add_argument("--split-seed", type=int, default=7, help="Seed for the canonical split.")
    parser.add_argument(
        "--splits",
        nargs=3,
        type=float,
        default=[0.85, 0.05, 0.10],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train, validation and test fractions.",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=BUDGETS, help="Training budgets.")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Subset seeds.")
    args = parser.parse_args()

    if not args.dataset_path:
        parser.error("Set --dataset-path or DATASET_PATH.")
    if abs(sum(args.splits) - 1.0) > 1e-9:
        parser.error("--splits must sum to 1.0.")

    return args


def collect_files(dataset_path):
    files = sorted(
        set(
            chain(
                glob(os.path.join(dataset_path, "*.npz")),
                glob(os.path.join(dataset_path, "*", "*.npz")),
            )
        ),
        key=str.lower,
    )
    if not files:
        raise FileNotFoundError(f"No .npz files found for dataset path: {dataset_path}")
    return files


def canonical_split(all_files, seed, splits):
    import torch
    from torch.utils.data import random_split, TensorDataset

    n = len(all_files)
    train_len = int(n * splits[0])
    val_len = int(n * splits[1])
    test_len = n - train_len - val_len

    # Use a dummy dataset so random_split gives us indices
    dummy = TensorDataset(torch.zeros(n))
    train_sub, val_sub, test_sub = random_split(
        dummy,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )

    train_files = [all_files[i] for i in train_sub.indices]
    val_files = [all_files[i] for i in val_sub.indices]
    test_files = [all_files[i] for i in test_sub.indices]

    return train_files, val_files, test_files


def write_manifest(filepath, file_list):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as handle:
        for path in file_list:
            handle.write(path + "\n")


def main():
    args = parse_args()
    all_files = collect_files(args.dataset_path)
    train_files, val_files, test_files = canonical_split(all_files, args.split_seed, args.splits)

    print(f"Found {len(all_files)} files.")
    print(f"Split sizes: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    write_manifest(args.output_dir / "val.txt", val_files)
    write_manifest(args.output_dir / "test.txt", test_files)
    print(f"Wrote {args.output_dir / 'val.txt'}")
    print(f"Wrote {args.output_dir / 'test.txt'}")

    for budget in args.budgets:
        if budget > len(train_files):
            print(f"Warning: budget {budget} exceeds the training set size ({len(train_files)}); capping.")
            budget_actual = len(train_files)
        else:
            budget_actual = budget

        for seed in args.seeds:
            rng = random.Random(seed)
            subset = rng.sample(train_files, budget_actual)
            subset = sorted(subset, key=str.lower)

            manifest_path = args.output_dir / f"{budget}_events" / f"seed_{seed}.txt"
            write_manifest(manifest_path, subset)
            print(f"  budget={budget:>6d} seed={seed}: wrote {len(subset)} files to {manifest_path}")

    print("Finished writing manifests.")


if __name__ == "__main__":
    main()
