"""
Description: fingerprint of the train/val/test split of a dataset copy.

Uses the training discovery, ordering and seed-7 85/5/10 split, so a match
with another machine's value (e.g. the --expected_split_sha256 used by the
flash launchers) means identical splits.  Usage:
    python -m train.split_fingerprint --dataset_path '/path/events_v7.0*'
"""

import argparse
from collections import Counter
from pathlib import Path

from evaluation.create_csv_v7_ft import deterministic_split_files, discover_event_files
from utils.funcs import split_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()
    files = discover_event_files(args.dataset_path)
    for directory, count in sorted(Counter(Path(f).parent.name for f in files).items()):
        print(f"{directory}: {count} events")
    train, val, test = deterministic_split_files(files)
    print(f"total: {len(files)} events ({len(train)} train / {len(val)} val / {len(test)} test)")
    print(f"split fingerprint: {split_fingerprint(train, val, test)}")


if __name__ == "__main__":
    main()
