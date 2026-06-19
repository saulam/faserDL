from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import flatten_metrics


def load_metrics(run: Path, selection: str, split: str) -> dict:
    path = run / "evaluation" / selection / f"{split}_metrics.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def preferred_direction(name: str) -> str:
    higher_markers = ("accuracy", "precision", "recall", "f1")
    return "higher" if any(marker in name for marker in higher_markers) else "lower"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two-view and three-view metrics")
    parser.add_argument("--two-run", type=Path, required=True)
    parser.add_argument("--three-run", type=Path, required=True)
    parser.add_argument("--selection", default="taskwise")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {}
    markdown = [
        "# Two-view versus three-view comparison",
        "",
        "| Split | Metric | Two view | Three view | Three − two | Preferred |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split in ("val", "test"):
        two = flatten_metrics(load_metrics(args.two_run, args.selection, split))
        three = flatten_metrics(load_metrics(args.three_run, args.selection, split))
        rows = {}
        for metric in sorted(set(two).intersection(three)):
            if metric.startswith("runtime."):
                continue
            delta = three[metric] - two[metric]
            rows[metric] = {
                "two_view": two[metric],
                "three_view": three[metric],
                "delta": delta,
                "preferred": preferred_direction(metric),
            }
            markdown.append(
                f"| {split} | `{metric}` | {two[metric]:.6g} | "
                f"{three[metric]:.6g} | {delta:+.6g} | {preferred_direction(metric)} |"
            )
        comparison[split] = rows
    with (args.output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    (args.output_dir / "comparison.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(f"Comparison written to {args.output_dir}")


if __name__ == "__main__":
    main()
