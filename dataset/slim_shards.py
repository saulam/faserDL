"""
Description: slim tar shards of a FASERCal dataset for stage-2 use (JEPA
pretraining without relational labels, fine-tuning, evaluation).

Each event keeps only the arrays SparseFASERCALDataset reads when stage1 is
off (STAGE2_ARRAYS), copied byte-for-byte, under the same
<directory>/<file> name.  Loading, the train/val/test split and its
fingerprint are therefore unchanged, while the data shrink ~10x (the truth
arrays dominate the files).  Shards let a job stage the whole dataset to node
memory with a few large sequential reads (clariden/stage_slim_dataset.sh)
instead of reading ~1.2M small files from Lustre every epoch.

    python -m dataset.slim_shards --dataset_path '/path/events_v7.0*' --out /path/slim --shards 64
"""

import argparse
import io
import json
import tarfile
import time
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

STAGE2_ARRAYS = (
    "run_number", "event_id", "reco_hits", "ghost_mask", "primary_vertex", "is_cc", "is_tau",
    "in_neutrino_pdg", "in_neutrino_energy", "vis_sp_momentum", "out_lepton_momentum",
    "jet_momentum", "tau_vis_momentum", "tau_decay_mode", "is_charmed", "charm_decay",
    "muspec_info", "ecal_hits", "ahcal_hits",
)
_EPOCH = (1980, 1, 1, 0, 0, 0)   # fixed timestamps: identical inputs give identical shards


def slim_npz(path):
    """The stage-2 arrays of one .npz event file, as a new .npz (bytes)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(buffer, "w") as target:
        for name in STAGE2_ARRAYS:
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, source.read(f"{name}.npy"))
    return buffer.getvalue()


def write_shard(job):
    index, files, out = job
    final = Path(out) / f"shard-{index:04d}.tar"
    if final.exists():
        return index, len(files), 0.0
    partial = final.with_suffix(".tar.partial")
    start = time.time()
    with tarfile.open(partial, "w", format=tarfile.GNU_FORMAT) as tar:
        for path in files:
            data = slim_npz(path)
            info = tarfile.TarInfo(f"{Path(path).parent.name}/{Path(path).name}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    partial.rename(final)
    return index, len(files), time.time() - start


def main():
    from evaluation.create_csv_v7_ft import deterministic_split_files, discover_event_files
    from utils.funcs import split_fingerprint

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset_path", required=True, help="training dataset path or glob")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    files = discover_event_files(args.dataset_path)
    if not files:
        raise SystemExit(f"No events found at {args.dataset_path}")
    args.out.mkdir(parents=True, exist_ok=True)
    size = -(-len(files) // args.shards)
    jobs = [(i, files[i * size:(i + 1) * size], str(args.out)) for i in range(args.shards) if files[i * size:(i + 1) * size]]
    done = 0
    with Pool(args.workers) as pool:
        for index, count, seconds in pool.imap_unordered(write_shard, jobs):
            done += count
            print(f"shard {index:04d}: {count} events in {seconds:.0f} s ({done}/{len(files)})", flush=True)

    train, val, test = deterministic_split_files(files)
    manifest = {
        "source": args.dataset_path,
        "events": len(files),
        "directories": dict(sorted(Counter(Path(f).parent.name for f in files).items())),
        "split_sha256": split_fingerprint(train, val, test),
        "shards": len(jobs),
        "arrays": list(STAGE2_ARRAYS),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
