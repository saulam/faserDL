#!/bin/bash
# Extract the slim dataset shards (python -m dataset.slim_shards) into node
# memory, keeping the <directory>/<file> event names.  Run once per node:
#   srun --ntasks-per-node=1 bash clariden/stage_slim_dataset.sh SHARD_DIR TARGET_DIR
set -euo pipefail
src="$1"
dst="$2"
if [[ -f "$dst/.staged" ]] && cmp -s "$src/manifest.json" "$dst/.staged"; then
    echo "$(hostname): $dst is already staged"
    exit 0
fi
rm -rf "$dst"
mkdir -p "$dst"
start=$SECONDS
ls "$src"/shard-*.tar | xargs -P "${STAGE_PARALLEL:-32}" -I{} tar -xf {} -C "$dst"
cp "$src/manifest.json" "$dst/.staged"
echo "$(hostname): staged $(du -sh "$dst" | cut -f1) in $((SECONDS - start)) s"
