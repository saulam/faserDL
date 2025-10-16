#!/usr/bin/env bash

MAX_JOBS=20
CHUNKS=116  # number of reco files => indices 0..115

wait_for_jobs() {
    # count only running background jobs from this shell
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

# Brace expansion can't use variables, so use seq (or a C-style for loop).
for i in $(seq 0 $((CHUNKS-1))); do
    wait_for_jobs
    python read_root.py --number "$i" --chunks "$CHUNKS" --disable &
done

wait
