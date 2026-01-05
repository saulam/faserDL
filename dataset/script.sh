#!/usr/bin/env bash

MAX_JOBS=30
CHUNKS=83  # number of reco files => indices 0..N-1

wait_for_jobs() {
    # count only running background jobs from this shell
    while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

for i in $(seq 0 $((CHUNKS-1))); do
    wait_for_jobs
    python read_root_v7.py --number "$i" --chunks "$CHUNKS" --disable &
done

wait
