#!/bin/bash

# Define the maximum number of jobs to run in parallel
MAX_JOBS=20

# Function to wait if there are MAX_JOBS running
function wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}

# Loop
for i in {0..8}; do
    # Wait until there are fewer than MAX_JOBS running
    wait_for_jobs
    
    # Start a new process in the background
    python read_root.py --number $i --chunks 8 --disable &

done

# Wait for all background jobs to finish before exiting the script
wait

