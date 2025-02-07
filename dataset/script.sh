#!/bin/bash

# Define the maximum number of jobs to run in parallel
MAX_JOBS=10

# Function to wait if there are MAX_JOBS running
function wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}

# Loop
for i in {0..9}; do
    # Wait until there are fewer than MAX_JOBS running
    wait_for_jobs
    
    # Start a new process in the background
    python read_andre.py --number $i --chunks 10 --disable &

done

# Wait for all background jobs to finish before exiting the script
wait
