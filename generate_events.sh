#!/bin/bash

# Define the maximum number of jobs to run in parallel
MAX_JOBS=10

# Function to wait if there are MAX_JOBS running
function wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}

for i in {0..49}; do
    # Wait until there are fewer than MAX_JOBS running
    wait_for_jobs
    
    # Start a new process in the background
    /usr/bin/python3 read_root.py --number $i --chunks 50 --disable &

done

# Wait for all background jobs to finish before exiting the script
wait
