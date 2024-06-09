#!/bin/bash

# Parameters
step_size=1.0
init_scale=0.1
iters=1000

# Jump values
jumps=(60 65 70 72 75 77 80 82 85 87 90 92 95)

# Loop through each jump value and run the Python script
for jump in "${jumps[@]}"; do
    echo "Running script with jump value: $jump"
    python eos_deep.py --step-size $step_size --jump $jump --init-scale $init_scale --iters $iters 
done
