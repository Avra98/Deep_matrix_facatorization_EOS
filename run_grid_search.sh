#!/bin/bash

# Bash script to run the Python gradient descent script with varying 's' values

# Define the range and number of grid points
start=8
end=9
num_points=10

# Calculate the step size
step=$(echo "scale=10; ($end - $start) / ($num_points - 1)" | bc)

# Loop over the grid points
for i in $(seq 0 $(($num_points - 1)))
do
    s=$(echo "$start + $i * $step" | bc)
    echo "Running with s=$s"
    python scalar_gd_decoupled.py --s $s
done
