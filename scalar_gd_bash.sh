#!/bin/bash

# Define ranges for alpha, initial y, and power N
alpha_values=(0.01 0.02 0.05)
initial_y_values=(0.1 0.2 0.5)
power_N_values=(3)

# Loop over each combination of values
for alpha in "${alpha_values[@]}"; do
    for initial_y in "${initial_y_values[@]}"; do
        for power_N in "${power_N_values[@]}"; do
            echo "Running with alpha = $alpha, initial_y = $initial_y, power_N = $power_N"
            python scalar_gd.py --alpha $alpha --initial_y $initial_y --power_N $power_N
        done
    done
done
