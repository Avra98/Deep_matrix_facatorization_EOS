#!/bin/bash

# Define the combinations of alpha and initial_y
alphas=(0.01 0.05 0.1)
initial_ys=(0.1 0.2 0.5)
iterations=10000

# Directory to save the results
output_dir="gradient_descent_results"
mkdir -p $output_dir

# Loop over each combination of alpha and initial_y
for alpha in "${alphas[@]}"; do
    for initial_y in "${initial_ys[@]}"; do
        echo "Running gradient descent with alpha=${alpha}, initial_y=${initial_y}, iterations=${iterations}"

        # Run the Python script with the current parameters
        python scalar_gd_decoupled.py --alpha $alpha --initial_y $initial_y --iterations $iterations

        # Define the expected filenames
        png_filename="scalar_gd_decoupled_${alpha}_${initial_y}.png"
        svg_filename="scalar_gd_decoupled_${alpha}_${initial_y}.svg"

        # Move the result files to the output directory with unique names
        mv $png_filename $output_dir/scalar_gd_decoupled_${alpha}_${initial_y}.png
        mv $svg_filename $output_dir/scalar_gd_decoupled_${alpha}_${initial_y}.svg
    done
done

echo "All runs completed. Results saved in $output_dir"
