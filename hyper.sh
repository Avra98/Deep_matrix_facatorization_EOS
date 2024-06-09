#!/bin/bash

FILE="./output_directory"  # Replace with your desired output directory path
mkdir -p $FILE

alpha=1e-4  # Initialize alpha
beta=0.0    # Initialize beta
COUNTER=0

# Loop over hyperparameters
for optima in "GD"; do
    for fac in 3 4 5 6 7 8; do
        if [ "$fac" -gt 5 ]; then
            num_epochs=50000
        else
            num_epochs=15000
        fi

        for non_linearity in None 0.1; do
            if [ "$non_linearity" = "None" ]; then
                python3 -u deep_mat.py --optima $optima --fac $fac --alpha $alpha --beta $beta --num_epochs $num_epochs \
                >> "$FILE/fac_${fac}_alpha_${alpha}_non_linearity_${non_linearity}.out" &
            else
                python3 -u deep_mat.py --optima $optima --fac $fac --alpha $alpha --beta $beta --num_epochs $num_epochs \
                --non_linearity $non_linearity \
                >> "$FILE/fac_${fac}_alpha_${alpha}_non_linearity_${non_linearity}.out" &
            fi

            let COUNTER=COUNTER+1
            if [ $((COUNTER % 10)) -eq 0 ]; then
                wait
            fi
        done
    done
done
