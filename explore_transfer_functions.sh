#!/bin/bash
END=1000
for ((i=0;i<END;i++)); do
    python3 examples/render_volumes.py --ignore_seed --random_settings --x_dim 256 --y_dim 256 --z_dim 256 --splits 1
done
