#!/bin/bash
END=1000
for ((i=0;i<END;i++)); do
    python3 examples/render_volumes.py --ignore_seed --random_settings --x_dim 512 --y_dim 512 --z_dim 512 --splits 8
done
