#!/bin/bash
END=500
for ((i=0;i<END;i++)); do
    python3 run_cppn_3d.py --output_dir 3D_test/3d_1024_rgba --c_dim 4
done
