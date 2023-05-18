#!/bin/bash
END=10000
for ((i=0;i<END;i++)); do
    python3 examples/generate_image.py --conf examples/generate_2d_conf.yaml
done
