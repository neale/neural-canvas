#!/bin/bash
END=1000
for ((i=0;i<END;i++)); do
    python3 examples/generate_volume.py --conf examples/generate_3d_conf.yaml
done
