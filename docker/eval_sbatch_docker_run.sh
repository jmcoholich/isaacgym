#!/bin/bash
# wandb login dbf6e5f5f95834689237ec641e4bde9bf8f522f7
wandb disabled
cd /opt/isaacgym/python/rlgpu
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo Number of GPUs: $NUM_GPUS
eval $1

wait
echo All jobs are complete
