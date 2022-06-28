#!/bin/bash
wandb login dbf6e5f5f95834689237ec641e4bde9bf8f522f7
wandb disabled
cd /opt/isaacgym/python/rlgpu
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo Number of GPUs: $NUM_GPUS
echo $1
python evaluate_policy.py --run_name="$1" --des_dir_coef="$2" --box_len="$3" --nn_ft_dist="$4" --nn_ft_width="$5"

wait
echo All jobs are complete
