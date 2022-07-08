#!/bin/bash
wandb login dbf6e5f5f95834689237ec641e4bde9bf8f522f7
cd /opt/isaacgym/python/rlgpu
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo Number of GPUs: $NUM_GPUS

if [[ $NUM_GPUS > 1 ]]
then
    for ((i=0; i<$NUM_GPUS; i++))
    do
        echo Starting run on GPU $i
        $1 --headless --device_id $i --seed $i &
    done
else
    $1 --headless --device_id 0
fi

wait
echo All jobs are complete
