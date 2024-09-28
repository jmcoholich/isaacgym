#!/bin/bash -l
#SBATCH --qos short
#SBATCH -p kira-lab
#SBATCH -c 8
#SBATCH -G a40:1

conda activate rlgpu

$1
