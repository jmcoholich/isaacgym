#!/bin/bash
set -e
set -u

docker run \
    --gpus \"device=${NV_GPU}\" \
    --cpuset-cpus="$(taskset -c -p $$ | cut -f2 -d ':' | awk '{$1=$1};1')" \
    --name isaacgym_container_$2 \
    --mount type=bind,source=/nethome/jcoholich3/isaacgym/python/rlgpu/nn,destination=/opt/isaacgym/python/rlgpu/nn/ \
    isaacgym_$2 /opt/isaacgym/docker/sbatch_docker_run.sh "$1"

echo slurm job all finished
exit
