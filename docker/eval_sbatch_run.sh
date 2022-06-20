#!/bin/bash
set -e
set -u

docker run \
    --gpus \"device=${NV_GPU}\" \
    --cpuset-cpus="$(taskset -c -p $$ | cut -f2 -d ':' | awk '{$1=$1};1')" \
    --name isaacgym_container_$2 \
    --user=1068369:2626 \
    --mount type=bind,source=/nethome/jcoholich3/isaacgym/python/rlgpu/data,destination=/opt/isaacgym/python/rlgpu/data/ \
    isaacgym_$2 /opt/isaacgym/docker/eval_sbatch_docker_run.sh "$1"

echo slurm job all finished
exit
