#!/bin/bash
set -e
set -u

docker run \
    --gpus \"device=${NV_GPU}\" \
    --cpuset-cpus="$(taskset -c -p $$ | cut -f2 -d ':' | awk '{$1=$1};1')" \
    --name isaacgym_container_$2 \
    --mount type=bind,source=/nethome/jcoholich3/isaacgym/python/rlgpu/data,destination=/opt/isaacgym/python/rlgpu/data/ \
    isaacgym_$2 /opt/isaacgym/docker/eval_sbatch_docker_run.sh "$1" $3 $4 $5 $6

echo Finished evaluation run, copying data now

# Now move the data from SkyNet to my personal workstation
TEST="$1"
if [[ "${TEST:0:1}" == "H" ]]
then
    DIR_NAME=$(python -c "print('$1'.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '') + 'dd_' + '$3' + '_bb_' + '$4'.replace('.', '_') + '_dist_' + str('$5').replace('.', '_') + '_width_' + str('$6').replace('.', '_'))")
else
    DIR_NAME=$(python -c "print('$1'.replace(' ', '_').replace('(', '').replace(')', '').replace('.', ''))")
fi

DATA_DIR="python/rlgpu/data/$DIR_NAME"
PWS_DIR="jcoholich@143.215.128.197:/home/jcoholich/isaacgym/python/rlgpu/data/"  # personal workstation directory
echo Moving data from:
echo $DATA_DIR
echo
echo to:
echo $PWS_DIR
rsync -r $DATA_DIR $PWS_DIR

if [ $? -eq 0 ]; then
    echo Files copied successfully. Restarting docker container to delete files as root.
    docker start isaacgym_container_$2
    docker exec isaacgym_container_$2 bash -c "rm -r /opt/isaacgym/python/rlgpu/data/$DIR_NAME"
    docker stop isaacgym_container_$2

else
    echo Something went wrong copying files with rsync
fi

echo slurm job all finished
exit
