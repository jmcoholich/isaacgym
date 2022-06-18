#!/bin/bash -l

# if [[ $(docker info | sed -n '15 p') != " Server Version: 20.10.7" ]]; then
#     echo On node $(hostname)
#     echo docker version is: $(docker info | sed -n '15 p')
#     echo exiting ...
#     exit
# fi
hostname
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# cd ~/rl_games
# git checkout master
# git pull origin master

# cd ~/isaacgym
# git checkout master
# git pull origin master

bash docker/build.sh $2
docker rm isaacgym_container_$2
bash docker/eval_sbatch_run.sh "$1" $2 # this script needs to do all the work

# Now move the data from SkyNet to my personal workstation
DIR_NAME=$(python -c "print('$1'.replace(' ', '_').replace('(', '').replace(')', '').replace('.', ''))")_debug
DATA_DIR="rlgpu/python/data/$DIR_NAME"
PWS_DIR="jcoholich@143.215.128.197:/home/jcoholich/isaacgym/python/rlgpu/data/$DIR_NAME"  # personal workstation directory
echo Moving data from:
echo $DATA_DIR
echo
echo to:
echo $PWS_DIR
rsync --remove-source-files -r $DATA_DIR $PWS_DIR