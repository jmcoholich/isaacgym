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

# yes | docker system prune -a
bash docker/build.sh $2
docker rm isaacgym_container_$2
time bash docker/eval_sbatch_run.sh "$1" $2 $3 $4 $5 $6 # this script needs to do all the work