# Isaacgym

Modified by Jeremiah Coholich for use in training on the [Unitree Aliengo](https://www.unitree.com/products/aliengo/) robot for the project [Learning High-Value Footstep Placements for Quadruped Robots](https://www.jeremiahcoholich.com/publication/quadruped_footsteps/).
Original code from NVIDIA:
https://developer.nvidia.com/isaac-gym (Preview Release 2)

Models are trained with my fork of the rl_games repo, which includes support for logging with [Weights and Biases](https://wandb.ai/site), among other things.

rl_games fork: https://github.com/jmcoholich/rl_games

This README contains instructions for installing both my modified versions of
isaacgym and the rl_games library.

The full documentation for IsaacGym can be found in `~/isaacgym/docs/`


## Features
Here is list of features I have added:
- fast vectorized analytical inverse kinematics for the Aliengo quadruped
- multi-GPU policy evaluation and data gathering pipeline
- procedural terrain generation
- logging with [Weights and Biases](https://wandb.ai/site)
- my [value-function footstep optimization method](https://www.jeremiahcoholich.com/publication/quadruped_footsteps/)
- scripts for generating videos from simulation cameras (vs screencap)
- [Augumented Random Search](https://arxiv.org/abs/1803.07055) as an alternative to [PPO](https://arxiv.org/abs/1707.06347)

## Prereqs

- Ubuntu 18.04 or 20.04.
- Python 3.6, 3.7 or 3.8.
- Minimum NVIDIA driver version: 460.32
    - Note: Even if you have no NVIDIA gpu, you will need to install an NVIDIA
    driver in order to run Isaacgym (I haven't found a better workaround).


To install an NVIDIA driver

    sudo apt update
    sudo apt install nvidia-driver-470

## To install IsaacGym + RL_Games locally

    cd ~
    git clone https://github.gatech.edu/jcoholich3/isaacgym.git
    cd isaacgym
    ./create_conda_env_rlgpu.sh
    conda activate rlgpu
    cd ~
    git clone https://github.gatech.edu/jcoholich3/rl_games.git
    cd rl_games
    pip install -e .

## To test installation

    cd ~/isaacgym/python/examples
    python joint_monkey.py

## Running on Skynet (Docker required due to Skynet using Ubuntu 16.04)

In a screens or tmux session, check out a node with:

    srun <args> --pty bash
    cd isaacgym
    bash docker/build.sh
    bash docker/run_sn.sh

This will start a Docker container where you can start training runs.

To train PMTG for trotting:

    python rlg_train.py --cfg_env pmtg_trot --seed 0 --device 0 --headless

To copy model files from docker container to SkyNet, ssh into the node you are
using, then within `isaacgym/docker/`, run

    bash copy_nn.sh

To visualize trained models from skynet:

    python rlg_train.py --play --checkpoint <run_id> --ws 7 --username jcoholich3

Replace jcoholich3 with your SkyNet username.

There is no need to copy the model from skynet to your local machine. Assuming
you have ssh set up, the program will remote into skynet and load the trained
model.

## Running on Skynet with sbatch
On the head node run:

    sbatch --gres gpu:4 submit.sh

replacing 4 with desired number of runs, up to 8 (one run per GPU).

If there is a desired node (perhaps with the docker image already built),
add the -w option.

    sbatch --gres gpu:4 -w clank submit.sh

submit.sh already has the other sbatch options, and it builds the docker image

submit.sh calls docker/sbatch_run.sh. This script starts the docker container
and runs docker/sbatch_docker_run.sh in it then cleans everything
up after sbatch_docker_run.sh finishes.

sbatch_docker_run.sh has the actual python command to start the training run.
Edit this script to change the python command.