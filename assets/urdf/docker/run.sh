#!/bin/bash
set -e
set -u

docker run -it --gpus all --name isaacgym_container isaacgym /bin/bash
