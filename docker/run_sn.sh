#!/bin/bash
set -e
set -u

docker run -it --gpus \"device=${NV_GPU}\" --cpuset-cpus="$(taskset -c -p $$ | cut -f2 -d ':' | awk '{$1=$1};1')" --name isaacgym_container isaacgym /bin/bash
