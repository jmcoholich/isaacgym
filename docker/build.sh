#!/bin/bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/../.."
docker build --network host -t isaacgym_$1 -f isaacgym/docker/Dockerfile .
if [[ $? !=  0 ]]
then
    echo Docker build failed, try pruning because space is typically an issue
    yes | docker system prune
    docker build --network host -t isaacgym_$1 -f isaacgym/docker/Dockerfile .
fi

if [[ $? !=  0 ]]
then
    echo Docker build failed, try pruning ALL because space is typically an issue
    yes | docker system prune -a
    docker build --network host -t isaacgym_$1 -f isaacgym/docker/Dockerfile .
fi

