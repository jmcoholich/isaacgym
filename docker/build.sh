#!/bin/bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/../.."
docker build --network host -t isaacgym_$1 -f isaacgym/docker/Dockerfile .
