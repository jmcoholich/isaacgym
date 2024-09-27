#!/bin/bash

set -ex

for d in 0.5 1 2 5 10 50;
do
bash generate_results.bash 240920120440498904 240920120114215145 des_dir_$d 1 $d
done
