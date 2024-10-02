#!/bin/bash
set -ex

# Start time measurement
START_TIME=$(date +%s)

# # All args required
# HIERARCHICAL_POLICY_ID=$1
# FLAT_POLICY_ID=$2
# RUN_NAME=$3
# # default 2
# JOBS_PER_GPU=${4:-1}
# DES_DIR_COEF=$5
# Evaluate the hierarchical policy
python evaluate_policy.py --id "240929215229583361" --flat_policy False --jobs_per_gpu 1 --des_dir_coef 0.0 --ws 7 --timeout 500
# Evaluate the flat policy
# python evaluate_policy.py --id $FLAT_POLICY_ID --flat_policy True --jobs_per_gpu $JOBS_PER_GPU

# # Generate plots/tables
python generate_all_plots.py --h_id "240929215229583361"  --save_dir plots/testing

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
# print in min and seconds
echo "Script execution time: $((ELAPSED_TIME / 60)) minutes and $((ELAPSED_TIME % 60)) seconds"

# # End time measurement
# END_TIME=$(date +%s)

# # Calculate and display the time taken
