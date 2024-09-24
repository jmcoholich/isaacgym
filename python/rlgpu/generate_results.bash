#!/bin/bash
set -ex

# Start time measurement
START_TIME=$(date +%s)

# All args required
HIERARCHICAL_POLICY_ID=$1
FLAT_POLICY_ID=$2
RUN_NAME=$3
# default 2
JOBS_PER_GPU=${4:-1}

# Evaluate the hierarchical policy
python evaluate_policy.py --id $HIERARCHICAL_POLICY_ID --flat_policy False --jobs_per_gpu $JOBS_PER_GPU

# Evaluate the flat policy
python evaluate_policy.py --id $FLAT_POLICY_ID --flat_policy True --jobs_per_gpu $JOBS_PER_GPU

# Generate plots/tables
python generate_all_plots.py --h_id $HIERARCHICAL_POLICY_ID --f_id $FLAT_POLICY_ID --save_dir plots/$RUN_NAME

# End time measurement
END_TIME=$(date +%s)

# Calculate and display the time taken
ELAPSED_TIME=$((END_TIME - START_TIME))
# print in min and seconds
echo "Script execution time: $((ELAPSED_TIME / 60)) minutes and $((ELAPSED_TIME % 60)) seconds"