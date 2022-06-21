"""
This is a script to submit <n_runs> sbatch jobs to slurm, instead of
requesting n_runs gpus all on the same machine, which is harder to get.

first arg =  number of runs
second arg = python command for run
third arg = name of runs
"""
import subprocess

num_runs = 5
job_nickname = "h_curr_1k_5k_long_steps"
python_cmd = "python rlg_train.py --cfg_env 12_H_long_steps --cfg_train 12_large_net --wandb_project aliengo_12"

slurm_options = [
    "--cpus-per-gpu", "7",
    "-p", "short",
    "--constraint", "2080_ti",
    "--gres", "gpu:1",
    "-x", "vincent",
]

# loop through random seeds
for i in range(num_runs):
    unique_name = job_nickname + str(i)
    dep_slurm_args = [
        "-o",  unique_name + ".log",
        "-J", unique_name
    ]
    cmd = ["sbatch"] + slurm_options + dep_slurm_args + ["submit.sh"] \
        + [python_cmd + " --seed " + str(i)] + [unique_name]
    print(" ".join(cmd))
    print()
    subprocess.run(cmd)
