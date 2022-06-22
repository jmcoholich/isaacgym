"""
This is a script to submit <n_runs> sbatch jobs to slurm, instead of
requesting n_runs gpus all on the same machine, which is harder to get.

first arg =  number of runs
second arg = python command for run
third arg = name of runs
"""
import subprocess

def get_blacklist():
    with open("blacklisted_nodes.txt", 'r') as f:
        x = f.readlines()[1:]
    blacklist = [i.rstrip() for i in x]
    return ','.join(blacklist)


num_runs = 5
job_nickname = "h_2_ahead_alt"
python_cmd = "python rlg_train.py --cfg_env 12_H_2_ahead_alt --cfg_train 12_large_net --wandb_project aliengo_12"

slurm_options = [
    "--cpus-per-gpu", "7",
    "-p", "short",
    "--constraint", "2080_ti",
    "--gres", "gpu:1",
    "-x", get_blacklist(),
]

# loop through random seeds
for i in [0, 1]:
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


