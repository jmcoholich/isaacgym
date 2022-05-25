"""
This is a script to submit <n_runs> sbatch jobs to slurm, instead of
requesting n_runs gpus all on the same machine, which is harder to get.

first arg =  number of runs
second arg = python command for run
third arg = name of runs
"""
import subprocess

num_runs = 3
job_nickname = "ez_env_high_tau"
python_cmd = "python rlg_train.py --tau 0.99 --cfg_env 12_H_ss_state_easier --cfg_train 12_large_net --wandb_project aliengo_12 --max_iterations 20000"

slurm_options = [
    "--cpus-per-gpu", "7",
    "-p", "long",
    "--constraint", "rtx_6000|a40",
    "--gres", "gpu:1",
    "-x", " heistotron,cyborg,deebot,qt-1,robby,dave,omgwth",
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
