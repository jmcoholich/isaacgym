"""
This is a script to submit <n_runs> sbatch jobs to slurm, instead of
requesting n_runs gpus all on the same machine, which is harder to get.

first arg =  number of runs
second arg = python command for run
third arg = name of runs
"""
import subprocess
import itertools
import random

def get_blacklist():
    with open("blacklisted_nodes.txt", 'r') as f:
        x = f.readlines()[1:]
    blacklist = [i.rstrip() for i in x]
    return ','.join(blacklist)

def main():
    num_runs = 5
    base_x_vel_coefs = [0.5, 1.0, 2.0]
    base_x_vel_clips = [0.125, 0.25, 0.5, 1.0]
    y_vel_pens = [0.05, 0.1, 0.2]
    smoothnesses = [0.0, 0.03125, 0.0625]
    prod = list(itertools.product(base_x_vel_coefs, base_x_vel_clips, y_vel_pens, smoothnesses))
    random.seed(1)
    random.shuffle(prod)
    for base_x_vel_coef, base_x_vel_clip, y_vel_pen, smoothness in prod[:1]:
        job_nickname = f"f_{base_x_vel_coef}_{base_x_vel_clip}_{y_vel_pen}_{smoothness}".replace('.', 'p')
        # python_cmd = "python rlg_train.py --cfg_env 12_F --cfg_train 12_large_net --wandb_project aliengo_12_F_sweep"
        python_cmd = ("python rlg_train.py --cfg_env 12_F --cfg_train 12_large_net --wandb_project aliengo_12_F_sweep "
        f" --wandb_run_name {job_nickname} "
        f" --base_x_vel_coef {base_x_vel_coef} --base_x_vel_clip {base_x_vel_clip} --y_vel_pen {y_vel_pen} --smoothness {smoothness} ")

        slurm_options = [
            "--cpus-per-gpu", "7",
            "-p", "overcap",
            "-A", "overcap",
            "--requeue",
            # "--constraint", "2080_ti|a40",
            "--gres", "gpu:1",
            "-x", get_blacklist(),
        ]

            # loop through random seeds
        for i in range(num_runs):
            unique_name = job_nickname + '__' + str(i)
            dep_slurm_args = [
                "-o",  unique_name + ".log",
                "-J", unique_name
            ]
            cmd = ["sbatch"] + slurm_options + dep_slurm_args + ["submit.sh"] \
                + [python_cmd + " --seed " + str(i)] + [unique_name]
            print(" ".join(cmd))
            print()
            subprocess.run(cmd)


if __name__ == "__main__":
    main()


