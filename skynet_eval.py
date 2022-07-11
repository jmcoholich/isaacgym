"""
Run python/rlgpu/evluate_policy.py on skynet in a docker container with
8 GPUs for maximum acceleration.
"""

import subprocess
from skynet_run import get_blacklist
import itertools
import random

def main():
    # des_dir_coefficients = [50, 75, 150]
    # box_lengths = [0.2, 0.225, 0.275]
    # nn_ft_dists = [0.05, 0.1, 0.2, 0.4]
    # nn_ft_widths = [0.0, 0.075, 0.15]
    # prod = list(itertools.product(des_dir_coefficients, box_lengths, nn_ft_dists, nn_ft_widths))
    # random.seed(1)
    # random.shuffle(prod)
    # prod = [(150, 0.2, 0.2, 0.15),
    #     (50, 0.275, 0.1, 0.15),
    #     (75, 0.225, 0.1, 0.075),
    #     (75, 0.225, 0.1, 0.15)]
    # prod = [(150, 0.2, 0.2, 0.15)]
    # for cf, bs, nn_ft_dist, nn_ft_width in prod:
    base_x_vel_coefs = [0.5, 1.0, 2.0]
    base_x_vel_clips = [0.125, 0.25, 0.5, 1.0]
    y_vel_pens = [0.05, 0.1, 0.2]
    smoothnesses = [0.0, 0.03125, 0.0625]
    prod = list(itertools.product(base_x_vel_coefs, base_x_vel_clips, y_vel_pens, smoothnesses))
    random.seed(1)
    random.shuffle(prod)
    for base_x_vel_coef, base_x_vel_clip, y_vel_pen, smoothness in prod[:1]:
        run_name = f"f_{base_x_vel_coef}_{base_x_vel_clip}_{y_vel_pen}_{smoothness}".replace('.', 'p')
        # run_name = "H ss state (0.2 rand new)"  # SOTA
        # run_name = "H new sota"
        # run_name = "F ss state final"
        des_dir_coef = -1  # NOTE dummy val # default: 50
        search_box_len = -1  # NOTE dummy val # default: 0.15
        nn_ft_dist = -1  # NOTE dummy val
        nn_ft_width = -1  # NOTE dummy val
        job_nickname = f"eval_{run_name}".replace(" ", "").replace(".", "").replace("(", "").replace(")", "").lower()
        if run_name[0] == "H":
            job_nickname += "dd_" + str(des_dir_coef) + "_bb_" + str(search_box_len).replace(".", "_") + "_dist_" + str(nn_ft_dist).replace(".", "_") + "_width_" + str(nn_ft_width).replace(".", "_")
        # python_cmd = f"python evaluate_policy.py --run_name='{run_name}' --debug"
        python_cmd = run_name

        slurm_options = [
            "--cpus-per-gpu", "7",
            "-p", "overcap",
            "-A", "overcap",
            # "--constraint", "rtx_6000|a40",
            "--gres", "gpu:8",
            "--requeue",
            "-x", get_blacklist() + ",cortana,fiona",
            # "-w", "zima, sophon, claptrap"
        ]

        # loop through random seeds
        # for i in range(num_runs):
        unique_name = job_nickname
        dep_slurm_args = [
            "-o",  'log/' + unique_name + ".log",
            "-J", unique_name
        ]
        cmd = ["sbatch"] + slurm_options + dep_slurm_args + ["eval_submit.sh"] \
            + [python_cmd] + [unique_name] + [str(des_dir_coef)] + [str(search_box_len)] + [str(nn_ft_dist)] + [str(nn_ft_width)]
        print(" ".join(cmd))
        print()
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
