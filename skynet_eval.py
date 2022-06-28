"""
Run python/rlgpu/evluate_policy.py on skynet in a docker container with
8 GPUs for maximum acceleration.
"""

import subprocess
from skynet_run import get_blacklist
import itertools
from random import shuffle

def main():
    des_dir_coefficients = [50, 75, 150]
    box_lengths = [0.2, 0.225, 0.275]
    nn_ft_dists = [0.05, 0.1, 0.2, 0.4]
    nn_ft_widths = [0.0, 0.075, 0.15]
    prod = list(itertools.product(des_dir_coefficients, box_lengths, nn_ft_dists, nn_ft_widths))
    shuffle(prod)
    for cf, bs, nn_ft_dist, nn_ft_width in prod:
        # run_name = "H ss state (0.2 rand new)"  # SOTA
        run_name = "H new sota"
        des_dir_coef = cf  # default: 50
        search_box_len = bs  # default: 0.15
        job_nickname = f"eval_{run_name}".replace(" ", "").replace(".", "").replace("(", "").replace(")", "").lower() + "dd_" + str(des_dir_coef) + "_bb_" + str(search_box_len).replace(".", "_") + "_dist_" + str(nn_ft_dist).replace(".", "_") + "_width_" + str(nn_ft_width).replace(".", "_")
        # python_cmd = f"python evaluate_policy.py --run_name='{run_name}' --debug"
        python_cmd = run_name

        slurm_options = [
            "--cpus-per-gpu", "7",
            "-p", "overcap",
            "-A", "overcap",
            "--constraint", "rtx_6000|a40",
            "--gres", "gpu:8",
            "-x", get_blacklist() + ",cortana,fiona",
            #"-w", "zima, sophon, claptrap"
        ]

        # loop through random seeds
        # for i in range(num_runs):
        unique_name = job_nickname
        dep_slurm_args = [
            "-o",  unique_name + ".log",
            "-J", unique_name
        ]
        cmd = ["sbatch"] + slurm_options + dep_slurm_args + ["eval_submit.sh"] \
            + [python_cmd] + [unique_name] + [str(des_dir_coef)] + [str(search_box_len)] + [str(nn_ft_dist)] + [str(nn_ft_width)]
        print(" ".join(cmd))
        print()
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
