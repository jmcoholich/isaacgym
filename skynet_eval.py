"""
Run python/rlgpu/evluate_policy.py on skynet in a docker container with
8 GPUs for maximum acceleration.
"""

import subprocess
from skynet_run import get_blacklist

for cf in [25, 50, 75, 100, 150, 200]:
    for bs in [0.15, 0.225, 0.3]:
        run_name = "H ss state (0.2 rand new)"  # SOTA
        #run_name = "H long steps curr 1k"
        des_dir_coef = cf  # default: 50
        search_box_len = bs  # default: 0.15
        job_nickname = f"eval_{run_name}".replace(" ", "").replace(".", "").replace("(", "").replace(")", "").lower() + "dd_" + str(des_dir_coef) + "_bb_" + str(search_box_len).replace(".", "_")
        # python_cmd = f"python evaluate_policy.py --run_name='{run_name}' --debug"
        python_cmd = run_name

        slurm_options = [
            "--cpus-per-gpu", "7",
            "-p", "overcap",
            "-A", "overcap",
            "--constraint", "rtx_6000|a40",
            "--gres", "gpu:8",
            "-x", get_blacklist(),
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
            + [python_cmd] + [unique_name] + [str(des_dir_coef)] + [str(search_box_len)]
        print(" ".join(cmd))
        print()
        subprocess.run(cmd)
