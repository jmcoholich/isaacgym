"""
Run python/rlgpu/evluate_policy.py on skynet in a docker container with
8 GPUs for maximum acceleration.
"""

import subprocess

run_name = "H curr"
job_nickname = f"eval_{run_name}".replace(" ", "").replace(".", "").replace("(", "").replace(")", "").lower()
# python_cmd = f"python evaluate_policy.py --run_name='{run_name}' --debug"
python_cmd = run_name

slurm_options = [
    "--cpus-per-gpu", "7",
    "-p", "short",
    # "--constraint", "rtx_6000|a40",
    "--gres", "gpu:8",
    "-x", "vicki",
]

# loop through random seeds
# for i in range(num_runs):
unique_name = job_nickname
dep_slurm_args = [
    "-o",  unique_name + ".log",
    "-J", unique_name
]
cmd = ["sbatch"] + slurm_options + dep_slurm_args + ["eval_submit.sh"] \
    + [python_cmd] + [unique_name]
print(" ".join(cmd))
print()
subprocess.run(cmd)
