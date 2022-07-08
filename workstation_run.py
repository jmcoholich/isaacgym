import subprocess

num_runs = 4
python_cmd = ["python rlg_train.py --cfg_env 12_H_ss_state_slim --cfg_train 12_large_net --wandb_project aliengo_12 --max_iterations 5000"]

# python_cmd = "conda activate rlgpu"

# subprocess.run(["tmux"])
# subprocess.run(["tmux", "set-option", "remain-on-exit"])

# loop through random seeds
for i in range(num_runs):
    cmd = [python_cmd + " --seed " + str(i)]
    print(" ".join(cmd))
    print()
    subprocess.run(cmd)
