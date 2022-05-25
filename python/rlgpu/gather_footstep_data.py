import torch
import matplotlib.pyplot as plt
from gather_stats_utils import gpu_parallel_cmd_runner, get_wandb_run_ids
import os
from subprocess import Popen, PIPE, run
import pickle


def gather_data():
    run_name = "H ss state (0.2 rand new)"
    run_ids = get_wandb_run_ids({run_name: None})[run_name]
    cmds = generate_commands(run_ids)
    gpu_parallel_cmd_runner(cmds)


def generate_commands(run_ids):
    cmds = []
    # directions = [i / 2.0 for i in range(4)]
    timeout = 5_000

    # stepping in place with optimization
    directions = [0]
    des_dir_coef = 0
    for direct in directions:
        for run_id, ws in run_ids:
            cmd = [
                "python", "rlg_train.py",
                "--play",
                "--headless",
                "--checkpoint", str(run_id),
                "--des_dir", str(direct),
                "--des_dir_coef", str(des_dir_coef),
                "--plot_values",
                "--footstep_targets_in_place",
                "--num_envs", "20",
                "--gather_stats", "20",
                "--timeout", str(timeout),
                "--no_ss",
                "--start_after", "60",
                # "--stop_after_footstep", "14",
                "--ws", str(ws)]
            save_fname = "data_" + "__".join(cmd[2:])
            cmd.append("--save_fname")
            cmd.append(save_fname)

            cmds.append(cmd)

    # stepping in place with random footstep selection
    for run_id, ws in run_ids:
        cmd = [
            "python", "rlg_train.py",
            "--play",
            "--headless",
            "--checkpoint", str(run_id),
            "--plot_values",
            "--footstep_targets_in_place",
            "--random_footsteps",
            "--num_envs", "20",
            "--gather_stats", "20",
            "--timeout", str(timeout),
            "--no_ss",
            "--start_after", "60",
            # "--stop_after_footstep", "14",
            "--ws", str(ws)]
        save_fname = "data_" + "__".join(cmd[2:])
        cmd.append("--save_fname")
        cmd.append(save_fname)

        cmds.append(cmd)

    # stepping in place without optimization
    for direct in directions:
        for run_id, ws in run_ids:
            cmd = [
                "python", "rlg_train.py",
                "--play",
                "--headless",
                "--checkpoint", str(run_id),
                "--footstep_targets_in_place",
                "--num_envs", "20",
                "--gather_stats", "20",
                "--timeout", str(timeout),
                "--no_ss",
                "--ws", str(ws)]
            save_fname = "data_" + "__".join(cmd[2:])
            cmd.append("--save_fname")
            cmd.append(save_fname)

            cmds.append(cmd)
    return cmds


if __name__ == "__main__":
    gather_data()
