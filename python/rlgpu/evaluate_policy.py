"""
This script evaluates the performance of a single trained policy on a specified
number of environments. It generates the files that will be read by
"process_footstep_data.py". Then, read the files, generate some statistics
and push them to wandb. This way, each run has information about both high
level and low-level performance.
"""

import argparse
from gather_stats_utils import gpu_parallel_cmd_runner, get_ws_from_run_id, \
    get_wandb_run_name_from_id, fname_parser, get_wandb_ids_from_run_name
from utils.aliengo_params_utils import detect_workstation_id
import os
import torch
import pickle
import wandb

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", type=str,
                       help="This is the id of the run to evalute.")
    group.add_argument("--run_name", type=str,
                       help="If passed, eval all runs with this name.")
    parser.add_argument("--timeout", type=int, default=5000,
                        help="Number of env steps to timeout after")
    parser.add_argument("--debug", action="store_true",
                        help="Sets params for fast runs to debugging the pipeline")
    parser.add_argument("--num_rollouts", type=int, default=20,
                        help="Number of episodes per policy to collect")
    parser.add_argument("--num_envs", type=int, default=20,
                        help="Number of envs to run in parallel")
    parser.add_argument("--des_dir_coef", type=int, default=50,
                        help="Coefficient for directional term in "
                        "value-function footstep target optimiation")
    return parser.parse_args()


def main():
    args = get_args()
    cmds = generate_commands(args)
    gpu_parallel_cmd_runner(cmds)


def generate_commands(args):
    cmds = []
    if args.debug:
        env_difficulties = [
            (None, None),  # this is for flat ground
            (.25, 0.0),  # (percent infill, stepping stone height variation)
            (.100, 0.1),
        ]
        args.timeout = 100
    else:
        env_difficulties = [
            # (None, None),  # this is for flat ground
            (.25, 0.0),  # (percent infill, stepping stone height variation)
            (.375, 0.0),
            (.50, 0.0),
            (.625, 0.0),
            (.75, 0.0),
            (.875, 0.0),
            (1.0, 0.0),

            (.25, 0.05),
            (.375, 0.05),
            (.50, 0.05),
            (.625, 0.05),
            (.75, 0.05),
            (.875, 0.05),
            (1.0, 0.05),

            (.25, 0.1),
            (.375, 0.1),
            (.50, 0.1),
            (.625, 0.1),
            (.75, 0.1),
            (.875, 0.1),
            (1.0, 0.1),
        ]

    if args.run_name:
        ids = get_wandb_ids_from_run_name(args.run_name)
        data_dir = "data/" + args.run_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "") + "dd_" + str(args.des_dir_coef)
        if args.debug:
            ids = ids[:2]
    else:
        ids = [args.id]
        data_dir = "data/" + args.id

    if args.debug:
        data_dir += "_debug"

    for id_ in ids:
        cmd_base = (
            "python", "rlg_train.py",
            "--play",
            "--headless",
            "--checkpoint", str(id_),
            "--num_envs", str(args.num_envs),
            "--gather_stats", str(args.num_rollouts),
            "--timeout", str(args.timeout),
            "--ws", str(determine_ws_arg(id_)),
            "--data_dir", data_dir
        )
        if args.run_name[0] == "H":
            cmds.extend(generate_in_place_cmds(args, cmd_base))
        cmds.extend(generate_training_reward_cmd(args, cmd_base))
        cmds.extend(generate_terrain_cmds(args, env_difficulties, cmd_base, id_))
    add_save_fname_arg(cmds)
    return cmds

def add_save_fname_arg(cmds):
    uninformative_args = {"--num_envs", "--gather_stats", "--ws", "--data_dir", "python", "--start_after"}
    single_uninformative_args = {"--play", "--headless"}
    for cmd in cmds:
        cmd_copy = list(cmd)
        i = 0
        while True:
            if cmd_copy[i] in uninformative_args:
                cmd_copy.pop(i)
                cmd_copy.pop(i)
            elif cmd_copy[i] in single_uninformative_args:
                cmd_copy.pop(i)
            else:
                i += 1
            if len(cmd_copy) <= i:
                break
        save_fname = "data_" + "__".join(cmd_copy)
        cmd.extend(["--save_fname", save_fname])

def determine_ws_arg(id):
    saved_ws = int(get_ws_from_run_id(id))  # workstation that the model file is saved on
    current_ws = int(detect_workstation_id())  # workstation that I'm running this script on
    if saved_ws == current_ws:
        return -1  # this is the default for the ws arg
    else:
        return saved_ws


def generate_training_reward_cmd(args, cmd_base):
    return [list(cmd_base)]


def generate_terrain_cmds(args, env_difficulties, cmd_base, id_):
    cmds = []

    for infill, height_var in env_difficulties:
        cmd = list(cmd_base)
        # cmd += ["--add_perturb", "100.0"]

        # if policy_type[0] == "H" and env != "Training_Reward":
        if get_wandb_run_name_from_id(id_)[0] == "H":
            cmd += ["--plot_values",
                    "--des_dir_coef", str(args.des_dir_coef),
                    "--des_dir", "0",
                    "--footstep_targets_in_place"]
        if infill is None:
            cmd += ["--no_ss"]
        else:
            cmd += ["--add_ss", "--ss_infill", str(infill),
                    "--ss_height_var", str(height_var)]
        cmds.append(cmd)
    return cmds


def generate_in_place_cmds(args, global_cmd_base):
    cmds = []

    cmd_base = global_cmd_base + (
        "--footstep_targets_in_place",
        "--no_ss",
        "--start_after", "60",
    )

    other_args = []

    # stepping in place with optimization on flat ground
    directions = [0]
    for direct in directions:
        other_args.append([
            "--des_dir", str(direct),
            "--des_dir_coef", str(args.des_dir_coef),
            "--plot_values",
        ])

    # stepping in place with random footstep selection
    other_args.append([
        "--plot_values",
        "--random_footsteps",
        "--des_dir_coef", "0",
    ])

    # stepping in place without optimization
    other_args.append([])

    for temp in other_args:
        cmd = list(cmd_base) + temp
        cmds.append(cmd)
    return cmds


if __name__ == "__main__":
    main()
