"""
This script will analyze the log files and generate all figures for an
experiment.

List of figures that I will generate, given a run name:
1. For now, just the metrics for all envs on the end-to-end implementation.

"""

import argparse
import torch
import matplotlib.pyplot as plt
import itertools
import os
import sys
import pickle
import io
import gzip
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import re

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def main():

    sota = {
        # "Proposed Method" : "H ss state (0.2 rand new)",
        "Proposed Method" : "H_2_ahead_3dd_76_bb_0_225",
        "End-to-end": "F ss state final",
    }
    args = get_args()
    random_id = str(torch.randint(1000000, (1,)).item())
    if not os.path.exists("plots"):
        os.makedirs("plots")
    save_dir = os.path.join("plots", random_id)
    os.mkdir(save_dir)
    args.save_dir = save_dir
    # s = time.time()
    # data = load_relevent_data(sota, args)
    e = time.time()
    # print(f"Serial loading took {e - s :.2f} seconds")
    # sota = {
    #     "Proposed Method" : "H ss state (0.2 rand new)",
    #     "End-to-end": "F ss state final",
    # }
    data = parallel_load_relevent_data(sota, args)
    print(f"Parallel loading took {time.time() - e :.2f} seconds")
    # sys.exit()
    generate_sup_plot(data, sota, args)
    generate_small_plot(data, sota, args)
    print_latex_table(data, sota, args)


def print_latex_table(data, sota, args):
    pass

def get_args():
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=False)
    # group.add_argument("--id", type=str,
    #                    help="This is the id of the run to evalute.")
    parser.add_argument("--run_name", type=str,
                       help="If passed, add this to plots in addition to H and F SOTA.")
    return parser.parse_args()


# def load_relevent_data(sota, args):
#     """load data relevent to the main performance super plot"""
#     names_to_analyze = sota
#     if args.run_name:
#         names = args.run_name.split(',')
#         names = [name.rstrip().lstrip() for name in names]
#         if any(name in sota.values() for name in names):
#             raise ValueError("run_name is already in SOTA")
#         else:
#             for name in names:
#                 names_to_analyze[name] = name

#     data = {}

#     for method in names_to_analyze:
#         data[method] = {}
#         data_dir = os.path.join("data", names_to_analyze[method].replace(" ", "_").replace("(", "").replace(")", "").replace(".", ""))
#         all_files = os.listdir(data_dir)
#         relevant_files = []
#         for file in all_files:
#             if file[-3:] == "pgz":
#                 relevant_files.append(file)
#         for file in relevant_files:
#             file_parts = file[5:-4].split("__")
#             if "--ss_height_var" in file_parts:
#                 heightvar = float(file_parts[file_parts.index("--ss_height_var") + 1])
#                 infill = float(file_parts[file_parts.index("--ss_infill") + 1])
#                 env_key = (infill, heightvar)
#             elif names_to_analyze[method][0] == "H" and all(x in file_parts for x in ["--no_ss", "--plot_values", "--des_dir_coef", "--des_dir"]):  # this is the special case for the flatground run
#                 env_key = "flatground"
#             elif names_to_analyze[method][0] == "F" and all(x in file_parts for x in ["--no_ss"]):
#                 env_key = "flatground"
#             elif len(file_parts) == 4:  # special case for getting training reward to normalize stats by
#                 env_key = "training_rew"
#             else:
#                 continue

#             checkpoint = file_parts[file_parts.index("--checkpoint") + 1]
#             if checkpoint not in data[method]:
#                 data[method][checkpoint] = {}
#             data[method][checkpoint][env_key] = {}
#             temp = data[method][checkpoint][env_key]

#             # with open(os.path.join(data_dir, file), 'rb') as f:
#             with gzip.GzipFile(os.path.join(data_dir, file), 'r') as f:
#                 x = CPU_Unpickler(f).load()
#             temp["successful"] = x["succcessful"]
#             temp["reward"] = x["reward"].squeeze().sum(dim=1)
#             temp["eps_len"] = x["still_running"].sum(dim=1).squeeze().to(torch.long)
#             idx = temp["eps_len"] - 1
#             temp["distance_traveled"] = x["base_position"][torch.arange(len(idx)), idx, 0]

#                 # common_footstep = x["current_footstep"].min().item()
#                 # num_runs = x["still_running"].shape[0]
#                 # rews_sum = 0.0
#                 # for i in range(num_runs):
#                 #     rews_sum += x["reward"][i, :termination_idcs[i], 0].sum()

#                 # if is_random:
#                 #     key = "random"
#                 # elif is_optim:
#                 #     key = "optim"
#                 # else:
#                 #     key = "in_place"
#                 # data[checkpoint][key] = {}
#                 # data[checkpoint][key]["Average rew per timestep"] = (rews_sum / termination_idcs.sum()).item()
#                 # data[checkpoint][key]["Average rew per footstep"] = (rews_sum / x['current_footstep'].sum()).item()
#                 # data[checkpoint][key]["Average rew per rollout"] = (rews_sum / num_runs).item()
#                 # data[checkpoint][key]["Average rollout length"] = (termination_idcs.float().mean()).item()
#                 # data[checkpoint][key]["Average footstep reached"] = (x['current_footstep'].float().mean()).item()
#                 # data[checkpoint][key]["Average timesteps per footstep"] = (termination_idcs.float().sum() / x['current_footstep'].float().sum()).item()

#                 # print(file)
#                 # for k, v in data[checkpoint][key].items():
#                 #     print(f"{k}: {v :0.2f}")
#                 # # print(f"Average rew per timestep: {rews_sum / termination_idcs.sum() :.3f}")
#                 # # print(f"Average rew per footstep: {rews_sum / x['current_footstep'].sum() :.2f}")
#                 # # print(f"Average rew per rollout: {rews_sum / num_runs :.2f}")
#                 # # print(f"Average rollout length: {termination_idcs.float().mean() :.1f}")
#                 # # print(f"Average footstep reached: {x['current_footstep'].float().mean() :.1f}")
#                 # # print(f"Average timesteps per footstep: {termination_idcs.float().sum() / x['current_footstep'].float().sum() :.1f}")
#                 # print("\n")

    return data

def parallel_load_relevent_data(sota, args):
    """load data relevent to the main performance super plot"""
    names_to_analyze = sota
    all_runs = os.listdir("data")
    for run in all_runs:
        if "H_new_sotadd" in run:
            names_to_analyze[run] = run
    # if args.run_name:
    #     names = args.run_name.split(',')
    #     names = [name.rstrip().lstrip() for name in names]
    #     if any(name in sota.values() for name in names):
    #         raise ValueError("run_name is already in SOTA")
    #     else:
    #         for name in names:
    #             names_to_analyze[name] = name

    # first populate the data
    data = {}
    futures = []
    with ThreadPoolExecutor() as executor:
        # future = executor.submit(pow, 323, 1235)
        # print(future.result())

        for method in names_to_analyze:
            data[method] = {}
            data_dir = os.path.join("data", names_to_analyze[method].replace(" ", "_").replace("(", "").replace(")", "").replace(".", ""))
            all_files = os.listdir(data_dir)
            relevant_files = []
            for file in all_files:
                if file[-3:] == "pgz":
                    relevant_files.append(file)
            for file in relevant_files:  # TODO
                file_parts = file[:-4].split("__")
                if file_parts[1] in {"flatground", "in_place_fixed", "in_place_opt", "in_place_rand", "training_rew"}:
                    env_key = file_parts[1]
                else:
                    temp = file_parts[1].split('_')
                    infill = float(temp[0].replace('p', '.'))
                    height_var = float(temp[1].replace('p', '.'))
                    env_key = (infill, height_var)
                    # if "--ss_height_var" in file_parts:
                    #     heightvar = float(file_parts[file_parts.index("--ss_height_var") + 1])
                    #     infill = float(file_parts[file_parts.index("--ss_infill") + 1])
                    #     env_key = (infill, heightvar)
                    # elif names_to_analyze[method][0] == "H" and all(x in file_parts for x in ["--no_ss", "--plot_values", "--des_dir_coef", "--des_dir"]):  # this is the special case for the flatground run
                    #     env_key = "flatground"
                    # elif names_to_analyze[method][0] == "F" and all(x in file_parts for x in ["--no_ss"]):
                    #     env_key = "flatground"
                    # elif len(file_parts) == 2:  # special case for getting training reward to normalize stats by
                    #     env_key = "training_rew"
                    # elif "--random_footsteps" in file_parts:
                    #     env_key = "in_place_rand"
                    # elif ""
                    # else:
                    #     continue

                checkpoint = file_parts[0]
                if checkpoint not in data[method]:
                    data[method][checkpoint] = {}
                data[method][checkpoint][env_key] = {}



                path = os.path.join(data_dir, file)
                futures.append(executor.submit(_load_and_process_data, path, method, checkpoint, env_key))
        '''
            with gzip.GzipFile(os.path.join(data_dir, file), 'r') as f:
                x = CPU_Unpickler(f).load()
            temp["successful"] = x["succcessful"]
            temp["reward"] = x["reward"].squeeze().sum(dim=1)
            temp["eps_len"] = x["still_running"].sum(dim=1).squeeze().to(torch.long)
            idx = temp["eps_len"] - 1
            temp["distance_traveled"] = x["base_position"][torch.arange(len(idx)), idx, 0] '''

                    # common_footstep = x["current_footstep"].min().item()
                    # num_runs = x["still_running"].shape[0]
                    # rews_sum = 0.0
                    # for i in range(num_runs):
                    #     rews_sum += x["reward"][i, :termination_idcs[i], 0].sum()

                    # if is_random:
                    #     key = "random"
                    # elif is_optim:
                    #     key = "optim"
                    # else:
                    #     key = "in_place"
                    # data[checkpoint][key] = {}
                    # data[checkpoint][key]["Average rew per timestep"] = (rews_sum / termination_idcs.sum()).item()
                    # data[checkpoint][key]["Average rew per footstep"] = (rews_sum / x['current_footstep'].sum()).item()
                    # data[checkpoint][key]["Average rew per rollout"] = (rews_sum / num_runs).item()
                    # data[checkpoint][key]["Average rollout length"] = (termination_idcs.float().mean()).item()
                    # data[checkpoint][key]["Average footstep reached"] = (x['current_footstep'].float().mean()).item()
                    # data[checkpoint][key]["Average timesteps per footstep"] = (termination_idcs.float().sum() / x['current_footstep'].float().sum()).item()

                    # print(file)
                    # for k, v in data[checkpoint][key].items():
                    #     print(f"{k}: {v :0.2f}")
                    # # print(f"Average rew per timestep: {rews_sum / termination_idcs.sum() :.3f}")
                    # # print(f"Average rew per footstep: {rews_sum / x['current_footstep'].sum() :.2f}")
                    # # print(f"Average rew per rollout: {rews_sum / num_runs :.2f}")
                    # # print(f"Average rollout length: {termination_idcs.float().mean() :.1f}")
                    # # print(f"Average footstep reached: {x['current_footstep'].float().mean() :.1f}")
                    # # print(f"Average timesteps per footstep: {termination_idcs.float().sum() / x['current_footstep'].float().sum() :.1f}")
                    # print("\n")
        for future in as_completed(futures):
            output, method, checkpoint, env_key = future.result()
            data[method][checkpoint][env_key] = output

    return data


def _load_and_process_data(path, method, checkpoint, env_key):
    output = {}
    with gzip.GzipFile(path, 'r') as f:
        x = CPU_Unpickler(f).load()
    output["successful"] = x["succcessful"]
    output["reward"] = x["reward"].squeeze().sum(dim=1)
    output["eps_len"] = x["still_running"].sum(dim=1).squeeze().to(torch.long)
    idx = output["eps_len"] - 1
    output["distance_traveled"] = x["base_position"][torch.arange(len(idx)), idx, 0]
    return output, method, checkpoint, env_key


def avg_across_seeds(data, metric, floats=False):
    """Takes means of all rollouts for each policy, then compute mean
    and std across the random seeds with these numbers as datapoints."""
    val = {}
    all_samples = torch.tensor([])
    eps_lengths = torch.tensor([])
    for seed in data.keys():
        for env in data[seed].keys():
            if env not in val:
                val[env] = torch.tensor([])
            val[env] = torch.cat([val[env], data[seed][env][metric].cpu().float().mean().unsqueeze(0)])

    if metric == "reward":
        training_rew = val['training_rew'].mean()
    val.pop('training_rew')
    for env in val:
        if metric == "reward":
            val[env] /= training_rew
        mean = val[env].mean()
        std = val[env].std()
        val[env] = {}
        val[env]["mean"] = mean
        val[env]["std"] = std
    return val

        # all_samples = torch.cat((all_samples, data[seed][metric].mean().unsqueeze(0)))
        # eps_lengths = torch.cat((all_samples, data[seed]["Episode_Length"].mean().unsqueeze(0)))
    # if floats:
    #     if val == "Reward" and ptype is not None:
    #         train_rews = {"H ss state (0.2 rand new)": (5053.30 + 5000.23 + 5010.39 + 4466.37 + 4296.00) / 5.0 / 1500,
    #         "F ss state final": 1.0}
    #         all_samples /= train_rews[ptype]
    #         all_samples /= eps_lengths.mean()
    #     return all_samples.mean().item(), all_samples.std().item()
    # else:  # return strings for csv
    #     return f"{all_samples.mean():.2f}" + u" \u00B1 " + f"{all_samples.std():.2f}"


def generate_sup_plot(data, sota, args):
    """Generate plots with performance on envs in increasing difficulty.
    Each policy type has two lines.

    The metrics are:
    - Success rate
    - distance traveled
    - reward as fraction of training reward
    - episode length
    """

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(15 / 1.25 - 1, 3.0 / 1.25 * 4.5 - 2))
    fig.subplots_adjust(hspace=0.5)
    axes = [ax1, ax2, ax3, ax4]
    metrics = ["successful", "reward", "eps_len", "distance_traveled"]
    flag = True
    fig.supxlabel("Stepping Stone Density / Stone Height Variation (m)", y=-0.025)
    fig.suptitle("Proposed Method vs End-to-end Policy", y=0.95, x=0.5)
    for metric, ax in zip(metrics, axes):
        marker = itertools.cycle(('s', 'o', 'H', 'D', "^", ">", "<", 's', 'd'))
        ax.grid()
        for method in data:
            vals = avg_across_seeds(data[method], metric, floats=True)
            fg_data = vals.pop("flatground")
            # sort into something that I can plot
            sorted_keys = list(vals.keys())
            sorted_keys.sort(key=lambda x: (x[1], -x[0]))
            y_points = [fg_data["mean"].item()]
            errors = [fg_data["std"].item()]
            x_labels = ["Flatground"]
            for key in sorted_keys:
                y_points.append(vals[key]["mean"].item())
                errors.append(vals[key]["std"].item())
                x_labels.append(f"{key[0] * 100}% / {key[1]}")
            if method == "Proposed Method":
                line_fmt = "--"
            else:
                line_fmt = "-"
            fmt = next(marker) + line_fmt
            ax.errorbar(x_labels, y_points, fmt=fmt, yerr=errors, label=method, capsize=3.0)
        if flag:
            # ax.legend(loc="upper right")
            flag = False
        # ax.set_xticks()
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.axvspan(1 - 0.5, 7 + 0.5, facecolor='b', alpha=0.1)
        ax.axvspan(7.5, 14.5, facecolor='g', alpha=0.1)
        ax.axvspan(14.5, 21.5, facecolor='r', alpha=0.1)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        if metric == "successful":
            ax.set_title("Terrain Traversal Success Rate")
            ax.set_ylabel("Success Rate")
        elif metric == "reward":
            ax.set_title("Fraction of Per-timestep Training Reward Achieved")
            ax.set_ylabel("Normalized Reward")
        elif metric == "distance_traveled":
            ax.set_title("Distance Traveled per Episode")
            ax.set_ylabel("Distance Traveled (m)")
        elif metric == "eps_len":
            ax.set_title("Episode Length")
            ax.set_ylabel("Episode Length (timesteps)")
    handles, labels = ax.get_legend_handles_labels()
    if args.run_name:
        fig.legend(handles, labels, bbox_to_anchor=(1.25, 0.9))
    else:
        fig.legend(handles, labels, loc=(0.775, 0.9125))
    # plt.show()

    # plt.title("3 random seeds,  30 rollouts each, 10k steps timeout, Error bars are std between seeds")
    # plt.show()
    path = os.path.join(args.save_dir, "supplot.svg")
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot at {path}")


def generate_small_plot(data, sota, args):
    """Generate small plot of just success rate on a more limited sweep
    of environments
    """
    envs_to_plot = [
        (1.0, 0.0), (0.75, 0.0), (0.5, 0.0), (0.25, 0.0),
        (1.0, 0.1), (0.75, 0.1), (0.5, 0.1), (0.25, 0.1)
    ]

    metric = "successful"
    plt.figure(figsize=(5.0, 2.5))
    plt.xlabel("Stepping Stone Density / Stone Height Variation (m)", y=-0.025)
    plt.title("Proposed Method vs End-to-end Policy", y=0.95, x=0.5)
    ax = plt.gca()
    marker = itertools.cycle(('s', 'o', 'H', 'D', "^", ">", "<", 's', 'd'))
    ax.grid()
    for method in data:
        vals = avg_across_seeds(data[method], metric)
        fg_data = vals.pop("flatground")
        # sort into something that I can plot
        # sorted_keys = list(vals.keys())
        sorted_keys = envs_to_plot
        sorted_keys.sort(key=lambda x: (x[1], -x[0]))
        y_points = [fg_data["mean"].item()]
        errors = [fg_data["std"].item()]
        x_labels = ["Flatground"]
        for key in sorted_keys:
            y_points.append(vals[key]["mean"].item())
            errors.append(vals[key]["std"].item())
            x_labels.append(f"{key[0] * 100}% / {key[1]}")
        if method == "Proposed Method":
            line_fmt = "--"
        else:
            line_fmt = "-"
        fmt = next(marker) + line_fmt
        ax.errorbar(x_labels, y_points, fmt=fmt, yerr=errors, label=method, capsize=3.0)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.axvspan(1 - 0.5, 4 + 0.5, facecolor='b', alpha=0.1)
    # ax.axvspan(7.5, 14.5, facecolor='g', alpha=0.1)
    ax.axvspan(4.5, 8.5, facecolor='r', alpha=0.1)
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(loc=(0.1, 1.02), ncol=2)
    if metric == "successful":
        ax.set_title("Terrain Traversal Success Rate", pad=25.0)
        ax.set_ylabel("Success Rate")
    elif metric == "reward":
        ax.set_title("Fraction of Per-timestep Training Reward Achieved")
        ax.set_ylabel("Normalized Reward")
    elif metric == "distance_traveled":
        ax.set_title("Distance Traveled per Episode")
        ax.set_ylabel("Distance Traveled (m)")
    elif metric == "eps_len":
        ax.set_title("Episode Length")
        ax.set_ylabel("Episode Length (timesteps)")
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels, loc=(0.775, 0.9125))
    # plt.show()

    # if not os.path.exists("plots"):
    #     os.makedirs("plots")
    # path = "plots/" + "smallplot" + '.svg'
    path = os.path.join(args.save_dir, "smallplot.svg")
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot at {path}")


if __name__ == "__main__":
    main()

