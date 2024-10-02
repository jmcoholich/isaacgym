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
import copy
import numpy as np
from math import log10, floor
import hashlib


def get_args():
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=False)
    # group.add_argument("--id", type=str,
    #                    help="This is the id of the run to evalute.")
    parser.add_argument("--h_id", type=str,
                       help="Run ID of the hierarchical method")
    parser.add_argument("--f_id", type=str,
                       help="Run ID of the flat policy")

    parser.add_argument("--run_name", type=str,
                       help="If passed, add this to plots in addition to H and F SOTA.")
    parser.add_argument("--save_dir", type=str, default="plots")
    parser.add_argument("--no_regen", action='store_true',
                       help="Skip regenerating the data file.")
    parser.add_argument("--score_runs", action='store_true',
                       help="score runs compared to a baseline")

    # parser.add_argument("--load_data", action="store_true",
    #                    help="If passed, add this to plots in addition to H and F SOTA.")
    # parser.add_argument("--load_data", action="store_true",
    #                    help="If passed, add this to plots in addition to H and F SOTA.")
    # parser.add_argument("--save_data", action="store_true",
    #                    help="If passed, add this to plots in addition to H and F SOTA.")
    return parser.parse_args()


def main():
    args = get_args()

    sota = {
        "Proposed Method" : args.h_id,
        "End-to-end": args.f_id,
    }
    if not args.no_regen:
        regen_data_file()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # data = get_data(sota, args)
    all_data = load_all_data_file()
    data = {}
    for key, val in sota.items():
        data[key] = all_data[val]

    if args.score_runs:
        score_runs(data, sota, args, "End-to-end")
    generate_sup_plot(data, sota, args)
    generate_small_plot(data, sota, args)
    generate_collision_small_plot(data, sota, args)
    generate_success_only_plot(data, sota, args)
    generate_in_place_results_table(data, sota, args)
    generate_optimized_footstep_trajectories_plot(data, sota, args)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def get_data(sota, args):
    """Steps
    1. Load the all_data file
    2. If all runs are in the all_data file, return the data I need
    3. Otherwise, load the required runs and add them to the all_data file,
    then return the relevent data.
    """
    names = sota
    # all_runs = [name for name in os.listdir('data') if os.path.isdir(os.path.join('data', name))]
    # for run in sorted(all_runs):
    #     if "f_" in run and 'debug' not in run:
    #         names[run] = run

    # if args.run_name:
    #     names = args.run_name.split(',')
    #     names = [name.rstrip().lstrip() for name in names]
    #     if any(name in sota.values() for name in names):
    #         raise ValueError("run_name is already in SOTA")
    #     else:
    #         for name in names:
    #             names[name] = name
    all_data = load_all_data_file()
    need_to_load = {}
    for name in names.values():
        if name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "") not in all_data:
            need_to_load[name] = name
    if need_to_load:
        raise NotImplementedError()
        # print(f"Need to load {len(need_to_load)} additional runs...")
        # extra_data = parallel_load_relevent_data(need_to_load)

    output_data = {}
    # return just the data for the runs that I actually need
    for key, value in names.items():
        output_data[key] = all_data[value.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")]
    return output_data


def score_runs(data, sota, args, baseline_run_name):
    """Given a baseline run, score runs based on how many environments they
    have a higher success rate than the baseline on."""
    scores = {}
    envs_to_plot = ['flatground',
        (.7, 0.0),
        (.8, 0.0),
        (.9, 0.0),
        (1.0, 0.0),

        (.7, 0.05),
        (.8, 0.05),
        (.9, 0.05),
        (1.0, 0.05),

        (.7, 0.075),
        (.8, 0.075),
        (.9, 0.075),
        (1.0, 0.075),

        (.7, 0.1),
        (.8, 0.1),
        (.9, 0.1),
        (1.0, 0.1),

        (.7, 0.125),
        (.8, 0.125),
        (.9, 0.125),
        (1.0, 0.125),
    ]
    baseline_scores = {}
    temp = avg_across_seeds(data[baseline_run_name], 'successful')
    for env in envs_to_plot:
        baseline_scores[env] = temp[env]["mean"]
    for method in data.keys():
        temp = avg_across_seeds(data[method], 'successful')
        score = 0
        for env in envs_to_plot:
            if temp[env]["mean"] > baseline_scores[env]:
                score += 1
        if score in scores:
            scores[score].append(method)
        else:
            scores[score] = [method]
    total = 0
    for score in sorted(scores.keys()):
        print(f"Score {score}: {len(scores[score])} runs")
        total += len(scores[score])
    print(total)
    print()
    print("Top Scoring runs")
    for score in reversed(sorted(scores.keys())[-3:]):
        print(f"Score {score}: {scores[score]}")
    sys.exit()


def generate_success_only_plot(data, sota, args):
    envs_to_plot = [
        (.7, 0.0),
        (.8, 0.0),
        (.9, 0.0),
        (1.0, 0.0),

        (.7, 0.05),
        (.8, 0.05),
        (.9, 0.05),
        (1.0, 0.05),

        (.7, 0.075),
        (.8, 0.075),
        (.9, 0.075),
        (1.0, 0.075),

        (.7, 0.1),
        (.8, 0.1),
        (.9, 0.1),
        (1.0, 0.1),

        (.7, 0.125),
        (.8, 0.125),
        (.9, 0.125),
        (1.0, 0.125),
    ]
    metric = "successful"
    figsize = (20.0, 5.0)
    name = "largeplot"
    _generate_single_plot(data, sota, args, envs_to_plot, metric, figsize, name)
    _generate_single_bar_plot(data, sota, args, envs_to_plot, metric, figsize, name)


def generate_small_plot(data, sota, args):
    envs_to_plot = [
        (.7, 0.0),
        (.8, 0.0),
        (.9, 0.0),
        (1.0, 0.0),

        (.7, 0.05),
        (.8, 0.05),
        (.9, 0.05),
        (1.0, 0.05),

        (.7, 0.075),
        (.8, 0.075),
        (.9, 0.075),
        (1.0, 0.075),

        (.7, 0.1),
        (.8, 0.1),
        (.9, 0.1),
        (1.0, 0.1),

        (.7, 0.125),
        (.8, 0.125),
        (.9, 0.125),
        (1.0, 0.125),
    ]
    metric = "successful"
    figsize = (5.0, 2.5)
    name = "smallplot"
    _generate_single_plot(data, sota, args, envs_to_plot, metric, figsize, name)
    _generate_single_bar_plot(data, sota, args, envs_to_plot, metric, figsize, name)

def generate_collision_small_plot(data, sota, args):
    envs_to_plot = [
        (.7, 0.0),
        (.8, 0.0),
        (.9, 0.0),
        (1.0, 0.0),

        (.7, 0.05),
        (.8, 0.05),
        (.9, 0.05),
        (1.0, 0.05),

        (.7, 0.075),
        (.8, 0.075),
        (.9, 0.075),
        (1.0, 0.075),

        (.7, 0.1),
        (.8, 0.1),
        (.9, 0.1),
        (1.0, 0.1),

        (.7, 0.125),
        (.8, 0.125),
        (.9, 0.125),
        (1.0, 0.125),
    ]
    metric = "collisions"
    figsize = (5.0, 2.5)
    name = "collisions_smallplot"
    # _generate_single_plot(data, sota, args, envs_to_plot, metric, figsize, name)
    _generate_single_bar_plot(data, sota, args, envs_to_plot, metric, figsize, name)


def generate_in_place_results_table(data, sota, args):
    results = process_in_place_results(data, sota, args)
    all_output = print_latex_table(results)
    # save to txt file
    with open(os.path.join(args.save_dir, "in_place_results.txt"), 'w') as f:
        f.write(all_output)

def process_in_place_results(data, sota, args):
    data = data["Proposed Method"]
    types = ["in_place_rand", "in_place_fixed", "in_place_opt"]
    # metrics = ["rew/timestep", "rew/footstep", "rew/episode", "episode_len", "targets_hit", "time/target"]
    output = dict.fromkeys(types)
    for typ in types:
        output[typ] = {}
        total_rew = 0
        total_timesteps = 0
        total_targets = 0
        total_episodes = 0
        for id_ in data.keys():
            total_rew += data[id_][typ]['reward'].sum().float().item()
            total_timesteps += data[id_][typ]['eps_len'].sum().float().item()
            total_targets += data[id_][typ]['footstep targets hit'].sum().float().item()
            total_episodes += data[id_][typ]['reward'].shape[0]
        output[typ]["rew/timestep"] = total_rew / total_timesteps
        output[typ]["rew/footstep"] = total_rew / total_targets
        output[typ]["rew/episode"] = total_rew / total_episodes
        output[typ]["episode_len"] = total_timesteps / total_episodes
        output[typ]["targets_hit"] = total_targets / total_episodes
        output[typ]["time/target"] = total_timesteps / total_targets

    # env_keys = ["flatground", "in_place_fixed", "in_place_opt", "in_place_rand", "training_rew"]
    """I would like these values to be weighted by timestep as well.
    So I should just save the unormalized values of stuff that I don't have yet
    then do a weighted average later"""

    # output["rew/timestep"] = (rews_sum / termination_idcs.sum()).item()
    # output["Average rew per footstep"] = (rews_sum / x['current_footstep'].sum()).item()
    # output["Average rew per rollout"] = (rews_sum / num_runs).item()
    # output["Average rollout length"] = (termination_idcs.float().mean()).item()
    # output["Average footstep reached"] = (x['current_footstep'].float().mean()).item()
    # output["Average timesteps per footstep"] = (termination_idcs.float().sum() / x['current_footstep'].float().sum()).item()
    return output


def print_latex_table(r):
    out = ""
    # temp = '#' * 10 + ' beginning latex code (copy to overleaf) ' + '#' * 10
    # print('#' * len(temp))
    # print(temp)
    # print('#' * len(temp) + '\n'*2)

    temp = '#' * 10 + ' beginning latex code (copy to overleaf) ' + '#' * 10
    out += '#' * len(temp) + '\n'
    out += temp + '\n'
    out += '#' * len(temp) + '\n'*2

    # print(r"\begin{table*}")
    # print(r"\begin{center}")
    # print(r"\begin{tabular}{ p{0.1\linewidth}|p{0.07\linewidth} p{0.07\linewidth} p{0.07\linewidth} p{0.09\linewidth} p{0.07\linewidth} p{0.07\linewidth} } ")
    # print(r"Footstep Target Selection Method & Reward per Timestep & Reward per Footstep & Reward per Episode & Episode Length (timesteps) & Footstep Targets Hit & Timesteps per Footstep\\ ")
    # print(r"\hline")
    # metrics = ["rew/timestep", "rew/footstep", "rew/episode", "episode_len", "targets_hit", "time/target"]
    metrics = ["rew/episode", "episode_len", "targets_hit"]
    keys = ["in_place_rand", "in_place_fixed", "in_place_opt"]
    titles = ["Random", "Fixed", "Optimized"]
    maxes = {key: (None, -float("inf")) for key in metrics}
    for title, key in zip(titles, keys):
        for met in metrics:
            if r[key][met] > maxes[met][1]:
                maxes[met] = (title, r[key][met])
    latex_table_code_string = ""
    for title, key in zip(titles, keys):
        latex_table_code_string += title
        for met in metrics:
            if title == maxes[met][0]:
                # bold the number
                latex_table_code_string += f" & \\textbf{{{r[key][met]:,.2f}}}"
            else:
                latex_table_code_string += f" & {r[key][met]:,.2f}"

        latex_table_code_string += r"\\"
        latex_table_code_string += "\n"
    # print(r"\label{table:in_place_results}")
    # print(r"\end{tabular}")
    # print(r"\end{center}")
    # print(r"\end{table*}")
    out += latex_table_code_string
    temp = '#' * 10 + ' end latex code ' + '#' * 10
    out += '\n'*2 + '#' * len(temp) + '\n'
    out += temp + '\n'
    out += '#' * len(temp) + '\n' + '\n'

    out += latex_to_readable_table(latex_table_code_string, ['Method'] + metrics)
    print(out)
    return out


def latex_to_readable_table(latex_string: str, col_headings: list):
    out = ""
    # Replace \textbf{...} with *...* for bold values
    latex_string = latex_string.replace('\\textbf{', '*').replace('}', '*')

    # Split the input string into rows by double backslash
    rows = latex_string.strip().split('\\\\')

    # Clean and process each row
    table_data = []
    max_lengths = [len(heading) for heading in col_headings]

    for row in rows:
        if row.strip():  # Skip any empty rows
            # Split each row by '&' to get the columns
            columns = [col.strip() for col in row.split('&')]
            table_data.append(columns)

            # Update max_lengths to track the maximum width of each column
            for i, col in enumerate(columns):
                max_lengths[i] = max(max_lengths[i], len(col))

    # Print the column headings first
    formatted_header = " | ".join(f"{heading:<{max_lengths[i]}}" for i, heading in enumerate(col_headings))
    # print(formatted_header)
    # print("-" * len(formatted_header))
    out += formatted_header + '\n'
    out += "-" * len(formatted_header) + '\n'

    # Print the formatted table rows
    for row in table_data:
        formatted_row = " | ".join(f"{col:<{max_lengths[i]}}" for i, col in enumerate(row))
        # print(formatted_row)
        out += formatted_row + '\n'
    out += '\n' * 2
    return out


def generate_optimized_footstep_trajectories_plot(data, sota, args):
    max_footstep_traj_len = 20

    xlb, xup = (-0.6, 0.6)
    ylb, yup = (-0.8, 0.8)
    k = 2.0
    fig, _ = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8 * k - 4.0, 5 * k))
    # fig.subplots_adjust(wspace=0.5)
    ax = plt.subplot(2, 3, 1)
    ax.set_aspect("equal")
    plt.grid()
    plt.xlim(xlb, xup)
    plt.ylim(ylb, yup)
    #    [[ 0.2135, -0.1493],
    #       [-0.2694,  0.1495]],

    #      [[ 0.2134,  0.1494],
    #       [-0.2693, -0.1492]]
    ax.plot([-0.1493, 0.1495, 0.1494, -0.1492], [0.2135, -0.2694, 0.2134, -0.2693], 'kD')
    ax.set_title('Fixed Targets')
    im = plt.imread('plots/cutout_aliengo.png')
    # The dimensions [left, bottom, width, height] of the new Axes. All quantities are in fractions of figure width and height.
    newax = fig.add_axes([0.1815, 0.583, 0.21, 0.213], zorder=1, anchor="SW")
    im[..., 3][im[..., 3].nonzero()] = 0.5  # adjust image transparency
    newax.imshow(im)
    newax.axis('off')
    line_colors = [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
    colors = torch.rand(max_footstep_traj_len, 3)

    for subplot_idx, id_ in enumerate(data["Proposed Method"].keys(), 2):
        x = data["Proposed Method"][id_]['in_place_opt']
        footsteps = x["footstep_targets"]
        env_idx = x["current_footstep"].argmax()
        idcs = x["current_footstep"]
        # plt.figure()
        # for i in range(footsteps.shape[1]):
        #     plt.scatter(footsteps[:, i, :, 1], footsteps[:, i, :, 0])
        #     # plt.scatter(footsteps[0, i, :, 0], footsteps[0, i, :, 1])
        # plt.axis("equal")
        # plt.grid()
        # plt.title("20 env footsteps scatter")

        ax = plt.subplot(2, 3, subplot_idx)
        alpha = torch.linspace(1, 0.5, 20)
        color = torch.rand(1, 3)
        last_footstep = np.zeros((4, 2))
        for i in range(min(x["current_footstep"].min(), max_footstep_traj_len)):
            # if i % 2 == 0:
            # color = torch.rand(1, 3)
            for j in range(2):
                temp = footsteps[:, i, j]
                x = -temp[..., 1].mean().cpu()
                y = temp[..., 0].mean().cpu()
                # plt.text(x, y, s=str(i))
                ax.scatter(x, y,
                    s=(temp[..., 0].var().cpu() + temp[..., 1].var().cpu() + 0.0001) / 2 * 1e5,
                    c=colors[int(i // 2)].reshape(1, 3),
                    alpha=alpha[i].item())
                if i not in {0, 1}:
                    # ax.plot([last_footstep[j + (i % 2) * 2, 0], x], [last_footstep[j + (i % 2) * 2, 1], y], color=line_colors[j + (i % 2) * 2])
                    ax.arrow(last_footstep[j + (i % 2) * 2, 0], last_footstep[j + (i % 2) * 2, 1], x - last_footstep[j + (i % 2) * 2, 0], y - last_footstep[j + (i % 2) * 2, 1], color=line_colors[j + (i % 2) * 2], head_width=0.02, width=0.0025)
                last_footstep[j + (i % 2) * 2, 0] = x
                last_footstep[j + (i % 2) * 2, 1] = y
        ax.set_aspect("equal")
        ax.set_title(f'Seed {subplot_idx - 2}')
        plt.grid()
        plt.xlim(xlb, xup)
        plt.ylim(ylb, yup)
        # plt.xlim(-2.0, 2.0)
        # plt.ylim(-2.0, 2.0)
    fig.suptitle(f"Optimized Footstep Trajectories", y=0.95, fontsize="x-large")
    fig.supxlabel("Footstep Y position (m)", y=0.05, fontsize="large")
    fig.supylabel("Footstep X position (m)", x=0.05, fontsize="large")
    path = os.path.join(args.save_dir, "optimized_footstep_trajectories" + '.svg')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def regen_data_file():
    all_runs = [name for name in os.listdir('data') if os.path.isdir(os.path.join('data', name))]
    names = {}
    for run in all_runs:
        names[run] = run
    fname = 'all_data.pgz'
    data = parallel_load_relevent_data(names)
    path = os.path.join('data', fname)
    print(f"Saving data file at {path}")
    with gzip.GzipFile(path, 'w') as f:
        pickle.dump(data, f)
    print(f"Saved.")
    # sys.exit()


def load_all_data_file():
    """Each data file is uniquely identified by the hash of the dict values."""
    fname = 'all_data.pgz'
    path = os.path.join('data', fname)
    print(f'Loading {path}')
    s = time.time()
    with gzip.GzipFile(path, 'r') as f:
        data = CPU_Unpickler(f).load()
    print(f"{path} loaded in {time.time() - s:.2f} seconds")
    return data


def parallel_load_relevent_data(names):
    """load data relevent to the main performance super plot"""
    s = time.time()
    print(f"\nLoading data for {len(names)} runs.")

    data = {}
    futures = []
    with ThreadPoolExecutor() as executor:
        # future = executor.submit(pow, 323, 1235)
        # print(future.result())

        for method in names:
            data[method] = {}
            data_dir = os.path.join("data", names[method].replace(" ", "_").replace("(", "").replace(")", "").replace(".", ""))
            all_files = os.listdir(data_dir)
            relevant_files = []
            for file in all_files:
                if file[-3:] == "pgz":
                    relevant_files.append(file)
            for file in relevant_files:
                file_parts = file[:-4].split("__")
                if file_parts[0][:4] == "data":  # this means it was generated using the old pipeline
                    continue
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
                    # elif names[method][0] == "H" and all(x in file_parts for x in ["--no_ss", "--plot_values", "--des_dir_coef", "--des_dir"]):  # this is the special case for the flatground run
                    #     env_key = "flatground"
                    # elif names[method][0] == "F" and all(x in file_parts for x in ["--no_ss"]):
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
                # for serial execution
                # output, method, checkpoint, env_key = _load_and_process_data(path, method, checkpoint, env_key)
                # data[method][checkpoint][env_key] = output

        for future in as_completed(futures):
            output, method, checkpoint, env_key = future.result()
            data[method][checkpoint][env_key] = output

    print(f"Loaded data in {time.time() - s :.2f} seconds\n")
    return data


def _load_and_process_data(path, method, checkpoint, env_key):
    output = {}
    with gzip.GzipFile(path, 'r') as f:
        x = CPU_Unpickler(f).load()
    output["successful"] = x["succcessful"]
    output["collisions"] = x["collisions"]
    output["reward"] = x["reward"].squeeze().sum(dim=1)
    output["eps_len"] = x["still_running"].sum(dim=1).squeeze().to(torch.long) - 2
    idx = output["eps_len"] - 1
    output["distance_traveled"] = x["base_position"][torch.arange(len(idx)), idx, 0]

    # if method == "Proposed Method" or method[0] == "H":
    if "current_footstep" in x:
        output["footstep targets hit"] = x['current_footstep'] - 1
        output['current_footstep'] = x['current_footstep']
        output['footstep_targets'] = x['footstep_targets']
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
        if np.isnan(std):
            # std is nan due to only a single seed
            std = np.ones_like(std)
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
            if method == "Proposed Method" or method[0] == "H":
                for other_metric in ['in_place_rand', 'in_place_opt', 'in_place_fixed']:
                    vals.pop(other_metric)
            # sort into something that I can plot
            sorted_keys = list(vals.keys())
            sorted_keys.sort(key=lambda x: (x[1], -x[0]))
            y_points = [fg_data["mean"].item()]
            # errors = [fg_data["std"].item()]
            x_labels = ["Flatground"]
            for key in sorted_keys:
                y_points.append(vals[key]["mean"].item())
                # errors.append(vals[key]["std"].item())
                x_labels.append(f"{key[0] * 100}% / {key[1]}")
            if method == "Proposed Method":
                line_fmt = "--"
            else:
                line_fmt = "-"
            fmt = next(marker) + line_fmt
            # ax.errorbar(x_labels, y_points, fmt=fmt, yerr=errors, label=method, capsize=3.0)
            ax.errorbar(x_labels, y_points, fmt=fmt, label=method, capsize=3.0)
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
    if len(data) > 2:
        fig.legend(handles, labels, bbox_to_anchor=(1.25, 0.9))
    else:
        fig.legend(handles, labels, loc=(0.775, 0.9125))
    # plt.show()

    # plt.title("3 random seeds,  30 rollouts each, 10k steps timeout, Error bars are std between seeds")
    # plt.show()
    path = os.path.join(args.save_dir, "supplot.svg")
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot at {path}")


def _generate_single_plot(data, sota, args, envs_to_plot, metric, figsize, name):
    """Generate small plot of just success rate on a more limited sweep
    of environments
    """

    plt.figure(figsize=figsize)
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
        # ax.errorbar(x_labels, y_points, fmt=fmt, yerr=errors, label=method, capsize=3.0)
        ax.errorbar(x_labels, y_points, fmt=fmt, label=method, capsize=3.0)
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
    path = os.path.join(args.save_dir, name + ".svg")
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot at {path}")


def _generate_single_bar_plot(data, sota, args, envs_to_plot, metric, figsize, name):
    """Generate small bar plot of success rate on a more limited sweep of environments"""

    plt.figure(figsize=figsize)
    plt.xlabel("Stepping Stone Density / Stone Height Variation (m)", y=-0.025)
    plt.title("Proposed Method vs End-to-end Policy", y=0.95, x=0.5)
    ax = plt.gca()
    marker = itertools.cycle(('s', 'o', 'H', 'D', "^", ">", "<", 's', 'd'))
    # ax.grid()

    width = 0.35  # width of the bars
    x_indices = np.arange(len(envs_to_plot) + 1)  # +1 for "Flatground"
    bar_offset = 0  # To shift bars horizontally for different methods

    for method in data:
        vals = avg_across_seeds(data[method], metric)
        fg_data = vals.pop("flatground")

        sorted_keys = envs_to_plot
        sorted_keys.sort(key=lambda x: (x[1], -x[0]))

        y_points = [fg_data["mean"].item()]
        errors = [fg_data["std"].item()]
        x_labels = ["Flatground"]

        for key in sorted_keys:
            y_points.append(vals[key]["mean"].item())
            errors.append(vals[key]["std"].item())
            x_labels.append(f"{key[0] * 100}% / {key[1]}")

        # ax.bar(x_indices + bar_offset, y_points, width=width, yerr=errors, label=method, capsize=3.0)
        y_points = [pt * 100 for pt in y_points]
        ax.bar(x_indices + bar_offset, y_points, width=width, label=method, capsize=3.0)
        for i, v in enumerate(y_points):
            ax.text(x_indices[i] + bar_offset - width / 2 * 0, v + 0.02, f"{v:.0f}", ha="center", va="bottom", fontsize=6)
        bar_offset += width  # Shift next method's bars horizontally

    ax.set_xticks(x_indices + width / 2)  # Align ticks in the middle of grouped bars
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # ax.axvspan(1 - 0.5, 4 + 0.5, facecolor='b', alpha=0.1)
    # ax.axvspan(4.5, 8.5, facecolor='r', alpha=0.1)

    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(loc=(0.1, 1.02), ncol=2)

    if metric == "successful":
        ax.set_title("Terrain Traversal Success Rate", pad=25.0)
        ax.set_ylabel("Success Rate (%)")
    elif metric == "reward":
        ax.set_title("Fraction of Per-timestep Training Reward Achieved")
        ax.set_ylabel("Normalized Reward")
    elif metric == "distance_traveled":
        ax.set_title("Distance Traveled per Episode")
        ax.set_ylabel("Distance Traveled (m)")
    elif metric == "eps_len":
        ax.set_title("Episode Length")
        ax.set_ylabel("Episode Length (timesteps)")
    elif metric == "collisions":
        ax.set_title("Collisions")
        ax.set_ylabel("Average Number of Collisions")

    path = os.path.join(args.save_dir, name + "_bars.svg")
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot at {path}")

if __name__ == "__main__":
    main()

