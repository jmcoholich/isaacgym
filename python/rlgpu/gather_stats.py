import wandb
import torch
from subprocess import Popen, PIPE, run
import csv
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import os
import numpy as np
import itertools
import time

from gather_stats_utils import get_wandb_run_ids

"""
TODO
- write metadata to csv file (params)
- add metadata to plots
- make a way to reduce num_envs for high memory configurations

"""
def main():
    data_dir = "data"
    load_data_ = True
    generate_csv_ = False
    # csv_name = "agg2"
    csv_name = None  # uses timestamp instead"
    generate_plots_ = True
    use_all_gpus = True
    add_perturb = False
    device_id = 1  # if not using all gpus
    # load_fnames = "best_aggregate.csv"
    # load_fnames = ["data03_02_2022_19_47.pkl", "data04_02_2022_11_26.pkl", "data04_02_2022_16_59.pkl"]
    # load_fnames = ["data08_02_2022_19_30.pkl"]
    # load_fnames = ["data25_02_2022_18_38.pkl"]  # H ss state (0.2 rand new)
    # load_fnames = ["data27_02_2022_11_26.pkl"]  # F ss state final
    load_fnames = ["data25_02_2022_18_38.pkl", "data27_02_2022_11_26.pkl"]  # F ss state final

    params = {
        "num_rollouts": 20,
        # "timeout": 1000,
        "timeout": 10_000,
        "des_dir_coef": 50,
    }

    # key is the name of the wandb run (all seeds have same name)
    # value is the name of the configuation used in plot
    # policy_types = {
    #     "H flat ground (0.99 tau)":     "Hierarchical, trained on Flat Ground",
    #     "H flat ground + TG":           "Hierarchical, trained on Flat Ground w/ TG",
    #     "H ss state":                   "Hierarchical, trained on Stepping Stones w/ state and TG",
    #     # "F flat ground":                "Flat, trained on Flat Ground",
    #     "F flat ground + TG":           "Flat, trained on Flat Ground w/ TG",
    #     "F ss state":                   "Flat, trained on Stepping Stones w/ state and TG",
    # }

    policy_types = {
        "H ss state (0.2 rand new)": "Proposed Method",
        "F ss state final": "End-to-end",
    }

    # policy_types = {
    #     "H ss state":                   "164 point scan, [512, 256] MLP (original)",
    #     "H ss state huge":              "644 point scan, [512, 256] MLP",
    #     "H ss state huge + huge net":   "644 point scan, [1024, 512] MLP",
    #     "H ss state (easier env)":             "Slightly easier environment (min stone density 0.5)",
    #     "H ss state (easier env, 0.99 tau)":   "Slightly easier environment (min stone density 0.5), lambda = 0.99",
    # }

    # policy_types = {
    #     "H ss state":                   "164 point scan, [512, 256] MLP (original)",
    #     "H ss state huge":              "644 point scan, [512, 256] MLP",
    #     "H ss state huge + huge net":   "644 point scan, [1024, 512] MLP",
    #     "H ss state (easier env)":             "Slightly easier environment (min stone density 0.5)",
    #     "H ss state (easier env, 0.99 tau)":   "Slightly easier environment (min stone density 0.5), lambda = 0.99",
    # }

    envs = [
        "Training_Reward",
        "Flat_Ground",
        "Stepping_Stones_25%_infill_0.0_height_variation",
        "Stepping_Stones_37.5%_infill_0.0_height_variation",
        "Stepping_Stones_50%_infill_0.0_height_variation",
        "Stepping_Stones_62.5%_infill_0.0_height_variation",
        "Stepping_Stones_75%_infill_0.0_height_variation",
        "Stepping_Stones_87.5%_infill_0.0_height_variation",
        "Stepping_Stones_100%_infill_0.0_height_variation",

        "Stepping_Stones_25%_infill_0.05_height_variation",
        "Stepping_Stones_37.5%_infill_0.05_height_variation",
        "Stepping_Stones_50%_infill_0.05_height_variation",
        "Stepping_Stones_62.5%_infill_0.05_height_variation",
        "Stepping_Stones_75%_infill_0.05_height_variation",
        "Stepping_Stones_87.5%_infill_0.05_height_variation",
        "Stepping_Stones_100%_infill_0.05_height_variation",

        "Stepping_Stones_25%_infill_0.1_height_variation",
        "Stepping_Stones_37.5%_infill_0.1_height_variation",
        "Stepping_Stones_50%_infill_0.1_height_variation",
        "Stepping_Stones_62.5%_infill_0.1_height_variation",
        "Stepping_Stones_75%_infill_0.1_height_variation",
        "Stepping_Stones_87.5%_infill_0.1_height_variation",
        "Stepping_Stones_100%_infill_0.1_height_variation",
    ]
    # envs = [
    #     # "Training_Reward",
    #     # "Flat_Ground",
    #     "Stepping_Stones_25%_infill_0.0_height_variation",
    #     "Stepping_Stones_38%_infill_0.0_height_variation",
    #     "Stepping_Stones_50%_infill_0.0_height_variation",
    #     "Stepping_Stones_60%_infill_0.0_height_variation",
    #     "Stepping_Stones_25%_infill_0.1_height_variation",
    #     "Stepping_Stones_38%_infill_0.1_height_variation",
    #     "Stepping_Stones_50%_infill_0.1_height_variation",
    #     "Stepping_Stones_60%_infill_0.1_height_variation",
    # ]
    if not load_data_:
        st = time.time()
        data = collect_data(policy_types, envs, params, device_id,
                            use_all_gpus, add_perturb)
        et = time.time()
        print(f"Data collection elapsed time: {et - st :.1f}")
        fname = save_data(data, data_dir)
    else:
        fnames = load_fnames
        data = load_data(fnames, data_dir)

    if generate_csv_:
        if csv_name is None:
            csv_name = fname
        generate_csv(data, csv_name, envs, data_dir, policy_types)
    if generate_plots_:
        generate_small_plot(data, policy_types, params, "Successful")
        generate_sup_plot(data, policy_types, params, ["Distance_Traveled", "Successful", "Episode_Length", "Reward"])
        # generate_plots(data, policy_types, "Distance_Traveled", params)
        # generate_plots(data, policy_types, "Successful", params)
        # generate_plots(data, policy_types, "Episode_Length", params)
        # generate_plots(data, policy_types, "Reward", params)


def generate_csv(data, fname, envs, data_dir, policy_types):
    """pass"""
    average_all = True

    path = os.path.join(data_dir, fname + '.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Policy Type"] + envs)  # This is the header
        for policy_type in data.keys():
            line = [policy_types[policy_type]]
            for env in envs:
                if env == "Training_Reward":
                    val = "Reward"
                else:
                    val = "Distance_Traveled"
                infostr = avg_across_seeds(data[policy_type][env], val)
                line.append(infostr)
            writer.writerow(line)


def avg_across_seeds(data, val, floats=False, ptype=None):
    """Takes means of all rollouts for each policy, then compute mean
    and std across the random seeds with these numbers as datapoints."""
    all_samples = torch.tensor([])
    eps_lengths = torch.tensor([])
    for seed in data.keys():
        all_samples = torch.cat((all_samples, data[seed][val].mean().unsqueeze(0)))
        eps_lengths = torch.cat((all_samples, data[seed]["Episode_Length"].mean().unsqueeze(0)))
    if floats:
        if val == "Reward" and ptype is not None:
            train_rews = {"H ss state (0.2 rand new)": (5053.30 + 5000.23 + 5010.39 + 4466.37 + 4296.00) / 5.0 / 1500,
            "F ss state final": 1.0}
            all_samples /= train_rews[ptype]
            all_samples /= eps_lengths.mean()
        return all_samples.mean().item(), all_samples.std().item()
    else:  # return strings for csv
        return f"{all_samples.mean():.2f}" + u" \u00B1 " + f"{all_samples.std():.2f}"


def generate_plots(data, policy_types, metric, params):
    """Generate plots with performance on envs in increasing difficulty.
    Each policy type has two lines.
    """

    # env_order1 = [
    #     "Flat_Ground",
    #     "Stepping_Stones_100%_infill_0.0_height_variation",
    #     "Stepping_Stones_75%_infill_0.0_height_variation",
    #     "Stepping_Stones_50%_infill_0.0_height_variation",
    #     "Stepping_Stones_25%_infill_0.0_height_variation",
    #     "Stepping_Stones_100%_infill_0.1_height_variation",
    #     "Stepping_Stones_75%_infill_0.1_height_variation",
    #     "Stepping_Stones_50%_infill_0.1_height_variation",
    #     "Stepping_Stones_25%_infill_0.1_height_variation",
    # ]
    # env_order1 = {
    #     "Stepping_Stones_25%_infill_0.0_height_variation",
    #     "Stepping_Stones_38%_infill_0.0_height_variation",
    #     "Stepping_Stones_50%_infill_0.0_height_variation",
    #     "Stepping_Stones_60%_infill_0.0_height_variation",
    #     "Stepping_Stones_25%_infill_0.1_height_variation",
    #     "Stepping_Stones_38%_infill_0.1_height_variation",
    #     "Stepping_Stones_50%_infill_0.1_height_variation",
    #     "Stepping_Stones_60%_infill_0.1_height_variation",
    # }
    env_order1 = [
        "Flat_Ground",
        "Stepping_Stones_100%_infill_0.0_height_variation",
        "Stepping_Stones_87.5%_infill_0.0_height_variation",
        "Stepping_Stones_75%_infill_0.0_height_variation",
        "Stepping_Stones_62.5%_infill_0.0_height_variation",
        "Stepping_Stones_50%_infill_0.0_height_variation",
        "Stepping_Stones_37.5%_infill_0.0_height_variation",
        "Stepping_Stones_25%_infill_0.0_height_variation",

        "Stepping_Stones_100%_infill_0.05_height_variation",
        "Stepping_Stones_87.5%_infill_0.05_height_variation",
        "Stepping_Stones_75%_infill_0.05_height_variation",
        "Stepping_Stones_62.5%_infill_0.05_height_variation",
        "Stepping_Stones_50%_infill_0.05_height_variation",
        "Stepping_Stones_37.5%_infill_0.05_height_variation",
        "Stepping_Stones_25%_infill_0.05_height_variation",

        "Stepping_Stones_100%_infill_0.1_height_variation",
        "Stepping_Stones_87.5%_infill_0.1_height_variation",
        "Stepping_Stones_75%_infill_0.1_height_variation",
        "Stepping_Stones_62.5%_infill_0.1_height_variation",
        "Stepping_Stones_50%_infill_0.1_height_variation",
        "Stepping_Stones_37.5%_infill_0.1_height_variation",
        "Stepping_Stones_25%_infill_0.1_height_variation",
    ]
    # env_order2 = [
    #     "Flat_Ground",
    #     "Stepping_Stones_100%_infill_0.1_height_variation",
    #     "Stepping_Stones_75%_infill_0.1_height_variation",
    #     "Stepping_Stones_50%_infill_0.1_height_variation",
    #     "Stepping_Stones_25%_infill_0.1_height_variation",
    # ]

    x_labels = [
        "Flat Ground",
        "100%  / 0.0",
        "87.5%  / 0.0",
        "75%  / 0.0",
        "62.5%  / 0.0",
        "50%  / 0.0",
        "37.5%  / 0.0",
        "25%  / 0.0",

        "100% / 0.05",
        "87.5% / 0.05",
        "75% / 0.05",
        "62.5% / 0.05",
        "50% / 0.05",
        "37.5% / 0.05",
        "25% / 0.05",

        "100% / 0.1",
        "87.5% / 0.1",
        "75% / 0.1",
        "62.5% / 0.1",
        "50% / 0.1",
        "37.5% / 0.1",
        "25% / 0.1",
    ]
    # x_labels = [
    #     "25%",
    #     "38%",
    #     "50%",
    #     "60%",
    #     "25% (+)",
    #     "38% (+)",
    #     "50% (+)",
    #     "60% (+)",
    # ]
    marker = itertools.cycle(('s', 'o', 'H', 'D', "^", ">", "<", 's', 'd'))
    plt.figure(figsize=(15 / 1.25, 3.0 / 1.25))
    plt.grid()
    for policy_type in data.keys():
        vals = []
        errors = []
        for env in env_order1:
            val, error = avg_across_seeds(data[policy_type][env], metric, floats=True, ptype=policy_type)
            vals.append(val)
            errors.append(error)
        if policy_type[0] == "H":
            line_fmt = "--"
        else:
            line_fmt = "-"
        fmt = next(marker) + line_fmt
        label = policy_types[policy_type]
        plt.errorbar(x_labels, vals, fmt=fmt, yerr=errors, label=label, capsize=3.0)
    plt.legend()
    plt.set_xlabel("Stepping Stone Density / Stone Height Variation (m)")
    plt.set_xticks(rotation=45)
    plt.ylabel(metric)
    plt.suptitle(metric)
    if metric == "Successful":
        plt.suptitle("Terrain Traversal Success Rate")
        plt.ylabel("Success Rate")
    elif metric == "Reward":
        plt.suptitle("Fraction of Per-timestep Training Reward Achieved")
        plt.ylabel("Normalized Reward")
    elif metric == "Distance_Traveled":
        plt.suptitle("Distance Traveled per Episode")
        plt.ylabel("Distance Traveled (m)")
    elif metric == "Episode_Length":
        plt.suptitle("Episode Length")
        plt.ylabel("Episode Length (timesteps)")

    # plt.title("3 random seeds,  30 rollouts each, 10k steps timeout, Error bars are std between seeds")
    # plt.show()
    if not os.path.exists("plots"):
        os.makedirs("plots")
    path = "plots/" + metric + "_benchmark" + '.svg'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot at {path}")


def generate_sup_plot(data, policy_types, params, metrics):
    """Generate plots with performance on envs in increasing difficulty.
    Each policy type has two lines.
    """

    env_order1 = [
        "Flat_Ground",
        "Stepping_Stones_100%_infill_0.0_height_variation",
        "Stepping_Stones_87.5%_infill_0.0_height_variation",
        "Stepping_Stones_75%_infill_0.0_height_variation",
        "Stepping_Stones_62.5%_infill_0.0_height_variation",
        "Stepping_Stones_50%_infill_0.0_height_variation",
        "Stepping_Stones_37.5%_infill_0.0_height_variation",
        "Stepping_Stones_25%_infill_0.0_height_variation",

        "Stepping_Stones_100%_infill_0.05_height_variation",
        "Stepping_Stones_87.5%_infill_0.05_height_variation",
        "Stepping_Stones_75%_infill_0.05_height_variation",
        "Stepping_Stones_62.5%_infill_0.05_height_variation",
        "Stepping_Stones_50%_infill_0.05_height_variation",
        "Stepping_Stones_37.5%_infill_0.05_height_variation",
        "Stepping_Stones_25%_infill_0.05_height_variation",

        "Stepping_Stones_100%_infill_0.1_height_variation",
        "Stepping_Stones_87.5%_infill_0.1_height_variation",
        "Stepping_Stones_75%_infill_0.1_height_variation",
        "Stepping_Stones_62.5%_infill_0.1_height_variation",
        "Stepping_Stones_50%_infill_0.1_height_variation",
        "Stepping_Stones_37.5%_infill_0.1_height_variation",
        "Stepping_Stones_25%_infill_0.1_height_variation",
    ]

    x_labels = [
        "Flat Ground",
        "100%  / 0.0",
        "87.5%  / 0.0",
        "75%  / 0.0",
        "62.5%  / 0.0",
        "50%  / 0.0",
        "37.5%  / 0.0",
        "25%  / 0.0",

        "100% / 0.05",
        "87.5% / 0.05",
        "75% / 0.05",
        "62.5% / 0.05",
        "50% / 0.05",
        "37.5% / 0.05",
        "25% / 0.05",

        "100% / 0.1",
        "87.5% / 0.1",
        "75% / 0.1",
        "62.5% / 0.1",
        "50% / 0.1",
        "37.5% / 0.1",
        "25% / 0.1",
    ]

    # plt.figure()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(15 / 1.25 - 1, 3.0 / 1.25 * 4.5 - 2))
    fig.subplots_adjust(hspace=0.5)
    axes = [ax1, ax2, ax3, ax4]
    flag = True
    fig.supxlabel("Stepping Stone Density / Stone Height Variation (m)", y=-0.025)
    fig.suptitle("Proposed Method vs End-to-end Policy", y=0.95, x=0.5)
    for metric, ax in zip(metrics, axes):
        marker = itertools.cycle(('s', 'o', 'H', 'D', "^", ">", "<", 's', 'd'))
        ax.grid()
        for policy_type in data.keys():
            vals = []
            errors = []
            for env in env_order1:
                val, error = avg_across_seeds(data[policy_type][env], metric, floats=True, ptype=policy_type)
                vals.append(val)
                errors.append(error)
            if policy_type[0] == "H":
                line_fmt = "--"
            else:
                line_fmt = "-"
            fmt = next(marker) + line_fmt
            label = policy_types[policy_type]
            ax.errorbar(x_labels, vals, fmt=fmt, yerr=errors, label=label, capsize=3.0)
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
        if metric == "Successful":
            ax.set_title("Terrain Traversal Success Rate")
            ax.set_ylabel("Success Rate")
        elif metric == "Reward":
            ax.set_title("Fraction of Per-timestep Training Reward Achieved")
            ax.set_ylabel("Normalized Reward")
        elif metric == "Distance_Traveled":
            ax.set_title("Distance Traveled per Episode")
            ax.set_ylabel("Distance Traveled (m)")
        elif metric == "Episode_Length":
            ax.set_title("Episode Length")
            ax.set_ylabel("Episode Length (timesteps)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.775, 0.9125))
    # plt.show()

    # plt.title("3 random seeds,  30 rollouts each, 10k steps timeout, Error bars are std between seeds")
    # plt.show()
    if not os.path.exists("plots"):
        os.makedirs("plots")
    path = "plots/" + "supplot" + '.svg'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot at {path}")


def generate_small_plot(data, policy_types, params, metric):
    """Generate plots with performance on envs in increasing difficulty.
    Each policy type has two lines.
    """

    env_order1 = [
        "Flat_Ground",
        "Stepping_Stones_100%_infill_0.0_height_variation",
        # "Stepping_Stones_87.5%_infill_0.0_height_variation",
        "Stepping_Stones_75%_infill_0.0_height_variation",
        # "Stepping_Stones_62.5%_infill_0.0_height_variation",
        "Stepping_Stones_50%_infill_0.0_height_variation",
        # "Stepping_Stones_37.5%_infill_0.0_height_variation",
        "Stepping_Stones_25%_infill_0.0_height_variation",

        # "Stepping_Stones_100%_infill_0.05_height_variation",
        # "Stepping_Stones_87.5%_infill_0.05_height_variation",
        # "Stepping_Stones_75%_infill_0.05_height_variation",
        # "Stepping_Stones_62.5%_infill_0.05_height_variation",
        # "Stepping_Stones_50%_infill_0.05_height_variation",
        # "Stepping_Stones_37.5%_infill_0.05_height_variation",
        # "Stepping_Stones_25%_infill_0.05_height_variation",

        "Stepping_Stones_100%_infill_0.1_height_variation",
        # "Stepping_Stones_87.5%_infill_0.1_height_variation",
        "Stepping_Stones_75%_infill_0.1_height_variation",
        # "Stepping_Stones_62.5%_infill_0.1_height_variation",
        "Stepping_Stones_50%_infill_0.1_height_variation",
        # "Stepping_Stones_37.5%_infill_0.1_height_variation",
        "Stepping_Stones_25%_infill_0.1_height_variation",
    ]

    x_labels = [
        "Flat Ground",
        "100%  / 0.0",
        # "87.5%  / 0.0",
        "75%  / 0.0",
        # "62.5%  / 0.0",
        "50%  / 0.0",
        # "37.5%  / 0.0",
        "25%  / 0.0",

        # "100% / 0.05",
        # "87.5% / 0.05",
        # "75% / 0.05",
        # "62.5% / 0.05",
        # "50% / 0.05",
        # "37.5% / 0.05",
        # "25% / 0.05",

        "100% / 0.1",
        # "87.5% / 0.1",
        "75% / 0.1",
        # "62.5% / 0.1",
        "50% / 0.1",
        # "37.5% / 0.1",
        "25% / 0.1",
    ]

    plt.figure(figsize=(5.0, 2.5))
    plt.xlabel("Stepping Stone Density / Stone Height Variation (m)", y=-0.025)
    plt.title("Proposed Method vs End-to-end Policy", y=0.95, x=0.5)
    ax = plt.gca()
    marker = itertools.cycle(('s', 'o', 'H', 'D', "^", ">", "<", 's', 'd'))
    ax.grid()
    for policy_type in data.keys():
        vals = []
        errors = []
        for env in env_order1:
            val, error = avg_across_seeds(data[policy_type][env], metric, floats=True, ptype=policy_type)
            vals.append(val)
            errors.append(error)
        if policy_type[0] == "H":
            line_fmt = "--"
        else:
            line_fmt = "-"
        fmt = next(marker) + line_fmt
        label = policy_types[policy_type]
        ax.errorbar(x_labels, vals, fmt=fmt, yerr=errors, label=label, capsize=3.0)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.axvspan(1 - 0.5, 4 + 0.5, facecolor='b', alpha=0.1)
    # ax.axvspan(7.5, 14.5, facecolor='g', alpha=0.1)
    ax.axvspan(4.5, 8.5, facecolor='r', alpha=0.1)
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(loc=(0.1, 1.02), ncol=2)
    if metric == "Successful":
        ax.set_title("Terrain Traversal Success Rate", pad=25.0)
        ax.set_ylabel("Success Rate")
    elif metric == "Reward":
        ax.set_title("Fraction of Per-timestep Training Reward Achieved")
        ax.set_ylabel("Normalized Reward")
    elif metric == "Distance_Traveled":
        ax.set_title("Distance Traveled per Episode")
        ax.set_ylabel("Distance Traveled (m)")
    elif metric == "Episode_Length":
        ax.set_title("Episode Length")
        ax.set_ylabel("Episode Length (timesteps)")
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels, loc=(0.775, 0.9125))
    # plt.show()

    if not os.path.exists("plots"):
        os.makedirs("plots")
    path = "plots/" + "smallplot" + '.svg'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot at {path}")


def load_data(load_fnames, data_dir):
    all_data = {}
    for load_fname in load_fnames:
        path = os.path.join(data_dir, load_fname)
        with open(path, 'rb') as f:
            print(f"Loading file at: {path}")
            data = pickle.load(f)
            print(f"done")
        all_data.update(data)
    return all_data


def save_data(data, data_dir):
    time_stamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    fname = 'data' + time_stamp + '.pkl'
    path = os.path.join(data_dir, fname)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(path, 'wb') as f:
        print(f"Writing gathered data to: {path}")
        pickle.dump(data, f)
        print(f"done")
    return fname



def collect_data(policy_types, envs, params, device_id, use_all_gpus,
                 add_perturb):
    if not use_all_gpus:
        raise NotImplementedError()
    run_ids = get_run_ids(policy_types)
    cmds, data, cmd2data = generate_commands(policy_types, envs, run_ids,
                                             params, add_perturb)
    total_cmds = len(cmds)
    # now, run all this stuff in parallel
    num_gpus = detect_num_gpus()
    jobs = [None] * num_gpus
    cmd_keys = [None] * num_gpus
    cmd_counter = 0
    # init jobs
    for i in range(num_gpus):
        cmd = cmds.pop()
        cmd_keys[i] = "".join(cmd)
        cmd_counter += 1
        print(f"Running command {cmd_counter}/{total_cmds}")
        print(" ".join(cmd) + '\n')
        jobs[i] = Popen(cmd + ["--device_id", str(i)], stdout=PIPE,
                        stderr=PIPE, text=True)

    while True:
        for i, job in enumerate(jobs):  # check if any of the jobs are done
            if job is not None:
                # job.poll() returns None if the job is still running
                # job.poll() returns an exit code if the job has finished
                poll = job.poll()
            else:
                poll = None

            if poll in {0, -11, 139}:  # successful exit or segmentation fault
                output = job.stdout.readlines()
                process_output(output, cmd2data, cmd_keys[i],
                               params["num_rollouts"])

                if not cmds:  # if there are no more runs
                    jobs[i] = None
                    if all(not k for k in jobs):  # if all jobs are done, break
                        break
                elif cmds:  # start another run if there are still runs left
                    cmd = cmds.pop()
                    cmd_keys[i] = "".join(cmd)
                    cmd_counter += 1
                    # time.sleep(10.0)  # give previous run plenty of time to exit to avoid issues.
                    print(f"Running command {cmd_counter}/{total_cmds}")
                    print(" ".join(cmd) + '\n')
                    jobs[i] = Popen(cmd + ["--device_id", str(i)], stdout=PIPE,
                                    stderr=PIPE, text=True)
            elif poll is not None and poll != 0:
                print(job.stderr.readlines())
                breakpoint()

        if all(not k for k in jobs):  # if all jobs are done, break
            break
        time.sleep(1.0)

    return data


def generate_commands(policy_types, envs, run_ids, params, add_perturb):
    cmd2data = {}  # maps from command strings to entry in the data dict
    cmds = []
    data = {}

    # I need a structure with  dimensions
    # (Policy type, env, random_seed, meausrement, rollouts)
    # the last dimension will be a torch.tensor
    # the first 4 will be dicts
    for policy_type in policy_types.keys():
        data[policy_type] = {}
        for env in envs:
            data[policy_type][env] = {}
            for run_id, workstation in run_ids[policy_type]:
                data[policy_type][env][run_id] = {}

                cmd = [
                    "python", "rlg_train.py",
                    "--play",
                    "--headless",
                    "--checkpoint", run_id,
                    "--ws", workstation,
                    "--timeout", str(params["timeout"]),
                    "--gather_stats", str(params["num_rollouts"])]

                if add_perturb:
                    cmd += ["--add_perturb", "100.0"]


                if policy_type[0] == "H" and env != "Training_Reward":
                    cmd += ["--plot_values",
                            "--des_dir_coef", str(params["des_dir_coef"]),
                            "--footstep_targets_in_place",
                            "--num_envs", "20"]
                else:
                    cmd += ["--num_envs", str(params["num_rollouts"])]

                if env == "Flat_Ground":
                    cmd += ["--no_ss"]

                if env[:15] == "Stepping_Stones":
                    temp = env.split("_")
                    infill = str(float(temp[2][:-1]) / 100.0)
                    height_var = temp[4]
                    cmd += ["--add_ss", "--ss_infill", str(infill),
                            "--ss_height_var", str(height_var)]
                cmds.append(cmd)
                key = "".join(cmd)
                assert key not in cmd2data
                cmd2data[key] = data[policy_type][env][run_id]
    return cmds, data, cmd2data



def process_output(output, cmd2data, cmd_key, num_rollouts):
    """Take raw stdout string from IsaacGym and extract the data."""
    stuff = torch.zeros(num_rollouts)
    i = 0
    finished_tensor = False
    output = "".join(output)
    for item in output.split()[::-1]:
        if item == "Loaded":
            return
        if finished_tensor:
            cmd2data[cmd_key][item] = stuff
            stuff = torch.zeros(num_rollouts)
            i = 0
            finished_tensor = False
            continue
        stuff[i] = float(item)
        i += 1
        if i == num_rollouts:
            finished_tensor = True


if __name__ == "__main__":
    main()

# subproc_test.py
# from subprocess import Popen, PIPE

# https://shuzhanfan.github.io/2017/12/parallel-processing-python-subprocess/


# cmds_list = [["sleep", "5"] for _ in range(5)]
# procs_list = [Popen(cmd, stdout=PIPE, stderr=PIPE, text=True) for cmd in cmds_list]
# while True:
#     for proc in procs_list:
#         print(proc.poll())
# print(proc.stdout.readlines())
# breakpoint()

