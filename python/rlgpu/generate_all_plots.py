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
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--id", type=str,
                       help="This is the id of the run to evalute.")
    group.add_argument("--run_name", type=str,
                       help="If passed, eval all runs with this name.")
    return parser.parse_args()


def main():

    sota = {
        "Proposed Method" : "H ss state (0.2 rand new)",
        "End-to-end": "F ss state final",
    }
    args = get_args()
    if args.id:
        raise NotImplementedError()
    data = load_relevent_data(sota, args)
    generate_sup_plot(data, sota)


def load_relevent_data(sota, args):
    """load data relevent to the main performance super plot"""
    names_to_analyze = sota
    if args.run_name or args.id:
        if args.run_name[0] == "H":
            names_to_analyze["Proposed Method"] = args.run_name
        else:
            names_to_analyze["End-to-end"] = args.run_name

    data = {}

    for method in names_to_analyze:
        data[method] = {}
        data_dir = os.path.join("data", names_to_analyze[method].replace(" ", "_").replace("(", "").replace(")", "").replace(".", ""))
        all_files = os.listdir(data_dir)
        relevant_files = []
        for file in all_files:
            if file[-3:] == "pkl":
                relevant_files.append(file)
        for file in relevant_files:
            file_parts = file[5:-4].split("__")
            if "--ss_height_var" in file_parts:
                heightvar = float(file_parts[file_parts.index("--ss_height_var") + 1])
                infill = float(file_parts[file_parts.index("--ss_infill") + 1])
                env_key = (infill, heightvar)
            elif method == "Proposed Method" and all(x in file_parts for x in ["--no_ss", "--plot_values", "--des_dir_coef", "--des_dir"]):  # this is the special case for the flatground run
                env_key = "flatground"
                print(method)
                print(file)
                print()
            elif method == "End-to-end" and all(x in file_parts for x in ["--no_ss"]):
                env_key = "flatground"
                print(method)
                print(file)
                print()
            else:
                continue

            checkpoint = file_parts[file_parts.index("--checkpoint") + 1]
            if checkpoint not in data[method]:
                data[method][checkpoint] = {}
            data[method][checkpoint][env_key] = {}
            temp = data[method][checkpoint][env_key]

            with open(os.path.join(data_dir, file), 'rb') as f:
                x = pickle.load(f)
                temp["successful"] = x["succcessful"]
                temp["reward"] = x["reward"]
                temp["eps_len"] = x["still_running"].sum(dim=1).squeeze().to(torch.long)
                idx = temp["eps_len"] - 1
                temp["distance_traveled"] = x["base_position"][torch.arange(len(idx)), idx, 0]

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

    return data


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

    for env in val:
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


def generate_sup_plot(data, sota):
    """Generate plots with performance on envs in increasing difficulty.
    Each policy type has two lines.

    The metrics are:
    - Success rate
    - distance traveled
    - reward as fraction of training reward
    - episode length
    """

    # env_order1 = [
    #     "Flat_Ground",
    #     "Stepping_Stones_100%_infill_0.0_height_variation",
    #     "Stepping_Stones_87.5%_infill_0.0_height_variation",
    #     "Stepping_Stones_75%_infill_0.0_height_variation",
    #     "Stepping_Stones_62.5%_infill_0.0_height_variation",
    #     "Stepping_Stones_50%_infill_0.0_height_variation",
    #     "Stepping_Stones_37.5%_infill_0.0_height_variation",
    #     "Stepping_Stones_25%_infill_0.0_height_variation",

    #     "Stepping_Stones_100%_infill_0.05_height_variation",
    #     "Stepping_Stones_87.5%_infill_0.05_height_variation",
    #     "Stepping_Stones_75%_infill_0.05_height_variation",
    #     "Stepping_Stones_62.5%_infill_0.05_height_variation",
    #     "Stepping_Stones_50%_infill_0.05_height_variation",
    #     "Stepping_Stones_37.5%_infill_0.05_height_variation",
    #     "Stepping_Stones_25%_infill_0.05_height_variation",

    #     "Stepping_Stones_100%_infill_0.1_height_variation",
    #     "Stepping_Stones_87.5%_infill_0.1_height_variation",
    #     "Stepping_Stones_75%_infill_0.1_height_variation",
    #     "Stepping_Stones_62.5%_infill_0.1_height_variation",
    #     "Stepping_Stones_50%_infill_0.1_height_variation",
    #     "Stepping_Stones_37.5%_infill_0.1_height_variation",
    #     "Stepping_Stones_25%_infill_0.1_height_variation",
    # ]

    # x_labels = [
    #     "Flat Ground",
    #     "100%  / 0.0",
    #     "87.5%  / 0.0",
    #     "75%  / 0.0",
    #     "62.5%  / 0.0",
    #     "50%  / 0.0",
    #     "37.5%  / 0.0",
    #     "25%  / 0.0",

    #     "100% / 0.05",
    #     "87.5% / 0.05",
    #     "75% / 0.05",
    #     "62.5% / 0.05",
    #     "50% / 0.05",
    #     "37.5% / 0.05",
    #     "25% / 0.05",

    #     "100% / 0.1",
    #     "87.5% / 0.1",
    #     "75% / 0.1",
    #     "62.5% / 0.1",
    #     "50% / 0.1",
    #     "37.5% / 0.1",
    #     "25% / 0.1",
    # ]

    # plt.figure()
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
    rand_id = str(torch.randint(100000, (1,)).item())
    path = "plots/" + "supplot" + rand_id + '.svg'
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot at {path}")


if __name__ == "__main__":
    main()

