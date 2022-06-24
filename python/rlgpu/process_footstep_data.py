import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from gather_stats_utils import fname_parser

def analyze_rews():
    # data_dir = os.path.join("footstep_data", "box_size_0p15")
    data_dir = os.path.join("footstep_data")
    all_files = os.listdir(data_dir)
    relevant_files = []
    for file in all_files:
        if file[-3:] == "pkl":
            relevant_files.append(file)

    data = {}
    for file in relevant_files:
        checkpoint, is_optim, is_random = fname_parser(file)
        if checkpoint not in data.keys():
            data[checkpoint] = {}
        with open(os.path.join(data_dir, file), 'rb') as f:
            x = pickle.load(f)
            termination_idcs = x["still_running"].sum(dim=1).squeeze().to(torch.int)
            common_footstep = x["current_footstep"].min().item()
            num_runs = x["still_running"].shape[0]
            # print(file)
            # print(termination_idcs)
            # print()
            # print()
            # continue

            # rew_per_timestep = torch.zeros(num_runs)
            # rew_per_footstep = torch.zeros(num_runs)
            rews_sum = 0.0
            for i in range(num_runs):
                rews_sum += x["reward"][i, :termination_idcs[i], 0].sum()

            if is_random:
                key = "random"
            elif is_optim:
                key = "optim"
            else:
                key = "in_place"
            data[checkpoint][key] = {}
            data[checkpoint][key]["Average rew per timestep"] = (rews_sum / termination_idcs.sum()).item()
            data[checkpoint][key]["Average rew per footstep"] = (rews_sum / x['current_footstep'].sum()).item()
            data[checkpoint][key]["Average rew per rollout"] = (rews_sum / num_runs).item()
            data[checkpoint][key]["Average rollout length"] = (termination_idcs.float().mean()).item()
            data[checkpoint][key]["Average footstep reached"] = (x['current_footstep'].float().mean()).item()
            data[checkpoint][key]["Average timesteps per footstep"] = (termination_idcs.float().sum() / x['current_footstep'].float().sum()).item()

            print(file)
            for k, v in data[checkpoint][key].items():
                print(f"{k}: {v :0.2f}")
            # print(f"Average rew per timestep: {rews_sum / termination_idcs.sum() :.3f}")
            # print(f"Average rew per footstep: {rews_sum / x['current_footstep'].sum() :.2f}")
            # print(f"Average rew per rollout: {rews_sum / num_runs :.2f}")
            # print(f"Average rollout length: {termination_idcs.float().mean() :.1f}")
            # print(f"Average footstep reached: {x['current_footstep'].float().mean() :.1f}")
            # print(f"Average timesteps per footstep: {termination_idcs.float().sum() / x['current_footstep'].float().sum() :.1f}")
            print("\n")

    return data


def print_table(data):
    num_seeds = len(list(data.keys()))
    for metric in data[list(data.keys())[0]]["optim"]:
        print(metric)
        for key in ["random", "in_place", "optim"]:
            sum_ = 0
            for val in data.values():
                sum_ += val[key][metric]
            avg_metric = sum_/num_seeds
            print(f"\t{key}: {avg_metric:.2f}")
            if key == "in_place":
                in_place = avg_metric
            elif key == "optim":
                print(f"\t% change: {(avg_metric - in_place) / in_place :.1%}")
        print()


def make_plots(data):
    X = data.keys()
    optim_rew = [data[checkpoint]["optim"]["Average rew per timestep"] for checkpoint in X]
    in_place_rew = [data[checkpoint]["in_place"]["Average rew per timestep"] for checkpoint in X]

    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.2, optim_rew, 0.4, label='Optimized footstep targets')
    plt.bar(X_axis + 0.2, in_place_rew, 0.4, label='Fixed footstep targets')

    plt.xticks(X_axis, X_axis)
    plt.xlabel("Random Seeds")
    plt.ylabel("Average Reward Per Timestep")
    plt.title("Effect of value-function optimization on stepping in place")
    plt.legend()
    plt.show()

def make_footstep_plots():
    max_footstep_traj_len = 20
    data_dir = os.path.join("footstep_data")
    all_files = os.listdir(data_dir)
    relevant_files = []
    for file in all_files:
        if file[-3:] == "pkl":
            relevant_files.append(file)


    xlb, xup = (-0.6, 0.6)
    ylb, yup = (-0.8, 0.8)
    data = {}
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
    subplot_idx = 2
    line_colors = [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
    colors = torch.rand(max_footstep_traj_len, 3)
    for file in relevant_files:
        checkpoint, is_optim, is_random = fname_parser(file)
        if is_random or not is_optim: continue
        with open(os.path.join(data_dir, file), 'rb') as f:
            x = pickle.load(f)
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
            subplot_idx += 1
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
    path = os.path.join("plots", "optimized_footstep_trajectories" + '.svg')
    plt.savefig(path, bbox_inches='tight')
    # plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    data = analyze_rews()
    print_table(data)
    # make_footstep_plots()
    # make_plots(data)
