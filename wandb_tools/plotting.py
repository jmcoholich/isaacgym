import wandb
import numpy as np
import matplotlib.pyplot as plt



def main():
    names = ["H ss state no pmtg final", "H ss state (0.2 rand new)"]
    # first, get all the model names for the SOTA runs
    api = wandb.Api()
    runs = api.runs("jcoholich/aliengo_12")

    data = {}
    for name in names:
        data[name] = {}
    for run_ in runs:
        if run_.name in names:
            n_samples = 15000
            data[run_.name][run_.id] = {}
            data[run_.name][run_.id]["rewards"] = run_.history(samples=n_samples)["rewards"].values
            data[run_.name][run_.id]["frame"] = run_.history(samples=n_samples)["frame"].values
            dim = min(data[run_.name][run_.id]["rewards"].shape[0], data[run_.name][run_.id]["frame"].shape[0])


    plt.figure(figsize=(5, 3))
    colors = iter([[0.0, 0.0, 1.0], [0.0, 0.5, 0.0]])
    linestyles = iter(['--', '-'])
    labels = iter(["Without TG", "With TG (Proposed Method)"])
    for config in data.values():
        n_pts = min([np.count_nonzero(~np.isnan(run["rewards"])) for run in config.values()])
        n_runs = len(list(config.keys()))
        rews = np.zeros((n_runs, n_pts))
        for i, run_ in enumerate(config.values()):
            # grab all non-NaN values
            idcs = ~np.isnan(run_["rewards"])
            rews[i] = run_["rewards"][idcs][:n_pts]
        mean = rews.mean(axis=0)
        stderr = rews.std(axis=0) * n_runs**-0.5
        stddev = rews.std(axis=0)
        error = stddev
        color = next(colors)
        linestyle = next(linestyles)
        plt.plot(run_["frame"][idcs][:n_pts], mean, label=next(labels), color=color, linestyle=linestyle)
        plt.fill_between(run_["frame"][idcs][:n_pts], mean - error, mean + error, alpha=0.3, color=color, linestyle=linestyle)
    plt.grid()
    plt.title("LLP Training Reward")
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Reward")
    plt.savefig("LLP_Training_Rew.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
