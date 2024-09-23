import wandb
from subprocess import Popen, PIPE, run


def detect_num_gpus():
    output = run("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l",
               stdout=PIPE, shell=True, text=True)
    return int(output.stdout.strip())


def fname_parser(fname):
    fname = fname.split("__")
    checkpoint = fname[fname.index("--checkpoint") + 1]
    is_optim = "--plot_values" in fname
    is_random = "--random_footsteps" in fname
    return checkpoint, is_optim, is_random


def get_wandb_run_name_from_id(id):
    api = wandb.Api()
    runs = api.runs("jcoholich/aliengo_12_F_sweep")  # TODO
    for run_ in runs:
        if run_.config["run_id"] == str(id):
            return run_.name


def get_ws_from_run_id(id):
    api = wandb.Api()
    runs = api.runs("jcoholich/aliengo_12_F_sweep")  # TODO
    for run_ in runs:
        if run_.config["run_id"] == str(id):
            return run_.tags[0]


def get_wandb_ids_from_run_name(name, project_name, username):
    """Return a list of run ids from a single run name string."""
    run_ids = []

    # first, get all the model names for the SOTA runs
    api = wandb.Api()
    runs = api.runs(f"{username}/{project_name}")

    for run_ in runs:
        if run_.name == name:
            run_ids.append(run_.config["run_id"])

    print(f"\nRun IDs with name: '{name}'\n")
    for id_ in run_ids:
        print(id_)
    print()

    return run_ids


def get_num_runs_per_name():
    """Return a list of run ids from a single run name string."""
    names = {}

    # first, get all the model names for the SOTA runs
    api = wandb.Api()
    runs = api.runs("jcoholich/aliengo_12_F_sweep")  # TODO

    for run_ in runs:
        if run_.name in names:
            names[run_.name] += 1
        else:
            names[run_.name] = 1

    for key, val in names.items():
        if val > 5:
            print(key)

    return names

def get_wandb_run_ids(names):

    run_ids = {}
    for key in names.keys(): run_ids[key] = []

    # first, get all the model names for the SOTA runs
    api = wandb.Api()
    runs = api.runs("jcoholich/aliengo_12_F_sweep")  # TODO

    for run_ in runs:
        if run_.name in names.keys():
            run_ids[run_.name].append((run_.config["run_id"], run_.tags[0]))

    # print run_ids gathered
    print()
    for key, value in run_ids.items():
        print(key)
        for i in range(len(value)):
            print("\t" + value[i][0])
        print()

    return run_ids


def gpu_parallel_cmd_runner(cmds):
    total_cmds = len(cmds)
    # now, run all this stuff in parallel
    num_gpus = detect_num_gpus()
    jobs = [None] * num_gpus
    cmd_counter = 0
    # init jobs
    for i in range(num_gpus):
        cmd = cmds.pop()
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

            if poll in {0, -11, 139}:  # successful exit or segmentation fault (I don't care)
                if not cmds:  # if there are no more runs
                    jobs[i] = None
                else:  # start another run if there are still runs left
                    cmd = cmds.pop()
                    cmd_counter += 1
                    # time.sleep(10.0)  # give previous run plenty of time to exit to avoid issues.
                    print(f"Running command {cmd_counter}/{total_cmds}")
                    print(" ".join(cmd) + '\n')
                    jobs[i] = Popen(cmd + ["--device_id", str(i)], stdout=PIPE,
                                    stderr=PIPE, text=True)
            elif poll is not None and poll != 0:  # unsuccessful exit
                trace = job.stderr.readlines()
                for line in trace:
                    print(line, end='')
                breakpoint()

        if all(not k for k in jobs):  # if all jobs are done, break
            break


if __name__ == "__main__":
    names = get_num_runs_per_name()
    breakpoint()
    # print(get_ws_from_run_id(220422175144086636))
    # print(get_wandb_run_name_from_id(220422175144086636))
