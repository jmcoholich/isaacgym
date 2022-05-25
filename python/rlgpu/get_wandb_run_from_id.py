import wandb
import argparse

def main():
    run_id = input("enter run id: ")
    api = wandb.Api()
    runs = api.runs("jcoholich/aliengo_12")

    for run_ in runs:
        if run_.config["run_id"] == run_id:
            print(run_.name)


if __name__ == "__main__":
    main()
