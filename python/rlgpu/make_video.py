import subprocess
import time
import sys
import os

def main():
    # the name is the first commandline arg
    name = sys.argv[1]
    frames_dir = "frames"
    results_dir = "videos"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    date_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    compile_video(date_time_str, frames_dir, results_dir, name)


def run_cmd(command, env=None):
    print('------------------------------------------')
    print("Running command:", command)
    if env is None:
        env = {}
    completed_process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    # Print the output and error messages
    print("Standard Output:")
    print(completed_process.stdout)

    print("Standard Error:")
    print(completed_process.stderr)

    # Print the return code
    print("Return Code:", completed_process.returncode)
    print('------------------------------------------')
    print()


def compile_video(date_time_str, frames_dir, results_dir, name):  # TODO make sure framerate is correct and add a speed to the video
    command = f"ffmpeg -framerate 60 -i {frames_dir}/frame-%04d.png -c:v libx264 -pix_fmt yuv420p {results_dir}/{name}_{date_time_str}.mp4"
    run_cmd(command, env={'LD_PRELOAD': '/usr/lib/x86_64-linux-gnu/libffi.so.7'})


if __name__=="__main__":
    main()
