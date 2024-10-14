import subprocess
import time
import sys
import os
from glob import glob
import cv2

def main():
    """load all frames from "frames". The files are FL-frame-0001.png, FL-frame-0002.png, ...
    For each of FL, FR, RL, RR, we will load all frames and compile them into a video.
    There is also the camera frames where are named frame-0001.png, frame-0002.png, ...
    """

    # load all frames
    frames_dict = {}
    feet = ['FL', 'FR', 'RL', 'RR']
    # Load frames for each camera
    for foot in feet:
        frame_pattern = os.path.join('frames', f"{foot}-frame-*.png")
        frames_dict[foot] = sorted(glob(frame_pattern))

    # Load general camera frames
    general_frame_pattern = os.path.join('frames', "frame-*.png")
    frames_dict['camera'] = sorted(glob(general_frame_pattern))

    # assemble and save frames
    frames_dir = "final_frames"
    if os.path.exists(frames_dir):
        os.system(f"rm -rf {frames_dir}")
    os.makedirs(frames_dir)
    frame_idx = 0
    while True:
        try:
            frame = frames_dict['camera'][frame_idx + 1]
            fl = frames_dict['FL'][frame_idx]
            fr = frames_dict['FR'][frame_idx]
            rl = frames_dict['RL'][frame_idx]
            rr = frames_dict['RR'][frame_idx]
        except IndexError:
            break
        img = cv2.imread(frame)
        fl = cv2.imread(fl)
        fr = cv2.imread(fr)
        rl = cv2.imread(rl)
        rr = cv2.imread(rr)

        # add each foot plot to its respective corner of the frame
        img[:fl.shape[0], :fl.shape[1]] = fl
        img[:fr.shape[0], -fr.shape[1]:] = fr
        img[-rl.shape[0]:, :rl.shape[1]] = rl
        img[-rr.shape[0]:, -rr.shape[1]:] = rr

        cv2.imwrite(os.path.join(frames_dir, f"frame-{frame_idx:04d}.png"), img)
        frame_idx += 1


    # the name is the first commandline arg
    try:
        name = sys.argv[1]
    except IndexError:
        print("Please provide a name for the video")
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

    if completed_process.returncode != 0:
        # Print the output and error messages
        print("Standard Output:")
        print(completed_process.stdout)

        print("Standard Error:")
        print(completed_process.stderr)

        # Print the return code
        print("Return Code:", completed_process.returncode)
        print('------------------------------------------')
        print()
    else:
        print("Done!")
        print('------------------------------------------')


def compile_video(date_time_str, frames_dir, results_dir, name):  # TODO make sure framerate is correct and add a speed to the video
    command = f"ffmpeg -framerate 60 -i {frames_dir}/frame-%04d.png -c:v libx264 -pix_fmt yuv420p {results_dir}/{name}_{date_time_str}.mp4"
    run_cmd(command, env={'LD_PRELOAD': '/usr/lib/x86_64-linux-gnu/libffi.so.7'})


if __name__=="__main__":
    main()
