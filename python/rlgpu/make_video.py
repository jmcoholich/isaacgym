import subprocess
import time
import sys
import os
from glob import glob
import cv2
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def main():
    """load all frames from "frames". The files are FL-frame-0001.png, FL-frame-0002.png, ...
    For each of FL, FR, RL, RR, we will load all frames and compile them into a video.
    There is also the camera frames where are named frame-0001.png, frame-0002.png, ...
    """
    try:
        name = sys.argv[1]
    except IndexError:
        print("Please provide a name for the video")

    # load all frames
    frames_dict = {}
    feet = ['FL', 'FR', 'RL', 'RR']
    # Load frames for each camera
    for foot in feet:
        frame_pattern = os.path.join('H_frames', f"{foot}-frame-*.png")
        frames_dict[foot] = sorted(glob(frame_pattern))

    # Load general camera frames
    general_frame_pattern = os.path.join('H_frames', "frame-*.png")
    frames_dict['H'] = sorted(glob(general_frame_pattern))
    general_frame_pattern = os.path.join('F_frames', "frame-*.png")
    frames_dict['F'] = sorted(glob(general_frame_pattern))

    # assemble and save frames
    frames_dir = "final_frames"
    if os.path.exists(frames_dir):
        os.system(f"rm -rf {frames_dir}")
    os.makedirs(frames_dir)
    frame_idx = 0
    colorbar = cv2.imread("colorbar.png")
    # scale by 0.6
    scale = 0.55
    colorbar = cv2.resize(colorbar, (int(colorbar.shape[1]*scale), int(colorbar.shape[0]*scale)))

    legend = cv2.imread("legend_optimal_footstep_target.png")
    scale = 0.8
    legend = cv2.resize(legend, (int(legend.shape[1]*scale), int(legend.shape[0]*scale)))

    with ThreadPoolExecutor() as executor:
        while True:
            try:
                H_frame = frames_dict['H'][frame_idx + 1]
                F_frame = frames_dict['F'][frame_idx + 1]
                fl = frames_dict['FL'][frame_idx]
                fr = frames_dict['FR'][frame_idx]
                rl = frames_dict['RL'][frame_idx]
                rr = frames_dict['RR'][frame_idx]
            except IndexError:
                break
            with ThreadPoolExecutor() as executor2:
                futures = []
                futures.append(executor2.submit(cv2.imread, H_frame))
                futures.append(executor2.submit(cv2.imread, F_frame))
                futures.append(executor2.submit(cv2.imread, fl))
                futures.append(executor2.submit(cv2.imread, fr))
                futures.append(executor2.submit(cv2.imread, rl))
                futures.append(executor2.submit(cv2.imread, rr))
                H_img = futures[0].result()
                F_img = futures[1].result()
                fl = futures[2].result()
                fr = futures[3].result()
                rl = futures[4].result()
                rr = futures[5].result()

            # add each foot plot to its respective corner of the frame
            H_img[:fl.shape[0], :fl.shape[1]] = fl
            H_img[:fr.shape[0], -fr.shape[1]:] = fr
            H_img[-rl.shape[0]:, :rl.shape[1]] = rl
            H_img[-rr.shape[0]:, -rr.shape[1]:] = rr
            # colorbar should be centered on the right edge of the H_img
            mid = 1080//2
            cbar_half = colorbar.shape[0]//2
            H_img[mid-cbar_half:mid+colorbar.shape[0]-cbar_half, -colorbar.shape[1]:] = colorbar
            img = np.zeros((1080, 1920, 3))
            # legend should go right under the colorbar, aligned with the right edge
            H_img[mid+colorbar.shape[0]-cbar_half-legend.shape[0]:mid+colorbar.shape[0]-cbar_half, -legend.shape[1]:] = legend


            # H img on the right, L img on the left
            img[:, :H_img.shape[1]] = H_img
            img[:, -F_img.shape[1]:] = F_img


            # cv2.imwrite(os.path.join(frames_dir, f"frame-{frame_idx:04d}.png"), img)
            executor.submit(cv2.imwrite, os.path.join(frames_dir, f"frame-{frame_idx:04d}.png"), img)
            frame_idx += 1

    # the name is the first commandline arg
    results_dir = "videos"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    date_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    for fps in [15, 30, 60]:
        compile_video(date_time_str, frames_dir, results_dir, name, fps)


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


def compile_video(date_time_str, frames_dir, results_dir, name, fps):
    command = f"ffmpeg -framerate {fps} -i {frames_dir}/frame-%04d.png -c:v libx264 -pix_fmt yuv420p {results_dir}/{name}_{fps}fps_{date_time_str}.mp4"
    run_cmd(command, env={'LD_PRELOAD': '/usr/lib/x86_64-linux-gnu/libffi.so.7'})


if __name__=="__main__":
    main()
