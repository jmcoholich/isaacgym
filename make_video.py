import cv2
import os
import time
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_name', type=str, default=None)
parser.add_argument('--file_prefix', type=str, default="value_search")
args = parser.parse_args()

img_height = 1080
img_width = 1920
num_channels = 4
layers = 3
fps = 15
if args.output_name is None:
    args.output_name = 'test_vid'
filename = "videos/" + args.output_name + ".avi"
no_plot = False


# load frames
robot_frames = []
plot_frames = []
i = -1
location = "python/rlgpu/test_imgs/"
started = False

if os.path.exists(filename):
    input(f"Overwriting existing file at {filename}."
          f" Press any key to continue.")

while True:
    robot_frame = cv2.imread(os.path.join(location, f"{args.file_prefix}-cam-{i}.png"), cv2.IMREAD_UNCHANGED)
    plot_frame = cv2.imread(os.path.join(location, f"{args.file_prefix}-{i}.png"), cv2.IMREAD_UNCHANGED)

    i += 1
    if robot_frame is None or plot_frame is None:
        if started:
            break
        else:
            continue
    started = True
    assert robot_frame.shape[1] + plot_frame.shape[1] == img_width
    robot_frames.append(robot_frame)
    plot_frames.append(plot_frame)

# preprocess frames
video_speed = np.around(fps / 60.0, 3)
full_frames = np.zeros((len(plot_frames), img_height, img_width, 3),
                       dtype=np.uint8)
graph_height = plot_frames[0].shape[0]
top_margin_height = (img_height - graph_height) // 2
bottom_margin_height = img_height - graph_height - top_margin_height
margin_width = plot_frames[0].shape[1]
top_margin = np.zeros((top_margin_height, margin_width, num_channels))
bottom_margin = np.zeros((bottom_margin_height, margin_width, num_channels))
no_plot_margin = np.zeros((img_height, margin_width, num_channels))
for i in range(len(robot_frames)):
    graph = plot_frames.pop(0)
    if no_plot:
        left_side = no_plot_margin
    else:
        left_side = np.concatenate((top_margin, graph, bottom_margin), axis=0)
    full_frame = np.concatenate((robot_frames.pop(0), left_side), axis=1)
    full_frame = cv2.putText(
        full_frame, f"x{video_speed} Speed" , (10, img_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0))
    full_frames[i] = full_frame[:, :, :-1]

# save video
size = (img_width, img_height)
if not os.path.exists('videos/'):
    os.makedirs('videos/')

out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
for i in range(full_frames.shape[0]):
    out.write(full_frames[i].astype(np.uint8))
out.release()
print('Video saved')
