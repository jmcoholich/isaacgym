import subprocess
import random
import itertools

running_job_nicknames = subprocess.run('squeue -u jcoholich3 --format="%.100j"', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

names = running_job_nicknames.stdout.split()[1:]
print(names)
print(len(names))
base_x_vel_coefs = [0.5, 1.0, 2.0]
base_x_vel_clips = [0.125, 0.25, 0.5, 1.0]
y_vel_pens = [0.05, 0.1, 0.2]
smoothnesses = [0.0, 0.03125, 0.0625]
prod = list(itertools.product(base_x_vel_coefs, base_x_vel_clips, y_vel_pens, smoothnesses))
random.seed(1)
random.shuffle(prod)
for base_x_vel_coef, base_x_vel_clip, y_vel_pen, smoothness in prod[30:60]:
    job_nickname = f"f_{base_x_vel_coef}_{base_x_vel_clip}_{y_vel_pen}_{smoothness}".replace('.', 'p')
    if job_nickname not in names:
        print(job_nickname)
