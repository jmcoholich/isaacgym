import torch
from utils import *


class ExpertPolicy:
    def __init__(self, task):
        self.task = task
        self.gym = task.gym
        self.viewer = task.viewer
        self.device = task.device
        self.num_envs = task.num_envs
        self.timesteps_per_traj = 60
        self.timesteps_rest = 30
        self.step_height = 0.15
        self.starting_foot_pos = torch.tensor(
            [[0.2134, 0.1494, 0.0285],  # FL
             [0.2135, -0.1493, 0.0285],  # FR
             [-0.2694, 0.1495, 0.0285],  # RL
             [-0.2693, -0.1492, 0.0285]],  # RR
            device=self.device)
        self.step_order = torch.tensor([2, 0, 3, 1], device=self.device)
        self.prev_output = None
        self.prev_prev_output = None

    def mean_dist(self):
        if self.prev_prev_output is None:
            output = self.prev_output
        else:
            output = self.prev_prev_output
        d = (output - self.task.foot_center_pos).norm(dim=2)
        return -d.mean(dim=1)

    def reward(self):
        if self.prev_prev_output is None:
            output = self.prev_output
        else:
            output = self.prev_prev_output
        # d = ((output - self.task.foot_center_pos) * 25.0).pow(4)
        d = ((output - self.task.foot_center_pos)).pow(2)
        return -d.sum(dim=(1, 2))

    def step_in_place_traj(self):
        """Return a 3D point for each foot."""
        output = self.starting_foot_pos.tile(
            self.num_envs, 1, 1)
        foot_period = self.timesteps_per_traj + self.timesteps_rest
        total_period = foot_period * 4

        steps = self.task.progress_buf.clone() % total_period
        step_idx = steps // foot_period
        leg_idx = self.step_order[step_idx]
        in_step_phase = (self.task.progress_buf.clone() % foot_period) <= self.timesteps_per_traj
        # breakpoint()
        if in_step_phase.any():
            start = self.starting_foot_pos[leg_idx][in_step_phase]
            traj = self.generate_flat_traj(start, start)
            phase = (self.task.progress_buf.clone() % foot_period)[in_step_phase] / self.timesteps_per_traj
            output[in_step_phase, leg_idx[in_step_phase]] = traj(phase)
            # print(output[0])
            # breakpoint()
        # breakpoint()
        # print(output[0])
        if self.prev_output is not None:
            self.prev_prev_output = self.prev_output.clone()
        self.prev_output = output.clone()
        return output.clone()

    def generate_flat_traj(self, start, end):
        """Generate batched triangular trajectories between two points, both on
        flat ground. """
        center = (start + end) / 2.0
        center[:, 2] += self.step_height
        device = start.device
        if not self.task.headless:
            """ Only draw lines for the 0th env, if it is pass in"""
            self.gym.clear_lines(self.viewer)

            vertices = torch.cat([start[0], center[0], center[0], end[0]], dim=0)
            colors = torch.zeros(2, 3)
            self.gym.add_lines(self.viewer, self.task.envs[0],
                               2, vertices.cpu().numpy(), colors.numpy())

        def traj(phase):
            """Output a 3D point of where the robot should track.
            Phase is in range [0, 1]"""
            num_to_gen = phase.shape[0]
            assert (0.0 <= phase).all() and (phase <= 1.0).all()
            output = torch.zeros(num_to_gen, 3, device=device)
            idcs = phase > 0.5
            temp = ((phase[idcs] - 0.5) * 2.0).unsqueeze(-1)
            output[idcs] = temp * end[idcs] + (1 - temp) * center[idcs]

            idcs = phase <= 0.5
            temp = (phase[idcs] * 2.0).unsqueeze(-1)
            output[idcs] = temp * center[idcs] + (1 - temp) * start[idcs]
            return output
        return traj


if __name__ == "__main__":
    pol = ExpertPolicy()
    n = 2
    device = 'cuda:0'
    start = torch.zeros(n, 3, device=device)
    end = torch.zeros(n, 3,  device=device)
    end[:, 0] = 1.0
    end[:, 1] = 1.0
    traj = pol.generate_flat_traj(start, end)
    phase = torch.ones(n, device='cuda:0')
    # traj(phase)
    breakpoint()