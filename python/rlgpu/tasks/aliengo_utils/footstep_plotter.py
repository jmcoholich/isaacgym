"""
This is for plotting where non-hierarchical agents make contact with the
terrain.
"""

import torch
from .trot_footstep_generator import get_circle_lines

class FootstepPlotter:
    def __init__(self, task):
        self.device = task.device
        self.task = task
        self.gym = task.gym
        self.reset()

    def reset(self):
        self.gym.clear_lines(self.task.viewer)

    def __call__(self):
        """This records the footsteps that have been hit, then sends
        them to the plotter.
        """
        foot_rgb = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
                                 [0.0, 0.0, 0.5], [0.5, 0.5, 0.0]])
        new_contacts = ((self.task.prev_foot_z_contact_forces[0] <= 0.0)
                        & (self.task.foot_contact_forces[0, :, 2] > 0.0))
        if not new_contacts.any():
            return
        centers = self.task.foot_center_pos[0, new_contacts]
        centers[..., 2] -= self.task.foot_collision_sphere_r - 0.01
        lines, _ = get_circle_lines(
            centers, radius=0.02, rand_colors=False, foot_colors=False)
        colors = foot_rgb[new_contacts]
        colors = colors.repeat_interleave(lines.shape[0] // colors.shape[0], 0)

        self.gym.add_lines(self.task.viewer, self.task.envs[0],
            lines.shape[0] // 2, lines.cpu().numpy(),
            colors.cpu().numpy())



