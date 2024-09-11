from rlgpu.tasks.aliengo_utils.utils import batch_z_2D_rot_mat
import torch
from isaacgym import gymapi as g


class RandomFootstepGenerator:
    """This is a class for generating footstep targets
    and keeping track of current footstep.
    """

    def __init__(self, task):
        self.device = task.device

        # these are in foot idx order
        self.starting_foot_pos= torch.tensor(
            [[0.2134, 0.1494, 0.0285],  # FL
             [0.2135, -0.1493, 0.0285],  # FR
             [-0.2694, 0.1495, 0.0285],  # RL
             [-0.2693, -0.1492, 0.0285]],  # RR
            device=self.device)

        self.task = task
        self.gym = task.gym
        self.viewer = task.viewer
        self.envs = task.envs
        self.cfg = task.cfg['footstep_target_parameters']
        self.rand_every_timestep = self.cfg.get("rand_every_timestep", False)
        self.center_rew_multiplier = self.cfg.get("center_rew_multiplier", 0.5)
        # self.update_period = self.cfg.get("update_period", 1)
        self.num_envs = task.num_envs
        self.current_footstep = torch.ones(self.num_envs, dtype=torch.long,
                                           device=self.device)
        self.curr = self.curriculum()
        self.vis = not self.task.headless  # TODO make sure its not just this preventing me from rendering to vid with headless
        # self.n_foosteps_hit = None
        if self.vis:
            self.curr_step_body = None
            self.step_body_ids = []
        # self.footsteps = torch.zeros(self.num_envs, 2 * self.cfg['n_cycles'] \
        #     + 2, 2, 2 + self.include_z, device=self.device)
        self.env_arange = torch.arange(self.num_envs, device=self.device)
        self.footsteps = self.generate_footsteps(self.env_arange)
        if self.rand_every_timestep:
            self.actual_footsteps = self.footsteps.clone()

        self.counter = torch.zeros(self.num_envs, device=self.device)
        self.last_time_hit_footstep = torch.zeros(self.num_envs,
                                                  device=self.device)

    def curriculum(self):
        current_epoch = 0
        if hasattr(self.task, "epochs"):
            current_epoch = self.task.epochs

        if self.task.args.adaptive_curriculum != -1 and self.task.wandb_log_counter % self.task.cfg["steps_num"]:  # I only want to adjust when an epoch is calculated.
            if self.current_footstep.float().mean() >= 10:
                self.curr += self.task.adaptive_curriculum
                self.curr = min(self.curr, 1.0)
        if self.task.cfg["curriculum_length"] != -1:
            max_epochs = self.task.cfg["curriculum_length"]
            return min(current_epoch / max_epochs, 1.0)
        else:
            return 1.0  # NOTE this disables the curriculum

    def generate_footsteps(self, env_ids):
        footsteps = torch.zeros(len(env_ids), self.cfg['n_cycles'], 4, 2, device=self.device)
        footsteps[:, :] = self.starting_foot_pos[:, :2]

        for i in range(1, self.cfg['n_cycles'], 2):
            # pick one foot to move
            foot_to_move = torch.randint(4, (len(env_ids),), device=self.device)
            # randomly generate x and y perturbations
            x_perturb = (torch.rand(len(env_ids), device=self.device) - 0.5) * self.cfg['footstep_rand']
            y_perturb = (torch.rand(len(env_ids), device=self.device) - 0.5) * self.cfg['footstep_rand']
            # add perturbations to the foot
            # footsteps[:, i] = footsteps[:, i - 1].clone()
            footsteps[:, i, foot_to_move, 0] += x_perturb
            footsteps[:, i, foot_to_move, 1] += y_perturb
        return footsteps

    def plot_current_targets(self):
        add_height = 0
        self.gym.clear_lines(self.viewer)
        centers = self.footsteps[0, self.current_footstep[0]]
        centers = torch.cat((centers, torch.zeros(4, 1, device=self.device)), -1)
        centers[:, 2] += add_height
        vertices, colors = get_circle_lines(centers.reshape(4, 3), rand_colors=True)
        self.gym.add_lines(self.viewer, self.envs[0],
                        vertices.shape[0] // 2, vertices.cpu().numpy(),
                        colors.cpu().numpy())

    def plot_all_targets(self):
        self.gym.clear_lines(self.viewer)
        centers = self.footsteps[0]
        centers = torch.cat((centers, torch.zeros(self.footsteps[0].shape[0], 4, 1, device=self.device)), -1)
        centers[:, :, 2] += add_height
        vertices, _ = get_circle_lines(centers.reshape(self.footsteps[0].shape[0] * self.footsteps[0].shape[1], 3), radius=0.02, rand_colors=True)
        colors = torch.rand(self.footsteps[0].shape[0], 3,
                            device=self.device)
        colors = colors.repeat_interleave(vertices.shape[0] // colors.shape[0], 0)
        self.gym.add_lines(self.viewer, self.envs[0],
                        vertices.shape[0] // 2, vertices.cpu().numpy(),
                        colors.cpu().numpy())

    def get_footstep_idcs(self, current_footstep):
        """
        The foot indices follow a pattern of
        0. (0, 3)
        1. (1, 2)

        2. (0, 3)
        3. (1, 2)

        So the indices can be determined by whether we are going for an
        even (including zero) or odd-numbered footstep target.
        """
        foot_idcs = torch.tensor([0, 3], device=self.device).tile(self.num_envs, 1)
        foot_idcs[current_footstep % 2 == 1] = torch.tensor(
            [1, 2], device=self.device)
        return foot_idcs

    def get_footstep_distance(self, current_footstep):
        """Return 2D XY vector from current feet to current targets."""
        # if current_footstep is None:
        #     current_footstep = self.current_footstep

        target_location = self.footsteps[self.env_arange, current_footstep]
        # target_location[:, :, 2] += self.task.foot_collision_sphere_r

        # foot_idcs = self.get_footstep_idcs(current_footstep)
        foot_center_pos = self.task.foot_center_pos[:, :, :2]
        return target_location - foot_center_pos

    def hit_random_footsteps(self, current_footstep=None):
        if current_footstep is None:
            current_footstep = self.current_footstep
        dist = self.get_footstep_distance(current_footstep).norm(dim=2)
        in_range = (dist < self.cfg['footstep_distance_threshold'])
        in_contact = (
            self.task.foot_contact_forces[self.env_arange, :, 2]
            > self.cfg["contact_force_treshold"]
            )
        reached = torch.logical_and(in_range, in_contact)
        if self.task.cfg["actionSpace"] == "pmtg":
            # also ensure that the TG is in contact phase for the feet in question
            phase_offset = torch.tensor(self.task.cfg["pmtg"]["phase_offset"], device=self.device)
            z_traj_input = (torch.ones(self.num_envs, 4, device=self.device)
                            * self.task.t.unsqueeze(-1) + phase_offset + 0.5)
            norm_phases = (z_traj_input % 1.0) * 4 - 2.0
            in_contact_phase = norm_phases < 0.0
            foot_idcs = self.get_footstep_idcs(current_footstep)
            tg_in_contact_phase = in_contact_phase[
                self.env_arange.unsqueeze(-1), foot_idcs]
            reached = torch.logical_and(reached, tg_in_contact_phase)
        hit_all = reached.all(-1)
        # add extra reward for hitting center of footstep
        reached_float = reached.float()
        if reached.any():
            reached_float[reached] += self.center_rew_multiplier \
                * (1 - dist[reached] / self.cfg['footstep_distance_threshold'])
        return reached_float.sum(dim=1), hit_all

    def rand_next_footstep(self):
        """Add randomness to the footstep targets at every timestep
        as a value function experiment. The observation reflects this since
        update_tensors() is called before observe() in post-physics step.
        """
        self.footsteps[self.env_arange, self.current_footstep] = \
            self.actual_footsteps.clone()[self.env_arange, self.current_footstep] \
            + (torch.rand(self.num_envs, 2, 2, device=self.device) - 0.5) \
            * self.cfg["every_timestep_rand_value"]

    def compute_rewards(self):
        """Calculate rewards and store them for later, since the observation
        needs to reflect the next footstep target if the current target is
        hit. However, rewards need to reflect the last action and observation.
        """
        rew_dict = {}

        # hit_footstep
        rew_dict["hit_footstep"], hit_all = self.hit_random_footsteps()
        rew_dict["hit_footstep"][hit_all] *= self.cfg["hit_all_multiplier"]

        # # foot stay
        # rew_dict["foot_stay"], _ = self.hit_trot_footstep_pair(
        #     current_footstep=(self.current_footstep - 1))

        # foot velocity
        rew_dict["foot_velocity"] = self.velocity_towards_footsteps()
        return rew_dict, hit_all

    def update_next_next_targets(self):
        """The next next targets should be at the location of the shoulder
        joints plus some delta in the des_dir_direction."""
        x_offset = 0.2
        idcs = self.get_footstep_idcs(self.current_footstep - 1)
        hip_pos = self.task.hip_pos[..., :2].clone()
        hip_pos[:, :, 0] += self.task.args.nn_ft_dist

        yaw = self.task.base_euler[:, 2]
        rot_mat = batch_z_2D_rot_mat(yaw)
        width_addition = torch.tensor([[0.0], [1.0]], device=self.device)
        width_addition = (rot_mat @ width_addition).squeeze(-1)

        hip_pos[:, [0, 2], 1] += self.task.args.nn_ft_width
        hip_pos[:, [1, 3], 1] -= self.task.args.nn_ft_width
        hip_pos = hip_pos[self.env_arange.unsqueeze(-1), idcs]
        self.footsteps[self.env_arange, self.current_footstep + 1] = hip_pos

    def update(self):
        """Calculate rewards and store them for later, since the observation
        needs to reflect the next footstep target if the current target is
        hit. However, rewards need to reflect the last action and observation.
        """
        self.curr = self.curriculum()
        self.rew_dict, hit_targets = self.compute_rewards()
        self.current_footstep[hit_targets] += 1
        self.last_time_hit_footstep[hit_targets] = \
            self.counter[hit_targets].clone()
        self.counter += 1

        if hit_targets[0] and not self.task.args.plot_all_targets:
            self.plot_current_targets()


    def velocity_towards_footsteps(self):
        """Return velocity of current foot towards footsteps. This is just
        xy velocity, since I'm not keeping track of target z anymore.
        """
        # get global feet velocity
        # foot_idcs = self.get_footstep_idcs(self.current_footstep)
        foot_vels = self.task.foot_vel[:, :, :2]

        # compute unit vectors from foot to targets
        target_pos = self.footsteps[self.env_arange, self.current_footstep]
        foot_pos = self.task.foot_center_pos[:, :, :2]
        diff = target_pos - foot_pos
        unit_vec = diff / (diff.norm(dim=-1, keepdim=True) + 1e-6)
        # dot product to find velocity component towards footstep target
        return (unit_vec * foot_vels).sum(-1)

    def out_of_footsteps(self):
        """We return timeout once the current footstep is the last
        footstep in the sequence, not once the last footstep is hit.
        This avoids indexing errors and prevents needing to generate
        a bogus observation on the last step."""
        # assert (self.current_footstep <= self.cfg['n_cycles'] * 4).all()
        return self.current_footstep == (self.cfg['n_cycles'] + 1) * 2 - 2

    def no_footstep_in(self, no_footstep_timeout):
        return (self.counter - self.last_time_hit_footstep) >= no_footstep_timeout

    def reset(self, env_ids):
        """Randomly generate a new set of footsteps and set the
        'current_footstep' back to zero. """

        self.counter[env_ids] = 0
        self.last_time_hit_footstep[env_ids] = 0
        self.current_footstep[env_ids] = 1
        self.footsteps[env_ids] = self.generate_footsteps(env_ids)
        if self.rand_every_timestep:
            self.actual_footsteps[env_ids] = self.footsteps[env_ids].clone()
        if env_ids[0] == 0 and self.task.args.plot_all_targets:
            """ Only draw lines for the 0th env, if it is pass in"""
            self.plot_all_targets()


    def get_footstep_target_distance(self):
        """obs is len 16 = 4 feet * 2 steps * 2 dims"""
        yaw = self.task.base_euler[:, 2]
        output = torch.zeros(self.num_envs, 16, device=self.device)
        curr_dists = self.get_footstep_distance(self.current_footstep)
        next_dists = self.get_footstep_distance(self.current_footstep + 1)
        dists = torch.cat([curr_dists, next_dists], 1)
        rot_mat = batch_z_2D_rot_mat(-yaw).view(self.num_envs, 1, 2, 2)
        dists = (rot_mat @ dists.unsqueeze(-1)).squeeze(-1)
        return dists.view(self.num_envs, 16)

    def get_footstep_target_distance_2_ahead_alt(self):
        """This is the observation that gets returned to the agent."""
        yaw = self.task.base_euler[:, 2]
        rot_mat = batch_z_2D_rot_mat(-yaw).view(self.num_envs, 1, 2, 2)
        output = torch.zeros(self.num_envs, 16, device=self.device)
        for i in range(2):
            dists = self.get_footstep_pair_distance(self.current_footstep - i)
            dists = (rot_mat @ dists.unsqueeze(-1)).squeeze(-1)
            foot_idcs = self.get_footstep_idcs(self.current_footstep - i)
            output_idcs = (foot_idcs.unsqueeze(-1) * 2
                           + torch.arange(2, device=self.device).view(1, 1, 2))
            output_idcs = output_idcs.view(self.num_envs, 4)
            output[self.env_arange.unsqueeze(-1), output_idcs] = \
                dists.view(self.num_envs, 4)
        i = -1
        dists = self.get_footstep_pair_distance(self.current_footstep - i)
        dists = (rot_mat @ dists.unsqueeze(-1)).squeeze(-1)
        foot_idcs = self.get_footstep_idcs(self.current_footstep - i)
        output_idcs = (foot_idcs.unsqueeze(-1) * 2 + torch.arange(2, device=self.device).view(1, 1, 2))
        output_idcs = output_idcs.view(self.num_envs, 4) + 8
        output[self.env_arange.unsqueeze(-1), output_idcs] = \
            dists.view(self.num_envs, 4)
        return output

    def get_current_foot_one_hot(self):
        foot_idcs = self.get_footstep_idcs(self.current_footstep)
        output = torch.zeros(self.num_envs, 4, device=self.device)
        output[self.env_arange.unsqueeze(-1), foot_idcs] = 1.0
        return output



def get_circle_lines(centers, radius=0.02, rand_colors=False, foot_colors=False, stack_height=1):
    stack_delta = 0.01
    num_lines = int(radius / 0.01) * 5  # this is per circle
    foot_rgb = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
                            [0.0, 0.0, 0.5], [0.5, 0.5, 0.0]])
    height = 0.01
    num_circles = centers.shape[0]

    # axes mean (centers, num_lines, start/end point, xyz position)
    lines = torch.zeros(stack_height, num_circles, num_lines, 2, 3, device=centers.device)
    lines[:, :, :, :, 2] = centers[:, 2].view(1, num_circles, 1, 1) + height + torch.arange(stack_height, device=centers.device).reshape(stack_height, 1, 1, 1) * stack_delta  # set height of all lines
    x_center_offset = radius * 2 * ((torch.arange(num_lines, device=centers.device) + 1) / (num_lines + 1)) - radius
    lines[:, :, :, :, 0] = centers[:, 0].view(1, num_circles, 1, 1) + x_center_offset.view(1, 1, num_lines, 1)
    temp = (radius**2 - x_center_offset**2)**0.5
    lines[:, :, :, 0, 1] = centers[:, 1].view(1, num_circles, 1) + temp.view(1, 1, num_lines)
    lines[:, :, :, 1, 1] = centers[:, 1].view(1, num_circles, 1) - temp.view(1, 1, num_lines)

    colors = torch.zeros(stack_height, num_circles, num_lines, 3, device=centers.device)
    if rand_colors:
        colors[:] = torch.rand(1, num_circles, 1, 3)
    elif foot_colors:
        for i in range(4):
            colors[:, i::4] = foot_rgb[i]
    else:
        colors[:] = torch.tensor([1.0, 0.0, 0.0])
    colors = colors.reshape(stack_height * num_circles, num_lines, 3)
    lines = lines.reshape(stack_height * num_circles * num_lines * 2, 3)
    return lines, colors


def main():
    from env import AliengoEnv
    import yaml
    import time
    import os

    path = os.path.join(os.path.dirname(__file__),
                        '../config/default_footstep.yaml')
    with open(path) as f:
        params = yaml.full_load(f)
    params = params['env_params']
    params['render'] = True
    params['fixed'] = False
    params['vis'] = True
    env = AliengoEnv(**params)
    env.reset()
    i = 0
    print()
    while True:
        i += 1
        time.sleep(1/240.0)
        env.client.stepSimulation()
        env.quadruped.footstep_generator.footstep_reached(
            params['reward_parts']['footstep_reached'][1])
        if i % 10 == 0:
            vel = env.quadruped.footstep_generator.velocity_towards_footstep()
            print("Velocity Towards current footstep: {:0.2f}".format(vel))


if __name__ == "__main__":
    main()
