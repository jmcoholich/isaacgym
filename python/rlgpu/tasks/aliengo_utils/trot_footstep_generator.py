from rlgpu.tasks.aliengo_utils.utils import batch_z_2D_rot_mat
import torch
from isaacgym import gymapi as g


class TrotFootstepGenerator:
    """This is a class for generating footstep targets
    and keeping track of current footstep.
    """

    def __init__(self, task):
        self.device = task.device

        # these are in walking order (legacy from walk_footstep_generator.py)
        self.starting_foot_pos = torch.tensor(
            [[-0.2694, 0.1495, 0.0285],  # RL
             [0.2134, 0.1494, 0.0285],  # FL
             [-0.2693, -0.1492, 0.0285],  # RR
             [0.2135, -0.1493, 0.0285]],  # FR
            device=self.device)

        # these are in foot idx order
        self.starting_foot_pos_in_order = torch.tensor(
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

    def generate_footsteps(self, env_ids):
        """Stochastically generate footsteps for a trotting gait.
        FR and RL will be first up for hitting targets.
        The self.footsteps tensor is 4 dimensional where the dimensions are
        (envs, footstep pair idx, footstep idx within pair, xy of target)
        The footstep pairs alternate between (FL, RR) and (FR, RL). The
        front foot is index 0 within the pair, the rear foot is idx 1.
        """
        num_to_gen = len(env_ids)
        # dimensions are (envs, first two target pairs, two feet, xy)
        footsteps = torch.zeros(num_to_gen, 2, 2, 2, device=self.device)
        # each footstep is an x, y, z position

        # ADD INFO FOR CURRICULUM!!
        current_epoch = 0
        if hasattr(self.task, "epochs"):
            current_epoch = self.task.epochs
        if self.task.args.max_iterations > 0:
            max_epochs = self.task.args.max_iterations
        else:
            max_epochs = self.task.cfg['max_epochs']
        curr = min(2.0 * current_epoch / max_epochs, 1.0)
        step_len = (self.cfg['step_length'] * curr
                    + ((torch.rand(num_to_gen, device=self.device) - 0.5)
                       * self.cfg['step_length_rand'] * curr))


        # step_len = (self.cfg['step_length']
        #             + ((torch.rand(num_to_gen, device=self.device) - 0.5)
        #                * self.cfg['step_length_rand']))


        # width = (self.cfg['step_width']
        #          + ((torch.rand(num_to_gen, device=self.device) - 0.5)
        #             * self.cfg['step_width_rand']))
        # length = self.cfg['base_length']
        # len_offset = self.cfg['length_offset']

        starting_foot_pos = self.starting_foot_pos[..., :2].clone()

        footsteps[:, 0, 0] = starting_foot_pos[1]  # should be FL
        footsteps[:, 0, 1] = starting_foot_pos[2]  # should be RR

        footsteps[:, 1, 0] = starting_foot_pos[3]  # should be FR
        footsteps[:, 1, 1] = starting_foot_pos[0]  # should be RL

        footsteps = footsteps.tile(1, self.cfg['n_cycles'] + 1, 1, 1)

        headings = ((torch.rand(num_to_gen, device=self.device) - 0.5)
                    * (self.cfg['radial_range'][1] - self.cfg['radial_range'][0])
                    + (self.cfg['radial_range'][1] + self.cfg['radial_range'][0])
                    / 2.0) * 3.1415926 / 180.0
        x_addition = torch.arange(
            (self.cfg['n_cycles'] + 1) * 2 - 1,
            device=self.device).unsqueeze(-1) * step_len.view(num_to_gen, 1, 1)
        y_addition = torch.arange(
            (self.cfg['n_cycles'] + 1) * 2 - 1,
            device=self.device).unsqueeze(-1) * step_len.view(num_to_gen, 1, 1)
        footsteps[:, 1:, :, 0] += x_addition * torch.cos(headings).unsqueeze(-1).unsqueeze(-1)
        footsteps[:, 1:, :, 1] += y_addition * torch.sin(headings).unsqueeze(-1).unsqueeze(-1)

        # linearly increase randomization for the first n footsteps
        full_rand_by_footstep_num = 10
        schedule = torch.linspace(0, 1, full_rand_by_footstep_num,
            device=self.device).view(1, full_rand_by_footstep_num, 1, 1)
        noise = (torch.rand_like(footsteps) - 0.5) * self.cfg['footstep_rand']
        noise[:, :full_rand_by_footstep_num] *= schedule
        footsteps[:] += noise * curr
        return footsteps

    def plot_footstep_targets(self, current_only=False):
        hit_so_far = True
        if self.task.is_stepping_stones:
            add_height = self.task.cfg['stepping_stones']['height']
        elif self.task.is_rough_terrain_blocks:
            add_height = self.task.cfg['rough_terrain_blocks']['height']
        else:
            add_height = 0

        if current_only:
            num_cycles = 1
        else:
            num_cycles = self.cfg['n_cycles'] + 1
        self.gym.clear_lines(self.viewer)
        if hit_so_far:
            num = self.current_footstep[0] + 1
            centers = self.footsteps[0, 0:num]
            centers = torch.cat((centers, torch.zeros(num, 2, 1, device=self.device)), -1)
            centers[:, :, 2] += add_height
            vertices, colors = get_circle_lines(centers.reshape(num * 2, 3), foot_colors=True)
        elif current_only:
            centers = self.footsteps[0, self.current_footstep[0] - 1:self.current_footstep[0] + 1]
            centers = torch.cat((centers, torch.zeros(2, 2, 1, device=self.device)), -1)
            centers[:, :, 2] += add_height
            vertices, colors = get_circle_lines(centers.reshape(4, 3))
        else:
            centers = self.footsteps[0]
            centers = torch.cat((centers, torch.zeros(self.footsteps[0].shape[0], 2, 1, device=self.device)), -1)
            centers[:, :, 2] += add_height
            vertices, _ = get_circle_lines(centers.reshape(self.footsteps[0].shape[0] * 2, 3), radius=0.02, rand_colors=True)
            colors = torch.rand(self.footsteps[0].shape[0], 3,
                                device=self.device)
            colors = colors.repeat_interleave(vertices.shape[0] // colors.shape[0], 0)

        self.gym.add_lines(self.viewer, self.envs[0],
                        vertices.shape[0] // 2, vertices.cpu().numpy(),
                        colors.cpu().numpy())
            # colors = torch.tensor([[1.0, 0.0, 0.0]] * num_cycles).repeat_interleave(4, dim=0)
            # vertices = self.footsteps[0, self.current_footstep[0] - 1:self.current_footstep[0] + 1]
            # vertices = vertices.view(num_cycles * 4, 3).clone().repeat_interleave(2, dim=0).cpu()
        # else:
            # colors = torch.rand(num_cycles, 3).repeat_interleave(4, dim=0)
            # base_footsteps = torch.cat((self.footsteps[0].clone(), torch.zeros(num_cycles * 2, 2, 1, device=self.device)), -1)
            # vertices = base_footsteps.view(num_cycles * 4, 3).clone().repeat_interleave(2, dim=0).cpu()
        # vertices[1::2, 2] += torch.tensor([0.1, 0.1, 0.2, 0.2] * num_cycles) * 0.25 + add_height
        # self.gym.add_lines(self.viewer, self.envs[0],
        #                    num_cycles * 4, vertices.numpy(),
        #                    colors.numpy())

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

    def get_footstep_pair_distance(self, current_footstep=None):
        """Return 2D XY vector from current feet to current targets."""
        if current_footstep is None:
            current_footstep = self.current_footstep

        target_location = self.footsteps[self.env_arange, current_footstep]
        # target_location[:, :, 2] += self.task.foot_collision_sphere_r

        foot_idcs = self.get_footstep_idcs(current_footstep)
        foot_center_pos = self.task.foot_center_pos[..., :2][
            self.env_arange.unsqueeze(-1), foot_idcs]
        return target_location - foot_center_pos

    def get_footstep_pair_contact_force(self, current_footstep=None):
        if current_footstep is None:
            current_footstep = self.current_footstep

        foot_idcs = self.get_footstep_idcs(current_footstep)
        contact_forces = self.task.foot_contact_forces[
            self.env_arange.unsqueeze(-1), foot_idcs]
        return contact_forces

    def hit_trot_footstep_pair(self, current_footstep=None):
        if current_footstep is None:
            current_footstep = self.current_footstep
        dist = self.get_footstep_pair_distance(
            current_footstep=current_footstep).norm(dim=2)
        in_range = (dist < self.cfg['footstep_distance_threshold'])
        in_contact = (self.get_footstep_pair_contact_force(current_footstep=current_footstep)[..., 2]
                      > self.cfg["contact_force_treshold"])
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
        hit_both = reached.all(-1)
        # add extra reward for hitting center of footstep
        reached_float = reached.float()
        if reached.any():
            reached_float[reached] += self.center_rew_multiplier \
                * (1 - dist[reached] / self.cfg['footstep_distance_threshold'])
        return reached_float.sum(dim=1), hit_both

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
        rew_dict["hit_footstep"], hit_both = self.hit_trot_footstep_pair()
        rew_dict["hit_footstep"][hit_both] *= self.cfg["hit_both_multiplier"]

        # foot stay
        rew_dict["foot_stay"], _ = self.hit_trot_footstep_pair(
            current_footstep=(self.current_footstep - 1))

        # foot velocity
        rew_dict["foot_velocity"] = self.velocity_towards_footsteps()
        return rew_dict, hit_both

    def update(self):
        """Calculate rewards and store them for later, since the observation
        needs to reflect the next footstep target if the current target is
        hit. However, rewards need to reflect the last action and observation.
        """

        self.rew_dict, hit_targets = self.compute_rewards()
        self.current_footstep[hit_targets] += 1
        self.last_time_hit_footstep[hit_targets] = \
            self.counter[hit_targets].clone()
        self.counter += 1

        if self.rand_every_timestep:
            self.rand_next_footstep()
            if self.vis:
                self.plot_footstep_targets(current_only=True)

    def velocity_towards_footsteps(self):
        """Return velocity of current foot towards footsteps. This is just
        xy velocity, since I'm not keeping track of target z anymore.
        """
        # get global feet velocity
        foot_idcs = self.get_footstep_idcs(self.current_footstep)
        foot_vels = self.task.foot_vel[..., :2][
            self.env_arange.unsqueeze(-1), foot_idcs]

        # compute unit vectors from foot to targets
        target_pos = self.footsteps[self.env_arange, self.current_footstep]
        foot_pos = self.task.foot_center_pos[..., :2][
            self.env_arange.unsqueeze(-1), foot_idcs]
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
        return self.current_footstep == (self.cfg['n_cycles'] + 1) * 2 - 1

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
        if self.vis and env_ids[0] == 0:
            """ Only draw lines for the 0th env, if it is pass in"""
            self.plot_footstep_targets()

    def get_footstep_target_distance(self):
        """This is the observation that gets returned to the agent."""
        yaw = self.task.base_euler[:, 2]
        rot_mat = batch_z_2D_rot_mat(-yaw).view(self.num_envs, 1, 2, 2)
        output = torch.zeros(self.num_envs, 8, device=self.device)
        for i in range(2):
            dists = self.get_footstep_pair_distance(self.current_footstep - i)
            dists = (rot_mat @ dists.unsqueeze(-1)).squeeze(-1)
            foot_idcs = self.get_footstep_idcs(self.current_footstep - i)
            output_idcs = (foot_idcs.unsqueeze(-1) * 2
                           + torch.arange(2, device=self.device).view(1, 1, 2))
            output_idcs = output_idcs.view(self.num_envs, 4)
            output[self.env_arange.unsqueeze(-1), output_idcs] = \
                dists.view(self.num_envs, 4)
        return output

    def get_current_foot_one_hot(self):
        foot_idcs = self.get_footstep_idcs(self.current_footstep)
        output = torch.zeros(self.num_envs, 4, device=self.device)
        output[self.env_arange.unsqueeze(-1), foot_idcs] = 1.0
        return output



def get_circle_lines(centers, radius=0.02, rand_colors=False, foot_colors=False):
    foot_rgb = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
                             [0.0, 0.0, 0.5], [0.5, 0.5, 0.0]])
    height = 0.01
    num_lines = int(radius / 0.01) * 5
    num_circles = centers.shape[0]

    # axes mean (centers, num_lines, start/end point, xyz position)
    lines = torch.zeros(num_circles, num_lines, 2, 3, device=centers.device)
    lines[:, :, :, 2] = centers[:, 2].view(num_circles, 1, 1) + height  # set height of all lines
    x_center_offset = radius * 2 * ((torch.arange(num_lines, device=centers.device) + 1) / (num_lines + 1)) - radius
    lines[:, :, :, 0] = centers[:, 0].view(num_circles, 1, 1) + x_center_offset.view(1, num_lines, 1)
    temp = (radius**2 - x_center_offset**2)**0.5
    lines[:, :, 0, 1] = centers[:, 1].view(num_circles, 1) + temp.view(1, num_lines)
    lines[:, :, 1, 1] = centers[:, 1].view(num_circles, 1) - temp.view(1, num_lines)

    colors = torch.zeros(num_circles, num_lines, 3)
    if rand_colors:
        colors[:] = torch.rand(num_circles, 1, 3)
    elif foot_colors:
        for i in range(4):
            colors[i::4] = foot_rgb[i]
    else:
        colors[:] = torch.tensor([1.0, 0.0, 0.0])
    lines = lines.reshape(num_circles * num_lines * 2, 3)
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
