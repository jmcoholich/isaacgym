from rlgpu.tasks.aliengo_utils.utils import batch_z_rot_mat
import torch
from isaacgym import gymapi as g


class WalkFootstepGenerator:
    """This is a class for generating footstep targets
    and keeping track of current footstep.
    """
    def __init__(self, task):
        raise NotImplementedError
        self.device = task.device
        self.starting_foot_pos = torch.tensor(
            [[-0.2694, 0.1495, 0.0285],
             [0.2134, 0.1494, 0.0285],
             [-0.2693, -0.1492, 0.0285],
             [0.2135, -0.1493, 0.0285]],
            device=self.device)

        self.task = task
        self.gym = task.gym
        self.viewer = task.viewer
        self.envs = task.envs
        self.cfg = task.cfg['footstep_target_parameters']
        self.num_envs = task.num_envs
        self.current_footstep = torch.zeros(self.num_envs, dtype=torch.long,
                                            device=self.device)
        self.footsteps = None
        self.footstep_idcs = None
        self.vis = not self.task.headless
        # self.n_foosteps_hit = None
        if self.vis:
            self.curr_step_body = None
            self.step_body_ids = []
        self.footsteps = torch.zeros(self.num_envs, 4 * self.cfg['n_cycles'] + 4, 3, device=self.device)
        self.footstep_idcs = torch.zeros(self.num_envs, 4, dtype=torch.long,
                                         device=self.device)
        self.generate_footsteps(torch.arange(self.num_envs, device=self.device))

        self.counter = torch.zeros(self.num_envs, device=self.device)

    def get_current_foot_one_hot(self):
        foot = self.footstep_idcs[
            torch.arange(self.num_envs, device=self.device),
            self.current_footstep % 4]
        output = torch.zeros(self.num_envs, 4, device=self.device)
        output[torch.arange(self.num_envs, device=self.device), foot] = 1.0
        return output

    def generate_footsteps(self, env_idcs):
        """Stochastically generate footsteps in a straight line
        for a walking gait for the environments with the given indices.
        Foot idx order: FL, FR, RL, RR
        """
        if self.cfg['gait'] != 'walk':
            raise NotImplementedError
        num_to_gen = len(env_idcs)
        footsteps = torch.zeros(num_to_gen, 4, 3, device=self.device)
        # each footstep is an x, y, z position
        step_len = (self.cfg['step_length']
                    + ((torch.rand(num_to_gen, device=self.device) - 0.5)
                       * self.cfg['step_length_rand']))
        width = (self.cfg['step_width']
                 + ((torch.rand(num_to_gen, device=self.device) - 0.5)
                    * self.cfg['step_width_rand']))
        length = self.cfg['base_length']
        len_offset = self.cfg['length_offset']

        # randomly chose right or left side of robot to start walking
        samples = torch.rand(num_to_gen, device=self.device)
        footstep_idcs = torch.zeros(num_to_gen, 4, dtype=torch.long,
                                    device=self.device)
        '''
        idcs = samples > 1.0
        footstep_idcs[idcs] = torch.tensor(
            [3, 1, 2, 0], dtype=torch.long, device=self.device)
        # if np.random.random_sample() > 0.5:
        # RR
        footsteps[idcs, 0, 0] = -length/2.0 + len_offset + step_len[idcs]
        footsteps[idcs, 0, 1] = -width[idcs]/2.0

        # FR
        footsteps[idcs, 1, 0] = length/2.0 + len_offset + step_len[idcs]
        footsteps[idcs, 1, 1] = -width[idcs]/2.0

        # RL
        footsteps[idcs, 2, 0] = -length/2.0 + len_offset + 2 * step_len[idcs]
        footsteps[idcs, 2, 1] = width[idcs]/2.0

        # FL
        footsteps[idcs, 3, 0] = length/2.0 + len_offset + 2*step_len[idcs]
        footsteps[idcs, 3, 1] = width[idcs]/2.0
        '''

        idcs = samples <= 1.0

        footstep_idcs[idcs] = torch.tensor(
            [2, 0, 3, 1], dtype=torch.long, device=self.device)
        # RL
        footsteps[idcs, 0, 0] = -length/2.0 + len_offset + step_len[idcs]
        footsteps[idcs, 0, 1] = width[idcs]/2.0

        # FL
        footsteps[idcs, 1, 0] = length/2.0 + len_offset + step_len[idcs]
        footsteps[idcs, 1, 1] = width[idcs]/2.0

        # RR
        footsteps[idcs, 2, 0] = -length/2.0 + len_offset + 2*step_len[idcs]
        footsteps[idcs, 2, 1] = -width[idcs]/2.0

        # FR
        footsteps[idcs, 3, 0] = length/2.0 + len_offset + 2*step_len[idcs]
        footsteps[idcs, 3, 1] = -width[idcs]/2.0

        footsteps[:, :, 2] = 0.0
        footsteps = footsteps.tile(1, self.cfg['n_cycles'], 1)
        footsteps[:, :, 0] += (torch.arange(self.cfg['n_cycles'], device=self.device).repeat_interleave(4).unsqueeze(0)
                               * step_len.unsqueeze(-1) * 2)
        footsteps[:, :, :-1] += (torch.rand_like(footsteps[:, :, :-1])
                                 - 0.5) * self.cfg['footstep_rand']

        self.footstep_idcs[env_idcs] = footstep_idcs
        starting_steps = self.starting_foot_pos.tile(num_to_gen, 1, 1)
        starting_steps[..., 2] = 0.0
        self.footsteps[env_idcs] = torch.cat([starting_steps, footsteps],
                                             dim=1)

        if self.vis and env_idcs[0] == 0:
            """ Only draw lines for the 0th env, if it is pass in"""
            self.gym.clear_lines(self.viewer)
            # self.gym.viewer_camera_look_at(
            #     self.viewer, self.envs[0], g.Vec3(3.0, 0.0, 1.0),
            #     g.Vec3(0.0, 0, 0.0))
            vertices = self.footsteps[0, :, :].clone().repeat_interleave(2, dim=0).cpu()
            vertices[1::2, 2] = torch.tensor([0.1, 0.2, 0.3, 0.4] * (self.cfg['n_cycles'] + 1))
            colors = torch.rand(self.cfg['n_cycles'] + 1, 3).repeat_interleave(4, dim=0)
            self.gym.add_lines(self.viewer, self.envs[0],
                               self.cfg['n_cycles'] * 4 + 4, vertices.numpy(),
                               colors.numpy())

    # def compute_vel_towards_footstep_rew(self, coef):
    #     return coef * self.velocity_towards_footstep()

    def other_feet_lifted(self):
        """Return 1.0 for each non-current foot that is not in contact with
        its last footstep target. Only consider the presence of contact
        and the distance threshold, any contact force is fine. Ignore feet
        that have no previous footstep target. """
        output = torch.zeros(self.num_envs, device=self.device)
        current_footstep = self.current_footstep.clone()
        for _ in range(2):
            current_footstep -= 1
            mask = current_footstep >= 0  # Only count the env_idcs that are True
            assert mask.all()
            current_footstep = current_footstep.clamp(0)  # do this to avoid throwing an indexing error
            foot = self.footstep_idcs[
                torch.arange(self.num_envs, device=self.device),
                current_footstep % 4]
            in_contact = (self.task.foot_contact_forces[torch.arange(self.num_envs, device=self.device), foot] > 0.0).any(-1)

            target_location = self.footsteps[torch.arange(self.num_envs, device=self.device), current_footstep]
            target_location[:, 2] += self.task.foot_collision_sphere_r
            foot_center_pos = self.task.foot_center_pos[torch.arange(self.num_envs, device=self.device), foot]
            dist = (target_location - foot_center_pos).norm(dim=1)
            in_place = torch.logical_and(in_contact, dist <= self.cfg['footstep_distance_threshold'])
            env_idcs = torch.logical_and(~in_place, mask)
            output[env_idcs] += 1.0
        return output

    def other_feet_lifted_smooth(self):
        """Same as the above, but instead of a hard cutoff if contact is
        in place, we vary smoothly based on contact force and height above
        ground. Now the reward varies between plus or minus one."""
        height_clamp = self.cfg['footstep_distance_threshold']
        total_output = torch.zeros(self.num_envs, device=self.device)
        current_footstep = self.current_footstep.clone()
        for _ in range(2):
            output = torch.zeros(self.num_envs, device=self.device)
            current_footstep -= 1
            valid_envs = current_footstep >= 0  # Only count the env_idcs that are True
            assert valid_envs.all()
            current_footstep = current_footstep.clamp(0)  # do this to avoid throwing an indexing error
            foot = self.footstep_idcs[
                torch.arange(self.num_envs, device=self.device),
                current_footstep % 4]

            contact_rew = self.task.foot_contact_forces[
                torch.arange(self.num_envs, device=self.device), foot]  # shape (n, 3)
            contact_rew = contact_rew.max(dim=1)[0]  # shape (n,)
            output -= (contact_rew.clamp(0.0, self.cfg['contact_force_treshold'])
                       / (self.cfg['contact_force_treshold'] + 1e-4))

            target_location = self.footsteps[torch.arange(self.num_envs, device=self.device), current_footstep]
            target_location[:, 2] += self.task.foot_collision_sphere_r
            foot_center_pos = self.task.foot_center_pos[torch.arange(self.num_envs, device=self.device), foot]

            foot_height_penalty = (foot_center_pos[:, 2] - self.task.foot_collision_sphere_r).clamp(0.0, height_clamp) / height_clamp
            output += foot_height_penalty

            dist = (target_location - foot_center_pos).norm(dim=1)

            # if I am outside the dist threshold, max penalty since I moved
            output[dist > self.cfg['footstep_distance_threshold']] = 1.0

            # if I am on a footstep that has previous footsteps to check, just return zero
            output[~valid_envs] = 0.0
            total_output += output
        return total_output

    def hit_config(self):
        """Return 1.0 if all feet are in the correct place. Zero otherwise"""

        current_footstep = self.current_footstep.clone()
        output = torch.zeros(self.num_envs, device=self.device)
        for _ in range(4):
            foot = self.footstep_idcs[torch.arange(self.num_envs, device=self.device), current_footstep % 4]
            dist = self.get_current_footstep_distance(current_footstep=current_footstep).norm(dim=1)
            in_contact = (self.task.foot_contact_forces[torch.arange(self.num_envs, device=self.device), foot] > self.cfg['contact_force_treshold']).any(-1)
            this_foot_good = torch.logical_and(in_contact, dist <= self.cfg['footstep_distance_threshold'])
            output += this_foot_good * 1.0

            current_footstep -= 1
        output[output == 4.0] += 4.0
        return output

    def update(self):
        """Called by update_tensors() in aliengo.py. Check if footstep has
        been hit, so observation will return distance to next footstep."""

        # dist = self.get_current_footstep_distance().norm(dim=1)
        # first check if current foot is in contact
        # foot = self.footstep_idcs[torch.arange(self.num_envs, device=self.device), self.current_footstep % 4]
        # in_contact = (self.task.foot_contact_forces[torch.arange(self.num_envs, device=self.device), foot] > self.cfg['contact_force_treshold']).any(-1)
        # env_idcs = torch.logical_and(in_contact, dist <= self.cfg['footstep_distance_threshold'])

        # need to calculate reward before current_footstep is incremented
        self.reached = self.hit_config()
        # self.reached[env_idcs] = 1.0
        # # the reward can be up to 1.5x by hitting center of target
        # self.reached[env_idcs] += 0.5 * (1 - dist[env_idcs] / self.cfg['footstep_distance_threshold'])

        env_idcs = torch.logical_and(
            self.reached == 8.0,
            self.counter % self.cfg['update_period'] == 0)
        self.current_footstep[env_idcs] += 1

        self.counter += 1

    def get_current_footstep_distance(self, current_footstep=None):
        """Returns xyz distance of current quadruped
        foot location to next footstep location.
        This is a vector in the global coordinate frame. For observations,
        it should be rotated to align with the robot yaw.
        """
        if current_footstep is None:
            current_footstep = self.current_footstep
        foot = self.footstep_idcs[torch.arange(self.num_envs, device=self.device), current_footstep % 4]
        target_location = self.footsteps[torch.arange(self.num_envs, device=self.device), current_footstep]
        target_location[:, 2] += self.task.foot_collision_sphere_r
        foot_center_pos = self.task.foot_center_pos[torch.arange(self.num_envs, device=self.device), foot]
        return target_location - foot_center_pos

    def velocity_towards_footstep(self, current_footstep=None):
        """Return velocity of current foot towards footstep.
        This method is called by the reward class.
        """
        if current_footstep is None:
            current_footstep = self.current_footstep
        foot = self.footstep_idcs[torch.arange(self.num_envs, device=self.device), current_footstep % 4]
        # velocity vector
        vel = self.task.foot_vel[torch.arange(self.num_envs, device=self.device), foot]
        # calculate unit vector in direction of footstep target
        pos = self.get_current_footstep_distance()
        pos_unit = pos/pos.norm(dim=1, keepdim=True)
        return (pos_unit * vel).sum(dim=1)  # dot product

    def velocity_towards_footsteps(self):
        """Return velocity of current foot towards footstep.
        This method is called by the reward class.
        """
        output = torch.zeros(self.num_envs, device=self.device)
        current_footstep = self.current_footstep.clone()
        for i in range(4):
            output += self.velocity_towards_footstep(
                current_footstep=current_footstep)
            current_footstep -= 1

        return output


    def footstep_reached(self):
        """Return 1 if the footstep has been reached, else 0.
        This method is called by the reward computation.
        """
        reward = self.reached.clone()
        self.reached = torch.zeros_like(self.reached)  # reset for next step
        return reward

    def is_timeout(self):
        """We return timeout once the current footstep is the last
        footstep in the sequence, not once the last footstep is hit.
        This avoids indexing errors and prevents needing to generate
        a bogus observation on the last step."""
        # assert (self.current_footstep <= self.cfg['n_cycles'] * 4).all()
        return self.current_footstep == ((self.cfg['n_cycles'] + 1) * 4 - 1)

    def reset(self, env_ids):
        """Randomly generate a new set of footsteps and set the
        'current_footstep' back to zero. """
        # if self.vis:
        #     for i in range(len(self.step_body_ids)):
        #         self.client.removeBody(self.step_body_ids[i])
        #         # self.client.removeBody(self.text_ids[i])
        #     if self.curr_step_body is not None:
        #         self.client.removeBody(self.curr_step_body)
        #         self.client.removeAllUserDebugItems()
        # self.n_foosteps_hit = self.current_footstep
        self.counter[env_ids] = 0
        self.current_footstep[env_ids] = 3
        self.generate_footsteps(env_ids)
        # self.generate_footsteps(self.params)

    def get_footstep_target_distance(self):

        output = torch.zeros(self.num_envs, 12, device=self.device)
        yaw = self.task.base_euler[:, 2]
        rot_mat = batch_z_rot_mat(-yaw)
        current_footstep = self.current_footstep.clone()
        for _ in range(3):
            mask = current_footstep < 0  # if this mask is true, don't set the output to anything
            assert (~mask).all()
            # current_footstep = current_footstep.clamp(0)  # do this to avoid throwing an indexing error
            positions = self.get_current_footstep_distance(current_footstep)

            positions = (rot_mat @ positions.unsqueeze(-1)).squeeze(-1)
            foot = self.footstep_idcs[torch.arange(self.num_envs, device=self.device), current_footstep % 4]
            idcs = torch.arange(3, device=self.device).unsqueeze(0) + foot.unsqueeze(-1) * 3
            output[torch.arange(self.num_envs, device=self.device).unsqueeze(-1), idcs] = positions
            current_footstep -= 1
        return output

        # output = torch.zeros(self.num_envs, 12, device=self.device)
        # positions = self.get_current_footstep_distance()
        # # rotate distances by negative yaw
        # yaw = self.task.base_euler[:, 2]
        # rot_mat = batch_z_rot_mat(-yaw)
        # positions = (rot_mat @ positions.unsqueeze(-1)).squeeze(-1)
        # foot = self.footstep_idcs[torch.arange(self.num_envs, device=self.device), self.current_footstep % 4]
        # idcs = torch.arange(3, device=self.device).unsqueeze(0) + foot.unsqueeze(-1) * 3
        # output[torch.arange(self.num_envs, device=self.device).unsqueeze(-1), idcs] = positions
        # return output


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
