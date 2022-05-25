import torch
from rlgpu.tasks.aliengo_utils.utils import batch_z_rot_mat

class SSTrotFootstepGenerator:
    def __init__(self, task):
        self.task = task
        self.cfg = task.cfg["ss_footstep_target_parameters"]
        assert task.cfg['stepping_stones']['density'] == 1.0, \
            "This is currently only setup to work with full density for easy"\
            "indexing purposes."
        self.gym = task.gym
        self.num_envs = task.num_envs
        self.device = self.task.device
        self.ss = self.task.stepping_stones
        self.stone_pos = self.task.stepping_stones.stone_pos
        self.n_cycles = self.cfg["n_cycles"]
        self.y_rand = self.cfg.get("y_rand", 3)
        assert self.y_rand >= 0 and self.y_rand != 2
        self.env_arange = torch.arange(self.num_envs, device=self.device)
        raise NotImplementedError("env_y_offsets is obsolete. Use task.env_offsets")
        self.env_y_offsets = \
            ((self.env_arange - 1).view(self.num_envs, 1).clamp(0).float()
             * self.task.cfg["stepping_stones"]["robot_spacing"])

        self.start_foot_pos = torch.tensor(
            [[0.2134, 0.1494, 0.0285],  # FL
            [0.2135, -0.1493, 0.0285],  # FR
            [-0.2694, 0.1495, 0.0285],  # RL
            [-0.2693, -0.1492, 0.0285]],  # RR
            device=self.device)

        self.start_foot_pos[[0, 2], 1] -= 0.02
        self.start_foot_pos[[1, 3], 1] += 0.02
        self.vis = not self.task.headless  # TODO make sure its not just this preventing me from rendering to vid with headless

        # targets are stepping stone indices
        self.targets = torch.zeros(
            self.num_envs, self.n_cycles * 2, 2, device=self.device,
            dtype=torch.long)

        self.current_footstep = torch.ones(self.num_envs, device=self.device,
                                           dtype=torch.long)

        self.counter = torch.zeros(self.num_envs, device=self.device,
                                   dtype=torch.long)
        self.last_time_hit_footstep = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.reset(self.env_arange)

    def compute_rewards(self):
        """Computes all reward terms for the footstep generator, even if
        they are not specified in the yaml file. Return a reward dict. The
        policy will receive whichever rewards from there."""
        rew_dict = {}

        # hit_footstep
        rew_dict["hit_footstep"], hit_both, bad_foot_collisions_current = \
            self.hit_trot_footstep_pair()
        rew_dict["hit_footstep"][hit_both] *= 3.0

        # foot stay
        rew_dict["foot_stay"], _, bad_foot_collisions_last = \
            self.hit_trot_footstep_pair(current_footstep=(
                self.current_footstep - 1))

        # foot velocity
        rew_dict["foot_velocity"] = self.velocity_towards_footsteps()

        # wrong stepping stone collision penalty
        rew_dict["wrong_ss_collision_penalty"] = \
            bad_foot_collisions_current + bad_foot_collisions_last

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

        # if self.rand_every_timestep:
        #     self.rand_next_footstep()
        #     if self.vis:
        #         self.plot_footstep_targets(current_only=True)


    def hit_trot_footstep_pair(self, current_footstep=None):
        """Check if a pair of footstep targets is hit. Considers
        - contact force threshold
        - bounds of stepping stone
        - pmtg phase (if applicable)

        Return
        - reward output that is 1 if one ss is hit, 2 if both are hit
        - boolean if both are hit
        - penalty output that is + 1.0 for each foot hitting a stepping stone
        that is not the target
        """

        if current_footstep is None:
            current_footstep = self.current_footstep
        dist = self.get_footstep_pair_distance(current_footstep=current_footstep).abs().max(dim=2)[0]
        in_range = (dist < self.ss.cfg['stone_dim'] / 2.0)

        # only care about z contact force
        in_contact = (self.get_footstep_pair_contact_force(current_footstep=current_footstep)[:, :, 2]
                      > self.cfg["contact_force_threshold"])
        reached = torch.logical_and(in_range, in_contact)

        bad_foot_collisions = (self.get_footstep_pair_contact_force(
            current_footstep=current_footstep).abs() > 0.0).any(-1).float().\
            sum(-1)

        if self.task.cfg["actionSpace"] == "pmtg":
            # also ensure that the TG is in contact phase for the feet in question
            phase_offset = torch.tensor(self.task.cfg["pmtg"]["phase_offset"], device=self.device)
            z_traj_input = (torch.ones(self.num_envs, 4, device=self.device)
                            * self.task.t.unsqueeze(-1) + phase_offset + 0.5)
            norm_phases = (z_traj_input % 1.0) * 4 - 2.0
            in_contact_phase = norm_phases < 0.0
            foot_idcs = self.get_footstep_idcs(current_footstep)
            tg_in_contact_phase = in_contact_phase[self.env_arange.unsqueeze(-1), foot_idcs]

            reached = torch.logical_and(reached, tg_in_contact_phase)
        reached_float = reached.float()
        hit_both = torch.logical_and(reached[:, 0], reached[:, 1])
        # add extra reward for hitting center of footstep
        # if reached.any():
        #     reached_float[reached] += self.center_rew_multiplier \
        #         * (1 - dist[reached] / self.cfg['footstep_distance_threshold'])
        return reached_float.sum(dim=1), hit_both, bad_foot_collisions

    def gen_targets(self, env_ids):
        num_envs = env_ids.shape[0]
        targets = torch.zeros(num_envs, self.n_cycles * 2, 2, device=self.device,
                              dtype=torch.long)

        start_foot_pos_vec = self.start_foot_pos[:, :2].repeat((num_envs, 1, 1))

        # NOTE the clamp weirdness is required because, for some reason, IsaacGym
        # puts the first two envs in the same place
        start_foot_pos_vec[:, :, 1] += ((env_ids.view(num_envs, 1) - 1).clamp(0)
            * self.task.cfg["stepping_stones"]["robot_spacing"])

        # the first cycle (one cycle = two target pairs) is just the
        # nearest starting footsteps
        start_ss, _ = self.ss.pos2idcs(start_foot_pos_vec)
        targets[:, 0] = start_ss[:, [0, 3]]  # FL, RR
        targets[:, 1] = start_ss[:, [1, 2]]  # FR, RL

        # for now, each target will only advance fwd one stone
        trans = torch.zeros(num_envs, (self.n_cycles - 1) * 2, 2, 2,
                            device=self.device, dtype=torch.long)
        # the x translation will be increasing one stone fwd each step
        temp = torch.arange((self.n_cycles - 1) * 2, device=self.device) + 1
        trans[..., 0] = temp.view((self.n_cycles - 1) * 2, 1)
        if self.y_rand != 0:
            trans[..., 1] = (torch.randint(
                self.y_rand,
                (num_envs, (self.n_cycles - 1) * 2),
                device=self.device)
                - self.y_rand // 2).unsqueeze(-1)
        idcs = targets[:, :2, :].repeat((1, self.n_cycles - 1, 1))
        targets[:, 2:] = self.ss.idcs2idcs(idcs, trans)[0]

        return targets

    def plot_footstep_targets(self, plot_all=True):
        if plot_all:
            self.gym.clear_lines(self.task.viewer)
            colors = torch.rand(self.num_envs * self.n_cycles * 2, 3).repeat_interleave(2, dim=0)
            self.target_pos = self.stone_pos[self.targets]
            vertices = self.target_pos.view(self.num_envs * self.n_cycles * 4,
                3).clone().repeat_interleave(2, dim=0).cpu()
            vertices[1::2, 2] += torch.tensor([0.1, 0.1, 0.2, 0.2] * self.n_cycles * self.num_envs)
            self.gym.add_lines(self.task.viewer, self.task.envs[0],
                            self.num_envs * self.n_cycles * 4, vertices.numpy(),
                            colors.numpy())
        else:  # only plot the first environment's targets
            self.gym.clear_lines(self.task.viewer)
            colors = torch.rand(self.n_cycles, 3).repeat_interleave(4, dim=0)
            self.target_pos = self.stone_pos[self.targets[0]]
            vertices = self.target_pos.view(self.n_cycles * 4, 3).clone().repeat_interleave(2, dim=0).cpu()
            vertices[1::2, 2] += torch.tensor([0.1, 0.1, 0.2, 0.2] * self.n_cycles)
            self.gym.add_lines(self.task.viewer, self.task.envs[0],
                            self.n_cycles * 4, vertices.numpy(),
                            colors.numpy())

    def reset(self, env_ids):
        self.current_footstep[env_ids] = 1
        self.counter[env_ids] = 0
        self.last_time_hit_footstep[env_ids] = 0
        self.targets[env_ids] = self.gen_targets(env_ids)
        if self.vis:
            self.plot_footstep_targets()

    def get_current_foot_one_hot(self):
        foot_idcs = self.get_footstep_idcs(self.current_footstep)
        output = torch.zeros(self.num_envs, 4, device=self.device)
        output[self.env_arange.unsqueeze(-1), foot_idcs] = 1.0
        return output

    def is_timeout(self, no_footstep_timeout):
        """We return timeout once the current footstep is the last
        footstep in the sequence, not once the last footstep is hit.
        This avoids indexing errors and prevents needing to generate
        a bogus observation on the last step."""
        # assert (self.current_footstep <= self.cfg['n_cycles'] * 4).all()

        on_last_footstep = self.current_footstep == (self.n_cycles * 2 - 1)
        no_footstep_in = (self.counter - self.last_time_hit_footstep) >= no_footstep_timeout
        return torch.logical_or(on_last_footstep, no_footstep_in)

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

    def get_footstep_target_distance(self):
        """This is the observation that gets returned to the agent."""
        yaw = self.task.base_euler[:, 2]
        rot_mat = batch_z_rot_mat(-yaw).view(self.num_envs, 1, 3, 3)
        output = torch.zeros(self.num_envs, 12, device=self.device)
        for i in range(2):
            dists = self.get_footstep_pair_distance(self.current_footstep - i)
            dists = (rot_mat @ dists.unsqueeze(-1)).squeeze(-1)
            foot_idcs = self.get_footstep_idcs(self.current_footstep - i)
            output_idcs = (foot_idcs.unsqueeze(-1) * 3
                           + torch.arange(3, device=self.device).view(1, 1, 3))
            output_idcs = output_idcs.view(self.num_envs, 6)
            output[self.env_arange.unsqueeze(-1),
                   output_idcs] = dists.view(self.num_envs, 6)
        return output

    def get_footstep_pair_distance(self, current_footstep=None):
        if current_footstep is None:
            current_footstep = self.current_footstep

        stone_idcs = self.targets[self.env_arange, current_footstep]
        target_location = self.stone_pos[stone_idcs]
        target_location[:, :, 2] += self.task.foot_collision_sphere_r

        foot_idcs = self.get_footstep_idcs(current_footstep)
        foot_center_pos = self.task.foot_center_pos[
            self.env_arange.unsqueeze(-1),
            foot_idcs]

        # correction, since I'm using stepping stones that belong to env 0
        cor_foot_center_pos = foot_center_pos.clone()
        cor_foot_center_pos[:, :, 1] += self.env_y_offsets

        return target_location - cor_foot_center_pos

    def get_footstep_pair_contact_force(self, current_footstep=None):
        if current_footstep is None:
            current_footstep = self.current_footstep

        foot_idcs = self.get_footstep_idcs(current_footstep)
        contact_forces = self.task.foot_contact_forces[
            torch.arange(self.num_envs, device=self.device).unsqueeze(-1),
            foot_idcs]
        return contact_forces

    def velocity_towards_footsteps(self):
        """Compute reward corresponding to velocity of feet towards
        the next footstep targets."""

        # get global feet velocity
        foot_idcs = self.get_footstep_idcs(self.current_footstep)
        foot_vels = self.task.foot_vel[
            self.env_arange.unsqueeze(-1),
            foot_idcs]

        # compute unit vectors from foot to targets
        stone_idcs = self.targets[self.env_arange, self.current_footstep]
        target_pos = self.stone_pos[stone_idcs]
        foot_pos = self.task.foot_center_pos[
            self.env_arange.unsqueeze(-1),
            foot_idcs]
        foot_pos[:, :, 1] += self.env_y_offsets
        diff = target_pos - foot_pos
        unit_vec = diff / diff.norm(dim=-1, keepdim=True)

        # dot product to find velocity component towards footstep target
        return (unit_vec * foot_vels).sum(-1)




