import torch
from .utils import *


class Observation():
    def __init__(self, parts, task, ignore_footstep_targets=False,
                 prev_obs_stacking=0, env_cfg=None):
        """Take a list of things to include in the observation
        and a AliengoQuadruped object.
        Return arbitrary observation upper and lower bound
        (values are not used) vectors of the correct length.
        """
        assert isinstance(parts, list)
        self.ignore_footstep_targets = ignore_footstep_targets
        if task is not None:
            self.num_envs = task.num_envs
            self.device = task.device
        self.parts = parts
        self.task = task
        if env_cfg is None:
            env_cfg = self.task.cfg
            self.add_noise = env_cfg.get("obs_noise", False)
        # self.prev_obs = None
        # self.action_len = action_len
        # values are function handle to call, length of observation,
        # std of gaussian noise added to observation
        self.handles = {
            'base_position': (self.get_base_position, 3, None),
            'base_6d_orientation': (self.get_base_6d_orientation, 6, None),
            'base_velocity': (self.get_base_velocity, 3, None),
            'base_angular_velocity': (self.get_base_angular_velocity, 3, None),

            'joint_torques': (self.get_applied_torques, 12, 0.1),
            'joint_positions': (self.get_joint_positions, 12, None),
            'joint_velocities': (self.get_joint_velocities, 12, None),

            'foot_positions': (self.get_foot_positions, 12, 0.001),
            'foot_velocities': (self.get_foot_velocities, 12, 0.02),

            'trajectory_generator_state': (self.get_tg_state, 5, None),
            'trajectory_generator_phase': (self.get_tg_phase, 2, None),
            'foot_contact_binary': (self.get_foot_contact_binary, 4, None),
            'footstep_target_distance': (self.get_footstep_target_distance, 8, None),
            'high_level_foot': (self.get_high_level_foot, 4, None),
            'zero_step_token': (self.get_zero_step_token, 1, None),
            'one_step_token': (self.get_one_step_token, 1, None),
            'foot_contact_forces': (self.get_foot_contact_forces, 12, None),
            'previous_action': (self.get_last_action, env_cfg['numActions'], None),
            'previous_previous_action': (self.get_last_last_action, env_cfg['numActions'], None),
            'footstep_generator_clock': (self.get_footstep_generator_clock, 4, None),

            'vision': (self.get_vision, 0, None),  # size of vision is determine by other parameters
            'stepping_stone_state': (self.get_stepping_stone_state, 112, 0.005),
            'stepping_stone_state_large': (self.get_stepping_stone_state_large, 164, 0.005),
            'stepping_stone_state_huge': (self.get_stepping_stone_state_huge, 644, 0.005),
            'stepping_stone_state_slim': (self.get_stepping_stone_state_slim, 284, 0.005),

            # 'base_orientation': (self.get_base_orientation, 3),  # euler angles
            'base_roll': (self.get_base_roll, 1, 0.05),
            'base_pitch': (self.get_base_pitch, 1, 0.05),
            'base_yaw': (self.get_base_yaw, 1, 0.05),
            'base_roll_velocity': (self.get_base_roll_velocity, 1, 0.1),
            'base_pitch_velocity': (self.get_base_pitch_velocity, 1, 0.1),
            'base_yaw_velocity': (self.get_base_yaw_velocity, 1, None),
            # 'expert_targets': (self.get_expert_targets, 12),
            'base_z': (self.get_base_z, 1, None),

            'footstep_generator_current_foot_one_hot': (self.get_footstep_generator_current_foot_one_hot, 4, None),


            # 'next_footstep_distance': (self.get_next_footstep_distance, 3),
            # 'current_footstep_foot_one_hot': (self.get_current_foot_one_hot, 4),
            # 'footstep_distance_one_hot': (self.get_footstep_distance_one_hot,
            #                               12),

            # 'robo_frame_foot_position': (self.get_foot_position, 12),
            # 'robo_frame_foot_velocity': (self.get_foot_velocity, 12),

            # 'noise': (self.get_noise, 1),
            # 'constant_zero': (self.get_zero, 1),
            # 'constant_one': (self.get_one, 1),
            # 'one_joint_only': (self.get_one_joint_only, 1),

            # 'start_token': (self.get_start_token, 1),
            # 'previous_observation': (None, None),
            # 'previous_action': (None, None),

        }
        assert all(part in self.handles.keys() for part in parts)
        # ensure env is invariant to order of obs parts listed in config file
        self.parts.sort()

        self.stack_prev = prev_obs_stacking != 0
        if self.stack_prev:
            self.prev_obs = []
            for _ in range(prev_obs_stacking):
                self.prev_obs.append(torch.zeros(
                    self.num_envs,
                    self.task.cfg['pre_stack_numObservations'],
                    device=self.device))

        # self.obs_len = 0
        # for part in self.parts:
        #     if part not in ['previous_observation', 'previous_action']:
        #         self.obs_len += self.handles[part][1]
        # if 'previous_observation' in self.parts:
        #     self.obs_len *= 2
        # if 'previous_action' in self.parts:
        #     self.obs_len += self.action_len

        # the below bounds are arbitrary and not used in the RL algorithms
        # self.observation_lb = -np.ones(self.obs_len)
        # self.observation_ub = np.ones(self.obs_len)

    def compute_obs_size_proprioception(self):
        """Computes the size of the observation, but without vision OR stepping
        stone state size since these parameters are not passed to this
        object."""
        val = 0
        for part in self.parts:
            val += self.handles[part][1]
        return val

    def get_footstep_obs_start_idx(self):
        output = 0
        for part in self.parts:
            if part == "footstep_target_distance":
                return output
            if self.handles[part][1] == 0:
                raise ValueError("Variable-length observation item encountered."
                                 " Unable to determine footstep obs start idx.")
            output += self.handles[part][1]
        raise ValueError("'footstep_target_distance' is not in observation.")

    def __call__(self, recalculating_obs=False, parts=None, add_noise=None):
        # for part in self.parts:
        #     print(part)
        #     print(self.handles[part][0]()[0])
        #     print()
        # print('#' * 150)
        # breakpoint()
        if add_noise is None:
            add_noise = self.add_noise

        # obs = torch.cat([self.handles[part][0]() for part in self.parts if part != 'vision'],
        #                 dim=1)
        if parts is None:
            parts = self.parts

        obs = []
        for part in parts:
            if part != "vision":
                temp = self.handles[part][0]()
                if add_noise and self.handles[part][2] is not None:
                    temp += torch.randn_like(temp) * self.handles[part][2]
                obs.append(temp)

        obs = torch.cat(obs, dim=1)

        if 'vision' in parts:
            obs = torch.cat((obs, self.get_vision()), dim=1)

        if not self.stack_prev:
            return obs
        elif recalculating_obs:
            raise NotImplementedError("Oldest previous observation has already"
            " been discarded, need to fix this if I want to use obs stacking")
        else:
            # the last observation is always the most recent
            output = torch.cat(self.prev_obs + [obs], dim=1)
            self.prev_obs.pop(0)
            self.prev_obs.append(obs)
            return output

        # if ('previous_observation' in self.parts
        #         and self.env.eps_step_counter != 0):
        #     obs = np.concatenate((obs, self.prev_obs))
        #     self.prev_obs = obs[:int(len(obs)/2)]
        # elif 'previous_observation' in self.parts:
        #     obs = np.concatenate((obs, obs))
        #     self.prev_obs = obs[:int(len(obs)/2)]

        # if 'previous_action' in self.parts and prev_action is not None:
        #     obs = np.concatenate((obs, prev_action))
        # elif 'previous_action' in self.parts:
        #     assert self.env.eps_step_counter == 0
        #     obs = np.concatenate((obs, np.zeros(self.action_len)))
        # return obs
    def get_footstep_generator_current_foot_one_hot(self):
        return self.task.footstep_generator.get_current_foot_one_hot()

    # def get_expert_targets(self):
    #     expert_pos = self.task.expert_policy.step_in_place_traj()
    #     return self.task.get_foot_frame_foot_positions(
    #         foot_center_pos=expert_pos).view(self.num_envs, 12)

    def get_base_z(self):
        return self.task.base_pos[:, 2:3]

    def get_foot_positions(self):
        return self.task.get_foot_frame_foot_positions().view(self.num_envs, 12)

    def get_foot_velocities(self):
        return self.task.get_foot_frame_foot_velocities().view(self.num_envs, 12)

    def get_footstep_generator_clock(self):
        clock = self.task.footstep_generator.counter % self.task.footstep_generator.cfg['update_period']
        foot = self.task.footstep_generator.footstep_idcs[
            torch.arange(self.num_envs, device=self.device),
            self.task.footstep_generator.current_footstep % 4]
        output = torch.zeros(self.num_envs, 4, device=self.device)
        output[torch.arange(self.num_envs, device=self.device), foot] = clock
        return output

    def get_foot_contact_forces(self):
        # if (self.task.foot_contact_forces < 0.0).any():
        #     print(self.task.foot_contact_forces)
        #     print(self.task.progress_buf[0])
        #     print()
        #     breakpoint()
        return self.task.foot_contact_forces.view(self.num_envs, 12)

    def get_last_action(self):
        return self.task.last_action.clone()

    def get_last_last_action(self):
        return self.task.last_last_action.clone()

    def get_zero_step_token(self):
        return (self.task.progress_buf == 0).float().unsqueeze(-1)

    def get_one_step_token(self):
        return (self.task.progress_buf == 1).float().unsqueeze(-1)

    def get_high_level_foot(self):
        """Return a one-hot vector representing which footstep is active. """
        idcs = self.task.foot_idcs[self.task.current_foot]
        output = torch.zeros(self.num_envs, 4, device=self.device)
        output[torch.arange(self.num_envs, device=self.device), idcs] = 1.0
        return output

    def get_footstep_target_distance(self):
        # if self.ignore_footstep_targets:
        #     return torch.zeros(self.num_envs, 12, device=self.device)
        return self.task.footstep_generator.get_footstep_target_distance()

    def get_stepping_stone_state(self):
        if self.task.is_stepping_stones:
            return self.task.stepping_stones.get_state(self.task.foot_center_pos.clone())
        elif self.task.is_rough_terrain_blocks:
            raise NotImplementedError()
        else:
            return self.spoof_ss_state(self.handles["stepping_stone_state"][1])

    def get_stepping_stone_state_large(self):
        if self.task.is_stepping_stones:
            return self.task.stepping_stones.get_large_state(self.task.foot_center_pos.clone())
        elif self.task.is_rough_terrain_blocks:
            raise NotImplementedError()
        else:
            return self.spoof_ss_state(self.handles["stepping_stone_state_large"][1])

    def get_stepping_stone_state_huge(self):
        if self.task.is_stepping_stones:
            return self.task.stepping_stones.get_huge_state(self.task.foot_center_pos.clone())
        elif self.task.is_rough_terrain_blocks:
            raise NotImplementedError()
        else:
            return self.spoof_ss_state(self.handles["stepping_stone_state_huge"][1])

    def get_stepping_stone_state_slim(self):
        if self.task.is_stepping_stones:
            return self.task.stepping_stones.get_slim_state(self.task.foot_center_pos.clone())
        elif self.task.is_rough_terrain_blocks:
            raise NotImplementedError()
        else:
            return self.spoof_ss_state(self.handles["stepping_stone_state_slim"][1])

    def spoof_ss_state(self, size):
        """This is for running policies trained with ss_state on flat ground.
        I just return the negative foot center distance, for the size of the
        observation
        """
        output = torch.zeros(self.num_envs, 4, size // 4, device=self.device)
        output[:] = -self.task.foot_center_pos[..., 2:].clone()
        return output.clamp(-1.0, 1.0).reshape(self.num_envs, size)

    def get_base_position(self):
        return self.task.base_pos

    def get_base_6d_orientation(self):
        return batch_quat_to_6d(self.task.base_quat)

    def get_base_velocity(self):
        return self.task.base_vel

    def get_base_angular_velocity(self):
        return self.task.base_avel

    def get_applied_torques(self):
        return self.task.joint_torques

    def get_joint_positions(self):
        return self.task.joint_positions

    def get_joint_velocities(self):
        return self.task.joint_velocities

    def get_tg_state(self):
        return torch.cat(((self.task.t % 1.0).sin().unsqueeze(1),
                          (self.task.t % 1.0).cos().unsqueeze(1),
                          self.task.tg_state), dim=1)

    def get_tg_phase(self):
        return torch.cat(((self.task.t % 1.0).sin().unsqueeze(1),
                          (self.task.t % 1.0).cos().unsqueeze(1)), dim=1)

    def get_foot_contact_binary(self):
        return (self.task.foot_contact_forces[:, :, 2] > 0.0).float()

    def get_vision(self):
        return self.task.cam_obs

    # def get_start_token(self):
    #     return np.array([float(self.env.eps_step_counter == 0)])

    # def get_base_velocity(self):
    #     return self.quadruped.base_vel

    # def get_base_orientation(self):
    #     return self.quadruped.base_euler

    # def get_base_position(self):
    #     return self.quadruped.base_position

    # def get_applied_torques(self):
    #     return self.quadruped.applied_torques

    # def get_joint_positions(self):
    #     return self.quadruped.joint_positions

    # def get_joint_velocities(self):
    #     return self.quadruped.joint_velocities

    # def get_base_angular_velocity(self):
    #     return self.quadruped.base_avel

    def get_base_roll(self):
        return self.task.base_euler[:, 0:1]

    def get_base_pitch(self):
        return self.task.base_euler[:, 1:2]

    def get_base_yaw(self):
        return self.task.base_euler[:, 2:3]

    def get_base_roll_velocity(self):
        return self.task.base_avel[:, 0:1]

    def get_base_pitch_velocity(self):
        return self.task.base_avel[:, 1:2]

    def get_base_yaw_velocity(self):
        return self.task.base_avel[:, 2:3]

    # def get_tg_phase(self):
    #     return np.array([np.sin(self.quadruped.phases[0]),
    #                      np.cos(self.quadruped.phases[0])])

    # def get_next_footstep_distance(self):
    #     """The footstep_generator.get_current_footstep_distance()
    #     returns the xyz vector aligned with the global coordinate system.
    #     The robot does not know its own yaw, so the vector is transformed
    #     to align with robot front direction.
    #     """
    #     yaw = self.quadruped.base_euler[2]
    #     vec = self.quadruped.footstep_generator.get_current_footstep_distance()
    #     # rotate by negative yaw angle to get vectors in robot frame
    #     rot_mat = np.array([[np.cos(-yaw), -np.sin(-yaw), 0.0],
    #                         [np.sin(-yaw), np.cos(-yaw), 0.0],
    #                         [0.0, 0.0, 1.0]])
    #     return (rot_mat @ np.expand_dims(vec, 1)).squeeze()

    # def get_footstep_distance_one_hot(self):
    #     output = np.zeros(12)
    #     foot = self.quadruped.footstep_generator.footstep_idcs[
    #         self.quadruped.footstep_generator.current_footstep % 4]
    #     vec = self.get_next_footstep_distance()
    #     output[foot*3 : (foot+1)*3] = vec
    #     return output

    # def get_noise(self):
    #     return (np.random.random_sample(1) - 0.5) * 2.0

    # def get_zero(self):
    #     return np.zeros(1)

    # def get_one_joint_only(self):
    #     return self.quadruped.joint_positions[np.newaxis, 2]

    # def get_current_foot_one_hot(self):
    #     foot = self.quadruped.footstep_generator.footstep_idcs[
    #         self.quadruped.footstep_generator.current_footstep % 4]
    #     one_hot = np.zeros(4)
    #     one_hot[foot] = 1.0
    #     return one_hot

    # def get_foot_position(self):
    #     return self.quadruped.get_foot_frame_foot_positions().flatten()

    # def get_foot_velocity(self):
    #     return self.quadruped.get_foot_frame_foot_velocities().flatten()

    # def get_foot_contact_state(self):
    #     return self.quadruped.get_foot_contact_states()

    # def get_one(self):
    #     return np.ones(1)
