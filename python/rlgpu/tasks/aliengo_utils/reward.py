import torch
from .utils import *


class Reward():
    def __init__(self, reward_parts, task):
        assert isinstance(reward_parts, dict)

        self.reward_parts = reward_parts
        self.task = task
        self.num_envs = self.task.num_envs
        self.device = self.task.device
        # self.action_size = action_size

        self.all_terms = {
            'base_x_vel': self.base_x_vel,
            'torque_penalty': self.torque_penalty,
            'euler_angle_pen': self.euler_angle_pen,
            'collision_penalty': self.collision_penalty,
            'hit_footstep': self.hit_footstep,
            # 'velocity_towards_footstep': self.velocity_towards_footstep,
            'velocity_towards_footsteps': self.velocity_towards_footsteps,
            # 'foot_lift_penalty': self.foot_lift_penalty,
            # 'foot_lift_penalty_smooth': self.foot_lift_penalty_smooth,
            'current_footstep_for_logging': self.current_footstep_for_logging,
            'smoothness': self.smoothness,
            'foot_smooth': self.foot_smooth,
            'base_smooth': self.base_smooth,
            'existence': self.existence,
            # 'stay_put': self.stay_put,
            'foot_lift': self.foot_lift,
            'slip_penalty': self.slip_penalty,
            'hit_config': self.hit_config,
            'follow_expert': self.follow_expert,
            'follow_expert_mean_dist': self.follow_expert_mean_dist,
            'y_vel_pen': self.y_vel_pen,
            'contact_rew': self.contact_rew,
            'foot_stay': self.foot_stay,
            'base_x_pos': self.base_x_pos,
            'wrong_ss_collision_penalty': self.wrong_ss_collision_penalty,
        }

        assert all(part in self.all_terms.keys()
                   for part in self.reward_parts.keys())

    def __call__(self):
        total_rew = torch.zeros(self.num_envs, device=self.device)
        rew_dict = {}
        for part in self.reward_parts.keys():
            term, raw_value_tensor = self.all_terms[part](*self.reward_parts[part])
            assert term.shape[0] == self.num_envs
            total_rew += term
            rew_dict["rewards/" + part] = raw_value_tensor
        return total_rew, rew_dict

    def follow_expert(self, k):
        term = self.task.expert_policy.reward()
        return k * term + 1.0, term

    def contact_rew(self, k):
        term = self.task.contact_rew()
        return k * term, term

    def foot_stay(self, k):
        term = self.task.footstep_generator.rew_dict["foot_stay"]
        return k * term, term

    def wrong_ss_collision_penalty(self, k):
        term = self.task.footstep_generator.rew_dict[
            "wrong_ss_collision_penalty"]
        return -k * term, term

    def y_vel_pen(self, k):
        term = self.task.base_vel[:, 1].square()
        return -k * term, term

    def follow_expert_mean_dist(self, k):
        term = self.task.expert_policy.mean_dist()
        return k * term, term

    def hit_config(self, k):
        term = self.task.footstep_generator.hit_config()
        return k * term, term

    def foot_lift(self, k):
        term = self.task.foot_lift_rew()
        return k * term, term

    def slip_penalty(self, k):
        term = self.task.get_slip_penalty()
        return -k * term, term

    # def stay_put(self, k):
    #     term = -(self.task.base_vel[:, 0]**2 + self.task.base_vel[:, 1]**2)**0.5
    #     return k * term, term.mean()

    def smoothness(self, k):
        diff1 = (self.task.last_action - self.task.last_last_action)
        diff2 = (self.task.last_last_action - self.task.last_last_last_action)
        term = -(diff1 - diff2).norm(dim=1)
        return k * term, term

    def foot_smooth(self, k):
        # term = -(self.task.foot_vel - self.task.prev_foot_vel).view(self.num_envs, 12).norm(dim=1)
        term = -self.huber((self.task.foot_vel - self.task.prev_foot_vel).view(self.num_envs, 12))
        return k * term, term

    def base_smooth(self, k):
        # term = -(self.task.cur_base_vel - self.task.prev_base_vel).norm(dim=1)
        term = -self.huber(self.task.cur_base_vel - self.task.prev_base_vel)
        return k * term, term

    def huber(self, val):
        delta = 1.0
        sq = val.abs() <= delta
        val[sq] = 0.5 * val[sq] * val[sq]
        val[~sq] = delta * (val[~sq].abs() - 0.5 * delta)
        return val.mean(dim=1)

    def base_x_vel(self, k, clip):
        term = self.task.base_vel[:, 0].clamp(-clip, clip)
        return k * term, term

    def base_x_pos(self, k):
        term = self.task.base_pos[:, 0]
        return k * term, term

    def existence(self, k):
        term = torch.ones(self.num_envs, device=self.device)
        return k * term, term

    def torque_penalty(self, k):
        term = self.task.joint_torques.square().sum(dim=1)
        return -term * k, term

    def euler_angle_pen(self, x, y, z):
        base_euler = batch_quat_to_euler(self.task.base_quat)
        coefs = torch.tensor([x, y, z], device=self.device)
        term = (coefs * base_euler.abs()).sum(dim=1)
        return -term, base_euler.abs().sum(dim=1)

    def collision_penalty(self, k):  # TODO this doesn't consider contacts at joints right?
        term = (self.task.body_contact_forces.abs() > 0.0).any(-1).count_nonzero(-1).float()
        return -k * term, term

    def hit_footstep(self, k):
        term = self.task.footstep_generator.rew_dict["hit_footstep"]
        return k * term, term

    # def velocity_towards_footstep(self, k, clip):
    #     term = self.task.footstep_generator.velocity_towards_footstep().clamp(-clip, clip)
    #     return k * term, term.mean()

    def velocity_towards_footsteps(self, k, clip):
        term = self.task.footstep_generator.rew_dict["foot_velocity"].clamp(
            -clip, clip).sum(-1)
        return k * term, term

    # def foot_lift_penalty(self, k):
    #     term = self.task.footstep_generator.other_feet_lifted()
    #     return -k * term, term.mean()

    # def foot_lift_penalty_smooth(self, k):
    #     term = self.task.footstep_generator.other_feet_lifted_smooth()
    #     return -k * term, term.mean()

    def current_footstep_for_logging(self, ignore):
        current_footstep = self.task.footstep_generator.current_footstep.float()
        return torch.zeros(self.num_envs, device=self.device), current_footstep


    # def joint_torques_sq(self, k):
    #     term = (self.quadruped.applied_torques
    #             * self.quadruped.applied_torques).sum()
    #     return k * term, term

    # def fwd_vel(self, k, lb, ub):
    #     term = np.clip(self.quadruped.base_vel[0], lb, ub)
    #     return k * term, term

    # def footstep_reached(self, k, distance_threshold):
    #     term = self.quadruped.footstep_generator.footstep_reached(
    #         distance_threshold)
    #     return k * term, term

    # def velocity_towards_footstep(self, k, min_, max_):
    #     term = np.clip(
    #         self.quadruped.footstep_generator.velocity_towards_footstep(),
    #         min_,
    #         max_)
    #     return k * term, term

    # def existance(self, k):
    #     return k, 1.0

    # def smoothness_sq(self, k):
    #     """The maximum reward is zero, and the minimum reward is -1.0."""
    #     if self.prev_action is not None:
    #         diff = self.action - self.prev_action
    #         term = -(diff * diff).sum() / self.action_size / 2.0
    #     else:
    #         term = 0.0
    #     self.prev_action = self.action
    #     return k * term, term

    # def orientation(self, k, x, y, z):
    #     coeffs = np.array([x, y, z])
    #     term = (abs(self.quadruped.base_euler) * coeffs).sum()
    #     return k * term, term

    # def lift_feet(self, k, height):
    #     # if feet are in the middle 50% of the lifing half of the pmtg phase
    #     swing_feet = ((1.25 * np.pi < self.quadruped.phases)
    #                   & (self.quadruped.phases < 1.75 * np.pi)).nonzero()[0]
    #     if len(swing_feet) == 0:
    #         term = 0.0
    #     else:
    #         global_pos = self.quadruped.get_global_foot_positions()
    #         above = len((global_pos[swing_feet, 2] > height).nonzero()[0])
    #         term = above / len(swing_feet)
    #     return k * term, term
