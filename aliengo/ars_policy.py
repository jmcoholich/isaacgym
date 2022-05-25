import torch
from isaacgym import gymapi as g
from copy import deepcopy

from torch_running_mean_std import RunningMeanStd


class ARSPolicy:
    def __init__(self, obs_size, params, device="cuda:0"):
        # self.root_tensor = root_tensor
        # self.dof_states = dof_states
        # self.device = device
        # self.n_envs = self.root_tensor.shape[0]
        # assert self.dof_states.shape[0]/self.n_envs == 12

        # self.base_pos = self.root_tensor[:, 0:3]
        # self.base_euler = self.root_tensor[:, 3:7]
        # self.base_vel = self.root_tensor[:, 7:10]
        # self.base_avel = self.root_tensor[:, 10:13]

        # self.joint_positions = self.dof_states[:, 0].view((self.n_envs, 12))
        # self.joint_velocities = self.dof_states[:, 1].view((self.n_envs, 12))

        # observation = self.get_observation()
        self.mean_std = RunningMeanStd(shape=(obs_size))
        self.old_mean_std = RunningMeanStd(shape=(obs_size))
        self.params = params
        self.policy = torch.zeros(
            (12, obs_size),
            device=device)
        self.candidate_policies = torch.zeros(
            (self.params["n_dirs"] * 2, 12, obs_size),
            device=device)
        self.perturbations = torch.zeros((self.params["n_dirs"], 12, obs_size),
                                         device=device)

    def __call__(self, obs):
        norm_obs = (obs - self.old_mean_std.mean) / torch.sqrt(self.old_mean_std.var)
        action = self.policy @ norm_obs.unsqueeze(-1)
        return action.flatten()

    def generate_candidates(self):
        self.candidate_policies = self.policy.tile(
            (self.params["n_dirs"] * 2, 1, 1))
        self.perturbations = torch.randn_like(self.perturbations)
        self.candidate_policies[::2] -= (self.perturbations
                                         * self.params['delta_std'])
        self.candidate_policies[1::2] += (self.perturbations
                                          * self.params['delta_std'])

    def query_candidates(self, obs):
        norm_obs = (obs - self.old_mean_std.mean) / torch.sqrt(self.old_mean_std.var)
        action = self.candidate_policies @ norm_obs.unsqueeze(-1)
        self.mean_std.update(obs)
        return action.flatten()

    def update(self, rewards):
        rewards = rewards.view(self.params['n_dirs'], 2)
        sort_idcs = rewards.max(axis=1).values.argsort(axis=0, descending=True)
        self.perturbations = self.perturbations[sort_idcs]
        rewards = rewards[sort_idcs]

        rew_diff = rewards[:self.params["top_dirs"], 1] - rewards[:self.params["top_dirs"], 0]
        update = (rew_diff.unsqueeze(-1).unsqueeze(-1)
                  * self.perturbations[:self.params['top_dirs']]).mean(axis=0)

        norm_lr = self.params['lr']/(rewards[:self.params['top_dirs']].std() + 1e-5)
        self.policy += norm_lr * update

    def update_mean_std(self):
        self.old_mean_std = deepcopy(self.mean_std)

