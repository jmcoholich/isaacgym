import torch
from torch import nn
from .observation import Observation
from .reward import Reward
from .walk_footstep_generator import WalkFootstepGenerator
from .trot_footstep_generator import TrotFootstepGenerator
import os
from isaacgym import gymapi as g
from .utils import *




class LowLevelNetwork(nn.Module):
    """ This was created with the following constraints on the low-level:
    - 2-layer MLP with ELU activation
    - input normalization"""
    def __init__(
            self,
            task,
            filename='210828200458491248',
            saved_nn_folder=True,
            ws=-1,
            nn_size=256,
            action_size=12,
            ll_obs_parts=None,
            play=False):

        super().__init__()
        self.task = task
        self.num_envs = task.num_envs
        self.device = task.device
        self.play = play
        if ll_obs_parts is None:
            ll_obs_parts = [
                "base_position",
                "base_6d_orientation",
                "base_velocity",
                "base_angular_velocity",
                "joint_positions",
                "joint_velocities",
                "joint_torques",
                "foot_contact_binary",
                "footstep_target_distance"
            ]
        self.observe = Observation(ll_obs_parts, self.task,
                                   ignore_footstep_targets=True)
        obs_size = self.observe.compute_obs_size_proprioception()
        state = self.load_checkpoint(filename, ws, saved_nn_folder=saved_nn_folder)
        self.model = torch.nn.Sequential(
            nn.Linear(obs_size, nn_size),
            nn.ELU(),
            nn.Linear(nn_size, nn_size),
            nn.ELU(),
            nn.Linear(nn_size, action_size)
        )
        self.set_weights(state['model'])
        self.mean = state['running_mean_std']['running_mean']
        self.std = state['running_mean_std']['running_var'] ** 0.5 + 1e-5

        if self.play:
            rew_parts = {
                'base_x_vel': [0.0, 1.0],  # rew term coef, clip value
                'torque_penalty': [0.000001],
                'collision_penalty': [0.25],
                'hit_footstep': [3.0],
                'velocity_towards_footstep': [0.0, 1000000.0],  # coef, clip
                'foot_lift_penalty': [0.125],
                'current_footstep_for_logging': [None]  # this is purely just for logging, using the structure I already set up
            }
            self.reward = Reward(rew_parts, task)
            self.rew_sum = 0.0

    def get_action(self, high_level_action):
        """The high level action is just a vector for xyz. """
        obs = self.observe()
        if self.play:
            # rew, rew_dict = self.reward()
            # self.rew_sum += rew[0]
            # if self.task.reset_buf[0]:
            #     print("reward: {}".format(self.rew_sum))
            #     self.rew_sum = 0.0
            # self.task.gym.viewer_camera_look_at(
            #     self.task.viewer, self.task.envs[0], g.Vec3(3.0, 0.0, 0.1),
            #     g.Vec3(0.0, 0, 0.0))
            # obs = self.observe()
            self.plot_footstep_goals(high_level_action)
            # ll_action = self(obs).clamp(-1.0, 1.0)
            # return self.task.foot_positions_action(ll_action)
        else:
            obs[:, 19:31] = self.compute_footstep_target_distance(high_level_action)
        ll_action = self(obs).clamp(-1.0, 1.0)
        return self.task.foot_positions_action(ll_action)

    def plot_footstep_goals(self, high_level_action):
        raise NotImplementedError  # TODO fix this
        self.task.gym.clear_lines(self.task.viewer)
        x = high_level_action.view(1, 2)[0, 0]
        y = high_level_action.view(1, 2)[0, 1]
        self.task.gym.add_lines(self.task.viewer,
                                self.task.envs[0],
                                1,  # num vertices
                                torch.tensor([x, y, 0, x, y, 1.0]).numpy(),
                                tuple(torch.rand(3).numpy()))  # color

    def compute_footstep_target_distance(self, high_level_actions):
        """Return a vector of size 12 containing the footstep distances in
        robot yaw frame. The high level actions are xy footstep placements
        on the ground relative to the foot in question."""
        obs = torch.zeros(self.num_envs, 12, device=self.device)
        foot = self.task.foot_idcs[self.task.current_foot]
        z_diff = self.task.foot_center_pos[torch.arange(self.num_envs, device=self.device), foot, 2]
        z_diff -= self.task.foot_collision_sphere_r
        z_diff *= -1.0
        # self.positions = torch.cat(
        #     (self.task.foot_center_pos[:, :-1] - high_level_actions,
        #      z_diff.unsqueeze(-1)),
        #     dim=1)
        self.positions = torch.cat(
            (high_level_actions.view(self.num_envs, 2),
             z_diff.unsqueeze(-1)),
            dim=1)
        # rotate distances by negative yaw
        # yaw = self.task.base_euler[:, 2]
        # rot_mat = batch_z_rot_mat(-yaw)
        # self.positions = (rot_mat @ self.positions.unsqueeze(-1)).squeeze(-1)
        idcs = torch.arange(3, device=self.device).unsqueeze(0) + foot.unsqueeze(-1) * 3
        obs[torch.arange(self.num_envs, device=self.device).unsqueeze(-1), idcs] = self.positions
        return obs

    def set_weights(self, state):
        with torch.no_grad():
            self.model[0].weight = nn.Parameter(state['a2c_network.actor_mlp.0.weight'])
            self.model[0].bias = nn.Parameter(state['a2c_network.actor_mlp.0.bias'])
            self.model[2].weight = nn.Parameter(state['a2c_network.actor_mlp.2.weight'])
            self.model[2].bias = nn.Parameter(state['a2c_network.actor_mlp.2.bias'])
            self.model[4].weight = nn.Parameter(state['a2c_network.mu.weight'])
            self.model[4].bias = nn.Parameter(state['a2c_network.mu.bias'])

    def forward(self, obs):
        with torch.no_grad():
            obs = (obs - self.mean.to(torch.float)) / self.std.to(torch.float)
            out = self.model(obs)
        return out

    def load_checkpoint(self, filename, ws, saved_nn_folder=False):
        if ws == -1:
            print("=> loading checkpoint '{}'".format(filename))
            directory = './nn' if not saved_nn_folder else './saved_nn'
            path = os.path.join(directory, filename, 'Aliengo.pth')
            state = torch.load(path, map_location=self.device)
        else:
            print("=> loading checkpoint '{}' from workstation {}".format(filename,
                                                                        ws))
            import paramiko
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ws_ip = ['143.215.128.18',
                    '143.215.131.33',
                    '143.215.131.34',
                    '143.215.128.16',
                    '143.215.131.25',
                    '143.215.131.23',
                    '130.207.124.148']  # last one is sky1.cc.gatech.edu
            print('\n\nOpening Remote SSH Client...\n\n')
            if ws <= 6:
                ssh_client.connect(ws_ip[ws - 1], 22, 'jcoholich')
            else:
                ssh_client.connect(ws_ip[ws - 1], 22, 'jcoholich3')
            print('Connected!\n\n')
            # ssh_client.exec_command('cd hutter_kostrikov; cd trained_models')
            sftp_client = ssh_client.open_sftp()
            path = os.path.join('isaacgym/python/rlgpu/nn', filename, 'Aliengo.pth')
            remote_file = sftp_client.open(path, 'rb')
            state = torch.load(remote_file, map_location=self.device)
            print('Agent Loaded\n\n')

        return state