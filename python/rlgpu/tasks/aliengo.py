from typing import Sequence
from typing_extensions import get_args
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym.gymtorch import wrap_tensor as wr
from isaacgym.gymtorch import unwrap_tensor as unwr
from isaacgym import gymapi as g
import torch
import numpy as np
import os
import imageio
import shutil


import time
import wandb

from torch._C import device
from .aliengo_utils.observation import Observation
from .aliengo_utils.reward import Reward
from .aliengo_utils.termination import Terminiations
from .aliengo_utils.low_level_network import LowLevelNetwork
from .aliengo_utils.utils import *
from .aliengo_utils.walk_footstep_generator import WalkFootstepGenerator
from .aliengo_utils.footstep_plotter import FootstepPlotter
from .aliengo_utils.trot_footstep_generator import TrotFootstepGenerator
from .aliengo_utils.expert_policy import ExpertPolicy
from .aliengo_utils.stepping_stones import SteppingStones
from .aliengo_utils.rough_terrain_blocks import RoughTerrainBlocks
from .aliengo_utils.stats_gatherer import StatsGatherer
from .aliengo_utils.ss_trot_footstep_generator import SSTrotFootstepGenerator


class Aliengo(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id,
                 headless):
        if cfg['env']['plot_values']:
            img_dir = "H_frames"
        else:
            img_dir = "F_frames"
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.mkdir(img_dir)
        self.img_dir = img_dir
        # self.global_min_norm = 1e5
        # self.global_lowest_foot = 1e5
        print("Using {} CPU threads".format(torch.get_num_threads()))
        if device_type == 'cuda':
            torch.cuda.set_device(device_id)

        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)

        self.wandb_log_counter = 0
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        self.cfg = cfg["env"]
        self.args = cfg["args"]
        self.sim_params = sim_params
        self.headless = headless
        self.physics_engine = physics_engine
        self.num_envs = self.cfg["numEnvs"]
        self.fix_base = self.cfg.get("fix_base", False)
        self.gravity = -9.81
        # self.fix_base = True
        # self.gravity = 0.0
        self.is_footsteps = 'footstep_target_parameters' in self.cfg

        self.is_stepping_stones = "stepping_stones" in self.cfg
        self.is_rough_terrain_blocks = "rough_terrain_blocks" in self.cfg
        assert not (self.is_stepping_stones and self.is_rough_terrain_blocks)

        if self.is_stepping_stones:
            self.stepping_stones = SteppingStones(
                self,
                self.cfg['stepping_stones'],
                self.num_envs,
                self.device)
        elif self.is_rough_terrain_blocks:
            self.terrain = RoughTerrainBlocks(
                self.cfg['rough_terrain_blocks'],
                self.num_envs,
                self.device)

        # these are in foot idx order
        self.starting_foot_pos = torch.tensor(
            [[0.2134, 0.1494, 0.0285],  # FL
             [0.2135, -0.1493, 0.0285],  # FR
             [-0.2694, 0.1495, 0.0285],  # RL
             [-0.2693, -0.1492, 0.0285]],  # RR
            device=self.device)

        self.start_height = 0.48 - 0.045 + self.fix_base * 0.1
        if self.is_stepping_stones:
            self.start_height += self.cfg['stepping_stones']['height']
            self.starting_foot_pos[:, 2] += self.cfg['stepping_stones']['height']
        elif self.is_rough_terrain_blocks:
            self.start_height += self.cfg['rough_terrain_blocks']['height']
            self.starting_foot_pos[:, 2] += self.cfg['rough_terrain_blocks']['height']


        super().__init__(cfg=cfg)  # this creates self.gym and calls self.create_sim()
        # robot info from URDF
        self.is_successful = torch.ones(self.num_envs, device=self.device,
                                        dtype=torch.bool)
        self.env_offsets = torch.zeros(self.num_envs, 2, device=self.device)
        for i in range(self.num_envs):
            o = self.gym.get_env_origin(self.envs[i])
            # assert self.env_offsets[i, 0] == o.x
            # assert self.env_offsets[i, 1] == o.y
            self.env_offsets[i, 0] = o.x
            self.env_offsets[i, 1] = o.y

        # I know this is dumb
        self.last_action = torch.zeros(self.num_envs, self.cfg['numActions'],
                                       device=self.device)
        self.last_last_action = torch.zeros(self.num_envs, self.cfg['numActions'],
                                            device=self.device)
        self.last_last_last_action = torch.zeros(self.num_envs, self.cfg['numActions'],
                                            device=self.device)

        if "ss_footstep_target_parameters" in self.cfg:
            self.footstep_generator = SSTrotFootstepGenerator(self)
        if self.is_footsteps:
            if self.cfg['footstep_target_parameters']['gait'] == 'walk':
                self.footstep_generator = WalkFootstepGenerator(self)
            elif self.cfg['footstep_target_parameters']['gait'] == 'trot':
                self.footstep_generator = TrotFootstepGenerator(self)
            else:
                raise NotImplementedError

        if 'follow_expert' in self.cfg['reward']:
            self.expert_policy = ExpertPolicy(self)

        hip = 0.037199
        thigh = 0.660252
        knee = -1.200187
        self.reset_joint_angles = torch.tensor(
            [hip, thigh, knee,
             -hip, thigh, knee,
             hip, thigh, knee,
             -hip, thigh, knee],
            device=self.device)
        # self.reset_joint_angles = torch.tensor(
        #     [0.037199, 0.660252, -1.200187, -0.028954, 0.618814, -1.183148, 0.048225, 0.690008, -1.254787, -0.050525, 0.661355, -1.243304],
        #     device=self.device)

        self.prev_foot_center_pos = torch.zeros(self.num_envs, 4, 3,
                                                device=self.device)
        self.prev_foot_vel = torch.zeros(self.num_envs, 4, 3,
                                         device=self.device)
        self.prev_base_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.cur_base_vel = torch.zeros(self.num_envs, 3, device=self.device)
        # self.prev_prev_foot_center_pos = torch.zeros(self.num_envs, 4, 3,
        #                                         device=self.device)
        self.prev_foot_z_contact_forces = torch.zeros(self.num_envs, 4,
                                                      device=self.device)

        self.foot_collision_sphere_r = 0.0265 + self.gym.get_sim_params(self.sim).physx.contact_offset
        self.l1_val = -0.25
        self.l2_val = -0.25
        self.l3_val = -0.083
        self.joint_angle_lb = torch.tensor(self.cfg["joint_angle_lb"], device=self.device)
        self.joint_angle_ub = torch.tensor(self.cfg["joint_angle_ub"], device=self.device)

        self.ik_foot_center_pos_lb = torch.tensor(self.cfg["foot_pos_lb"], device=self.device)
        self.ik_foot_center_pos_ub = torch.tensor(self.cfg["foot_pos_ub"], device=self.device)

        # link indices (link 0 is the base)
        self.hip_idcs = [1, 5, 9, 13]
        self.thigh_idcs = [2, 6, 10, 14]
        self.knee_idcs = [3, 7, 11, 15]
        self.feet_idcs = [4, 8, 12, 16]
        self.non_feet_idcs = [0] + self.hip_idcs + self.thigh_idcs + self.knee_idcs

        # joint indices
        self.hip_joint_idcs = [0, 3, 6, 9]
        self.thigh_joint_idcs = [1, 4, 7, 10]
        self.knee_joint_idcs = [2, 5, 8, 11]

        self._acquire_tensors()
        self.update_tensors()
        self.observe = Observation(self.cfg["observation"], self,
                                   prev_obs_stacking=self.cfg["prev_obs_stacking"])
        self.reward = Reward(self.cfg["reward"], self)
        self.terminations = Terminiations(self, self.cfg["termination"])
        self.epochs = 0
        if self.cfg["actionSpace"] == 'high_level':
            self.low_level_policy = LowLevelNetwork(self, play=self.args.play)
        self.current_foot = torch.randint(4, (self.num_envs,), device=self.device)
        self.foot_idcs = torch.tensor([3, 1, 2, 0], device=self.device).to(torch.long)

        if self.args.gather_stats != -1:
            self.gather_stats = StatsGatherer(self, self.args.gather_stats)

        if self.args.plot_contact_locations:
            self.plot_footsteps = FootstepPlotter(self)

        # env = 0
        # if self.is_stepping_stones:
        #     h = self.cfg["stepping_stones"]["height"]
        # elif self.is_rough_terrain_blocks:
        #     h = self.cfg["rough_terrain_blocks"]["height"]
        # else:
        #     h = 0.0
        # self.gym.viewer_camera_look_at(
        #     self.viewer, self.envs[env],
        #     # side view
        #     g.Vec3(self.base_pos[env, 0] + 0.5, self.base_pos[env, 1] + 1.75, 0.75 + h),
        #     g.Vec3(self.base_pos[env, 0] + 0.5, self.base_pos[env, 1], 0.25 + h),
        #     # top view
        #     # g.Vec3(self.base_pos[env, 0] + 0.5, self.base_pos[env, 1]+ 0.01, 1.75 + h),
        #     # g.Vec3(self.base_pos[env, 0] + 0.5, self.base_pos[env, 1], 0.25 + h),
        #     )


    def _acquire_tensors(self):
        if self.is_stepping_stones:
            start_idx = self.stepping_stones.num_stones
        elif self.is_rough_terrain_blocks:
            start_idx = self.terrain.num_blocks
        else:
            start_idx = 0

        # acquire and define robot state variables
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _torques = self.gym.acquire_dof_force_tensor(self.sim)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        # _jac = self.gym.acquire_jacobian_tensor(self.sim, "aliengo")

        self.dof_state = wr(_dof_state_tensor)  # unaffected by additional (single-body) actors
        self.joint_positions = self.dof_state[:, 0].view(self.num_envs, 12)
        self.joint_velocities = self.dof_state[:, 1].view(self.num_envs, 12)

        self.all_root_tensor = wr(_root_tensor)  # will need to add another dimension for actor index
        self.root_tensor = self.all_root_tensor[start_idx:].view(self.num_envs, 13)
        self.base_pos = self.root_tensor[:, 0:3]
        self.base_quat = self.root_tensor[:, 3:7]
        self.base_vel = self.root_tensor[:, 7:10]
        self.base_avel = self.root_tensor[:, 10:13]

        self.all_rb_states = wr(_rb_states)
        self.num_rbs = self.all_rb_states.shape[0]
        self.rb_start_idx = start_idx
        # 17 rigid bodies per aliengo, 13 DOF per rigid body
        self.rb_states = self.all_rb_states[start_idx:].view(self.num_envs, 17, 13)

        self.contact_forces = wr(_contact_forces)
        self.contact_forces = self.contact_forces[start_idx:].view(self.num_envs, 17, 3)
        # self.full_jac = wr(_jac)
        self.joint_torques = wr(_torques).view(self.num_envs, 12)  # unaffected by additional (single-body) actors

        if self.cfg["actionSpace"] == "pmtg_delta":
            self.t = torch.zeros(self.num_envs, device=self.device)
            # amplitude, standing_height, frequency
            self.start_tg_state = [0.0, -0.44, 0.0]
            self.tg_state = torch.tensor(self.start_tg_state,
                device=self.device).tile(self.num_envs, 1)
            self.tg_state_ub = torch.tensor(self.cfg['pmtg']['ub'],
                                            device=self.device)
            self.tg_state_lb = torch.tensor(self.cfg['pmtg']['lb'],
                                            device=self.device)

            self.pmtg_action_bound = torch.tensor(
                self.cfg["pmtg"]["max_change"] + self.cfg["pmtg"]["residual"]
                * 4, device=self.device)
            if self.cfg["pmtg"].get("use_jump", False):
                self.x_traj_fn = x_traj_jump
                self.z_traj_fn = z_traj_jump
            else:
                self.x_traj_fn = x_traj
                self.z_traj_fn = z_traj
        elif self.cfg["actionSpace"] == "pmtg":
            # pmtg action space:
            # amplitude, standing_height, frequency + 12 * residuals
            self.t = torch.zeros(self.num_envs, device=self.device)
            res = self.cfg['pmtg']["residual"] * 4
            neg_res = [-x for x in res]
            self.tg_action_ub = torch.tensor(self.cfg['pmtg']['ub'] + res,
                                            device=self.device)
            self.tg_action_lb = torch.tensor(self.cfg['pmtg']['lb'] + neg_res,
                                            device=self.device)
            self.tg_action_mean = (self.tg_action_lb + self.tg_action_ub) / 2.0
            self.tg_action_half_range = (self.tg_action_ub - self.tg_action_lb) / 2.0
            self.pmtg_reset_action = torch.tensor(
                [0.0, -0.44, 0.0] + 12 * [0.0], device=self.device)
            if self.cfg["pmtg"].get("use_jump", False):
                self.x_traj_fn = x_traj_jump
                self.z_traj_fn = z_traj_jump
            else:
                self.x_traj_fn = x_traj
                self.z_traj_fn = z_traj

        elif self.cfg["actionSpace"] == "foot_positions_delta":
            self.foot_pos_action_bound = torch.tensor(
                self.cfg["foot_pos_change_limits"], device=self.device)

        elif self.cfg["actionSpace"] == "foot_positions":
            foot_pos_ub = torch.tensor(
                self.cfg["foot_pos_ub"], device=self.device)
            foot_pos_lb = torch.tensor(
                self.cfg["foot_pos_lb"], device=self.device)
            self.foot_pos_action_mean = (foot_pos_ub + foot_pos_lb) / 2.0
            self.foot_pos_action_half_range = (foot_pos_ub - foot_pos_lb) / 2.0
            self.foot_pos_action_reset_action = \
                torch.tensor([[-0.0248,  0.0983, -0.4550],
                              [-0.0248, -0.0983, -0.4550],
                              [-0.0248,  0.0983, -0.4550],
                              [-0.0248, -0.0983, -0.4550]],
                             device=self.device).flatten()

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        actions = actions.view(self.num_envs, self.cfg['numActions'])
        self.last_last_last_action[:] = self.last_last_action.clone()
        self.last_last_action[:] = self.last_action.clone()
        self.last_action[:] = actions.clone()
        if self.args.play or not self.headless:
            if self.is_stepping_stones:
                h = self.cfg["stepping_stones"]["height"]
            elif self.is_rough_terrain_blocks:
                h = self.cfg["rough_terrain_blocks"]["height"]
            else:
                h = 0.0
            env = 0
            self.gym.viewer_camera_look_at(
                self.viewer, self.envs[env],
                # side view
                g.Vec3(self.base_pos[env, 0] + 0.5, self.base_pos[env, 1] + 1.75, 0.75 + h),
                g.Vec3(self.base_pos[env, 0] + 0.5, self.base_pos[env, 1], 0.25 + h),
                # top view
                # g.Vec3(self.base_pos[env, 0] + 0.5, self.base_pos[env, 1]+ 0.01, 1.75 + h),
                # g.Vec3(self.base_pos[env, 0] + 0.5, self.base_pos[env, 1], 0.25 + h),
                )

        actions = actions.clamp(-1.0, 1.0)
        if self.cfg["actionSpace"] == "pmtg_delta":
            joint_positions = self.pmtg_delta(actions)
        elif self.cfg["actionSpace"] == "pmtg":
            joint_positions = self.pmtg(actions)
        elif self.cfg["actionSpace"] == "joints":
            joint_positions = self.unnormalize_actions(actions)
        elif self.cfg["actionSpace"] == "foot_positions":
            joint_positions = self.foot_positions_action(actions)
        elif self.cfg["actionSpace"] == "foot_positions_delta":
            joint_positions = self.foot_positions_delta_action(actions)
        elif self.cfg["actionSpace"] == "high_level":
            joint_positions = self.low_level_policy.get_action(actions)
        else:
            raise ValueError("Invalid action space")

        # joint_positions = self._test_ik()
        # joint_positions = self._visual_test_ik()
        # joint_positions = self.test_pmtg()
        # joint_positions = self._tune_pd()
        # joint_positions = self.test_expert()
        _joint_positions = unwr(joint_positions)
        self.gym.set_dof_position_target_tensor(self.sim, _joint_positions)

        if "perturbations" in self.cfg.keys():

            forces = torch.zeros(self.num_rbs, 3, device=self.device)
            env_view = forces[self.rb_start_idx:].view(self.num_envs, 17, 3)
            aliengo_rb_forces = torch.zeros(17, 3, device=self.device)
            torso_force = torch.zeros(3, device=self.device)
            i = torch.randint(0, 5, (1,))
            if i == 0:
                torso_force[0] = -1000.0
            elif i == 1:
                torso_force[0] = 1000.0
            elif i == 2:
                torso_force[1] = -1000.0
            elif i == 3:
                torso_force[1] = 1000.0
            aliengo_rb_forces[0] = torso_force
            env_view[self.progress_buf % self.args.add_perturb == 0] = aliengo_rb_forces
            forces_ = unwr(forces)
            self.gym.apply_rigid_body_force_tensors(self.sim, forces_, None,
                                                    g.LOCAL_SPACE)

        # if self.progress_buf[0] > 5:
        #     joint_positions = self.expert_policy()

    def test_expert(self):
        try:
            print((self.expert_global_pos[0] - self.foot_center_pos[0]).norm(dim=1).mean().item())
        except AttributeError:
            pass

        expert_global_pos = self.expert_policy.step_in_place_traj()
        self.expert_global_pos = expert_global_pos
        foot_frame_expert_pos = self.get_foot_frame_foot_positions(
            foot_center_pos=expert_global_pos)
        # breakpoint()
        return self.analytic_inv_kinematics(foot_frame_expert_pos)

    def expert_policy(self):
        # return self.reset_joint_angles.tile((self.num_envs, 1))
        pos = self.get_foot_frame_foot_positions().clone()
        # breakpoint()
        # pos[..., 2] -= self.foot_collision_sphere_r
        # pos[:, [0, 1], 2] -= 0.0125
        # pos[:, [2, 3], 2] -= 0.005
        # pos[:, 2, 2] += 0.001
        joint_pos = self.analytic_inv_kinematics(pos)
        # assert torch.allclose(joint_pos, self.joint_positions, atol=1e-4, rtol=0.0), breakpoint()
        joint_pos = self.joint_positions.clone()
        return joint_pos

    def _tune_pd(self):
        temp = self.reset_joint_angles.tile((self.num_envs, 1))
        if (self.wandb_log_counter // 20) % 2 == 0:
            temp[:, self.thigh_joint_idcs] += .2
        else:
            temp[:, self.thigh_joint_idcs] -= .2
        return temp

    def check_if_hit_target(self, contact_force=5, dist_thresh=0.05):
        try:
            dist = self.low_level_policy.positions.norm(dim=1)
        except AttributeError:
            return
        # first check if current foot is in contact
        foot = self.foot_idcs[self.current_foot]
        in_contact = (self.foot_contact_forces[torch.arange(self.num_envs, device=self.device), foot] > contact_force).any(-1)
        # self.reached = torch.zeros(self.num_envs, device=self.device)  # for reward function
        env_idcs = torch.logical_and(in_contact, dist <= dist_thresh)
        # self.reached[env_idcs] = 1.0
        if self.args.play:
            if len(env_idcs) != 0:
                print("REACHED!!", torch.rand(1).item())
        self.current_foot[env_idcs] = (self.current_foot[env_idcs] + 1) % 4

    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        #    - call self.gym.create_sim
        #    - create ground plane
        #    - set up environments

        self.sim_params.up_axis = g.UP_AXIS_Z
        self.sim_params.gravity = g.Vec3(0.0, 0.0, self.gravity)
        self.sim_params.physx.contact_offset = 0.02  # default 0.02
        self.sim_params.substeps = 2  # default 2
        self.graphics_device_id = self.device_id
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def _create_envs(self):
        start = time.time()
        # asset_root = "../assets"
        # asset_file = "urdf/aliengo/urdf/aliengo.urdf"
        asset_options = g.AssetOptions()
        asset_options.fix_base_link = self.fix_base
        # asset_options.armature = 0.01

        asset = self.gym.load_asset(self.sim,
                                    self.cfg['asset']['assetRoot'],
                                    self.cfg['asset']['assetFileName'],
                                    asset_options)
        if self.is_stepping_stones:
            envs_per_row = self.cfg['stepping_stones']['num_rows']
            self.env_rows = -int(self.num_envs  # ceiling division
                       // -self.cfg['stepping_stones']['num_rows'])
            lower = g.Vec3(0.0, 0.0, 0.0)
            upper = g.Vec3(self.cfg['stepping_stones']['robot_x_spacing'],
                           self.cfg['stepping_stones']['robot_y_spacing'],
                           0.0)
            # self.env_offsets = torch.zeros(self.num_envs, 2,
            #                                device=self.device)
            # self.env_offsets[:, 0] = torch.arange(self.cfg['stepping_stones']['num_rows'],
            #     device=self.device).repeat(self.env_rows)[:self.num_envs] \
            #     * self.cfg['stepping_stones']['robot_x_spacing']

            # self.env_offsets[:, 1] = torch.arange(self.env_rows,
            #     device=self.device).repeat_interleave(self.cfg['stepping_stones']['num_rows'])[:self.num_envs] \
            #     * self.cfg['stepping_stones']['robot_y_spacing']

        elif self.is_rough_terrain_blocks:
            spacing = self.cfg['rough_terrain_blocks']['robot_spacing']
            envs_per_row = 1
            lower = g.Vec3(0.0, 0.0, 0.0)
            upper = g.Vec3(0.0, spacing, 0.0)
        else:
            spacing = self.cfg['envSpacing']
            envs_per_row = int(self.num_envs ** 0.5)
            lower = g.Vec3(0.0, 0.0, 0.0)
            upper = g.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.actor_handles = []
        self.cam_tensors = []
        pose = g.Transform()  # this doesn't matter, will be reset anyways
        # pose.p = g.Vec3(0.0, 0.0, self.start_height)
        # pose.r = g.Quat(0.0, 0.0, 0.0, 1.0)
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, envs_per_row)
            self.envs.append(env)
            if self.is_stepping_stones and i == 0:
                self.stepping_stones.create(self.gym, self.sim, self.envs[0])
            elif self.is_rough_terrain_blocks and i == 0:
                self.terrain.create(self.gym, self.sim, self.envs[0])
            aliengo_robot = self.gym.create_actor(env, asset, pose, 'aliengo', i, 0)
            dof_props = self.gym.get_actor_dof_properties(env, aliengo_robot)
            dof_props["driveMode"].fill(g.DOF_MODE_POS)
            dof_props["stiffness"].fill(self.cfg["joint_stiffness"])
            dof_props["damping"].fill(self.cfg["joint_damping"])
            # set P gain for knee joints a little bit higher
            # knee_pgain = self.cfg["joint_stiffness"] * 3.0
            # dof_props[8][6] = knee_pgain
            # dof_props[11][6] = knee_pgain
            # dof_props[2][6] = knee_pgain
            # dof_props[5][6] = knee_pgain
            self.gym.set_actor_dof_properties(env, aliengo_robot, dof_props)

            # link_props = self.gym.get_actor_rigid_shape_properties(env, aliengo_robot)
            # assert len(link_props) == 17
            # for i in range(17):
            #     link_props[i].friction = -1000.0  # 1.0 is the default.
            #     link_props[i].rolling_friction = -1000.0  # 0.0 is the default.
            # self.gym.set_actor_rigid_shape_properties(env, aliengo_robot, link_props)

            self.gym.enable_actor_dof_force_sensors(env, aliengo_robot)
            self.actor_handles.append(aliengo_robot)
            if "vision" in self.cfg:
                self._create_cameras(i)
            if self.args.save_images and i == 0:
                self._create_camera_for_video()
        print("elapsed time: {}".format(time.time() - start))  # 1.5 s for vectorized 100 envs

    def _create_camera_for_video(self):
        view = "back"
        view = "side"
        props = g.CameraProperties()
        # props.width = 960  # this is half of 1080 resolution
        # props.width = 660
        # breakpoint()
        # props.width = 1920 if view == "side" else 660
        props.width = int(1920 * 3/5 - 5) if self.cfg['plot_values'] else int(1920 * 2/5 - 5)
        props.height = 1080
        # props.enable_tensors = False
        props.enable_tensors = True

        self.video_camera = self.gym.create_camera_sensor(self.envs[0], props)

        # to fix camera behind the robot
        # x = 0.0 - 0.75
        # y = 0.0
        # z = 0.5
        # self.gym.set_camera_location(
        #     self.video_camera, self.envs[0], g.Vec3(x, y, z),
        #     g.Vec3(0.0, y, 0.0))

        # to have the camera follow the robot from a fixed distance
        local_transform = g.Transform()

        if view == "back":
            local_transform.p = g.Vec3(-0.75, 0.0, 0.0)
        elif view == "side":
            if self.cfg['plot_values']:
                y = 1.5
            else:
                y = 1.0
            local_transform.p = g.Vec3(0.1, y, 0.5)
            local_transform.r = g.Quat.from_euler_zyx(0.0, 15.0 * 3.14159 / 180, -90.0 * 3.14159 / 180)
        else:
            raise ValueError("Invalid view")

        self.gym.attach_camera_to_body(
            self.video_camera, self.envs[0], self.actor_handles[0],
            local_transform, g.FOLLOW_POSITION)

        cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.video_camera, g.IMAGE_COLOR)
        # cam_tensor = self.gym.get_camera_image(self.sim, self.envs[0], self.video_camera, g.IMAGE_COLOR)
        self.video_camera_image = wr(cam_tensor)

    def _create_cameras(self, i):
        # TODO is it possible to not see the robot with the camera??
        props = g.CameraProperties()
        props.width = self.cfg["vision"]["size"]
        props.height = self.cfg["vision"]["size"]
        props.enable_tensors = True
        cam_handle = self.gym.create_camera_sensor(self.envs[i], props)
        local_transform = g.Transform()
        local_transform.p = g.Vec3(0.3235, 0.0, 0.0)
        local_transform.r = g.Quat.from_axis_angle(g.Vec3(0, 1, 0), 40 * 3.14159/180)
        self.gym.attach_camera_to_body(cam_handle, self.envs[i], self.actor_handles[i], local_transform, g.FOLLOW_TRANSFORM)
        if self.cfg["vision"]["type"] == "depthmap":
            flag = g.IMAGE_DEPTH
        elif self.cfg["vision"]["type"] == "color":
            flag = g.IMAGE_COLOR
        else:
            raise NotImplementedError
        cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], cam_handle, flag)
        torch_cam_tensor = wr(cam_tensor)
        self.cam_tensors.append(torch_cam_tensor)

    def _create_ground_plane(self):
        plane_params = g.PlaneParams()
        plane_params.normal = g.Vec3(0, 0, 1)
        plane_params.distance = 0.0
        plane_params.static_friction = self.cfg["plane"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["plane"]["dynamicFriction"]
        plane_params.restitution = self.cfg["plane"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def foot_positions_action(self, actions):
        """Take normalized cartesian foot positions and output
        joint angles. Only deal in foot center positions."""
        actions = actions.view(self.num_envs, 4, 3)
        actions *= self.foot_pos_action_half_range
        actions += self.foot_pos_action_mean
        joint_positions = self.analytic_inv_kinematics(actions)
        return joint_positions

    def foot_positions_delta_action(self, actions):
        """Take normalized change in cartesian foot positions and output
        joint angles. Only deal in foot center positions in foot frame."""
        actions = actions.view(self.num_envs, 4, 3)
        actions *= self.foot_pos_action_bound
        joint_positions = self.analytic_inv_kinematics(actions + self.get_foot_frame_foot_positions())
        return joint_positions

    def test_pmtg(self):
        # self.gym.viewer_camera_look_at(
        #     self.viewer, self.envs[0], g.Vec3(0.0, 3.0, 0.8),
        #     g.Vec3(0, 0, 0.2))
        actions = torch.zeros(self.num_envs, 15, device=self.device)
        actions[:, 0] = 1.0  # amplitude
        actions[:, 1] = 0.0  # stand height
        actions[:, 2] = 1.0  # frequency
        # time.sleep(0.1)
        return self.pmtg(actions)

    def pmtg_delta(self, actions):
        """First 3 actions are change in amplitude, standing_height, frequency.
        Remaining 12 actions are foot position residuals.
        """
        actions *= self.pmtg_action_bound
        self.tg_state += actions[:, :3]
        self.tg_state = torch.maximum(torch.minimum(self.tg_state,
            self.tg_state_ub), self.tg_state_lb)
        amplitude = self.tg_state[:, 0:1]
        standing_depth = self.tg_state[:, 1:2]

        foot_pos = torch.zeros(self.num_envs, 4, 3, device=self.device)
        phase_offset = torch.tensor(self.cfg["pmtg"]["phase_offset"], device=self.device)
        y_offset = 0.075
        # x_offset = -0.02109375
        x_offset = 0.0

        foot_pos[:, :, 0] = amplitude * self.x_traj_fn(
            torch.ones(self.num_envs, 4).to(self.device)
            * self.t.unsqueeze(-1) + phase_offset)

        foot_pos[:, :, 2] = self.cfg["pmtg"]["step_height"] * self.z_traj_fn(
            torch.ones(self.num_envs, 4, device=self.device)
            * self.t.unsqueeze(-1) + phase_offset + 0.5) + standing_depth

        foot_pos[:, :, 1] += torch.tensor(
            [1.0, -1.0, 1.0, -1.0], device=self.device) * y_offset
        foot_pos[:, :, 1] += x_offset

        foot_pos += actions[:, 3:].view(self.num_envs, 4, 3)
        # foot_pos = torch.tensor([[0, 0, -0.35]] * 4).tile(self.num_envs, 1 , 1).to(self.device)
        joint_positions = self.analytic_inv_kinematics(foot_pos)
        self.t += 1/60.0 * self.tg_state[:, 2]
        self.pmtg_freq = self.tg_state[:, 2]  # this is used in normalizing contact and foot lift rewards later
        return joint_positions

    def pmtg(self, actions):
        """First 3 actions are change in amplitude, standing_height, frequency.
        Remaining 12 actions are foot position residuals.
        """
        actions = actions * self.tg_action_half_range + self.tg_action_mean
        amplitude = actions[:, 0:1]
        standing_depth = actions[:, 1:2]

        foot_pos = torch.zeros(self.num_envs, 4, 3, device=self.device)
        phase_offset = torch.tensor(self.cfg["pmtg"]["phase_offset"], device=self.device)
        y_offset = 0.075
        # x_offset = -0.02109375
        x_offset = 0.0

        foot_pos[:, :, 0] = amplitude * self.x_traj_fn(
            torch.ones(self.num_envs, 4).to(self.device)
            * self.t.unsqueeze(-1) + phase_offset)

        foot_pos[:, :, 2] = self.cfg["pmtg"]["step_height"] * self.z_traj_fn(
            torch.ones(self.num_envs, 4, device=self.device)
            * self.t.unsqueeze(-1) + phase_offset + 0.5) + standing_depth

        foot_pos[:, :, 1] += torch.tensor(
            [1.0, -1.0, 1.0, -1.0], device=self.device) * y_offset
        foot_pos[:, :, 1] += x_offset

        foot_pos += actions[:, 3:].view(self.num_envs, 4, 3)
        # foot_pos = torch.tensor([[0, 0, -0.35]] * 4).tile(self.num_envs, 1 , 1).to(self.device)
        joint_positions = self.analytic_inv_kinematics(foot_pos)
        self.t += 1/60.0 * actions[:, 2]
        self.pmtg_freq = actions[:, 2]  # this is used in normalizing contact and foot lift rewards later
        return joint_positions

    def _visual_test_ik(self):
        """Oscillates between limits of foot position space. """
        try:
            self._visual_test_ik_counter += 1
        except AttributeError:
            self._visual_test_ik_counter = 0
        foot_pos = torch.ones(self.num_envs, 4, 3, device=self.device)
        # foot_pos[:, :, 0] = 0.0
        foot_pos[:, :, 1] = 0.0
        foot_pos[:, :, 2] = 0.5
        foot_pos *= torch.sin(
            torch.tensor([self._visual_test_ik_counter],
                         device=self.device) * 0.1)
        # foot_pos *= 0.1

        lb = self.ik_foot_center_pos_lb
        ub = self.ik_foot_center_pos_ub
        foot_pos *= (ub - lb) / 2.0
        foot_pos += (ub + lb) / 2.0
        return self.analytic_inv_kinematics(foot_pos)

    def _test_ik(self):
        """Set entire range of foot positions in action space,
        wait for positions to be hit, then check error. Also sets
        base to positions and orientations to rule out that as a factor.
        """

        assert self.fix_base
        assert self.gravity == 0.0
        wait = 100
        try:
            self._test_ik_counter += 1
        except AttributeError:
            self._test_ik_counter = 0
        if self._test_ik_counter % wait == 0:
            # check the error of the previous setting
            try:
                diff = self.get_foot_frame_foot_positions() - self._test_ik_foot_pos
                max_error = diff.abs().max()
                print("Max error is {}".format(max_error.item()))
                breakpoint()
            except AttributeError:
                pass
            finally:
                # set random position and orientation of base
                random_quats = 0.025 * (torch.rand(
                    self.num_envs, 3, device=self.device) - 0.5)
                real_part = (1.0 - random_quats.norm(
                    dim=1)).sqrt().unsqueeze(-1)
                random_quats = torch.cat((random_quats, real_part), dim=1)
                random_positions = torch.rand(self.num_envs, 3,
                                              device=self.device) + 1
                vel = torch.zeros(self.num_envs, 6, device=self.device)
                random_root_state = torch.cat((random_positions,
                                               random_quats,
                                               vel), dim=1)
                _random_root_state = unwr(random_root_state)
                self.gym.set_actor_root_state_tensor(
                    self.sim, _random_root_state)

                # generate random end_effector positions
                self._test_ik_foot_pos = 2.0 * (torch.rand(
                    self.num_envs, 4, 3, device=self.device) - 0.5)
                self._test_ik_foot_pos *= 0.25
                lb = self.ik_foot_center_pos_lb
                ub = self.ik_foot_center_pos_ub
                self._test_ik_foot_pos *= (ub - lb) / 2.0
                self._test_ik_foot_pos += (ub + lb) / 2.0
                self._test_ik_joint_positions = self.analytic_inv_kinematics(
                    self._test_ik_foot_pos)
        return self._test_ik_joint_positions

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.old_reset_buf = self.reset_buf.clone()
        self.old_progress_buf = self.progress_buf.clone()
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.update_tensors()
        if "vision" in self.cfg:
            if self.args.save_images:
                raise NotImplementedError  # I haven't used vision while also using the video camera
            self.update_graphics()
        self.obs_buf[:] = self.observe()
        self.terminations()
        self.compute_reward()
        if self.args.save_images:
            self.update_video_camera_graphics()
        if self.args.gather_stats > 0:
            self.gather_stats(self.rew_dict)
        if self.args.plot_contact_locations:
            self.plot_footsteps()

    def update_video_camera_graphics(self):
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        # if self.progress_buf[0] >= self.args.start_after:
        self.gym.start_access_image_tensors(self.sim)

        # save images
        img_dir = self.img_dir

        fname = os.path.join(
            img_dir,
            f"frame-{self.progress_buf[0]:04}.png")
        cam_img = self.video_camera_image.cpu().numpy()
        imageio.imwrite(fname, cam_img)
        self.gym.end_access_image_tensors(self.sim)
        # if (self.progress_buf[0] >= self.args.exit_after
        #         or (self.reset_buf[0]
        #             and self.progress_buf[0] > self.args.start_after)):

        # if plot values, the value optimization code will handle exit due to
        # need to clean up multiprocessing of plot saving
        if not self.cfg["plot_values"] and self.reset_buf[0] and self.progress_buf[0] > 10:
            os.sys.exit()

    def update_graphics(self):
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.raw_cam_obs = torch.stack(self.cam_tensors)
        self.gym.end_access_image_tensors(self.sim)
        # convert to greyscale, also ignore alpha
        self.cam_obs = (0.299 * self.raw_cam_obs[:, :, :, 0]
                        + 0.587 * self.raw_cam_obs[:, :, :, 1]
                        + 0.114 * self.raw_cam_obs[:, :, :, 2])
        # # save images
        # self.gym.viewer_camera_look_at(
        #     self.viewer, self.envs[0], g.Vec3(-2.0, 0.0, 11.0),
        #     g.Vec3(0, 0, 10.0))
        # img_dir = "test_imgs"
        # import os
        # import imageio
        # if not os.path.exists(img_dir):
        #     os.mkdir(img_dir)
        # fname = os.path.join(img_dir, "cam-1.png")
        # cam_img = self.cam_obs[0].cpu().numpy()
        # imageio.imwrite(fname, cam_img)
        # breakpoint()

        # normalize images
        self.cam_obs -= 128.0
        self.cam_obs /= 128.0
        # flatten() actually copies the tensor due to exclusion of alpha (no longer contiguous)
        self.cam_obs = self.cam_obs.flatten(start_dim=1, end_dim=-1)

    def contact_rew(self):
        assert self.cfg["actionSpace"] == {"pmtg", "pmtg_delta"}
        phase_offset = torch.tensor(self.cfg["pmtg"]["phase_offset"], device=self.device)

        z_traj_input = (torch.ones(self.num_envs, 4, device=self.device)
                        * self.t.unsqueeze(-1) + phase_offset + 0.5)
        norm_phases = (z_traj_input % 1.0) * 4 - 2.0
        should_be_contact = norm_phases < 0.0
        is_contact = (self.foot_contact_forces > 0.0).any(-1)
        reward = torch.logical_and(should_be_contact, is_contact).sum(dim=1).float()
        reward *= self.pmtg_freq
        return reward

    def foot_lift_rew(self):
        """
        This has essentially become a reward for hitting the foot lift
        stage of the pmtg. (i.e. encourage the robot to actually move).
        The foot lift height threshold reward doesn't seem to work well on
        the varied-height stepping stone environment.
        """
        lift_height = 0.0
        if self.is_rough_terrain_blocks:
            raise NotImplementedError()
        elif self.is_stepping_stones:
            lift_height += self.cfg['stepping_stones']['height']
        assert self.cfg["actionSpace"] in {"pmtg", "pmtg_delta"}
        phase_offset = torch.tensor(self.cfg["pmtg"]["phase_offset"], device=self.device)

        z_traj_input = (torch.ones(self.num_envs, 4, device=self.device)
                        * self.t.unsqueeze(-1) + phase_offset + 0.5)
        norm_phases = (z_traj_input % 1.0) * 4 - 2.0
        should_be_lift = torch.logical_and(0.75 <= norm_phases, norm_phases <= 1.25)
        is_lift = self.foot_center_pos.view(self.num_envs, 4, 3)[:, :, 2] > lift_height + self.foot_collision_sphere_r

        is_lift[:] = True  # NOTE

        reward = torch.logical_and(should_be_lift, is_lift).sum(dim=1).float()
        reward *= self.pmtg_freq
        return reward

    def update_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.jac = self.full_jac[:, self.jac_feet_idcs, :3, -12:]  # I am only interested in foot position (not orientation)

        # try block is for when this method is called in the constructor and
        # the cloned vectors don't exist yet
        try:
            # self.prev_prev_foot_center_pos = self.prev_foot_center_pos.clone()
            self.prev_base_vel = self.cur_base_vel.clone()
            self.cur_base_vel = self.base_vel.clone()
            self.prev_foot_vel = self.foot_vel.clone()
            self.prev_foot_center_pos = self.foot_center_pos.clone()
            self.prev_foot_z_contact_forces = \
                self.foot_contact_forces[:, :, 2].clone()
        except AttributeError:
            pass

        self.foot_center_pos = self.rb_states[:, self.feet_idcs, :3]
        self.foot_vel = self.rb_states[:, self.feet_idcs, 7:10]
        self.hip_pos = self.rb_states[:, self.hip_idcs, :3]
        self.thigh_pos = self.rb_states[:, self.thigh_idcs, :3]
        self.knee_pos = self.rb_states[:, self.knee_idcs, :3]

        self.foot_contact_forces = self.contact_forces[:, self.feet_idcs]
        self.body_contact_forces = self.contact_forces[:, self.non_feet_idcs]
        self.base_euler = batch_quat_to_euler(self.base_quat)
        # self.check_if_hit_target()

        if self.is_footsteps or "ss_footstep_target_parameters" in self.cfg:
            self.footstep_generator.update()

    def compute_reward(self):
        self.rew_buf[:], self.rew_dict = self.reward()
        self.wandb_log_counter += 1
        if self.wandb_log_counter % self.cfg["steps_num"] == 0:
            self.epochs += 1
        if self.epochs % self.cfg["wandb_log_interval"] == 0:
            mean_rew_dict = {}
            for key, value in self.rew_dict.items():
                mean_rew_dict[key] = value.mean()
            mean_rew_dict.update({
                'frame': self.wandb_log_counter * self.num_envs,
                'epochs': self.epochs})
            wandb.log(mean_rew_dict)
        return

    def get_slip_penalty(self):
        """Return a penalty of -1.0 for each foot that moves while in
        contact.
        """
        eps = 0.03  # this has to allow rolling still
        feet_of_concern = torch.logical_and(
            (self.prev_foot_z_contact_forces > 0.0),
            (self.foot_contact_forces[..., 2] > 0.0))
        moved = ((self.foot_center_pos[:, :, :2]
                  - self.prev_foot_center_pos[:, :, :2]).abs() > eps).any(-1)
        bad_feet = torch.logical_and(moved, feet_of_concern)
        return bad_feet.float().sum(-1)

    def reset_last_action_buffers(self, env_ids):
        if self.cfg["actionSpace"] in {"pmtg_delta", "foot_positions_delta"}:
            # pmtg_delta action space is all residuals, so set all to zero
            # tg_state is reset in the self.reset() fn below
            self.last_action[env_ids] = 0.0
            self.last_last_action[env_ids] = 0.0
            self.last_last_last_action[env_ids] = 0.0

        elif self.cfg["actionSpace"] == "pmtg":
            # pmtg action space:
            # amplitude, standing_height, frequency + 12 * residuals
            self.last_action[env_ids] = self.pmtg_reset_action
            self.last_last_action[env_ids] = self.pmtg_reset_action
            self.last_last_last_action[env_ids] = self.pmtg_reset_action

        elif self.cfg["actionSpace"] == "foot_positions":
            # action space is foot_frame_foot_positions(), so set all
            # to the starting foot frame foot position
            self.last_action[env_ids] = self.foot_pos_action_reset_action
            self.last_last_action[env_ids] = self.foot_pos_action_reset_action
            self.last_last_last_action[env_ids] = self.foot_pos_action_reset_action

        elif self.cfg["actionSpace"] == "joints":
            raise NotImplementedError()
        elif self.cfg["actionSpace"] == "high_level":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid action space")

    def reset(self, env_ids):
        self.is_successful[env_ids] = True
        self.reset_last_action_buffers(env_ids)

        self.prev_foot_center_pos[env_ids] = self.starting_foot_pos.clone()
        self.prev_foot_vel[env_ids] = 0.0
        self.prev_base_vel[env_ids] = 0.0
        self.cur_base_vel[env_ids] = 0.0
        # self.prev_prev_foot_center_pos[env_ids] = self.starting_foot_pos.clone()
        # Set them all to zero so that we can't get penalized for slipping
        # i.e. 0.0 contact force means not in contact
        self.prev_foot_z_contact_forces[env_ids] = torch.zeros(4,
            device=self.device)

        self.joint_positions[env_ids, :] = self.reset_joint_angles.tile((len(env_ids), 1))
        self.joint_velocities[env_ids, :] = torch.zeros((len(env_ids), 12), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.base_pos[env_ids, :] = torch.tensor([0.0, 0.0, self.start_height], device=self.device).tile((len(env_ids), 1))
        # self.base_quat[env_ids, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).tile((len(env_ids), 1))
        # self.base_quat[env_ids, :] = torch.tensor([0.0, 0.0, 0.0871556693173223, 0.9961947045160647], device=self.device).tile((len(env_ids), 1))  # 10 deg z
        # self.base_quat[env_ids, :] = torch.tensor([0.0, 0.0871556693173223, 0.0, 0.9961947045160647], device=self.device).tile((len(env_ids), 1))  # 10 deg y rot
        # self.base_quat[env_ids, :] = torch.tensor([0.0871556693173223, 0.0, 0.0, 0.9961947045160647], device=self.device).tile((len(env_ids), 1))  # 10 deg X rot
        self.base_quat[env_ids, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).tile((len(env_ids), 1))
        self.base_vel[env_ids, :] = torch.zeros((len(env_ids), 3), device=self.device)
        self.base_avel[env_ids, :] = torch.zeros((len(env_ids), 3), device=self.device)
        if self.is_stepping_stones:
            # these need to become actor indices to be used to reset the robots
            env_ids_int32 += self.stepping_stones.num_stones
        elif self.is_rough_terrain_blocks:
            # these need to become actor indices to be used to reset the robots
            env_ids_int32 += self.terrain.num_blocks
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            unwr(self.all_root_tensor),
            unwr(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            unwr(self.dof_state),
            unwr(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            unwr(self.joint_positions.clone()),
            unwr(env_ids_int32), len(env_ids_int32))

        if hasattr(self, "footstep_generator"):
            self.footstep_generator.reset(env_ids)
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        if self.cfg["actionSpace"] == "pmtg_delta":
            self.t[env_ids] = 0.0
            self.tg_state[env_ids] = torch.tensor(self.start_tg_state,
                device=self.device)
        elif self.cfg["actionSpace"] == "pmtg":
            self.t[env_ids] = 0.0
        if self.args.plot_contact_locations:
            self.plot_footsteps.reset()

    def unnormalize_actions(self, actions):
        position_mean = (self.joint_angle_ub + self.joint_angle_lb)/2
        position_range = self.joint_angle_ub - self.joint_angle_lb
        unnormalized_action = actions * (position_range * 0.5) + position_mean
        # unnormalized_action = torch.Tensor(
        #     [0.037199, 0.660252, -1.200187, -0.028954, 0.618814, -1.183148,
        #      0.048225, 0.690008, -1.254787, -0.050525, 0.661355, -1.243304]).to("cuda:0").tile(10).unsqueeze(-1)
        return unnormalized_action

    def get_foot_frame_foot_positions(self, foot_center_pos=None):
        # TODO make more efficient by only doing 2D rotation matrix?
        if foot_center_pos is None:
            foot_center_pos = self.foot_center_pos
        hip_centered_pos = (foot_center_pos - self.hip_pos)
        yaw = batch_quat_to_euler(self.base_quat)[:, 2]
        rot_mat = batch_z_rot_mat(-yaw)
        foot_frame_foot_pos = (rot_mat.unsqueeze(1)
                               @ hip_centered_pos.unsqueeze(-1)).squeeze(-1)
        foot_frame_foot_pos[:, :, 2] -= self.foot_collision_sphere_r
        return foot_frame_foot_pos

    def get_foot_frame_foot_velocities(self):
        yaw = batch_quat_to_euler(self.base_quat)[:, 2]
        rot_mat = batch_z_rot_mat(-yaw)
        foot_frame_foot_vel = (rot_mat.unsqueeze(1)
                               @ self.foot_vel.unsqueeze(-1)).squeeze(-1)
        return foot_frame_foot_vel

    def foot_frame_pos_to_global(self, foot_frame_foot_pos):
        foot_frame_foot_pos[:, :, 2] += self.foot_collision_sphere_r
        yaw = batch_quat_to_euler(self.base_quat)[:, 2]
        rot_mat = batch_z_rot_mat(yaw)
        global_foot_pos = (rot_mat.unsqueeze(1)
                           @ foot_frame_foot_pos.unsqueeze(-1)).squeeze(-1)
        global_foot_pos += self.hip_pos
        return global_foot_pos

    # def _test_numerical_ik(self):
    #     joint_angles = self.numerical_ik(None)
    #     self.gym.viewer_camera_look_at(self.viewer, self.envs[0], g.Vec3(0.0, 1.0, 0.8), g.Vec3(0, 0, 0.8))
    #     pi = 3.14159265358979
    #     actions = (torch.rand(self.num_envs, 12).to(self.device) - 0.5) * 2.0
    #     joint_positions = self.unnormalize_actions(actions)
    #     # joint_positions[:, self.hip_joint_idcs] = 0.0
    #     # joint_positions[:, self.thigh_joint_idcs] = 0.0
    #     # joint_positions[:, self.knee_joint_idcs] = -pi/2
    #     # joint_positions[:, :3] = joint_angles
    #     return joint_positions

    # def numerical_ik(self, desired_foot_positions):

    #     _, _, foot_center_pos = self.fwd_kin_for_jac(self.joint_positions[0, 0], self.joint_positions[0, 1], self.joint_positions[0, 2])

    #     actual_pos = self.get_foot_frame_foot_positions()
    #     actual_pos[..., 2] += self.foot_collision_sphere_r
    #     error = (foot_center_pos - actual_pos[0, 0]).abs().max()
    #     print("Foot position error {}".format(error))
    #     return
    #     desired_foot_positions[..., 2] += self.foot_collision_sphere_r
    #     current_joint_angles = self.joint_positions.clone()[..., :3]
    #     for _ in range(10):
    #         jacobian = self.calc_jacobian(current_joint_angles)
    #         # Do DLS
    #         d = 0.05  # damping term
    #         current_joint_angles += something
    #         lmbda = torch.eye(3).to(self.device) * (d ** 2)
    #     return current_joint_angles

    # def fwd_kin_for_jac(self, hip_angle, thigh_angle, knee_angle):
    #     """Take joint angles and return the positions of hip, thigh,
    #     and knee joints."""
    #     # NOTE just dealing with the FL leg for now.
    #     breakpoint()
    #     temp = batch_y_rot_mat(knee_angle) @ torch.tensor([0.0, 0.0, -0.25], device=self.device).unsqueeze(-1)
    #     temp = batch_y_rot_mat(thigh_angle) @ (temp + torch.tensor([0.0, 0.0, -0.25], device=self.device).unsqueeze(-1))
    #     foot_center_pos = (batch_x_rot_mat(hip_angle) @ (temp + torch.tensor([0.0, 0.083, 0.0], device=self.device).unsqueeze(-1))).squeeze(-1)
    #     thigh_pos = None
    #     knee_pos = None
    #     return thigh_pos, knee_pos, foot_center_pos

    # def dls_inverse_kinematics(self, target_foot_positions):
    #     """Takes target_foot_positions of shape (num_envs, 4, 3).
    #     Returns joint targets of shape (num_envs, 12).
    #     Note: This performs very poorly due to the inability to perform
    #     more than one iteration, because an updated jacobian in unavailalbe
    #     (unless I were to write my own function that calculates the jac
    #     given a vector of joint positions)
    #     """

    #     # j_eef_T = torch.transpose(j_eef, 1, 2)
    #     # d = 0.05  # damping term
    #     # lmbda = torch.eye(6).to(device) * (d ** 2)
    #     # u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 9, 1)
    #     jac_T = self.jac.transpose(-1, -2)
    #     # d = 0.05
    #     d = 0.0
    #     lmbda = torch.eye(3).to(self.device) * (d ** 2)
    #     pos_error = (target_foot_positions - self.foot_center_pos).unsqueeze(-1)
    #     delta_th = (jac_T @ (self.jac @ jac_T + lmbda).inverse() @ pos_error).sum(dim=1).squeeze(-1)
    #     return delta_th + self.joint_positions

    # def calc_jacobian(self, joint_positions):
    #     # I need to find the current axis of rotation of the knee joint as a unit vec (I can do this with)
    #     shin_vec = self.foot_center_pos[0, 0] - self.knee_pos[0, 0]
    #     thigh_vec = self.thigh_pos[0, 0] - self.knee_pos[0, 0]
    #     knee_joint_unit_vec = torch.cross(thigh_vec, shin_vec, dim=-1)
    #     knee_joint_unit_vec /= torch.linalg.norm(knee_joint_unit_vec)
    #     knee_jac = torch.cross(knee_joint_unit_vec, shin_vec)  # this should be the first column of the jac

    #     hip_joint_unit_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device)
    #     thigh_joint_unit_vec = torch.cross(hip_joint_unit_vec, -thigh_vec)
    #     thigh_joint_unit_vec /= thigh_joint_unit_vec.norm()
    #     thigh_jac = torch.cross(thigh_joint_unit_vec, self.foot_center_pos[0, 0] - self.thigh_pos[0, 0])  # second column of jac

    #     hip_jac = torch.cross(hip_joint_unit_vec, self.hip_pos[0, 0] - self.foot_center_pos[0, 0])

    #     FL_leg_jac = torch.stack((knee_jac, thigh_jac, hip_jac)).T

    #     return FL_leg_jac

    # def _test_fwd_kinematics(self):
    #     """Return foot frame foot positions calculated via fwd kinematics."""
    #     hip_angle = self.joint_positions[:, [0, 3, 6, 9]]
    #     thigh_angle = self.joint_positions[:, [1, 4, 7, 10]]
    #     knee_angle = self.joint_positions[:, [2, 5, 8, 11]]
    #     y = self.get_foot_frame_foot_positions()
    #     y[..., -1] += 0.0265

    #     x = batch_y_rot_mat(knee_angle) @ torch.tensor([0, 0, -0.25]).to(self.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
    #     # x = x.squeeze(-1) + self.knee_pos - self.hip_pos
    #     # breakpoint()
    #     # max_error = (x - y).abs().max()
    #     # print("fwd_kinematics max error: {:.4f}".format(max_error))
    #     # return
    #     x += torch.tensor([0, 0, -0.25]).to(self.device).unsqueeze(-1)
    #     x = batch_y_rot_mat(thigh_angle) @ x + torch.tensor([[0, 0.083, 0],
    #                                                          [0, -0.083, 0],
    #                                                          [0, 0.083, 0],
    #                                                          [0, -0.083, 0]]).to(self.device).unsqueeze(-1)
    #     x = (batch_x_rot_mat(hip_angle) @ x).squeeze(-1)
    #     max_error = (x - y).abs().max()
    #     if max_error > 0.01:
    #         breakpoint()
    #     print("fwd_kinematics max error: {:.2}".format(max_error))

    def analytic_fwd_kinematics(self, hip, thigh, knee):
        x = torch.zeros(self.num_envs, 4, 3, device=self.device)
        l3 = torch.tensor([-self.l3_val, self.l3_val, -self.l3_val, self.l3_val], device=self.device)

        temp = self.l1_val * torch.cos(knee) + self.l2_val
        temp2 = (-torch.sin(thigh) * self.l1_val * torch.sin(knee)
                 + torch.cos(thigh) * temp)

        x[:, :, 0] = (self.l1_val * torch.sin(knee) * torch.cos(thigh)
                      + torch.sin(thigh) * temp)
        x[:, :, 1] = l3 * torch.cos(hip) - torch.sin(hip) * temp2
        x[:, :, 2] = l3 * torch.sin(hip) + torch.cos(hip) * temp2 - self.foot_collision_sphere_r
        return x

    # def _test_analytic_fwd_kinematics(self):
    #     hip = self.joint_positions[:, [0, 3, 6, 9]]
    #     thigh = self.joint_positions[:, [1, 4, 7, 10]]
    #     knee = self.joint_positions[:, [2, 5, 8, 11]]
    #     y = self.get_foot_frame_foot_positions()
    #     y[..., -1] += 0.0265
    #     x = torch.zeros_like(y)
    #     l1 = -0.25
    #     l2 = -0.25
    #     l3 = -0.083
    #     l3 = torch.tensor([-l3, l3, -l3, l3]).to(self.device)

    #     temp = l1 * torch.cos(knee) + l2
    #     temp2 = (-torch.sin(thigh) * l1 * torch.sin(knee)
    #              + torch.cos(thigh) * temp)

    #     x[:, :, 0] = (l1 * torch.sin(knee) * torch.cos(thigh)
    #                   + torch.sin(thigh) * temp)
    #     x[:, :, 1] = l3 * torch.cos(hip) - torch.sin(hip) * temp2
    #     x[:, :, 2] = l3 * torch.sin(hip) + torch.cos(hip) * temp2

    #     max_error = (x - y).abs().max()
    #     if max_error > 0.01:
    #         breakpoint()
    #     print("fwd_kinematics max error: {:.4}".format(max_error))

    # def _test_ik(self):
    #     """Test my forward and inverse kinematics implementation.
    #     Generate foot positions"""
    #     pass

    def analytic_inv_kinematics(self, foot_pos):
        """Take desired foot positions (in foot_frame) and output
        joint positions.
        Accounts for out-of-bound desired foot positions.
        Foot order: FL, FR, RL, RR
        NOTE: Right and left are switched from PyBullet! (for some reason)
        """
        # I don't care about foot positions above the robot's hip position.

        pi = 3.1415927
        foot_pos = foot_pos.clone()
        foot_pos[..., 2] += self.foot_collision_sphere_r

        # rotate from foot_frame to whole robot frame (ie normalize for
        # robot orienation)
        # breakpoint()
        euler = batch_quat_to_euler(self.base_quat)
        pitch_rot_mat = batch_y_rot_mat(-euler[:, 1])
        roll_rot_mat = batch_x_rot_mat(-euler[:, 0])
        foot_pos = pitch_rot_mat.unsqueeze(1) @ foot_pos.unsqueeze(-1)
        foot_pos = (roll_rot_mat.unsqueeze(1) @ foot_pos).squeeze(-1)

        # foot height clipping
        highest_z = -0.1
        idcs = foot_pos[:, :, 2] > highest_z
        foot_pos[:, :, 2][idcs] = highest_z

        # norm clipping
        eps = 1e-3
        norms = foot_pos.norm(dim=2)
        max_norm = (0.5**2 + 0.083**2)**0.5 - eps
        min_norm = 0.083 + eps
        idcs = norms > max_norm
        foot_pos[idcs.unsqueeze(-1).tile(1, 1, 3)] *= (max_norm/norms[idcs]).repeat(3)  # TODO
        idcs = norms < min_norm
        foot_pos[idcs.unsqueeze(-1).tile(1, 1, 3)] *= (min_norm/norms[idcs]).repeat(3)

        l3 = torch.tensor([0.083], device=self.device)
        l1 = torch.tensor([0.25], device=self.device)
        knee = (1.0 - (torch.linalg.norm(foot_pos, dim=-1).square() - l3.square())
                / (2.0 * l1.square())).clamp(-1.0, 1.0).acos() - pi

        # actual_knee = self.joint_positions[:, self.knee_joint_idcs]
        # knee_error = (knee - actual_knee).abs().max()
        # print(knee_error)
        # if knee_error > 1e-3:
        #     breakpoint()

        lol = (((-knee).cos() * l1 + l1).square()
               + (l1 * (-knee).sin()).square()).sqrt()
        lol2 = ((-knee).cos() * l1 + l1)
        thigh = (foot_pos[:, :, 0] / lol).clamp(-1.0, 1.0).acos()
        idcs = foot_pos[:, :, 2] > 0.0
        thigh[idcs] *= -1.0
        thigh -= pi/2
        correction = torch.atan2(l1 * (-knee).sin(), lol2)
        calc_thigh = thigh + correction

        # actual_thigh = self.joint_positions[:, self.thigh_joint_idcs]
        # thigh_error = (calc_thigh - actual_thigh).abs().max()
        # print(thigh_error)
        # if thigh_error > 1e-2:  # NOTE: will exceed 1e-3 in error when argument to acos is near -1 or 1, since derivative of acos is near inf, floating point error is amplified
        #     breakpoint()
        proj_thigh_shin_len = (lol * thigh.cos()).abs()
        calc_hip = (torch.atan2(proj_thigh_shin_len, torch.tensor([0.083, -0.083, 0.083, -0.083], device=self.device))
                    + torch.atan2(foot_pos[..., 2], foot_pos[..., 1]))

        # actual_hip = self.joint_positions[:, self.hip_joint_idcs].clone()
        # hip_error = (calc_hip - actual_hip).abs().max()
        # print(hip_error)
        # if hip_error > 1e-2:
        #     breakpoint()

        output = torch.zeros(self.num_envs, 12, device=self.device)
        output[:, self.hip_joint_idcs] = calc_hip
        output[:, self.thigh_joint_idcs] = calc_thigh
        output[:, self.knee_joint_idcs] = knee
        return output


if __name__ == "__main__":
    q = torch.rand(int(1e5), 4).to('cuda:0')
    for _ in range(10):
        s = time.time()
        y = batch_quat_to_6d(q)
        e = time.time()
        print(e - s)
