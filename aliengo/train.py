import argparse
import os
import time
import yaml



from isaacgym import gymapi as g
from isaacgym import gymtorch
from isaacgym.gymtorch import wrap_tensor as wr
from isaacgym.gymtorch import unwrap_tensor as unwr
import numpy as np
import torch

from ars_policy import ARSPolicy

# TODO don't keep recreating tensors that I use for resets, for example.
# TODO use torch jit scripts for additional speedup

def create_sim(n_envs, render):
    gym = g.acquire_gym()
    sim_params = g.SimParams()
    sim_params.up_axis = g.UP_AXIS_Z
    sim_params.gravity = g.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True
    compute_device_id = 0
    graphics_device_id = 0
    sim = gym.create_sim(compute_device_id, graphics_device_id, g.SIM_PHYSX,
                         sim_params)
    plane_params = g.PlaneParams()
    plane_params.normal = g.Vec3(0, 0, 1)
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0

    gym.add_ground(sim, plane_params)

    asset_root = "../assets"
    asset_file = "urdf/aliengo/urdf/aliengo.urdf"

    asset_options = g.AssetOptions()
    asset_options.fix_base_link = False
    # asset_options.armature = 0.01

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    spacing = 1.0
    envs_per_row = 10
    lower = g.Vec3(0.0, 0.0, 0.0)
    upper = g.Vec3(spacing, spacing, spacing)
    envs = []
    actor_handles = []
    for i in range(n_envs):
        env = gym.create_env(sim, lower, upper, envs_per_row)
        envs.append(env)

        pose = g.Transform()  # this doesn't matter, will be reset anyways
        pose.p = g.Vec3(0.0, 0.0, 0.48)
        pose.r = g.Quat(0.0, 0.0, 0.0, 1.0)

        aliengo_robot = gym.create_actor(env, asset, pose, '_', i, 1)
        props = gym.get_actor_dof_properties(env, aliengo_robot)
        props["driveMode"].fill(g.DOF_MODE_POS)
        # props["stiffness"].fill(1000.0)
        # props["damping"].fill(200.0)
        props["stiffness"].fill(1.0 * 1000)
        props["damping"].fill(0.1 * 1000)
        gym.set_actor_dof_properties(env, aliengo_robot, props)
        actor_handles.append(aliengo_robot)

    if render:
        cam_props = g.CameraProperties()
        viewer = gym.create_viewer(sim, cam_props)
    else:
        viewer = None
    gym.prepare_sim(sim)
    _root_tensor = gym.acquire_actor_root_state_tensor(sim)
    _dof_states = gym.acquire_dof_state_tensor(sim)
    root_tensor = wr(_root_tensor)
    dof_states = wr(_dof_states)
    return gym, sim, viewer, _root_tensor, root_tensor, _dof_states, dof_states, envs, actor_handles


def unnormalize_actions(actions, n_envs):
    actions = actions.clamp(-1.0, 1.0)
    position_lb = torch.tensor([-1.22173048, -1.570795  , -2.77507351, -1.22173048, -1.570795  ,
       -2.77507351, -1.22173048, -1.570795  , -2.77507351, -1.22173048,
       -1.570795  , -2.77507351], device="cuda:0")

    position_ub = torch.tensor([ 1.22173048,  1.570795  , -0.64577182,  1.22173048,  1.570795  ,
       -0.64577182,  1.22173048,  1.570795  , -0.64577182,  1.22173048,
        1.570795  , -0.64577182], device="cuda:0")
    position_mean = (position_ub + position_lb)/2
    position_range = position_ub - position_lb
    unnormalized_action = (actions.view((n_envs, 12)) * (position_range * 0.5) + position_mean).flatten()
    # unnormalized_action = torch.Tensor(
    #     [0.037199, 0.660252, -1.200187, -0.028954, 0.618814, -1.183148,
    #      0.048225, 0.690008, -1.254787, -0.050525, 0.661355, -1.243304]).to("cuda:0").tile(10).unsqueeze(-1)
    return unnormalized_action


def run_simulation(gym, sim, viewer, _root_tensor, root_tensor, _dof_states, dof_states, policy, n_envs, params, envs, actor_handles, eval=False):
    reset_sim(gym, sim, n_envs, _root_tensor, root_tensor, _dof_states, dof_states, envs, actor_handles, viewer)
    if not eval:
        policy.generate_candidates()
    rewards = torch.zeros(n_envs, device="cuda:0")
    old_terminations = torch.zeros(n_envs, device="cuda:0")
    for i in range(params['steps_per_eps']):
        # act
        obs = get_observation(root_tensor, dof_states, n_envs)
        if not eval:
            actions = policy.query_candidates(obs)
        else:
            actions = policy(obs)
        normalized_actions = unnormalize_actions(actions, n_envs)
        _normalized_actions = unwr(normalized_actions)
        gym.set_dof_position_target_tensor(sim, _normalized_actions)

        # run simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)

        # check termination
        termination_mask = get_termination_mask(root_tensor, dof_states, old_terminations)
        if torch.all(torch.logical_not(termination_mask)):
            break
        # get reward
        rewards += reward(root_tensor, dof_states, n_envs) * termination_mask

        if eval and viewer is not None:
        # if True and viewer is not None:
            idcs = torch.nonzero(torch.logical_not(termination_mask)).squeeze(-1)
            for j in range(len(idcs)):
                gym.set_rigid_body_color(envs[idcs[j]], actor_handles[idcs[j]], 0, g.MeshType.MESH_VISUAL, g.Vec3(0.0, 0.0, 0.0))
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    if eval:
        # print("Avg rew per step (m/s): {:.02f}".format(rewards.mean() / params['steps_per_eps']))
        print("Avg rew: {:.02f}".format(rewards.mean()) + "#" * 100)
    return rewards


def get_termination_mask(root_tensor, dof_states, old_terminations):
    """Returns True if environment is terminated."""
    height_term = torch.logical_or(root_tensor[:, 2] > 0.8, root_tensor[:, 2] < 0.3)
    attitude_term = (batch_quat_to_euler(root_tensor[:, 3:7]).abs()
                     > torch.tensor([0.25, 0.25, 0.25], device="cuda:0")
                     * 3.14159).any(axis=1)
    is_currently_terminal = torch.logical_or(height_term, attitude_term)
    is_terminated = torch.logical_or(is_currently_terminal, old_terminations)
    termination_mask = torch.logical_not(is_terminated)
    old_terminations = is_terminated.clone()
    return termination_mask


def batch_quat_to_euler(q):
    """https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
    """
    e = torch.zeros(q.shape[0], 3, device="cuda:0")
    e[:, 0] = torch.atan2(2 * (q[:, 3] * q[:, 0] + q[:, 1] * q[:, 2]),
                          1 - 2 * (torch.square(q[:, 0]) + torch.square(q[:, 1])))
    e[:, 1] = torch.asin(2 * (q[:, 3] * q[:, 1] - q[:, 2] * q[:, 0]))
    e[:, 2] = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
                          1 - 2 * (torch.square(q[:, 1]) + torch.square(q[:, 2])))
    return e


def reward(root_tensor, dof_states, n_envs):
    # return torch.ones(n_envs, device="cuda:0")
    return root_tensor[:, 7].clamp(-1.0, 1.0) + 1.0


def get_observation(root_tensor, dof_states, n_envs):

    assert dof_states.shape[0]/n_envs == 12

    base_pos = root_tensor[:, 0:3]
    base_euler = root_tensor[:, 3:7]
    base_vel = root_tensor[:, 7:10]
    base_avel = root_tensor[:, 10:13]

    joint_positions = dof_states[:, 0].view((n_envs, 12))
    joint_velocities = dof_states[:, 1].view((n_envs, 12))

    observation = torch.cat((
                            base_pos,
                            base_euler,
                            base_vel,
                            base_avel,
                            joint_positions,
                            joint_velocities,
                            torch.ones((n_envs, 1), dtype=torch.float32, device="cuda:0")
                            ), dim=1)
    return observation


def reset_sim(gym, sim, n_envs, _root_tensor, root_tensor, _dof_states, dof_states, envs, actor_handles, viewer):
    reset_root_tensor = torch.tensor([0, 0, 0.48]  # position
                                     + [0, 0, 0, 1]  # orientation
                                     + [0, 0, 0]  # velocity
                                     + [0, 0, 0]  # angular velocity
                                     , dtype=torch.float32).to("cuda:0").tile((n_envs, 1))  # TODO avoid recreating this everytime
    _reset_root_tensor = unwr(reset_root_tensor)
    positions = torch.Tensor(
        [0.037199, 0.660252, -1.200187, -0.028954, 0.618814, -1.183148,
         0.048225, 0.690008, -1.254787, -0.050525, 0.661355, -1.243304]).to("cuda:0").tile(n_envs).unsqueeze(-1)
    reset_dof_tensor = torch.cat((positions, torch.zeros(n_envs * 12, 1).to("cuda:0")), dim=1)
    _reset_dof_tensor = unwr(reset_dof_tensor)
    assert gym.set_actor_root_state_tensor(sim, _reset_root_tensor)
    assert gym.set_dof_state_tensor(sim, _reset_dof_tensor)
    _positions = unwr(positions)
    gym.set_dof_position_target_tensor(sim, _positions)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    if viewer is not None:
        for i in range(n_envs):
            gym.set_rigid_body_color(envs[i], actor_handles[i], 0, g.MeshType.MESH_VISUAL, g.Vec3(1.0, 1.0, 1.0))



def main(render, params):
    global_start = time.time()
    n_envs = params["n_dirs"] * 2
    obs_size = 38
    gym, sim, viewer, _root_tensor, root_tensor, _dof_states, dof_states, envs, actor_handles = create_sim(n_envs, render)
    policy = ARSPolicy(obs_size, params)
    for i in range(params['n_samples'] // (params['steps_per_eps'] * n_envs)):
        start = time.time()
        info = run_simulation(gym, sim, viewer, _root_tensor, root_tensor, _dof_states, dof_states,
                              policy, n_envs, params, envs, actor_handles)
        end = time.time()
        print("FPS: {:d}".format(int(params['steps_per_eps'] / (end - start))))
        policy.update(info)
        end2 = time.time()
        print("Samples per second: {:d}".format(int(params['steps_per_eps'] * n_envs / (end2 - start))))

        if i % params['eval_int'] == 0:
            run_simulation(gym, sim, viewer, _root_tensor, root_tensor, _dof_states, dof_states,
                           policy, n_envs, params, envs, actor_handles, eval=True)
        policy.update_mean_std()
        print("Total time: {}".format(time.time() - global_start))
        print("Total samples upper bound: {}".format(i * n_envs * params['steps_per_eps']))
        print()

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render",
                        action="store_true",
                        default=False)
    parser.add_argument("--config",
                        help="specify name of yaml config file. If none is given, use default.yaml",
                        type=str,
                        default="ars")
    args = parser.parse_args()
    with open(os.path.join('config', args.config + '.yaml')) as f:
        params = yaml.full_load(f)
    main(args.render, params)
