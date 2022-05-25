import time

from isaacgym import gymapi as g
from isaacgym import gymtorch
from isaacgym.gymtorch import wrap_tensor as wr
import numpy as np

gym = g.acquire_gym()
sim_params = g.SimParams()
sim_params.up_axis = g.UP_AXIS_Z
sim_params.gravity = g.Vec3(0.0, 0.0, -9.8)
sim_params.physx.use_gpu = True
# sim_params.use_gpu_pipeline = True
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

asset_root = "../../assets"
asset_file = "urdf/aliengo/urdf/aliengo.urdf"

asset_options = g.AssetOptions()
asset_options.fix_base_link = False
# asset_options.armature = 0.01

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

spacing = 2.0
n_envs = 100
envs_per_row = 10
lower = g.Vec3(-spacing, 0.0, -spacing)
upper = g.Vec3(spacing, spacing, spacing)
envs = []
actor_handles = []
for i in range(n_envs):
    env = gym.create_env(sim, lower, upper, envs_per_row)
    envs.append(env)

    pose = g.Transform()
    pose.p = g.Vec3(0.0, 0.0, 1.0)
    pose.r = g.Quat(-0.707107, 0.0, 0.0, 0.707107)

    aliengo_robot = gym.create_actor(env, asset, pose, '_', i, 1)
    props = gym.get_actor_dof_properties(env, aliengo_robot)
    props["driveMode"].fill(g.DOF_MODE_POS)
    props["stiffness"].fill(1000.0)
    props["damping"].fill(200.0)
    gym.set_actor_dof_properties(env, aliengo_robot, props)
    actor_handles.append(aliengo_robot)



cam_props = g.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)


gym.prepare_sim(sim)
root_tensor = wr(gym.acquire_actor_root_state_tensor(sim))
dof_states = wr(gym.acquire_dof_state_tensor(sim))
t = 0.0
while True:
    t += 0.01
    targets = np.sin(np.ones(12).astype('f') * t)
    print(root_tensor)
    # print(root_tensor)
    # time.sleep()
    # for i in range(n_envs):
    #     gym.set_actor_dof_position_targets(envs[i], actor_handles[i], targets)
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

