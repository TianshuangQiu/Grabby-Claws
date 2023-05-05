"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Isaac Gym Graphics Example
--------------------------
This example demonstrates the use of several graphics operations of Isaac
Gym, including the following
- Load Textures / Create Textures from Buffer
- Apply Textures to rigid bodies
- Create Camera Sensors
  * Static location camera sensors
  * Camera sensors attached to a rigid body
- Retrieve different types of camera images

Requires Pillow (formerly PIL) to write images from python. Use `pip install pillow`
 to get Pillow.
"""


import os
import numpy as np
import pdb
from numpy import sqrt
from isaacgym import gymapi
from isaacgym import gymutil
from PIL import Image as im
from datetime import datetime
from autolab_core.rigid_transformations import RigidTransform

# from isaacgym import gymtorch
# import torch

# acquire the gym interface
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Graphics Example",
    headless=True,
    custom_parameters=[],
)

# get default params
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.04
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# create sim
sim = gym.create_sim(
    args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
)
if sim is None:
    print("*** Failed to create sim")
    quit()

# Create a default ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

if not args.headless:
    # create viewer using the default camera properties
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    if viewer is None:
        raise ValueError("*** Failed to create viewer")

# set up the env grid
num_envs = 1
spacing = 0.5
num_per_row = int(sqrt(num_envs))
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

asset_root = "../assets"
bin_options = gymapi.AssetOptions()
# bin_options.use_mesh_materials = True
# bin_options.vhacd_enabled = True
# bin_options.vhacd_params.resolution = 300000
# bin_options.vhacd_params.max_convex_hulls = 1024
# bin_options.vhacd_params.max_num_vertices_per_ch = 1024

# load bin asset
bin_asset_file = "urdf/custom/pawn.urdf"
print("Loading asset '%s' from '%s'" % (bin_asset_file, asset_root))
bin_asset = gym.load_asset(sim, asset_root, bin_asset_file, bin_options)
bin_pose = gymapi.Transform()
bin_pose.p = gymapi.Vec3(-0.18, 0.0, 0.3)
bin_pose.r = gymapi.Quat.from_euler_zyx(-np.pi / 2, 0, 0)
# Create box asset
box = gym.create_box(sim, 0.15, 0.03, 0.05, bin_options)

yumi_asset_file = "urdf/yumi/yumi_no_lift.urdf"

robot_options = gymapi.AssetOptions()
robot_options.fix_base_link = True
robot_options.flip_visual_attachments = False
robot_options.vhacd_enabled = True
robot_options.disable_gravity = True
robot_options.vhacd_params.max_convex_hulls = 64
robot_options.vhacd_params.max_num_vertices_per_ch = 256
robot_options.armature = 0.001


mount_table = gym.create_box(sim, 0.5, 1.04, 0.5, gymapi.AssetOptions())

print("Loading asset '%s' from '%s'" % (yumi_asset_file, asset_root))
yumi_asset = gym.load_asset(sim, asset_root, yumi_asset_file, robot_options)


def deproject_point(
    cam_width, cam_height, pixel: tuple, depth_buffer, seg_buffer, view, proj
):
    vinv = np.linalg.inv(view)
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    depth_buffer[seg_buffer == 0] = -10001
    if depth_buffer[pixel] < -10000:
        return None
    centerU = cam_width / 2
    centerV = cam_height / 2
    u = -(pixel[1] - centerU) / (cam_width)  # image-space coordinate
    v = (pixel[0] - centerV) / (cam_height)  # image-space coordinate
    d = depth_buffer[pixel]  # depth buffer value
    X2 = [d * fu * u, d * fv * v, d, 1]  # deprojection vector
    p2 = X2 * vinv  # Inverse camera view to get world coordinates
    return [p2[0, 2], p2[0, 0], p2[0, 1]]


# Create environments
actor_handles = [[]]
camera_handles = [[]]
dof_handles = [[]]
envs = []

# create environments
for i in range(num_envs):
    actor_handles.append([])
    segmentation_id = 1
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    segmentation_id += 1

    # put table in
    # table_handle = gym.create_actor(env, mount_table, gymapi.Transform(), "table", i, 0)

    # put robot in
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 0.1)
    pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
    robot = gym.create_actor(env, yumi_asset, pose, None, 0, 0)
    props = gym.get_actor_dof_properties(env, robot)
    # pdb.set_trace()
    props["driveMode"] = gymapi.DOF_MODE_POS
    props["armature"] = 0.001
    props["stiffness"] = 5000.0
    props["damping"] = 0
    gym.set_actor_dof_properties(env, robot, props)
    mesh = gym.create_actor(env, bin_asset, gymapi.Transform(), None, 0, 0)
    actor_handles[i].append(robot)
    dof_handles.append([])
    for k in range(gym.get_actor_dof_count(env, robot)):
        dof_handles[i].append(gym.get_actor_dof_handle(env, robot, k))

    # current_dof = gym.get_actor_dof_states(env, robot, gymapi.STATE_POS)
    # target = np.array([0, 0], dtype=np.float32)
    # pdb.set_trace()
    # current_dof["pos"] = target
    # gym.set_actor_dof_states(env, robot, current_dof, gymapi.STATE_POS)
    for j, handle in enumerate(dof_handles[i]):
        gym.set_dof_target_position(env, handle, 0.1)

    # Create 2 cameras in each environment, one which views the origin of the environment
    # and one which is attached to the 0th body of the 0th actor and moves with that actor
    camera_handles.append([])
    camera_properties = gymapi.CameraProperties()
    camera_properties.width = 720
    camera_properties.height = 480
    camera_properties.use_collision_geometry = True

    # Set a fixed position and look-target for the first camera
    # position and target location are in the coordinate frame of the environment
    # h1 = gym.create_camera_sensor(envs[i], camera_properties)
    # camera_position = gymapi.Vec3(0, 0, 1)
    # camera_target = gymapi.Vec3(0.00001, 0, 0)

    # gym.set_camera_location(h1, envs[i], camera_position, camera_target)
    # camera_handles[i].append(h1)

    h2 = gym.create_camera_sensor(envs[i], camera_properties)
    camera_position = gymapi.Vec3(0.5, 0.5, 0.5)
    camera_target = gymapi.Vec3(0, 0, 0)
    gym.set_camera_location(h2, envs[i], camera_position, camera_target)
    camera_handles[i].append(h2)


if os.path.exists("graphics_images"):
    import shutil

    shutil.rmtree("graphics_images")
    os.mkdir("graphics_images")
frame_count = 0

gym.prepare_sim(sim)
# Main simulation loop
while True:
    # step the physics simulation
    # pdb.set_trace()
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # communicate physics to graphics system
    gym.step_graphics(sim)

    # render the camera sensors
    gym.render_all_camera_sensors(sim)

    for i in range(num_envs):
        # target = np.array([-0.05, 0.05], dtype=np.float32)
        # print(gym.set_actor_dof_position_targets(env, actor_handles[i][-1], target))
        current_dof = gym.get_actor_dof_states(
            env, actor_handles[i][-1], gymapi.STATE_POS
        )
        print(current_dof)

    # pdb.set_trace()

    if frame_count > 0 and np.mod(frame_count, 1) == 0:
        gym.step_graphics(sim)
        # render the camera sensors
        gym.render_all_camera_sensors(sim)
        for j in range(0, 1):
            # The gym utility to write images to disk is recommended only for RGB images.
            rgb_filename = f"../graphics_images/rgb_env{i}_cam{j}_frame{str(frame_count).zfill(4)}.png"
            gym.write_camera_image_to_file(
                sim,
                envs[i],
                camera_handles[i][j],
                gymapi.IMAGE_COLOR,
                rgb_filename,
            )

    if not args.headless:
        # render the viewer
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time to sync viewer with
        # simulation rate. Not necessary in headless.
        gym.sync_frame_time(sim)

        # Check for exit condition - user closed the viewer window
        if gym.query_viewer_has_closed(viewer):
            break

    if frame_count > 50:
        break

    frame_count = frame_count + 1

    print(frame_count, datetime.now())

print("Done")

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
