import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import mops_data

env = gym.make(
    # "Affordance-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    "AffordanceCasaKitchen-v1",
    # "RenderEnv-v1",
    num_envs=4,
    obs_mode="pointcloud",  # there is also "state_dict", "rgbd", ...
    # obs_mode="rgbd",
    # viewer_camera_configs=dict(shader_pack="rt-fast"),
    sensor_configs=dict(width=200, height=200, shader_pack="default"),
    use_distractors=True,
)

obs, _ = env.reset(seed=3)  # reset with a seed for determinism
done = False
step = 0
while not done:
    action = np.zeros_like(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(action)
    done = False

    if step < 20:
        step += 1

        continue

    xyzw = obs["pointcloud"]["xyzw"][3]  # Torch GPU tensor
    rgb = obs["pointcloud"]["rgb"][3]  # Torch GPU tensor

    print(xyzw.shape)
    print(rgb.shape)

    # Convert to numpy
    xyzw = xyzw.cpu().numpy()
    rgb = rgb.cpu().numpy()

    # remove w
    xyz = xyzw[:, :3]
    rgb = rgb / 255.0

    import open3d as o3d

    # Create a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Visualize
    o3d.visualization.draw_geometries([pcd])
    break
