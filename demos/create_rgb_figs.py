import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import mops_data

rng = np.random.default_rng(0)
env = gym.make(
    # "Affordance-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    # "AffordanceCasaKitchen-v1",
    "RenderEnv-v1",
    num_envs=1,  # Force CPU PhysX
    obs_mode="rgb+depth+segmentation",  # there is also "state_dict", "rgbd", ...
    viewer_camera_configs=dict(shader_pack="rt-fast"),
    sensor_configs=dict(width=1024, height=1024),
    np_rng=rng,
)

while True:
    obs, _ = env.reset(seed=0)  # reset with a seed for determinism
    done = False
    step = 0
    while not done:
        action = np.zeros_like(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        done = False

        if step < 20:
            step += 1
            continue

        base_cam_obs = obs["sensor_data"]["base_camera"]

        rgb = obs["sensor_data"]["base_camera_rt"]["rgb"].cpu()[0]  # (H, W, 3)
        depth = base_cam_obs["depth"].cpu()[0]  # (H, W, 1)

        # Add alpha channel to rgb image
        alpha = np.ones((rgb.shape[0], rgb.shape[1], 1)) * 255  # (H, W, 1)
        rgba = np.concatenate([rgb, alpha], axis=-1)  # (H, W, 4)
        rgba = rgba.astype(np.uint8)

        # Make everything transparent in rgba where depth == 0
        mask = (depth == 0).squeeze()

        rgba[mask, 3] = 0

        # save RGBA PNG
        plt.imsave("rgba.png", rgba)

        aff = base_cam_obs["affordance_segmentation"].cpu()[0]  # (H, W, C)
        aff = aff.argmax(dim=-1, keepdim=True)  # (H, W, 1)

        # colorize affordance segmentation using set2 colormap
        cmap = plt.get_cmap("tab10")
        aff = cmap(aff.squeeze().numpy())[:, :, :3] * 255
        aff = aff.astype(np.uint8)

        # add alpha channel
        alpha = np.ones((aff.shape[0], aff.shape[1], 1), dtype=np.uint8) * 255
        aff = np.concatenate([aff, alpha], axis=-1)

        # apply alpha mask
        aff[mask, 3] = 0

        # save affordance segmentation
        plt.imsave("aff.png", aff)

        break
        # Disable Rendering as it is buggy with activated camera observations
        # env.render()
    break
