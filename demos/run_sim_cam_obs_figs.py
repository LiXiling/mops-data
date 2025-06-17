import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import mops_data

print("CREATE ENV")

env = gym.make(
    # "Affordance-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    "AffordanceCasaKitchen-v1",
    # "RenderEnv-v1",
    num_envs=1,
    obs_mode="rgb+depth+segmentation+normal",  # there is also "state_dict", "rgbd", ...
    # obs_mode="rgbd",
    # viewer_camera_configs=dict(shader_pack="rt-fast"),
    sensor_configs=dict(width=1280, height=720, shader_pack="rt"),
    use_distractors=True,
    generative_textures="100p",
    pre_roll=1,
)

OUTPUT_DIR = os.path.join(".", "rm_figs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

while True:
    print("RESET ENV")
    obs, _ = env.reset(seed=0)  # reset with a seed for determinism
    done = False
    step = 0
    while not done:
        action = np.zeros_like(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        done = False

        if step < 5:
            step += 1

            continue

        print("MAKE FIGURES")

        print(obs["sensor_data"].keys())

        base_cam_obs = obs["sensor_data"]["fetch_head"]
        print(base_cam_obs.keys())

        # Plot the RGB Image
        rgb = base_cam_obs["rgb"].cpu()[0].numpy()
        rgb = rgb.astype(np.uint8)
        # save RGB PNG
        plt.imsave(os.path.join(OUTPUT_DIR, "rgb.png"), rgb)

        # Plot the Depth Image
        depth = base_cam_obs["depth"].cpu()[0]
        # save Depth PNG with viridis colormap
        plt.imsave(os.path.join(OUTPUT_DIR, "depth.png"), depth, cmap="viridis")

        if "segmentation" in base_cam_obs:
            # Plot the Part Segmentation
            segm = base_cam_obs["segmentation"].cpu()[0]
            # list of unique values in segm
            unique_values = torch.unique(segm)
            # replace sagm with index of unique values
            for i, val in enumerate(unique_values):
                segm[segm == val] = i

            # save Part Segmentation PNG with tab10 colormap
            plt.imsave(os.path.join(OUTPUT_DIR, "segm.png"), segm, cmap="nipy_spectral")

            # Plot the Class Segmentation
            class_Segm = base_cam_obs["class_segmentation"].cpu()[0]
            # list of unique values in class_Segm
            unique_values = torch.unique(class_Segm)
            # replace class_Segm with index of unique values
            for i, val in enumerate(unique_values):
                class_Segm[class_Segm == val] = i

            # save Class Segmentation PNG with viridis colormap
            plt.imsave(
                os.path.join(OUTPUT_DIR, "class_segm.png"),
                class_Segm,
                cmap="nipy_spectral",
            )

            aff = base_cam_obs["affordance_segmentation"].cpu()[0]
            aff = aff.argmax(dim=-1, keepdim=True)
            # list of unique values in aff
            unique_values = torch.unique(aff)
            # replace aff with index of unique values
            for i, val in enumerate(unique_values):
                aff[aff == val] = i

            # save Affordance Segmentation PNG with viridis colormap
            plt.imsave(os.path.join(OUTPUT_DIR, "aff.png"), aff, cmap="nipy_spectral")
            plt.imsave(os.path.join(OUTPUT_DIR, "aff2.png"), aff, cmap="Set2")

        if "normal" in base_cam_obs:
            # Plot the Normal Image
            normal = base_cam_obs["normal"].cpu()[0].numpy()
            # normal map is between -1 and 1, convert to 0 and 1
            normal = (normal + 1) / 2

            # save Normal PNG with viridis colormap
            plt.imsave(os.path.join(OUTPUT_DIR, "normal.png"), normal)

        break
        # Disable Rendering as it is buggy with activated camera observations
        # env.render()
    break
