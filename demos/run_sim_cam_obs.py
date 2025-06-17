import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import mops_data

env = gym.make(
    # "Affordance-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    # "RenderEnv-v1",
    "AffordanceCasaKitchen-v1",
    num_envs=1,
    obs_mode="rgb+depth+segmentation",  # there is also "state_dict", "rgbd", ...
    viewer_camera_configs=dict(shader_pack="rt-fast"),
    sensor_configs=dict(width=1280, height=720, shader_pack="default"),
    # use_distractors=True,
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

        print(obs["sensor_data"].keys())

        base_cam_obs = obs["sensor_data"]["base_camera"]
        fig, axs = plt.subplots(2, 3, figsize=(20, 8))

        # Plot the RGB Image
        axs[0, 0].imshow(base_cam_obs["rgb"].cpu()[0])
        axs[0, 0].set_title("RGB Image")

        # Plot the Depth Image
        axs[0, 1].imshow(base_cam_obs["depth"].cpu()[0])
        axs[0, 1].set_title("Depth Image")

        # Plot the Part Segmentation
        axs[0, 2].imshow(base_cam_obs["segmentation"].cpu()[0])
        axs[0, 2].set_title("Part Segmentation")

        # Plot the Instance Segmentation
        axs[1, 0].imshow(base_cam_obs["instance_segmentation"].cpu()[0])
        axs[1, 0].set_title("Instance Segmentation")

        # Plot the Class Segmentation
        axs[1, 1].imshow(base_cam_obs["class_segmentation"].cpu()[0])
        axs[1, 1].set_title("Class Segmentation")

        # Plot the Affordance Segmentation

        # Shape of aff = (H, W, C) with multilabel assignment
        aff = base_cam_obs["affordance_segmentation"].cpu()[0]
        aff = aff.sum(dim=-1, keepdim=True)
        axs[1, 2].imshow(aff)
        axs[1, 2].set_title("Affordance Segmentation")

        plt.tight_layout()
        plt.show()

        break
        # Disable Rendering as it is buggy with activated camera observations
        # env.render()
    break
