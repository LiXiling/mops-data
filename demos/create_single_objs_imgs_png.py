import os
import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mops_data

SEED = 1337
DATA_DIR = "data/mops_data/single_object_png"

import torch

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    anno_df = pd.read_csv("scripts/partnet_mobility_with_affordances.csv")
    all_ids = anno_df["dir_name"].unique()

    # Get Current time
    # current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

    for run in range(1):
        for i in range(20):
            obj_id = rng.choice(all_ids)

            obj_dir = os.makedirs(f"{DATA_DIR}/{obj_id}", exist_ok=True)

            print(run, i, obj_id)

            env = gym.make(
                "RenderEnv-v1",
                num_envs=1,
                obs_mode="rgb+depth+segmentation+normal",  # there is also "state_dict", "rgbd", ...
                viewer_camera_configs=dict(shader_pack="rt-fast"),
                sensor_configs=dict(width=400, height=400, shader_pack="default"),
                np_rng=rng,
                obj_id=obj_id,
            )

            obs, _ = env.reset(seed=0)

            action = None
            for step in range(3):
                obs, reward, terminated, truncated, info = env.step(action)

            base_cam_obs = obs["sensor_data"]["base_camera"]

            if "normal" in base_cam_obs:
                # Plot the Normal Image
                normal = base_cam_obs["normal"].cpu()[0].numpy()
                # normal map is between -1 and 1, convert to 0 and 1
                normal = (normal + 1) / 2

                # save Normal PNG with viridis colormap
                plt.imsave(
                    f"{DATA_DIR}/{obj_id}/normal.png",
                    normal,
                )
                continue

            # Plot the RGB Image
            rgb = base_cam_obs["rgb"].cpu()[0].numpy()
            rgb = rgb.astype(np.uint8)
            # save RGB PNG
            plt.imsave(f"{DATA_DIR}/{obj_id}/rgb.png", rgb)

            # Plot the Depth Image
            depth = base_cam_obs["depth"].cpu()[0]
            # save Depth PNG with viridis colormap
            plt.imsave(f"{DATA_DIR}/{obj_id}/depth.png", depth, cmap="viridis")

            if "segmentation" in base_cam_obs:
                # Plot the Part Segmentation
                segm = base_cam_obs["segmentation"].cpu()[0]
                # list of unique values in segm
                unique_values = torch.unique(segm)
                # replace sagm with index of unique values
                for i, val in enumerate(unique_values):
                    segm[segm == val] = i

                # create colormap with tab10, but 0 is black
                cmap = plt.get_cmap("tab10", len(unique_values))
                cmap.colors[0] = (0, 0, 0, 1)
                # save Segmentation PNG with tab10 colormap
                plt.imsave(f"{DATA_DIR}/{obj_id}/segm.png", segm, cmap=cmap)

                # Plot the Class Segmentation
                class_segm = base_cam_obs["class_segmentation"].cpu()[0]
                # list of unique values in class_Segm
                unique_values = torch.unique(class_segm)
                # replace class_Segm with index of unique values
                for i, val in enumerate(unique_values):
                    class_segm[class_segm == val] = i

                # update cmap
                cmap = plt.get_cmap("tab10", len(unique_values))
                cmap.colors[0] = (0, 0, 0, 1)
                # save Class Segmentation PNG with tab10 colormap
                plt.imsave(
                    f"{DATA_DIR}/{obj_id}/class_segm.png",
                    class_segm,
                    cmap=cmap,
                )

                aff = base_cam_obs["affordance_segmentation"].cpu()[0]
                aff = aff.argmax(dim=-1, keepdim=True)
                # list of unique values in aff
                unique_values = torch.unique(aff)
                # replace aff with index of unique values
                for i, val in enumerate(unique_values):
                    aff[aff == val] = i
                unique_values = torch.unique(aff)

                # update cmap
                cmap = plt.get_cmap("tab10", len(unique_values))
                cmap.colors[0] = (0, 0, 0, 1)
                # save Affordance Segmentation PNG with tab10 colormap
                plt.imsave(
                    f"{DATA_DIR}/{obj_id}/aff_segm.png",
                    aff,
                    cmap=cmap,
                )

            env.close()

    # afterwards time
    # print(f"Finished at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Print total execution duration
    # print(
    #    f"Total execution duration: {pd.Timestamp.now() - pd.Timestamp(current_time)}"
    # )
    # print("Done!")
