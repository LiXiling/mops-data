import os
import pickle

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mops_data

SEED = 42
DATA_DIR = "data/mops_data/single_object"


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    anno_df = pd.read_csv("scripts/partnet_mobility_with_affordances.csv")
    all_ids = anno_df["dir_name"].unique()

    # Get Current time
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

    for run in range(1):
        for i in range(len(all_ids)):
            obj_id = all_ids[i]

            # if folder exists continue
            if os.path.exists(f"{DATA_DIR}/{obj_id}/{obj_id}_{run}.pkl"):
                continue

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
            # Save obs as pickle
            os.makedirs(f"{DATA_DIR}/{obj_id}", exist_ok=True)
            with open(f"{DATA_DIR}/{obj_id}/{obj_id}_{run}.pkl", "wb") as f:
                pickle.dump(base_cam_obs, f)

            env.close()

    # afterwards time
    print(f"Finished at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Print total execution duration
    print(
        f"Total execution duration: {pd.Timestamp.now() - pd.Timestamp(current_time)}"
    )
    print("Done!")
