import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv

import mops_data

rng = np.random.default_rng(3)

env = gym.make(
    # "RenderEnv-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    "AffordanceCasaKitchen-v1",
    num_envs=1,
    obs_mode="none",  # there is also "state_dict", "rgbd", ...
    render_mode="human",
    parallel_in_single_scene=False,
    # viewer_camera_configs=dict(shader_pack="rt-fast"),
    viewer_camera_configs=dict(shader_pack="rt-fast"),
    # sensor_configs=dict(shader_pack="rt-fast"),
    # use_distractors=False,
    # np_rng=rng,
)

frame = 0
while True:
    obs, _ = env.reset(seed=0)  # reset with a seed for determinism
    done = False
    while not done:
        action = np.zeros_like(env.action_space.sample())
        # action = None
        obs, reward, terminated, truncated, info = env.step(action)
        done = False
        env.render()
        frame += 1

        # if frame == 10:
        #     unw_env: BaseEnv = env.unwrapped
        #     unw_env.viewer.set_camera_pose(
        #         # This is the winodw camera pose
        #         # unw_env.scene.human_render_cameras[
        #         #     "render_camera"
        #         # ].camera.global_pose.sp
        #         # unw_env.scene.sensors["base_camera"].camera.global_pose.sp
        #     )
