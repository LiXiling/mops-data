from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from mops_data.asset_manager.object_annotation_registry import ObjectAnnotationRegistry
from mops_data.asset_manager.partnet_mobility_loader import PartNetMobilityLoader
from mops_data.render.afford_obs_augmentor import AffordObsAugmentor
from mops_data.render.shader_config import RT_RGB_ONLY_CONFIG


@register_env("RenderEnv-v1", max_episode_steps=100)
class RenderEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "none"]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        obj_id=None,
        obj_index=None,
        np_rng: np.random.Generator = None,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        self.object_annotation_registry = ObjectAnnotationRegistry()
        self.partnet_mobility_loader = PartNetMobilityLoader(
            env=self,
            dir_path="data/partnet_mobility",
            registry=self.object_annotation_registry,
        )

        self.afford_augmentor = AffordObsAugmentor(
            registry=self.object_annotation_registry
        )

        if np_rng is None:
            np_rng = np.random.default_rng()
        self.np_rng = np_rng

        self.obj_id = obj_id
        self.obj_index = obj_index

        self.obj_center = np.asarray([10, 10, 10])

        # This calls self._load_scene() and creates all loaded objects
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options):
        rng: np.random.Generator = self.np_rng

        # if self.obj_id is None:
        #     obj_index = self.obj_index
        #     if obj_index is None:
        #         obj_index = 0

        #     self.partnet_mobility_loader.load_by_index(
        #         self.obj_index,
        #         self.obj_center + rng.uniform(-0.1, 0.1, 3),
        #         no_grav=True,
        #         scale=0.8,
        #     )
        # else:
        #     self.partnet_mobility_loader.load(
        #         self.obj_id,
        #         self.obj_center + rng.uniform(-0.1, 0.1, 3),
        #         no_grav=True,
        #         scale=0.8,
        #     )

        # self.partnet_mobility_loader.load_random_object(
        #     rng,
        #     [-0.5, -0.8, 0],
        #     no_grav=True,  # scale=1.0
        # )
        # self.partnet_mobility_loader.load_random_object(
        #     rng,
        #     [-0.5, +0.8, 0],
        #     no_grav=True,  # scale=1.0
        # )

        self.object_annotation_registry.register_missing_objects(self)

    def _get_obs_with_sensor_data(self, info, apply_texture_transforms=True):
        obs = super()._get_obs_with_sensor_data(info, apply_texture_transforms)
        augment_obs = self.afford_augmentor.augment(self, obs)
        return augment_obs

    ### ManiSkill Env BoilerPlate ###
    def _initialize_episode(self, env_idx, options):
        pass

    @property
    def _default_sensor_configs(self):
        obj_pos = np.asarray([10.0, 10.0, 10.0])

        # Sample Random point 1.0m away from the object
        target_pos = obj_pos + self.np_rng.uniform(-1, 1, size=3)
        # normalize to 1.0m length
        target_pos = target_pos / np.linalg.norm(target_pos) * 1.0
        target_pos = target_pos + obj_pos

        pose = sapien_utils.look_at(eye=[1.3, 0, 1.6], target=[-0.1, 0, 0.1])
        # pose = sapien_utils.look_at(eye=target_pos, target=[0, 0, 0])
        return [
            CameraConfig(
                "base_camera", pose=pose, width=128, height=128, fov=np.pi / 2
            ),
            CameraConfig(
                "base_camera_rt",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                shader_config=RT_RGB_ONLY_CONFIG,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
