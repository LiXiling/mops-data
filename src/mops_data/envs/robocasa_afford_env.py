import os

import numpy as np
from mani_skill.envs.tasks.mobile_manipulation.robocasa.kitchen import (
    RoboCasaKitchenEnv,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from mops_data.asset_manager.object_annotation_registry import ObjectAnnotationRegistry
from mops_data.asset_manager.partnet_mobility_loader import PartNetMobilityLoader
from mops_data.asset_manager.uniform_sampler import UniformRandomSampler
from mops_data.render.afford_obs_augmentor import AffordObsAugmentor
from mops_data.render.shader_config import RT_RGB_ONLY_CONFIG


@register_env(
    "AffordanceCasaKitchen-v1", max_episode_steps=100, asset_download_ids=["RoboCasa"]
)
class RoboCasaAffordanceKitchenEnv(RoboCasaKitchenEnv):
    SUPPORTED_ROBOTS = ["fetch", "none"]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_agentview_center",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        renderer="mujoco",
        renderer_config=None,
        init_robot_base_pos=None,
        seed=None,
        layout_and_style_ids=None,
        layout_ids=None,
        style_ids=None,
        scene_split=None,
        generative_textures="100p",
        obj_registries=...,
        obj_instance_split=None,
        use_distractors=True,
        translucent_robot=False,
        randomize_cameras=False,
        fixtures_only=False,
        pre_roll=0,
        **kwargs,
    ):
        self.object_annotation_registry = ObjectAnnotationRegistry()
        self.partnet_mobility_loader = PartNetMobilityLoader(
            env=self,
            dir_path="data/partnet_mobility",
            registry=self.object_annotation_registry,
        )
        self.afford_augmentor = AffordObsAugmentor(self.object_annotation_registry)
        # save all folder names in he flder "data/partnet_mobility" to a list

        # self.all_objects = [
        #     str(elem)
        #     for elem in list(
        #         self.partnet_mobility_loader.partnet_mob_annotations[
        #             "dir_name"
        #         ].unique()
        #     )
        # ]

        self.all_objects = [
            str(elem)
            for elem in list(
                self.partnet_mobility_loader.partnet_small_mob_annotations[
                    "dir_name"
                ].unique()
            )
        ]

        self.rendered_object_ids = []
        self.pre_roll = pre_roll

        print(style_ids)

        super().__init__(
            *args,
            robot_uids=robot_uids,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            base_types=base_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            use_object_obs=use_object_obs,
            reward_scale=reward_scale,
            reward_shaping=reward_shaping,
            placement_initializer=placement_initializer,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            renderer=renderer,
            renderer_config=renderer_config,
            init_robot_base_pos=init_robot_base_pos,
            seed=seed,
            layout_and_style_ids=layout_and_style_ids,
            layout_ids=layout_ids,
            style_ids=style_ids,
            scene_split=scene_split,
            generative_textures=generative_textures,
            obj_registries=obj_registries,
            obj_instance_split=obj_instance_split,
            use_distractors=use_distractors,
            translucent_robot=translucent_robot,
            randomize_cameras=randomize_cameras,
            fixtures_only=fixtures_only,
            **kwargs,
        )

    def _load_scene(self, options):
        for _ in range(self.pre_roll):
            for i in range(self.num_envs):
                self._batched_episode_rng[i].randint(0, 120)

        super()._load_scene(options)
        # self.scene.scene_offsets

        for env in range(self.num_envs):

            for k, v in self.unwrapped.scene.actors.items():

                if "counter" in k:
                    counter_pos = v.pose.p[0]

                    bounds: np.array = v.get_collision_meshes()[0].bounds
                    width = np.max(bounds, axis=0)[0] - np.min(bounds, axis=0)[0]
                    depth = np.max(bounds, axis=0)[1] - np.min(bounds, axis=0)[1]
                    area = width * depth

                    N = int(7 * area)
                    self.sampled_elements = np.random.choice(
                        self.all_objects, N, replace=False
                    )

                    x_min = counter_pos[0] - width / 2
                    x_max_ = counter_pos[0] + width / 2
                    y_min = counter_pos[1] - depth / 2
                    y_max_ = counter_pos[1] + depth / 2

                    sampler = UniformRandomSampler(
                        name="scenesampler",
                        x_range=(x_min, x_max_),
                        y_range=(y_min, y_max_),
                        z_offset=counter_pos[2],
                    )

                    for elem in self.sampled_elements:

                        x_pos = sampler._sample_x()
                        y_pos = sampler._sample_y()
                        z_pos = 1
                        q = sampler._sample_quat()

                        self.partnet_mobility_loader.load(
                            f"{elem}",
                            [
                                x_pos,
                                y_pos,
                                z_pos,
                            ],
                            no_grav=False,
                        )

            self.object_annotation_registry.register_missing_objects(self)
            # self.init_object_positions()

    def init_object_positions(self):
        for env in range(self.num_envs):
            scene_offset = (
                self.scene.scene_offsets_np[env]
                if hasattr(self.scene, "scene_offsets_np")
                else [0, 0, 0]
            )
            x_min = scene_offset[0]
            x_max_ = scene_offset[0] + self.sim_config.spacing
            y_min = scene_offset[1]
            y_max_ = scene_offset[1] + self.sim_config.spacing
            sampler = UniformRandomSampler(
                name="scenesampler",
                x_range=(x_min, x_max_),
                y_range=(y_min, y_max_),
                z_offset=scene_offset[2],
            )

        height = 1.0
        for key in self.scene.articulations.keys():
            if any(elem in key for elem in self.sampled_elements):
                x_pos = sampler._sample_x()
                y_pos = sampler._sample_y()
                z_pos = 1
                q = sampler._sample_quat()

                # new_pose = Pose.create_from_pq(p=[x_pos, y_pos, z_pos], q=q)
                new_pose = Pose.create_from_pq(p=[1.0, height, 1.0], q=q)
                height += 2.0
                # self.segmentation_id_map[obj_id].set_kinematic(False)
                self.scene.articulations[key].set_pose(new_pose)
                # self.segmentation_id_map[key].set_pose(new_pose)

    def _get_obs_with_sensor_data(self, info, apply_texture_transforms=True):
        obs = super()._get_obs_with_sensor_data(info, apply_texture_transforms)
        augment_obs = self.afford_augmentor.augment(self, obs)
        return augment_obs

    @property
    def _default_sensor_configs(self):
        # TODO (fix cameras to be where robocasa places them)
        pose = sapien_utils.look_at([3.0, -7.5, 2.5], [3.0, 0.0, 1.0])
        birdseye_pose = sapien_utils.look_at(eye=[2.0, -2, 4], target=[2, -2, 0.1])
        return [
            CameraConfig("base_camera", pose, 128, 128, 60 * np.pi / 180, 0.01, 100),
            CameraConfig(
                "birdseye_camera",
                pose=birdseye_pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                shader_pack="default",
            ),
            CameraConfig(
                "birdseye_camera_rt",
                pose=birdseye_pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                shader_config=RT_RGB_ONLY_CONFIG,
            ),
        ]
