from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.objects.kitchen_object_utils import (
    sample_kitchen_object,
)
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from transforms3d.quaternions import quat2mat

from .base_rendering_env import DatasetRenderEnv


@register_env("KitchenRenderEnv-v1", max_episode_steps=1)
class KitchenEnv(DatasetRenderEnv):
    """
    Builds a RoboCasa kitchen and places objects on counters to generate
    cluttered scene images from realistic viewpoints.
    """

    def __init__(
        self,
        *args,
        asset_df: pd.DataFrame = None,
        obj_registries=("objaverse",),
        obj_instance_split=None,
        **kwargs,
    ):
        self.asset_df = asset_df
        self.asset_ids = []
        self.target_fixture = None  # The fixture the camera will look at
        self.obj_registries = obj_registries
        self.obj_instance_split = obj_instance_split
        super().__init__(*args, **kwargs)

    def _load_lighting(self, options: Dict[str, Any]):
        """Adds randomized point lights inside the kitchen for better interior lighting."""
        self.scene.ambient_light = [0.3, 0.3, 0.3]
        if not hasattr(self, "target_fixture") or self.target_fixture is None:
            self.scene.add_point_light([0, 0, 4], [1, 1, 1], shadow=True)
            return

        base_intensity = np.random.uniform(1.1, 3.0)
        main_light_pos = self.target_fixture.pos + np.array(
            [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 2.0]
        )
        self.scene.add_point_light(
            main_light_pos,
            color=np.array([1, 1, 1]) * base_intensity * np.random.uniform(0.9, 1.1),
            shadow=True,
        )
        camera_pose = self._default_sensor_configs[0].pose
        self.scene.add_point_light(
            # This is somehow a nested list
            np.array(camera_pose.p[0]),
            color=np.array([1, 1, 1]) * base_intensity * np.random.uniform(0.4, 0.6),
            shadow=False,
        )

    def _update_sensor_pose(self):
        """Reposition the camera around the target fixture for a new viewpoint.

        Uses camera_azimuth as a lateral offset angle (radians mapped from degrees)
        around the fixture's forward axis, so different azimuth values produce
        meaningfully different viewpoints of the same counter.
        """
        if not (
            hasattr(self, "_sensors")
            and "base_camera" in self._sensors
            and hasattr(self, "target_fixture")
            and self.target_fixture is not None
        ):
            return
        target_pos = self.target_fixture.pos
        rot_mat = quat2mat(self.target_fixture.quat)
        forward_dir = rot_mat @ np.array([0.0, 1.0, 0.0])
        camera_pos = target_pos - forward_dir * 1.5
        camera_pos[2] = 1.6
        # Encode lateral jitter deterministically via camera_azimuth
        azimuth_rad = np.radians(self.camera_azimuth)
        camera_pos[0] += np.cos(azimuth_rad) * 0.3
        camera_pos[1] += np.sin(azimuth_rad) * 0.3
        new_pose = sapien_utils.look_at(camera_pos, target_pos, [0, 0, 1])
        pose_sp = new_pose.sp if hasattr(new_pose, "sp") else new_pose
        self._sensors["base_camera"].camera.set_local_pose(pose_sp)

    def _load_objects(self, options: Dict[str, Any]):
        """Load a kitchen, pick a counter, and place a mix of objects on it."""
        self.asset_ids = []  # Reset for each new render (env may be reused)
        # Preroll batched episode RNG to randomize kitchen env
        for _ in range(np.random.randint(1, 100)):
            for i in range(self.num_envs):
                self._batched_episode_rng[i].randint(0, 120)

        self.scene_builder = RoboCasaSceneBuilder(self)
        self.scene_builder.build()

        fixtures: dict = self.scene_builder.scene_data[0]["fixtures"]
        valid_counters = [
            f
            for name, f in fixtures.items()
            if "counter" in name and f.get_reset_regions(env=self, fixtures=fixtures)
        ]
        if not valid_counters:
            return

        self.target_fixture = np.random.choice(valid_counters)

        all_reset_regions = self.target_fixture.get_reset_regions(
            env=self, fixtures=fixtures
        )
        available_regions = list(all_reset_regions.values())

        if not available_regions:
            # This can happen if a counter has regions defined but they are all occupied.
            return

        # Get the fixture's world pose and rotation matrix, which are constant for all regions.
        target_pos = self.target_fixture.pos
        rot_mat = quat2mat(self.target_fixture.quat)
        z_spawn_offset = 0.02  # Spawn objects 2cm above the surface

        # --- Place custom assets, spreading them across different regions ---
        for _ in range(np.random.randint(5, 10)):
            # For each object, pick a NEW random region from the available list.
            reset_region = np.random.choice(available_regions)
            size, offset = reset_region["size"], reset_region["offset"]

            asset_info = self.asset_df.sample(1).iloc[0]
            self.asset_ids.append(str(asset_info["dir_name"]))

            local_pos = offset + np.array(
                [
                    np.random.uniform(-size[0] / 2, size[0] / 2),
                    np.random.uniform(-size[1] / 2, size[1] / 2),
                    0,
                ]
            )
            world_pos = target_pos + rot_mat @ local_pos
            world_pos[2] += z_spawn_offset

            euler = np.random.uniform(-np.pi, np.pi, size=3)
            self.partnet_mobility_loader.load(
                asset_info["dir_name"], world_pos, euler=euler
            )

        # --- Place random RoboCasa assets, also spreading them out ---
        for i in range(5):
            # Also pick a new random region for each RoboCasa object.
            reset_region = np.random.choice(available_regions)
            size, offset = reset_region["size"], reset_region["offset"]

            obj_kwargs, _ = self.sample_object()
            obj = MJCFObject(self.scene, name=f"distractor_{i}", **obj_kwargs)

            local_pos = offset + np.array(
                [
                    np.random.uniform(-size[0] / 2, size[0] / 2),
                    np.random.uniform(-size[1] / 2, size[1] / 2),
                    0,
                ]
            )
            world_pos = target_pos + rot_mat @ local_pos
            world_pos[2] += z_spawn_offset

            obj.set_pos(world_pos)

    def sample_object(self):
        """Helper to sample a random kitchen object from the RoboCasa dataset."""
        return sample_kitchen_object(
            groups="all",
            graspable=True,
        )

    @property
    def _default_sensor_configs(self):
        """Configure the camera to look at the target fixture."""
        if hasattr(self, "target_fixture") and self.target_fixture is not None:
            target_pos = self.target_fixture.pos
            rot_mat = quat2mat(self.target_fixture.quat)
            forward_dir = rot_mat @ np.array([0.0, 1.0, 0.0])
            camera_pos = target_pos - forward_dir * 1.5
            camera_pos[2] = 1.6
            camera_pos[0] += np.random.uniform(-0.3, 0.3)
            camera_pos[1] += np.random.uniform(-0.3, 0.3)
            pose = sapien_utils.look_at(camera_pos, target_pos, [0, 0, 1])
        else:
            pose = super()._default_sensor_configs[0].pose

        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=self.image_size[0],
                height=self.image_size[1],
                fov=np.pi / 3,
            )
        ]

    def is_valid_render(self, obs: Dict, min_segments: int = 5) -> bool:
        """Checks if the render has enough distinct objects."""
        instance_mask = obs["sensor_data"]["base_camera"]["instance_segmentation"]
        unique_ids = torch.unique(instance_mask)
        return len(unique_ids[unique_ids != 0]) >= min_segments

    def build_render_data(self, obs: Dict) -> Dict[str, np.ndarray]:
        """Extracts and formats final data from the observation, including bounding boxes."""
        render_data = super().build_render_data(obs)
        render_data["asset_id"] = self.asset_ids
        instance_mask = render_data["instance"].squeeze()
        semantic_mask = render_data["semantic"].squeeze()
        unique_instances = torch.unique(instance_mask)
        bboxes = []
        for instance_id in unique_instances:
            if instance_id == 0:
                continue
            mask = instance_mask == instance_id
            if not torch.any(mask):
                continue
            y_indices, x_indices = torch.where(mask)
            min_y, max_y = torch.min(y_indices), torch.max(y_indices)
            min_x, max_x = torch.min(x_indices), torch.max(x_indices)

            # Bounding box in [X, Y, W, H, Class_ID] format
            class_id = semantic_mask[y_indices[0], x_indices[0]]
            bboxes.append(
                [
                    min_x.item(),
                    min_y.item(),
                    (max_x - min_x + 1).item(),
                    (max_y - min_y + 1).item(),
                    class_id.item(),
                ]
            )

        render_data["bbox"] = (
            np.array(bboxes, dtype=np.int32)
            if bboxes
            else np.empty((0, 5), dtype=np.int32)
        )
        return render_data
