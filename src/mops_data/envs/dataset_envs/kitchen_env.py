from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sapien
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
from mani_skill.utils.scene_builder.robocasa.objects.kitchen_object_utils import (
    sample_kitchen_object,
)
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from transforms3d.euler import euler2quat
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

    def _load_objects(self, options: Dict[str, Any]):
        """
        Load a RoboCasa kitchen, pick a single counter, and place objects
        only on that counter to create a cluttered scene.
        """
        self.scene_builder = RoboCasaSceneBuilder(self)
        self.scene_builder.build()

        fixtures: dict = self.scene_builder.scene_data[0]["fixtures"]
        valid_counters = [
            f
            for name, f in fixtures.items()
            if "counter" in name and f.get_reset_regions(env=self, fixtures=fixtures)
        ]

        if not valid_counters:
            print("Warning: No valid counters with placement regions found.")
            return

        self.target_fixture = np.random.choice(valid_counters)
        reset_regions = self.target_fixture.get_reset_regions(
            env=self, fixtures=fixtures
        )
        reset_region = np.random.choice(list(reset_regions.values()))
        size = reset_region["size"]
        offset = reset_region["offset"]

        # Place a mix of custom and random RoboCasa objects
        self._place_custom_assets(size, offset)
        self._place_robocasa_assets(size, offset)

    def _place_robocasa_assets(self, size, offset, num_objects=5):
        """Place randomly sampled RoboCasa objects on the target fixture."""
        for i in range(num_objects):
            # Sample a random graspable object from the RoboCasa dataset
            obj_kwargs, obj_info = self.sample_object()
            obj = MJCFObject(self.scene, name=f"distractor_{i}", **obj_kwargs)

            # Correctly calculate the world position for the object
            world_pos = self._get_world_pos_from_local(size, offset)
            euler = np.random.uniform(-np.pi, np.pi, size=3)
            quat = euler2quat(*euler)
            obj.set_pos(world_pos)

    def _place_custom_assets(self, size, offset):
        """Place assets from the provided DataFrame on the target fixture."""

        num_objects = np.random.randint(
            5, 10
        )  # Randomly choose how many objects to place
        for _ in range(num_objects):
            asset_info = self.asset_df.sample(1).iloc[0]
            asset_id = asset_info["dir_name"]
            self.asset_ids.append(str(asset_id))

            # Correctly calculate the world position for the object
            world_pos = self._get_world_pos_from_local(size, offset)
            euler = np.random.uniform(-np.pi, np.pi, size=3)
            self.partnet_mobility_loader.load(asset_id, world_pos, euler=euler)

    def _get_world_pos_from_local(self, size, offset):
        """
        Samples a point in the fixture's local coordinate frame and transforms
        it to the world coordinate frame.
        """
        # a. Sample a position in the fixture's LOCAL coordinate frame.
        local_x = np.random.uniform(-size[0] / 2, size[0] / 2) + offset[0]
        local_y = np.random.uniform(-size[1] / 2, size[1] / 2) + offset[1]
        local_z = offset[2]
        local_pos = np.array([local_x, local_y, local_z])

        # b. Transform the local position to the WORLD coordinate frame.
        fixture_pos = self.target_fixture.pos
        fixture_quat = self.target_fixture.quat
        rot_mat = quat2mat(fixture_quat)
        world_pos = fixture_pos + rot_mat @ local_pos
        return world_pos

    def sample_object(self, *args, **kwargs):
        """Helper to sample a random kitchen object from the RoboCasa dataset."""
        return sample_kitchen_object(
            groups="all",
            graspable=True,
        )

    @property
    def _default_sensor_configs(self):
        """
        Configure the camera. If a target fixture is set, it computes a realistic
        "in-kitchen" pose. Otherwise, it falls back to a default orbital view.
        """
        # If a target fixture has been selected, calculate a realistic camera pose
        if hasattr(self, "target_fixture") and self.target_fixture is not None:
            # The target_fixture is the counter itself. We want to look at its center.
            target_pos = self.target_fixture.pos

            # Get the forward direction of the counter by rotating a local +Y vector
            # using the fixture's world rotation.
            rot_quat = self.target_fixture.quat
            rot_mat = quat2mat(rot_quat)
            local_forward_dir = np.array([0.0, 1.0, 0.0])
            forward_dir = rot_mat @ local_forward_dir

            # Position the camera in front of the counter and at a realistic height.
            camera_pos = target_pos - forward_dir * 1.5  # Stand 1.5m away
            camera_pos[2] = 1.6  # Eye-level height

            # Add some randomization for more varied shots
            camera_pos[0] += np.random.uniform(-0.3, 0.3)
            camera_pos[1] += np.random.uniform(-0.3, 0.3)

            # Ensure the camera looks at the center of the counter
            up_vector = np.array([0.0, 0.0, 1.0])
            pose = sapien_utils.look_at(camera_pos, target_pos, up_vector)
        else:
            # Fallback to the default orbital camera if no target is set
            pose = super()._default_sensor_configs[0].pose

        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=self.image_size[0],
                height=self.image_size[1],
                fov=np.pi / 3,
            ),
        ]

    def is_valid_render(self, obs: Dict, min_segments: int = 5) -> bool:
        """
        Checks if the render is useful, e.g., has enough distinct objects.
        """
        instance_mask = obs["sensor_data"]["base_camera"]["instance_segmentation"]
        unique_ids = torch.unique(instance_mask)
        # Count unique instances, excluding the background (ID 0)
        num_unique_objects = len(unique_ids[unique_ids != 0])
        return num_unique_objects >= min_segments

    def build_render_data(self, obs: Dict) -> Dict[str, np.ndarray]:
        """
        Extracts and formats final data from the observation, including bounding boxes.
        """
        render_data = super().build_render_data(obs)
        render_data["asset_id"] = self.asset_ids

        instance_mask = render_data["instance"].squeeze()
        semantic_mask = render_data["semantic"].squeeze()
        unique_instances = torch.unique(instance_mask)

        bboxes = []
        for instance_id in unique_instances:
            if instance_id == 0:  # Skip background
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
