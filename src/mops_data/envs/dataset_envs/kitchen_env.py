from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sapien
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
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

    def __init__(self, *args, asset_df: pd.DataFrame = None, **kwargs):
        self.asset_df = asset_df
        self.asset_ids = []
        self.target_fixture = None  # The fixture the camera will look at
        super().__init__(*args, **kwargs)

    def _load_objects(self, options: Dict[str, Any]):
        """
        Load a RoboCasa kitchen and place custom objects on its counter surfaces
        using the builder's helper functions.
        """
        self.scene_builder = RoboCasaSceneBuilder(self)
        self.scene_builder.build()

        cluttered_counters: List[Counter] = []

        # 1. Iterate through the builder's fixtures to find counters
        fixtures: dict = self.scene_builder.scene_data[0]["fixtures"]
        for name, fixture in fixtures.items():
            if "counter" not in name:
                continue

            # 2. Safely get all valid placement regions for the fixture.
            # Some "counters" might not have any valid placement areas.
            reset_regions = fixture.get_reset_regions(env=self, fixtures=fixtures)
            if not reset_regions:
                continue  # Skip this fixture if it has no valid regions.

            # 3. Choose one of the valid regions to place objects in.
            # We can now be sure that the list of regions is not empty.
            reset_region = self._episode_rng.choice(list(reset_regions.values()))
            size = reset_region["size"]
            offset = reset_region["offset"]

            # Define the placement area based on the region's properties
            x_range = np.array([-size[0] / 2, size[0] / 2]) + offset[0]
            y_range = np.array([-size[1] / 2, size[1] / 2]) + offset[1]
            z_pos = fixture.pos[2] + offset[2]

            # 3. Place a random number of custom assets in this valid region
            n_assets = self._episode_rng.randint(2, 4)
            for _ in range(n_assets):
                asset_info = self.asset_df.sample(1).iloc[0]
                asset_id = asset_info["dir_name"]
                self.asset_ids.append(str(asset_id))

                pos = np.array(
                    [
                        self._episode_rng.uniform(*x_range),
                        self._episode_rng.uniform(*y_range),
                        z_pos,
                    ]
                )
                euler = self._episode_rng.uniform(-np.pi, np.pi, size=3)
                self.partnet_mobility_loader.load(asset_id, pos, euler=euler)

            cluttered_counters.append(fixture)

        # 4. Pick one of the cluttered counters as the camera's focal point
        if cluttered_counters:
            self.target_fixture = self._episode_rng.choice(cluttered_counters)
            print(self.target_fixture)

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
            camera_pos[0] += self._episode_rng.uniform(-0.3, 0.3)
            camera_pos[1] += self._episode_rng.uniform(-0.3, 0.3)

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
