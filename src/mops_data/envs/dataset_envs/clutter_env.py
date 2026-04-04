from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from .base_rendering_env import DatasetRenderEnv


@register_env("ClutterRenderEnv-v1", max_episode_steps=1)
class ClutterEnv(DatasetRenderEnv):
    """Top-down tabletop clutter rendering environment.

    Scatters 8–15 random PartNet-Mobility objects on a table surface and
    renders from overhead viewpoints.  Produces bounding boxes alongside
    the standard segmentation/depth/affordance masks.
    """

    SUPPORTED_ROBOTS = ["none"]

    def __init__(
        self,
        *args,
        # Asset specification
        asset_df: pd.DataFrame = None,
        **kwargs,
    ):
        self.asset_df = asset_df
        self.asset_ids = []
        super().__init__(*args, **kwargs)

    def _load_objects(self, options: Dict[str, Any]):
        """Load scene with the specified object"""

        self.premade_scene = TableSceneBuilder(env=self)
        self.premade_scene.build()

        # random number 8 - 15
        n_assets = np.random.randint(8, 16)

        for i in range(n_assets):
            # Randomly select an asset from the DataFrame
            asset = self.asset_df.sample(1).iloc[0]
            mob_id = asset["dir_name"]
            self.asset_ids.append(str(mob_id))  # String for JSON serialization
            object_position = np.random.uniform(-0.5, 0.5, size=3)
            object_position[2] = 0.05 * (i + 1)

            object_euler = np.random.uniform(
                low=[-np.pi, -np.pi, -np.pi],
                high=[np.pi, np.pi, np.pi],
                size=3,
            )

            # Load the selected asset
            self.partnet_mobility_loader.load(
                mob_id,
                object_position,
                euler=object_euler,
            )

    def is_valid_render(self, obs: Dict, min_segments: int = 3) -> bool:
        """
        Check if render is valid based on part segmentation complexity.

        Args:
            obs: Observation dictionary
            min_segments: Minimum number of unique segmentation values required

        Returns:
            True if render has sufficient segmentation detail
        """
        camera_obs = obs["sensor_data"]["base_camera"]
        part_mask = camera_obs["instance_segmentation"].cpu()[0]
        unique_values = np.unique(part_mask)

        return len(unique_values) >= min_segments

    def build_render_data(self, obs: Dict) -> Dict[str, np.ndarray]:
        obs = super().build_render_data(obs)
        obs["asset_id"] = self.asset_ids

        instance_mask = obs["instance"].squeeze()
        semantic_mask = obs["semantic"].squeeze()

        unique_instances = torch.unique(instance_mask)

        bboxes = []
        for instance_id in unique_instances:
            # Skip the background instance, which is typically 0
            if instance_id == 0:
                continue

            # Create a mask for the current instance
            mask = instance_mask == instance_id
            if not torch.any(mask):
                continue

            # Find all pixel coordinates for the instance
            y_indices, x_indices = torch.where(mask)

            # Calculate bounding box coordinates
            min_y, max_y = torch.min(y_indices), torch.max(y_indices)
            min_x, max_x = torch.min(x_indices), torch.max(x_indices)

            # Convert to XYWH format
            x = min_x.item()
            y = min_y.item()
            w = (max_x - min_x + 1).item()
            h = (max_y - min_y + 1).item()

            # Get the semantic class ID for this instance.
            # All pixels of an instance have the same semantic ID.
            class_id = semantic_mask[y_indices[0], x_indices[0]].item()

            bboxes.append([x, y, w, h, class_id])

        # Store as a single NumPy array
        obs["bbox"] = np.array(bboxes, dtype=np.int32)
        return obs
