from typing import Any, Dict, Optional

import numpy as np
from mani_skill.utils.registration import register_env

from mops_data.envs.dataset_envs.base_rendering_env import DatasetRenderEnv


@register_env("SingleObjectRenderEnv-v1", max_episode_steps=10)
class SingleObjectRenderEnv(DatasetRenderEnv):
    """
    Simplified rendering environment for dataset generation.

    Loads PartNet Mobility objects and renders them with configurable
    camera, lighting, and background settings.
    """

    SUPPORTED_ROBOTS = ["none"]

    def __init__(
        self,
        *args,
        # Asset specification
        mob_id: Optional[str] = None,
        object_scale: float = 0.8,
        object_position: Optional[np.ndarray] = None,
        **kwargs,
    ):
        self.mob_id = mob_id
        self.object_scale = object_scale
        self.object_position = (
            object_position
            if object_position is not None
            else np.array([0.0, 0.0, 0.0])
        )

        super().__init__(*args, **kwargs)

    def _load_objects(self, options: Dict[str, Any]):
        """Load scene with the specified object"""
        # Load object using available identifier
        self.partnet_mobility_loader.load(
            self.mob_id,
            self.object_position,
            no_grav=True,
            scale=self.object_scale,
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
        part_mask = camera_obs["segmentation"].cpu()[0]
        unique_values = np.unique(part_mask)

        return len(unique_values) >= min_segments
