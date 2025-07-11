from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from mops_data.generation.base_config import BaseDatasetConfig


def generate_clutter_viewpoints(
    n_viewpoints: int, random_seed: Optional[int] = None
) -> List[Dict]:
    """
    Very simple version: just sample randomly within front-biased ranges.

    Args:
        n_viewpoints: Number of viewpoints to generate
        random_seed: Optional seed for reproducibility

    Returns:
        List of viewpoint dictionaries
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    viewpoints = []

    for _ in range(n_viewpoints):
        # Elevation in Degrees
        elevation = np.random.uniform(35, 70)

        # Azimuth: front 120-degree arc centered on 0
        azimuth = np.random.uniform(-60, 60)

        viewpoints.append({"elevation": elevation, "azimuth": azimuth})

    return viewpoints


@dataclass(kw_only=True)
class KitchenDatasetConfig(BaseDatasetConfig):
    """Configuration for cluttered object dataset generation."""

    dataset_name: str = "mops_kitchen"

    # Dataset distribution
    target_train_images_per_set: int = 3000
    target_test_images_per_set: int = 1000

    camera_distance: float = 0.5

    def get_viewpoints(self, n_viewpoints: int) -> List[Dict]:
        return generate_clutter_viewpoints(
            n_viewpoints=n_viewpoints, random_seed=self.random_seed
        )
