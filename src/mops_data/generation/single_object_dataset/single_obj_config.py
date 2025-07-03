from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from mops_data.generation.base_config import BaseDatasetConfig


def generate_simple_front_biased_viewpoints(
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
        # Elevation in degrees
        elevation = np.random.uniform(-30, 30)

        # Azimuth: front 120-degree arc centered on 0
        azimuth = np.random.uniform(-60, 60)

        viewpoints.append({"elevation": elevation, "azimuth": azimuth})

    return viewpoints


@dataclass(kw_only=True)
class SingleObjectDatasetConfig(BaseDatasetConfig):
    """Configuration for single object dataset generation."""

    dataset_name: str = "mops_single_object"

    # Dataset distribution
    target_train_images_per_set: int = 40
    target_test_images_per_set: int = 20

    def get_viewpoints(self, n_viewpoints: int) -> List[Dict]:
        return generate_simple_front_biased_viewpoints(
            n_viewpoints=n_viewpoints, random_seed=self.random_seed
        )
