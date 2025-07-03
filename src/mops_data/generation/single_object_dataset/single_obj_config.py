from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


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
        # Elevation: 10-40 degrees (avoid top-down and ground level)
        elevation = np.random.uniform(-30, 30)

        # Azimuth: front 120-degree arc centered on 0
        # This gives us -60 to +60 degrees as you suggested
        azimuth = np.random.uniform(-60, 60)

        viewpoints.append({"elevation": elevation, "azimuth": azimuth})

    return viewpoints


@dataclass
class SingleObjectDatasetConfig:
    """Configuration for single object dataset generation."""

    output_path: str
    dataset_name: str = "mops_single_object"

    # Dataset distribution
    target_train_images_per_class: int = 30
    target_test_images_per_class: int = 15
    test_asset_ratio: float = 0.3
    random_seed: int = 42

    # Asset requirements
    min_assets_per_class: int = 8
    classes_to_include: Optional[List[str]] = None

    # Rendering parameters
    image_size: Tuple[int, int] = (512, 512)
    camera_distance: float = 1.5
    obs_mode: str = "rgb+depth+segmentation+normal"

    # Validation
    min_segments_threshold: int = 3
    max_resampling_attempts: int = 3  # Max attempts to get valid render

    # Lighting variation ranges
    light_temp_range: Tuple[float, float] = (2700, 8000)  # Warm to cool daylight
    light_intensity_range: Tuple[float, float] = (0.8, 1.2)

    # Generation parameters
    viewpoints: List[Dict] = None
    lighting_types: List[str] = None

    def __post_init__(self):
        if self.viewpoints is None:
            # Diverse viewpoints for good coverage
            self.viewpoints = generate_simple_front_biased_viewpoints(n_viewpoints=48)

        if self.lighting_types is None:
            self.lighting_types = ["studio", "natural", "dramatic"]


ASSET_BLACKLIST = [
    10356,
    10546,
    11260,
    11538,
    11887,
    12071,
    12115,
    12542,
    12578,
    12584,
    12612,
    23724,
    25144,
    2780,
    29525,
    30857,
    32213,
    32746,
    3380,
    3593,
    39138,
    43142,
    7130,
    7138,
    7221,
    7306,
    7320,
    7347,
    8966,
    9918,
    9987,
    40069,
    41434,
]
