from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class OutputFormat(Enum):
    """Supported dataset output formats."""

    HDF5 = "hdf5"
    WEBDATASET = "webdataset"
    PARQUET = "parquet"
    DATASETS = "datasets"


@dataclass(kw_only=True)
class BaseDatasetConfig:
    """Shared configuration for all MOPS dataset generation pipelines.

    Subclasses must implement :meth:`get_viewpoints` to supply scene-appropriate
    camera positions.  Viewpoints and lighting types are lazy-initialised in
    ``__post_init__`` so subclass defaults take effect before the lists are built.
    """

    output_path: str
    dataset_name: str = "mops_dataset"
    output_format: OutputFormat = OutputFormat.DATASETS

    # Dataset distribution
    target_train_images_per_set: int = 40
    target_test_images_per_set: int = 20

    test_asset_ratio: float = 0.3
    random_seed: int = 42

    # Asset requirements
    min_assets_per_class: int = 8

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
    viewpoints: List[Dict] | None = None
    lighting_types: List[str] | None = None

    def __post_init__(self):
        if self.viewpoints is None:
            # Diverse viewpoints for good coverage
            self.viewpoints = self.get_viewpoints(n_viewpoints=48)

        if self.lighting_types is None:
            self.lighting_types = ["studio", "natural", "dramatic"]

    def get_viewpoints(self, n_viewpoints: int) -> List[Dict]:
        """Return *n_viewpoints* camera positions as dicts with ``azimuth`` and
        ``elevation`` keys (degrees).  Must be implemented by subclasses."""
        raise NotImplementedError(
            "get_viewpoint method must be implemented in subclasses"
        )


# PartNet-Mobility IDs that reliably cause simulation crashes (OOM, broken URDF,
# physics instability).  Excluded from all pipelines during asset filtering.
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
    101062,
    10686,
]
