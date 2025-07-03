import abc
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mops_data.generation.base_config import BaseDatasetConfig


class BaseDatasetPipeline(abc.ABC):
    """Base class for dataset generation pipelines."""

    def __init__(self, config: BaseDatasetConfig, partnet_mob_df: pd.DataFrame):
        self.config = config
        self.assets_df = partnet_mob_df
        np.random.seed(self.config.random_seed)

        self.filtered_df = self._filter_classes()
        self.base_variations = self._generate_base_variations()
        self.plan = self._create_plan()

    def _filter_classes(self) -> pd.DataFrame:
        """Filter assets to ensure each class has enough samples."""
        df = self.assets_df.copy()

        class_counts = df["model_cat"].value_counts()
        valid_classes = class_counts[
            class_counts >= self.config.min_assets_per_class
        ].index
        df = df[df["model_cat"].isin(valid_classes)].reset_index(drop=True)

        print(f"Filtered to {len(df)} assets across {len(valid_classes)} classes.")
        return df

    def _generate_base_variations(self) -> List[Dict]:
        """Generate base variation combinations from viewpoints and lighting types."""
        return [
            {"viewpoint": vp, "lighting": lt}
            for vp, lt in itertools.product(
                self.config.viewpoints, self.config.lighting_types
            )
        ]

    @abc.abstractmethod
    def _create_plan(self) -> Dict[str, Dict]:
        """Create a dataset generation plan."""
        pass
