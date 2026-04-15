import abc
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mops_data.generation.base_config import BaseDatasetConfig, OutputFormat
from mops_data.generation.variation_utils import (
    generate_base_variations,
    sample_variations_for_asset,
)


class BaseDatasetPipeline(abc.ABC):
    """Orchestrates dataset creation: asset filtering, variation sampling, and rendering.

    Concrete subclasses implement :meth:`_create_plan` (what to render) and
    :meth:`create_dataset` (how to render and write it).

    Args:
        config: Dataset configuration.
        partnet_mob_df: Full PartNet-Mobility annotations DataFrame, typically
            loaded via :func:`~mops_data.asset_manager.anno_handler.load_annotations`.
    """

    def __init__(self, config: BaseDatasetConfig, partnet_mob_df: pd.DataFrame):
        self.config = config
        self.assets_df = partnet_mob_df
        np.random.seed(self.config.random_seed)

        self.filtered_df = self._filter_classes()
        self.base_variations = generate_base_variations(
            self.config.viewpoints, self.config.lighting_types
        )
        self.plan = self._create_plan()

    def _filter_classes(self) -> pd.DataFrame:
        """Drop asset classes with fewer than ``config.min_assets_per_class`` instances."""
        df = self.assets_df.copy()

        class_counts = df["model_cat"].value_counts()
        valid_classes = class_counts[
            class_counts >= self.config.min_assets_per_class
        ].index
        df = df[df["model_cat"].isin(valid_classes)].reset_index(drop=True)

        print(f"Filtered to {len(df)} assets across {len(valid_classes)} classes.")
        return df

    def _sample_variations_for_asset(self, n_images: int) -> List[Dict]:
        """Sample variations for a single asset."""
        return sample_variations_for_asset(self.config, n_images, self.base_variations)

    @contextmanager
    def _open_writer(
        self,
        max_images_estimate: int,
        class_names: Optional[List[str]] = None,
    ):
        """Open the dataset writer configured by ``self.config.output_format``.

        Yields a writer instance with the common ``add_image()`` interface.
        """
        fmt = self.config.output_format

        if fmt == OutputFormat.HDF5:
            from mops_data.generation.hdf_writer import HDF5Writer

            with HDF5Writer(
                self.config.output_path,
                max_images_estimate=max_images_estimate,
                class_names=class_names,
            ) as writer:
                yield writer

        elif fmt == OutputFormat.WEBDATASET:
            from mops_data.generation.webdataset_writer import WebDatasetWriter

            with WebDatasetWriter(
                self.config.output_path,
                max_images_estimate=max_images_estimate,
                class_names=class_names,
            ) as writer:
                yield writer

        elif fmt == OutputFormat.PARQUET:
            from mops_data.generation.parquet_writer import ParquetWriter

            with ParquetWriter(
                self.config.output_path,
                max_images_estimate=max_images_estimate,
                class_names=class_names,
            ) as writer:
                yield writer

        else:
            raise ValueError(f"Unsupported output format: {fmt}")

    @abc.abstractmethod
    def _create_plan(self) -> Dict[str, Dict]:
        """Build a generation plan: map from scene/asset key → render spec.

        Called once in ``__init__``; result stored in ``self.plan``.
        """
        pass

    @abc.abstractmethod
    def create_dataset(self) -> None:
        """Execute the generation plan and write the dataset output."""
        pass
