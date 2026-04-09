"""Generate the MOPS single-object dataset.

Runs the single-object dataset pipeline, which renders isolated PartNet-Mobility
objects from multiple viewpoints with varied lighting and camera settings.

Usage
-----
# Quick debug run (few images, small resolution):
python scripts/generate_single_object.py --debug

# Full dataset generation:
python scripts/generate_single_object.py

# Custom output path:
python scripts/generate_single_object.py --output data/my_single_obj
"""

import argparse

from mops_data.generation.single_object_dataset.single_obj_config import (
    SingleObjectDatasetConfig,
)
from mops_data.generation.single_object_dataset.single_object_generation import generate

FULL_CONFIG = SingleObjectDatasetConfig(
    output_path="data/mops_data/mops_single_object_dataset.h5",
    target_train_images_per_set=40,
    target_test_images_per_set=20,
    min_assets_per_class=10,
    image_size=(512, 512),
    light_temp_range=(2000, 10000),
    light_intensity_range=(0.6, 1.5),
)

DEBUG_CONFIG = SingleObjectDatasetConfig(
    output_path="data/mops_data/debug_single_object.h5",
    target_train_images_per_set=5,
    target_test_images_per_set=5,
    min_assets_per_class=100,
    image_size=(128, 128),
    light_temp_range=(2000, 10000),
    light_intensity_range=(0.6, 1.5),
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the MOPS single-object dataset."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a small debug generation instead of the full dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the output directory path.",
    )
    args = parser.parse_args()

    config = DEBUG_CONFIG if args.debug else FULL_CONFIG
    if args.output:
        config.output_path = args.output

    generate(config)


if __name__ == "__main__":
    main()
