"""Generate the MOPS kitchen (affordance) dataset.

Runs the kitchen dataset pipeline, which renders objects placed inside RoboCasa
kitchen environments from overhead/table-level viewpoints with varied lighting.

Usage
-----
# Quick debug run (few images, small resolution):
python scripts/generate_kitchen.py --debug

# Full dataset generation:
python scripts/generate_kitchen.py

# Custom output path:
python scripts/generate_kitchen.py --output data/my_kitchen
"""

import argparse

from mops_data.generation.base_config import OutputFormat
from mops_data.generation.kitchen_dataset.kitchen_config import KitchenDatasetConfig
from mops_data.generation.kitchen_dataset.kitchen_generation import generate

FULL_CONFIG = KitchenDatasetConfig(
    output_path="data/mops_data/mops_kitchen_dataset_100k",
    target_train_images_per_set=90000,
    target_test_images_per_set=10000,
    min_assets_per_class=5,
    image_size=(640, 480),
    light_temp_range=(2000, 10000),
    light_intensity_range=(0.6, 1.5),
    obs_mode="rgb+segmentation+depth+normal",
)

FIVE_CONFIG = KitchenDatasetConfig(
    output_path="data/mops_data/mops_kitchen_dataset_5k",
    target_train_images_per_set=4000,
    target_test_images_per_set=1000,
    min_assets_per_class=5,
    image_size=(640, 480),
    light_temp_range=(2000, 10000),
    light_intensity_range=(0.6, 1.5),
    obs_mode="rgb+segmentation+depth+normal",
)

DEBUG_CONFIG = KitchenDatasetConfig(
    output_path="data/mops_data/debug_kitchen",
    target_train_images_per_set=2,
    target_test_images_per_set=2,
    min_assets_per_class=5,
    image_size=(640, 480),
    light_temp_range=(2000, 10000),
    light_intensity_range=(0.6, 1.5),
    obs_mode="rgb+depth+segmentation+normal",
)


def main():
    parser = argparse.ArgumentParser(description="Generate the MOPS kitchen dataset.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a small debug generation instead of the full dataset.",
    )
    parser.add_argument(
        "--five",
        action="store_true",
        help="Run a small 5k generation instead of the full dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the output directory path.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["webdataset", "hdf5"],
        default="webdataset",
        help="Output format (default: webdataset).",
    )
    args = parser.parse_args()

    config = DEBUG_CONFIG if args.debug else FIVE_CONFIG if args.five else FULL_CONFIG
    if args.output:
        config.output_path = args.output
    config.output_format = OutputFormat(args.format)

    generate(config)


if __name__ == "__main__":
    main()
