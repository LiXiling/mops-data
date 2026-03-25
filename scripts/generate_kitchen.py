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
python scripts/generate_kitchen.py --output data/my_kitchen.h5
"""

import argparse

from mops_data.generation.kitchen_dataset.kitchen_config import KitchenDatasetConfig
from mops_data.generation.kitchen_dataset.kitchen_generation import generate

FULL_CONFIG = KitchenDatasetConfig(
    output_path="data/mops_data/mops_kitchen_dataset.h5",
    target_train_images_per_set=3000,
    target_test_images_per_set=1000,
    min_assets_per_class=5,
    image_size=(512, 512),
    light_temp_range=(2000, 10000),
    light_intensity_range=(0.6, 1.5),
)

DEBUG_CONFIG = KitchenDatasetConfig(
    output_path="data/mops_data/debug_kitchen.h5",
    target_train_images_per_set=2,
    target_test_images_per_set=2,
    min_assets_per_class=5,
    image_size=(640, 360),
    light_temp_range=(2000, 10000),
    light_intensity_range=(0.6, 1.5),
)


def main():
    parser = argparse.ArgumentParser(description="Generate the MOPS kitchen dataset.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a small debug generation instead of the full dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override the output .h5 file path.",
    )
    args = parser.parse_args()

    config = DEBUG_CONFIG if args.debug else FULL_CONFIG
    if args.output:
        config.output_path = args.output

    generate(config)


if __name__ == "__main__":
    main()
