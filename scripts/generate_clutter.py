"""Generate the MOPS tabletop clutter dataset.

Runs the clutter dataset pipeline, which renders cluttered tabletop scenes with
multiple PartNet-Mobility objects from top-down viewpoints with varied lighting.

Usage
-----
# Quick debug run (few images, small resolution):
python scripts/generate_clutter.py --debug

# Full dataset generation:
python scripts/generate_clutter.py

# Custom output path:
python scripts/generate_clutter.py --output data/my_clutter
"""

import argparse

from mops_data.generation.base_config import OutputFormat
from mops_data.generation.clutter_dataset.clutter_config import ClutterDatasetConfig
from mops_data.generation.clutter_dataset.clutter_generation import generate

FULL_CONFIG = ClutterDatasetConfig(
    output_path="data/mops_data/mops_clutter_dataset_5k",
    target_train_images_per_set=4000,
    target_test_images_per_set=1000,
    min_assets_per_class=5,
    image_size=(640, 480),
    light_temp_range=(2000, 10000),
    light_intensity_range=(0.6, 1.5),
    obs_mode="rgb+segmentation+depth+normal",
)

DEBUG_CONFIG = ClutterDatasetConfig(
    output_path="data/mops_data/debug_clutter",
    target_train_images_per_set=5,
    target_test_images_per_set=5,
    min_assets_per_class=5,
    image_size=(128, 128),
    light_temp_range=(2000, 10000),
    light_intensity_range=(0.6, 1.5),
    obs_mode="rgb+segmentation+depth+normal",
)


def main():
    parser = argparse.ArgumentParser(description="Generate the MOPS clutter dataset.")
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
    parser.add_argument(
        "--format",
        type=str,
        choices=["webdataset", "hdf5"],
        default="webdataset",
        help="Output format (default: webdataset).",
    )
    args = parser.parse_args()

    config = DEBUG_CONFIG if args.debug else FULL_CONFIG
    if args.output:
        config.output_path = args.output
    config.output_format = OutputFormat(args.format)

    generate(config)


if __name__ == "__main__":
    main()
