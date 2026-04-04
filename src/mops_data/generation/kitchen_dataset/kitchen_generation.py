import time

import mops_data.asset_manager.anno_handler as mops_ah
from mops_data.generation.base_config import ASSET_BLACKLIST
from mops_data.generation.kitchen_dataset.kitchen_config import KitchenDatasetConfig
from mops_data.generation.kitchen_dataset.kitchen_pipeline import KitchenDatasetPipeline


def generate(dataset_config: KitchenDatasetConfig):
    """Run the kitchen scene generation pipeline end-to-end.

    Filters to non-large PartNet objects, applies the asset blacklist, builds a
    :class:`KitchenDatasetPipeline`, and calls ``create_dataset()``.
    Prefer invoking via ``scripts/generate_kitchen.py`` for CLI convenience.

    Args:
        dataset_config: Fully-configured :class:`KitchenDatasetConfig`.
    """
    print("--- Kitchen Dataset Generation Script ---")
    start_time = time.time()
    print(
        f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )

    df = mops_ah.load_annotations().partnet_mobility_df

    # TableTop Clutter - don't use large objects
    df = df[~df["is_large_object"]]
    print(df.shape)
    df = df.groupby("model_id").first().reset_index()

    # Blacklist of assets that are known to cause issues
    df = df[~df["dir_name"].isin(ASSET_BLACKLIST)]

    config = dataset_config

    pipeline = KitchenDatasetPipeline(config, df)
    pipeline.create_dataset()

    end_time = time.time()
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    elapsed = end_time - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")


if __name__ == "__main__":
    FULL_DATASET_CONFIG = KitchenDatasetConfig(
        output_path="data/mops_data/mops_kitchen_dataset_v2.h5",
        target_train_images_per_set=3000,
        target_test_images_per_set=1000,
        min_assets_per_class=5,
        image_size=(512, 512),
        light_temp_range=(2000, 10000),
        light_intensity_range=(0.6, 1.5),
    )

    DEBUG_DATASET_CONFIG = KitchenDatasetConfig(
        output_path="data/debug_mops.h5",
        target_train_images_per_set=2,
        target_test_images_per_set=2,
        min_assets_per_class=5,
        image_size=(640, 360),
        light_temp_range=(2000, 10000),
        light_intensity_range=(0.6, 1.5),
    )

    generate(DEBUG_DATASET_CONFIG)
