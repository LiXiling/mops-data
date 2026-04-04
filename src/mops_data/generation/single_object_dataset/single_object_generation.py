import time

import mops_data.asset_manager.anno_handler as mops_ah
from mops_data.generation.base_config import ASSET_BLACKLIST
from mops_data.generation.single_object_dataset.single_obj_config import (
    SingleObjectDatasetConfig,
)
from mops_data.generation.single_object_dataset.single_object_pipeline import (
    BalancedSingleObjectDatasetPipeline,
)


def generate(dataset_config: SingleObjectDatasetConfig):
    """Run the single-object generation pipeline end-to-end.

    Loads PartNet-Mobility annotations, applies the asset blacklist, builds a
    :class:`BalancedSingleObjectDatasetPipeline`, and calls ``create_dataset()``.
    Prefer invoking via ``scripts/generate_single_object.py`` for CLI convenience.

    Args:
        dataset_config: Fully-configured :class:`SingleObjectDatasetConfig`.
    """
    print("--- Single Object Dataset Generation Script ---")
    start_time = time.time()
    print(
        f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )

    df = mops_ah.load_annotations().partnet_mobility_df
    df = df.groupby("model_id").first().reset_index()

    # Blacklist of assets that are known to cause issues
    df = df[~df["dir_name"].isin(ASSET_BLACKLIST)]

    config = dataset_config

    pipeline = BalancedSingleObjectDatasetPipeline(config, df)
    pipeline.create_dataset()

    end_time = time.time()
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    elapsed = end_time - start_time
    print(f"Total elapsed time: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")


if __name__ == "__main__":
    FULL_DATASET_CONFIG = SingleObjectDatasetConfig(
        output_path="data/mops_data/mops_single_dataset_big_v2.h5",
        target_train_images_per_set=40,
        target_test_images_per_set=20,
        min_assets_per_class=10,
        image_size=(512, 512),
        light_temp_range=(2000, 10000),
        light_intensity_range=(0.6, 1.5),
    )

    DEBUG_DATASET_CONFIG = SingleObjectDatasetConfig(
        output_path="data/mops_data/debug_mops.h5",
        target_train_images_per_set=5,
        target_test_images_per_set=5,
        min_assets_per_class=100,
        image_size=(128, 128),
        light_temp_range=(2000, 10000),
        light_intensity_range=(0.6, 1.5),
    )

    generate(DEBUG_DATASET_CONFIG)
