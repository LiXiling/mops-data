import itertools
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mops_data.generation.base_pipeline import BaseDatasetPipeline
from mops_data.generation.hdf_writer import HDF5Writer
from mops_data.generation.subprocess_renderer import (
    SPLIT_SEED_OFFSETS,
    render_in_subprocess,
)

ENV_ID = "SingleObjectRenderEnv-v1"
ENV_MODULE = "mops_data.envs.dataset_envs"


class BalancedSingleObjectDatasetPipeline(BaseDatasetPipeline):
    """Pipeline for generating balanced single object dataset."""

    def _create_plan(self) -> Dict[str, Dict]:
        """Create a balanced generation plan for each class."""
        plan = {}
        for class_name in self.filtered_df["model_cat"].unique():
            class_assets = self.filtered_df[self.filtered_df["model_cat"] == class_name]
            n_test = max(1, int(len(class_assets) * self.config.test_asset_ratio))

            train_assets, test_assets = train_test_split(
                class_assets, test_size=n_test, random_state=self.config.random_seed
            )

            plan[class_name] = {
                "train_assets": train_assets.to_dict("records"),
                "test_assets": test_assets.to_dict("records"),
                "target_train": self.config.target_train_images_per_set,
                "target_test": self.config.target_test_images_per_set,
            }
        return plan

    def _build_env_kwargs(self, asset_info: Dict, variation: Dict) -> Dict:
        """Build kwargs dict for gym.make (must be picklable)."""
        return {
            k: v
            for k, v in {
                "render_mode": "rgb_array",
                "obs_mode": self.config.obs_mode,
                "image_size": self.config.image_size,
                "camera_distance": self.config.camera_distance,
                "camera_elevation": variation["viewpoint"]["elevation"],
                "camera_azimuth": variation["viewpoint"]["azimuth"],
                "lighting_type": variation["lighting"]["type"],
                "lighting_intensity": variation["lighting"]["intensity"],
                "light_temperature": variation["lighting"]["temperature"],
                "sensor_configs": dict(shader_pack="rt"),
                "mob_id": asset_info.get("dir_name"),
            }.items()
            if v is not None
        }

    def _render_with_retry(
        self,
        asset_info: Dict,
        variations: list,
        image_seed: int,
    ) -> tuple:
        """Render in a subprocess, retrying with new variations on failure."""
        attempts = [
            {
                "env_kwargs": self._build_env_kwargs(asset_info, var),
                "seed": image_seed + attempt_idx,
                "num_steps": 3,
                "min_segments": self.config.min_segments_threshold,
            }
            for attempt_idx, var in enumerate(variations)
        ]

        data, attempt_idx = render_in_subprocess(ENV_ID, ENV_MODULE, attempts)
        if data is not None:
            return data, variations[attempt_idx]
        return None, None

    def _generate_images_for_class_split(
        self,
        writer: HDF5Writer,
        assets: List[Dict],
        target_count: int,
        split: str,
        class_name: str,
    ):
        """Generate and save images for a specific class and split."""
        pbar = tqdm(
            total=target_count, desc=f"  {split.capitalize():<5} images", unit="img"
        )

        split_offset = SPLIT_SEED_OFFSETS[split]
        asset_cycler = itertools.cycle(enumerate(assets))
        generated_count = 0
        attempt_index = 0

        while generated_count < target_count:
            image_seed = self.config.random_seed + split_offset + attempt_index
            _, asset_info = next(asset_cycler)

            np.random.seed(image_seed)
            variations = self._sample_variations_for_asset(
                self.config.max_resampling_attempts
            )

            render_data, variation = self._render_with_retry(
                asset_info, variations, image_seed
            )
            attempt_index += 1

            if render_data is None:
                continue

            render_params = {
                "split": split,
                "variation": variation,
                "image_size": self.config.image_size,
            }
            asset_id = asset_info["dir_name"]

            writer.add_image(
                class_name=class_name,
                asset_id=asset_id,
                render_params=render_params,
                **render_data,
            )
            generated_count += 1
            pbar.update(1)

        pbar.close()

    def create_dataset(self):
        """Create the balanced dataset by rendering assets and writing to HDF5."""
        total_images = sum(
            plan["target_train"] + plan["target_test"] for plan in self.plan.values()
        )
        print(f"\n=== CREATING DATASET: {self.config.output_path} ===")
        print(f"Estimated total images: {total_images}")

        try:
            with HDF5Writer(
                self.config.output_path,
                int(total_images * 1.1),  # Pre-allocate with 10% buffer
                list(self.plan.keys()),
            ) as writer:
                for class_name, plan in tqdm(
                    self.plan.items(), desc="Total Progress", unit="class"
                ):
                    print(f"\nProcessing class: {class_name}")

                    # Generate training images
                    self._generate_images_for_class_split(
                        writer,
                        plan["train_assets"],
                        plan["target_train"],
                        "train",
                        class_name,
                    )
                    # Generate testing images
                    self._generate_images_for_class_split(
                        writer,
                        plan["test_assets"],
                        plan["target_test"],
                        "test",
                        class_name,
                    )

        except Exception as e:
            print(f"\nFATAL ERROR during dataset creation: {e}")
            raise

        print("\n=== DATASET CREATION COMPLETE ===")
        print(f"File saved to: {self.config.output_path}")
