import itertools
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mops_data.generation.base_pipeline import BaseDatasetPipeline
from mops_data.generation.hdf_writer import HDF5Writer
from mops_data.generation.single_object_dataset.single_obj_config import (
    SingleObjectDatasetConfig,
)


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

    def _create_render_env(self, asset_info: Dict, variation: Dict):
        """Create a render environment for a given asset and variation."""
        env_kwargs = {
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
        }
        return gym.make(
            "DatasetRenderEnv-v1",
            **{k: v for k, v in env_kwargs.items() if v is not None},
        )

    def _render_with_retry(self, asset_info: Dict) -> Optional[Dict[str, np.ndarray]]:
        """Render an asset, retrying with new variations on failure or low quality."""

        resampling_attempts = self.config.max_resampling_attempts

        asset_id_str = str(
            asset_info.get("model_id") or asset_info.get("dir_name", "N/A")
        )
        variations = self._sample_variations_for_asset(resampling_attempts)
        for attempt in range(resampling_attempts):
            current_variation = variations[attempt]
            env = self._create_render_env(asset_info, current_variation)
            try:
                obs, _ = env.reset(seed=self.config.random_seed + attempt)
                # Step environment a few times for stability
                for _ in range(3):
                    obs, _, _, _, _ = env.step(None)

                if env.unwrapped.is_valid_render(
                    obs, self.config.min_segments_threshold
                ):
                    obs = env.unwrapped.extract_render_data(obs)
                    env.close()
                    return obs, current_variation

                print(
                    f"Warning: Low quality render for {asset_id_str} (attempt {attempt+1}). Resampling variation."
                )
            except Exception as e:
                print(
                    f"Error rendering {asset_id_str} (attempt {attempt+1}): {e}. Retrying..."
                )
            finally:
                env.close()

        print(
            f"Error: Failed to get a valid render for {asset_id_str} after {self.config.max_resampling_attempts} attempts."
        )
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

        # Use itertools.cycle to loop through assets until target count is met
        asset_cycler = itertools.cycle(enumerate(assets))
        generated_count = 0

        while generated_count < target_count:
            asset_idx, asset_info = next(asset_cycler)

            # If Render is invalid, try the next asset
            render_data, variation = self._render_with_retry(asset_info)
            if render_data is None:
                continue

            render_params = {
                "split": split,
                "variation": variation,
                "image_size": self.config.image_size,
            }
            asset_id = str(asset_info.get("model_id", f"asset_{asset_idx}"))

            # Save Image
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
                list(self.plan.keys()),
                int(total_images * 1.1),  # Pre-allocate with 10% buffer
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

        print(f"\n=== DATASET CREATION COMPLETE ===")
        print(f"File saved to: {self.config.output_path}")
