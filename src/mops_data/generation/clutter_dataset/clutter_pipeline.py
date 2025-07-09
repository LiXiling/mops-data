import itertools
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mops_data.envs.dataset_envs.base_rendering_env import DatasetRenderEnv
from mops_data.generation.base_pipeline import BaseDatasetPipeline
from mops_data.generation.hdf_writer import HDF5Writer


class ClutterDatasetPipeline(BaseDatasetPipeline):
    """Pipeline for generating cluttered dataset."""

    def _create_plan(self) -> Dict[str, Dict]:
        """Create a balanced generation plan for each class."""
        train_dfs, test_dfs = [], []
        for class_name in self.filtered_df["model_cat"].unique():
            class_assets = self.filtered_df[self.filtered_df["model_cat"] == class_name]
            n_test = max(1, int(len(class_assets) * self.config.test_asset_ratio))

            train_assets, test_assets = train_test_split(
                class_assets, test_size=n_test, random_state=self.config.random_seed
            )

            train_dfs.append(train_assets)
            test_dfs.append(test_assets)

        train_df = pd.concat(train_dfs)
        test_df = pd.concat(test_dfs)

        plan = {
            "train_assets": train_df,
            "test_assets": test_df,
        }
        return plan

    def _create_render_env(self, asset_df: pd.DataFrame, variation: Dict):
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
            "asset_df": asset_df,
        }
        return gym.make(
            "ClutterRenderEnv-v1",
            **{k: v for k, v in env_kwargs.items() if v is not None},
        )

    def _render_with_retry(
        self, asset_df: pd.DataFrame
    ) -> Optional[Dict[str, np.ndarray]]:
        """Render an asset_subset, retrying with new variations on failure or low quality."""

        resampling_attempts = self.config.max_resampling_attempts
        variations = self._sample_variations_for_asset(resampling_attempts)
        for attempt in range(resampling_attempts):
            current_variation = variations[attempt]
            gym_env = self._create_render_env(asset_df, current_variation)
            try:
                obs, _ = gym_env.reset(seed=self.config.random_seed + attempt)
                # Step environment a few times for stability
                for _ in range(15):
                    obs, _, _, _, _ = gym_env.step(None)

                render_env: DatasetRenderEnv = gym_env.unwrapped

                if render_env.is_valid_render(obs, self.config.min_segments_threshold):
                    obs = render_env.build_render_data(obs)
                    return obs, current_variation

                print(
                    f"Warning: Low quality render. (attempt {attempt+1}). Resampling variation."
                )
            except Exception as e:
                print(f"Error rendering. (attempt {attempt+1}): {e}. Retrying...")
                raise e
            finally:
                gym_env.close()

        print(
            f"Error: Failed to get a valid render after {self.config.max_resampling_attempts} attempts."
        )
        return None, None

    def _generate_images_for_class_split(
        self,
        writer: HDF5Writer,
        assets: pd.DataFrame,
        target_count: int,
        split: str,
        class_name: str,
    ):
        """Generate and save images for a specific class and split."""
        pbar = tqdm(
            total=target_count, desc=f"  {split.capitalize():<5} images", unit="img"
        )

        generated_count = 0

        while generated_count < target_count:
            random_set = assets.sample(40)

            # If Render is invalid, try the next asset
            render_data, variation = self._render_with_retry(random_set)
            if render_data is None:
                continue

            render_params = {
                "split": split,
                "variation": variation,
                "image_size": self.config.image_size,
                # TODO: Mulitple Classes, Multiple IDs
            }

            # Save Image
            writer.add_image(
                class_name=class_name,
                render_params=render_params,
                **render_data,
            )
            generated_count += 1
            pbar.update(1)

        pbar.close()

    def create_dataset(self):
        """Create the balanced dataset by rendering assets and writing to HDF5."""
        total_images = (
            self.config.target_train_images_per_set
            + self.config.target_test_images_per_set
        )

        print(f"\n=== CREATING DATASET: {self.config.output_path} ===")
        print(f"Estimated total images: {total_images}")

        try:
            with HDF5Writer(
                self.config.output_path,
                int(total_images * 1.1),  # Pre-allocate with 10% buffer
            ) as writer:
                print("\n=== STARTING TRAIN DATASET CREATION ===")

                self._generate_images_for_class_split(
                    writer,
                    self.plan["train_assets"],
                    self.config.target_train_images_per_set,
                    "train",
                    "train_set",
                )

                self._generate_images_for_class_split(
                    writer,
                    self.plan["test_assets"],
                    self.config.target_test_images_per_set,
                    "test",
                    "test_set",
                )

        except Exception as e:
            print(f"\nFATAL ERROR during dataset creation: {e}")
            raise

        print(f"\n=== DATASET CREATION COMPLETE ===")
        print(f"File saved to: {self.config.output_path}")
