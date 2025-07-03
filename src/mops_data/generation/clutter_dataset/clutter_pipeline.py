import itertools
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mops_data.generation.clutter_dataset.clutter_config import ClutterDatasetConfig
from mops_data.generation.hdf_writer import HDF5Writer


class BalancedDatasetPipeline:
    """Pipeline for generating balanced single object dataset."""

    def __init__(self, config: ClutterDatasetConfig, partnet_mob_df: pd.DataFrame):
        self.config = config
        self.assets_df = partnet_mob_df
        np.random.seed(self.config.random_seed)

        self.filtered_df = self._filter_classes()
        self.balanced_plan = self._create_balanced_plan()
        self.base_variations = self._generate_base_variations()
        self._print_dataset_plan()

    def _filter_classes(self) -> pd.DataFrame:
        """Filter classes to those with enough assets."""
        df = self.assets_df.copy()

        class_counts = df["model_cat"].value_counts()
        valid_classes = class_counts[
            class_counts >= self.config.min_assets_per_class
        ].index
        df = df[df["model_cat"].isin(valid_classes)].reset_index(drop=True)

        print(f"Filtered to {len(df)} assets across {len(valid_classes)} classes.")
        return df

    def _create_balanced_plan(self) -> Dict[str, Dict]:
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
                "target_train": self.config.target_train_images_per_class,
                "target_test": self.config.target_test_images_per_class,
            }
        return plan

    def _generate_base_variations(self) -> List[Dict]:
        """Generate base variation combinations from viewpoints and lighting types."""
        return [
            {"viewpoint": vp, "lighting": lt}
            for vp, lt in itertools.product(
                self.config.viewpoints, self.config.lighting_types
            )
        ]

    def _sample_variations_for_asset(self, n_images: int) -> List[Dict]:
        """Sample n variations for a single asset with random lighting."""
        num_base = len(self.base_variations)
        indices = np.random.choice(num_base, size=n_images, replace=n_images > num_base)

        variations = []
        for i in indices:
            var = self.base_variations[i].copy()
            var["viewpoint"] = var["viewpoint"].copy()
            # Add small random perturbations to each sampled viewpoint
            var["viewpoint"]["azimuth"] += np.random.uniform(-10, 10)
            var["viewpoint"]["elevation"] += np.random.uniform(-5, 5)
            var["lighting"] = self._sample_random_lighting()
            variations.append(var)

        return variations

    def _sample_random_lighting(self) -> Dict:
        """Sample random lighting parameters."""
        return {
            "type": np.random.choice(self.config.lighting_types),
            "temperature": np.random.uniform(*self.config.light_temp_range),
            "intensity": np.random.uniform(*self.config.light_intensity_range),
        }

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

    def _render_with_retry(
        self, asset_info: Dict, variation: Dict
    ) -> Optional[Dict[str, np.ndarray]]:
        """Render an asset, retrying with new variations on failure or low quality."""
        asset_id_str = str(
            asset_info.get("model_id") or asset_info.get("dir_name", "N/A")
        )
        current_variation = variation.copy()
        for attempt in range(self.config.max_resampling_attempts):
            env = self._create_render_env(asset_info, current_variation)
            try:
                obs, _ = env.reset(seed=self.config.random_seed + attempt)
                # Step environment a few times for stability
                for _ in range(3):
                    obs, _, terminated, _, _ = env.step(None)
                    if terminated:
                        break

                if env.unwrapped.is_valid_render(
                    obs, self.config.min_segments_threshold
                ):
                    return env.unwrapped.extract_render_data(obs)

                print(
                    f"Warning: Low quality render for {asset_id_str} (attempt {attempt+1}). Resampling variation."
                )
                # Resample a new variation for the next attempt
                current_variation = self._sample_variations_for_asset(1)[0]

            except Exception as e:
                print(
                    f"Error rendering {asset_id_str} (attempt {attempt+1}): {e}. Retrying..."
                )
                current_variation = self._sample_variations_for_asset(1)[0]
            finally:
                obs = env.unwrapped.extract_render_data(obs)
                env.close()

        print(
            f"Error: Failed to get a valid render for {asset_id_str} after {self.config.max_resampling_attempts} attempts."
        )
        return obs

    def _generate_images_for_class_split(
        self,
        writer: HDF5Writer,
        assets: List[Dict],
        target_count: int,
        split: str,
        class_name: str,
    ):
        """Generate and save images for a specific class and split."""
        if not assets:
            print(f"    No assets available for {class_name} {split} split. Skipping.")
            return

        pbar = tqdm(
            total=target_count, desc=f"  {split.capitalize():<5} images", unit="img"
        )

        # Use itertools.cycle to loop through assets until target count is met
        asset_cycler = itertools.cycle(enumerate(assets))
        generated_count = 0

        while generated_count < target_count:
            asset_idx, asset_info = next(asset_cycler)
            variation = self._sample_variations_for_asset(1)[0]

            render_data = self._render_with_retry(asset_info, variation)
            if render_data is None:
                continue

            render_params = {
                "split": split,
                "variation": variation,
                "image_size": self.config.image_size,
            }
            asset_id = str(asset_info.get("model_id", f"asset_{asset_idx}"))

            writer.add_image(
                class_name=class_name,
                asset_id=asset_id,
                render_params=render_params,
                **render_data,
            )
            generated_count += 1
            pbar.update(1)

        pbar.close()

    def _print_dataset_plan(self):
        """Print the dataset creation plan."""
        print("\n=== BALANCED DATASET PLAN ===")
        print(
            f"Target: {self.config.target_train_images_per_class} train + {self.config.target_test_images_per_class} test images per class"
        )
        print(f"Classes: {len(self.balanced_plan)}")
        total_train = (
            len(self.balanced_plan) * self.config.target_train_images_per_class
        )
        total_test = len(self.balanced_plan) * self.config.target_test_images_per_class
        print(
            f"Estimated Total: {total_train + total_test} ({total_train} train, {total_test} test)"
        )
        print("-" * 30)

    def create_dataset(self):
        """Create the balanced dataset by rendering assets and writing to HDF5."""
        total_images = sum(
            plan["target_train"] + plan["target_test"]
            for plan in self.balanced_plan.values()
        )
        print(f"\n=== CREATING DATASET: {self.config.output_path} ===")
        print(f"Estimated total images: {total_images}")

        try:
            with HDF5Writer(
                self.config.output_path,
                list(self.balanced_plan.keys()),
                int(total_images * 1.1),  # Pre-allocate with 10% buffer
            ) as writer:
                for class_name, plan in tqdm(
                    self.balanced_plan.items(), desc="Total Progress", unit="class"
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
