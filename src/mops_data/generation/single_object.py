import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mops_data.generation.hdf_writer import HDF5Writer


@dataclass
class SingleObjectDatasetConfig:
    """Configuration for single object dataset generation."""

    output_path: str
    dataset_name: str = "mops_single_object"

    # Dataset distribution
    target_train_images_per_class: int = 30
    target_test_images_per_class: int = 15
    test_asset_ratio: float = 0.3
    random_seed: int = 42

    # Asset requirements
    min_assets_per_class: int = 8
    classes_to_include: Optional[List[str]] = None

    # Rendering parameters
    image_size: Tuple[int, int] = (512, 512)
    camera_distance: float = 1.5
    obs_mode: str = "rgb+depth+segmentation+normal"

    # Validation
    min_segments_threshold: int = 3
    max_resampling_attempts: int = 3  # Max attempts to get valid render

    # Lighting variation ranges
    light_temp_range: Tuple[float, float] = (2700, 8000)  # Warm to cool daylight
    light_intensity_range: Tuple[float, float] = (0.8, 1.2)

    # Generation parameters
    viewpoints: List[Dict] = None
    lighting_types: List[str] = None

    def __post_init__(self):
        if self.viewpoints is None:
            # Diverse viewpoints for good coverage
            elevations = [10, 15, 20, 25, 30, 35, 40, 45]
            azimuths = range(0, 360, 30)  # Every 30 degrees
            self.viewpoints = [
                {"elevation": elev, "azimuth": azim}
                for elev in elevations
                for azim in azimuths
            ]

        if self.lighting_types is None:
            self.lighting_types = ["studio", "natural", "dramatic"]


class BalancedDatasetPipeline:
    """Pipeline for generating balanced single object dataset."""

    def __init__(self, config: SingleObjectDatasetConfig, partnet_mob_df: pd.DataFrame):
        self.config = config
        self.assets_df = partnet_mob_df

        np.random.seed(self.config.random_seed)

        # Filter and plan dataset
        self.filtered_df = self._filter_classes()
        self.balanced_plan = self._create_balanced_plan()
        self.base_variations = self._generate_base_variations()

        self._print_dataset_plan()

    def _filter_classes(self) -> pd.DataFrame:
        """Filter classes to those with enough assets"""
        df = self.assets_df.copy()

        if "model_cat" not in df.columns:
            raise ValueError("DataFrame must have 'model_cat' column")

        if self.config.classes_to_include:
            df = df[df["model_cat"].isin(self.config.classes_to_include)]

        class_counts = df["model_cat"].value_counts()
        valid_classes = class_counts[
            class_counts >= self.config.min_assets_per_class
        ].index
        df = df[df["model_cat"].isin(valid_classes)].reset_index(drop=True)

        print(f"Filtered to {len(df)} assets across {len(valid_classes)} classes")
        return df

    def _create_balanced_plan(self) -> Dict[str, Dict]:
        """Create balanced generation plan for each class"""
        plan = {}

        for class_name in self.filtered_df["model_cat"].unique():
            class_assets = self.filtered_df[self.filtered_df["model_cat"] == class_name]
            total_assets = len(class_assets)

            n_test_assets = max(1, int(total_assets * self.config.test_asset_ratio))
            n_train_assets = total_assets - n_test_assets

            train_assets, test_assets = train_test_split(
                class_assets,
                test_size=n_test_assets,
                random_state=self.config.random_seed,
            )

            # Calculate images per asset
            train_images_per_asset = max(
                1, self.config.target_train_images_per_class // n_train_assets
            )
            test_images_per_asset = max(
                1, self.config.target_test_images_per_class // n_test_assets
            )

            train_remainder = self.config.target_train_images_per_class % n_train_assets
            test_remainder = self.config.target_test_images_per_class % n_test_assets

            plan[class_name] = {
                "train_assets": train_assets,
                "test_assets": test_assets,
                "train_images_per_asset": train_images_per_asset,
                "test_images_per_asset": test_images_per_asset,
                "train_remainder": train_remainder,
                "test_remainder": test_remainder,
                "total_train_images": self.config.target_train_images_per_class,
                "total_test_images": self.config.target_test_images_per_class,
                "n_train_assets": n_train_assets,
                "n_test_assets": n_test_assets,
            }

        return plan

    def _generate_base_variations(self) -> List[Dict]:
        """Generate base variation combinations"""
        combinations = list(
            itertools.product(self.config.viewpoints, self.config.lighting_types)
        )

        variations = []
        for viewpoint, lighting in combinations:
            variation = {"viewpoint": viewpoint, "lighting": lighting}
            variations.append(variation)

        return variations

    def _sample_variations_for_asset(self, n_images: int) -> List[Dict]:
        """Sample n variations for a single asset with random lighting"""
        if n_images <= len(self.base_variations):
            indices = np.random.choice(
                len(self.base_variations), size=n_images, replace=False
            )
            selected_variations = [self.base_variations[i] for i in indices]
        else:
            # Need more variations than base combinations
            base_cycles = n_images // len(self.base_variations)
            remainder = n_images % len(self.base_variations)

            selected_variations = []

            # Add full cycles with small perturbations
            for cycle in range(base_cycles):
                for base_var in self.base_variations:
                    var = base_var.copy()
                    if cycle > 0:
                        # Add small perturbations for repeated variations
                        var["viewpoint"] = var["viewpoint"].copy()
                        var["viewpoint"]["azimuth"] += np.random.uniform(-10, 10)
                        var["viewpoint"]["elevation"] += np.random.uniform(-3, 3)
                    selected_variations.append(var)

            # Add remainder variations
            if remainder > 0:
                indices = np.random.choice(
                    len(self.base_variations), size=remainder, replace=False
                )
                for i in indices:
                    var = self.base_variations[i].copy()
                    var["viewpoint"] = var["viewpoint"].copy()
                    var["viewpoint"]["azimuth"] += np.random.uniform(-15, 15)
                    selected_variations.append(var)

        # Add random lighting to each variation
        for var in selected_variations:
            var["lighting"] = self._sample_random_lighting()

        return selected_variations

    def _sample_random_lighting(self) -> Dict:
        """Sample random lighting parameters"""
        lighting_type = np.random.choice(self.config.lighting_types)

        # Sample temperature and intensity from ranges
        temp_min, temp_max = self.config.light_temp_range
        intensity_min, intensity_max = self.config.light_intensity_range

        temperature = np.random.uniform(temp_min, temp_max)
        intensity = np.random.uniform(intensity_min, intensity_max)

        return {
            "type": lighting_type,
            "temperature": temperature,
            "intensity": intensity,
        }

    def _create_render_env(self, asset_row: pd.Series, variation: Dict):
        """Create render environment for asset with variation"""
        env_kwargs = {
            "render_mode": "rgb_array",
            "num_envs": 1,
            "obs_mode": self.config.obs_mode,
            "image_size": self.config.image_size,
            "camera_distance": self.config.camera_distance,
            "camera_elevation": variation["viewpoint"]["elevation"],
            "camera_azimuth": variation["viewpoint"]["azimuth"],
            "lighting_type": variation["lighting"]["type"],
            "lighting_intensity": variation["lighting"]["intensity"],
            "light_temperature": variation["lighting"]["temperature"],
        }

        # Set asset identifier
        if "dir_name" in asset_row and pd.notna(asset_row["dir_name"]):
            env_kwargs["dir_name"] = asset_row["dir_name"]
        elif "model_id" in asset_row and pd.notna(asset_row["model_id"]):
            env_kwargs["asset_id"] = str(asset_row["model_id"])
        else:
            env_kwargs["asset_index"] = asset_row.name

        if "model_cat" in asset_row and pd.notna(asset_row["model_cat"]):
            env_kwargs["model_cat"] = asset_row["model_cat"]

        return gym.make("DatasetRenderEnv-v1", **env_kwargs)

    def _render_asset(
        self, asset_row: pd.Series, variation: Dict
    ) -> Dict[str, np.ndarray]:
        """Render a single asset with validation and resampling for low quality"""
        current_variation = variation.copy()

        for attempt in range(self.config.max_resampling_attempts):
            env = self._create_render_env(asset_row, current_variation)

            try:
                obs, info = env.reset(seed=0)

                # Step environment a few times for stability
                action = (
                    np.zeros_like(env.action_space.sample())
                    if hasattr(env.action_space, "sample")
                    else None
                )
                for _ in range(3):
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break

                # Validate render quality
                if env.unwrapped.is_valid_render(
                    obs, min_segments=self.config.min_segments_threshold
                ):
                    return env.unwrapped.extract_render_data(obs)
                else:
                    # Low quality render - try again with new variation
                    if attempt < self.config.max_resampling_attempts - 1:
                        asset_id = (
                            asset_row.get("dir_name")
                            or asset_row.get("model_id")
                            or asset_row.name
                        )
                        print(
                            f"Low quality render for asset {asset_id}, attempt {attempt + 1}, resampling..."
                        )

                        # Generate new variation with different viewpoint and lighting
                        new_base_variation = np.random.choice(self.base_variations)
                        current_variation = new_base_variation.copy()
                        current_variation["lighting"] = self._sample_random_lighting()

                        # Add some randomization to viewpoint
                        current_variation["viewpoint"] = current_variation[
                            "viewpoint"
                        ].copy()
                        current_variation["viewpoint"]["azimuth"] += np.random.uniform(
                            -30, 30
                        )
                        current_variation["viewpoint"][
                            "elevation"
                        ] += np.random.uniform(-10, 10)
                        current_variation["viewpoint"]["elevation"] = np.clip(
                            current_variation["viewpoint"]["elevation"], 5, 60
                        )
                    else:
                        # Final attempt failed, use what we have
                        asset_id = (
                            asset_row.get("dir_name")
                            or asset_row.get("model_id")
                            or asset_row.name
                        )
                        print(
                            f"Warning: Using low quality render for asset {asset_id} after {attempt + 1} attempts"
                        )
                        return env.unwrapped.extract_render_data(obs)

            except Exception as e:
                asset_id = (
                    asset_row.get("dir_name")
                    or asset_row.get("model_id")
                    or asset_row.name
                )
                if attempt == self.config.max_resampling_attempts - 1:
                    print(
                        f"Error rendering asset {asset_id} after {attempt + 1} attempts: {e}"
                    )
                    raise e
                else:
                    print(
                        f"Error rendering asset {asset_id}, attempt {attempt + 1}, retrying..."
                    )
                    # Generate new variation for retry
                    new_base_variation = np.random.choice(self.base_variations)
                    current_variation = new_base_variation.copy()
                    current_variation["lighting"] = self._sample_random_lighting()
            finally:
                env.close()

        # Should not reach here
        raise RuntimeError("Failed to render asset after all attempts")

    def _generate_images_for_class_split(
        self,
        writer: HDF5Writer,
        assets_df: pd.DataFrame,
        plan: Dict,
        split: str,
        class_name: str,
    ):
        """Generate images for a specific class and split"""
        images_per_asset = plan[f"{split}_images_per_asset"]
        remainder = plan[f"{split}_remainder"]

        # Randomly assign extra images to assets
        assets_with_extra = set()
        if remainder > 0:
            extra_indices = np.random.choice(
                len(assets_df), size=remainder, replace=False
            )
            assets_with_extra = set(extra_indices)

        total_images_generated = 0
        target_total = plan[f"total_{split}_images"]

        for asset_idx, (_, asset_row) in tqdm(
            enumerate(assets_df.iterrows()), desc=f"{split} assets"
        ):
            n_images = images_per_asset
            if asset_idx in assets_with_extra:
                n_images += 1

            if total_images_generated + n_images > target_total:
                n_images = target_total - total_images_generated
                if n_images <= 0:
                    break

            variations = self._sample_variations_for_asset(n_images)

            for variation in variations:
                if total_images_generated >= target_total:
                    break

                render_data = self._render_asset(asset_row, variation)

                render_params = {
                    "split": split,
                    "variation": variation,
                    "image_size": self.config.image_size,
                    "dataset_name": self.config.dataset_name,
                }

                asset_id = str(
                    asset_row.get(
                        "model_id", asset_row.get("dir_name", f"asset_{asset_idx}")
                    )
                )

                writer.add_image(
                    image=render_data["image"],
                    semantic_mask=render_data["semantic_mask"],
                    class_name=class_name,
                    asset_id=asset_id,
                    render_params=render_params,
                    part_mask=render_data.get("part_mask"),
                    instance_mask=render_data.get("instance_mask"),
                    affordance_mask=render_data.get("affordance_mask"),
                    depth=render_data.get("depth"),
                    normal=render_data.get("normal"),
                )

                total_images_generated += 1

        print(f"    Generated {total_images_generated} {split} images for {class_name}")

    def _print_dataset_plan(self):
        """Print dataset creation plan"""
        print("\n=== BALANCED DATASET PLAN ===")
        print(
            f"Target: {self.config.target_train_images_per_class} train + {self.config.target_test_images_per_class} test images per class"
        )
        print(f"Classes: {len(self.balanced_plan)}")
        print(f"Lighting: Random selection from {self.config.lighting_types}")
        print(
            f"Temperature range: {self.config.light_temp_range[0]:.0f}K - {self.config.light_temp_range[1]:.0f}K"
        )

        total_train = (
            len(self.balanced_plan) * self.config.target_train_images_per_class
        )
        total_test = len(self.balanced_plan) * self.config.target_test_images_per_class
        print(
            f"Total images: {total_train + total_test} ({total_train} train, {total_test} test)"
        )

    def create_dataset(self):
        """Create the balanced dataset"""
        total_images = sum(
            plan["total_train_images"] + plan["total_test_images"]
            for plan in self.balanced_plan.values()
        )

        print(f"\n=== CREATING DATASET ===")
        print(f"Output: {self.config.output_path}")
        print(f"Estimated total images: {total_images}")

        try:
            estimated_images = max(total_images, int(total_images * 1.2))

            with HDF5Writer(
                self.config.output_path,
                list(self.balanced_plan.keys()),
                estimated_images,
            ) as writer:

                for class_name in tqdm(self.balanced_plan.keys(), desc="Classes"):
                    plan = self.balanced_plan[class_name]

                    print(f"\nProcessing class: {class_name}")

                    print(f"  Generating {plan['total_train_images']} train images...")
                    self._generate_images_for_class_split(
                        writer, plan["train_assets"], plan, "train", class_name
                    )

                    print(f"  Generating {plan['total_test_images']} test images...")
                    self._generate_images_for_class_split(
                        writer, plan["test_assets"], plan, "test", class_name
                    )

        except Exception as e:
            print(f"Error during dataset creation: {e}")
            raise

        print(f"\n=== DATASET COMPLETE ===")
        print(f"File: {self.config.output_path}")


if __name__ == "__main__":
    # Example usage
    print("Please provide a PartNet Mobility DataFrame to create_dataset()")
    print("Expected columns: 'model_cat', 'model_id' (and optionally 'dir_name')")

    try:
        import mops_data.asset_manager.anno_handler as mops_ah

        df = mops_ah.load_annotations().partnet_mobility_df
        df = df.groupby("model_id").first().reset_index()

        blacklist = [12071]
        df = df[~df["model_id"].isin(blacklist)]

        # Create small test subset
        test_df = df.sample(n=20, random_state=64).reset_index(drop=True)

        config = SingleObjectDatasetConfig(
            output_path="data/test_dataset.h5",
            target_train_images_per_class=10,
            target_test_images_per_class=5,
            min_assets_per_class=2,
            image_size=(512, 512),  # Smaller for testing
            light_temp_range=(2700, 7000),  # Warm to daylight
            light_intensity_range=(0.8, 1.3),
        )

        pipeline = BalancedDatasetPipeline(config, test_df)
        pipeline.create_dataset()

    except ImportError:
        print("mops_data.asset_manager.anno_handler not available for testing")
