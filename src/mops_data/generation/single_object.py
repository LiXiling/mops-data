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
    """
    Configuration for the single object dataset generation.
    """

    output_path: str
    dataset_name: str = "mops_single_object"

    # Datapoint Distributions
    target_train_images_per_class: int = 30
    target_test_images_per_class: int = 15
    test_asset_ratio: float = 0.3

    random_seed: int = 42

    # Asset Requirements
    min_assets_per_class: int = 8
    classes_to_include: Optional[List[str]] = None  # If None, all classes are included

    # Rendering Parameters
    image_size: Tuple[int, int] = (512, 512)
    camera_distance: float = 1.5
    obs_mode: str = "rgb+depth+segmentation+normal"

    # Camera Fallback Parameters
    min_segments_threshold: int = 2
    max_fallback_attempts: int = 3

    # Generation parameters
    viewpoints: List[Dict] = None
    lighting_setups: List[Dict] = None
    backgrounds: List[Dict] = None

    def __post_init__(self):
        if self.viewpoints is None:
            # More diverse viewpoints for better coverage
            elevations = [10, 15, 20, 25, 30, 35, 40, 45]  # More elevation angles
            azimuths = range(0, 360, 30)  # Every 30 degrees instead of 45

            self.viewpoints = [
                {"elevation": elev, "azimuth": azim}
                for elev in elevations
                for azim in azimuths
            ]

        if self.lighting_setups is None:
            self.lighting_setups = [
                {"type": "studio", "intensity": 1.0},
                {"type": "natural", "intensity": 0.8},
                {"type": "dramatic", "intensity": 1.2},
            ]

        if self.backgrounds is None:
            self.backgrounds = [
                {"type": "white", "texture": None},
                {"type": "wood_floor", "texture": "wood_01"},
                {"type": "concrete_floor", "texture": "concrete_01"},
            ]


class BalancedDatasetPipeline:
    """
    Pipeline for generating a balanced single object dataset with camera fallback.
    """

    def __init__(self, config: SingleObjectDatasetConfig, partnet_mob_df: pd.DataFrame):
        """
        Initialize the balanced dataset creation pipeline

        Args:
            config: Dataset configuration
            partnet_mob_df: DataFrame with PartNet_Mobility asset information.
        """
        self.config = config
        self.assets_df = partnet_mob_df

        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)

        # Filter and balance classes
        self.filtered_df = self._filter_classes()
        self.balanced_plan = self._create_balanced_plan()

        # Generate base variation combinations
        self.base_variations = self._generate_base_variations()

        # Predefined good camera angles for fallback
        self.fallback_viewpoints = [
            {"elevation": 25, "azimuth": 45},  # Good diagonal views
            {"elevation": 30, "azimuth": 135},
            {"elevation": 20, "azimuth": 225},
            {"elevation": 35, "azimuth": 315},
            {"elevation": 15, "azimuth": 0},  # Cardinal directions
            {"elevation": 15, "azimuth": 90},
            {"elevation": 15, "azimuth": 180},
            {"elevation": 15, "azimuth": 270},
        ]

        # Statistics tracking
        self.fallback_stats = {"total_renders": 0, "fallback_used": 0}

        self._print_dataset_plan()

    def _is_viewpoint_valid(self, segmentation_mask: np.ndarray) -> bool:
        """
        Check if segmentation mask shows good object coverage

        Args:
            segmentation_mask: 2D numpy array with segmentation data

        Returns:
            (is_valid, metrics_dict)
        """

        unique_values = np.unique(segmentation_mask)
        num_segments = len(unique_values)

        return num_segments >= self.config.min_segments_threshold

    def _generate_fallback_viewpoint(self, attempt: int) -> Dict:
        """Generate a randomized fallback viewpoint"""
        if attempt == 1:
            # First fallback: use proven good angles (randomized)
            base_viewpoint = np.random.choice(self.fallback_viewpoints).copy()
            # Add small random perturbation
            base_viewpoint["azimuth"] += np.random.uniform(-15, 15)
            base_viewpoint["elevation"] += np.random.uniform(-5, 5)
            return base_viewpoint
        else:
            # Subsequent fallbacks: completely random
            return {
                "elevation": np.random.uniform(15, 45),
                "azimuth": np.random.uniform(0, 360),
            }

    def _print_dataset_plan(self):
        """Print the dataset creation plan"""
        print()
        print("=== BALANCED DATASET PLAN ===")
        print(
            f"Target: {self.config.target_train_images_per_class} train + {self.config.target_test_images_per_class} test images per class"
        )
        print(f"Classes: {len(self.balanced_plan)}")
        print(f"Total viewpoints available: {len(self.base_variations)}")

        total_train = (
            len(self.balanced_plan) * self.config.target_train_images_per_class
        )
        total_test = len(self.balanced_plan) * self.config.target_test_images_per_class
        print(
            f"Total images: {total_train + total_test} ({total_train} train, {total_test} test)"
        )

        print()
        print("Per-class breakdown:")
        for class_name, plan in list(self.balanced_plan.items())[:3]:
            print(f"  {class_name}:")
            print(
                f"    Train: {plan['n_train_assets']} assets × {plan['train_images_per_asset']}(+{plan['train_remainder']}) = {plan['total_train_images']} images"
            )
            print(
                f"    Test:  {plan['n_test_assets']} assets × {plan['test_images_per_asset']}(+{plan['test_remainder']}) = {plan['total_test_images']} images"
            )

        if len(self.balanced_plan) > 3:
            print(f"  ... and {len(self.balanced_plan) - 3} more classes")

    def _filter_classes(self) -> pd.DataFrame:
        """Filter classes to those with enough assets for proper splits"""
        df = self.assets_df.copy()

        if "model_cat" not in df.columns:
            raise ValueError(
                "DataFrame must have 'model_cat' column for object categories"
            )

        if self.config.classes_to_include:
            df = df[df["model_cat"].isin(self.config.classes_to_include)]

        class_counts = df["model_cat"].value_counts()
        valid_classes = class_counts[
            class_counts >= self.config.min_assets_per_class
        ].index
        df = df[df["model_cat"].isin(valid_classes)]
        df = df.reset_index(drop=True)

        print(f"Filtered to {len(df)} assets across {len(valid_classes)} classes")
        return df

    def _create_balanced_plan(self) -> Dict[str, Dict]:
        """Create a balanced plan for each class"""
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
            itertools.product(
                self.config.viewpoints,
                self.config.lighting_setups,
                self.config.backgrounds,
            )
        )

        variations = []
        for viewpoint, lighting, background in combinations:
            variation = {
                "viewpoint": viewpoint,
                "lighting": lighting,
                "background": background,
            }
            variations.append(variation)

        return variations

    def _sample_variations_for_asset(self, n_images: int) -> List[Dict]:
        """Sample n variations for a single asset, with repetition if needed"""
        if n_images <= len(self.base_variations):
            indices = np.random.choice(
                len(self.base_variations), size=n_images, replace=False
            )
            return [self.base_variations[i] for i in indices]
        else:
            base_cycles = n_images // len(self.base_variations)
            remainder = n_images % len(self.base_variations)

            variations = []

            for cycle in range(base_cycles):
                for base_var in self.base_variations:
                    var = base_var.copy()
                    if cycle > 0:
                        var["viewpoint"] = var["viewpoint"].copy()
                        var["viewpoint"]["azimuth"] += np.random.uniform(-5, 5)
                        var["lighting"] = var["lighting"].copy()
                        var["lighting"]["intensity"] *= np.random.uniform(0.95, 1.05)
                    variations.append(var)

            if remainder > 0:
                indices = np.random.choice(
                    len(self.base_variations), size=remainder, replace=False
                )
                for i in indices:
                    var = self.base_variations[i].copy()
                    var["viewpoint"] = var["viewpoint"].copy()
                    var["viewpoint"]["azimuth"] += np.random.uniform(-10, 10)
                    variations.append(var)

            return variations

    def _create_render_env(self, asset_row: pd.Series, variation: Dict):
        """Create a fresh render environment for each asset"""
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
            "background_type": variation["background"]["type"],
            "background_texture": variation["background"]["texture"],
        }

        if "dir_name" in asset_row and pd.notna(asset_row["dir_name"]):
            env_kwargs["dir_name"] = asset_row["dir_name"]
        elif "model_id" in asset_row and pd.notna(asset_row["model_id"]):
            env_kwargs["asset_id"] = str(asset_row["model_id"])
        else:
            env_kwargs["asset_index"] = asset_row.name

        if "model_cat" in asset_row and pd.notna(asset_row["model_cat"]):
            env_kwargs["model_cat"] = asset_row["model_cat"]

        env = gym.make("DatasetRenderEnv-v1", **env_kwargs)
        return env

    def _render_asset(
        self, asset_row: pd.Series, variation: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Render a single asset with camera fallback system
        """
        self.fallback_stats["total_renders"] += 1
        current_variation = variation.copy()

        for attempt in range(self.config.max_fallback_attempts + 1):
            env = self._create_render_env(asset_row, current_variation)

            try:
                # Reset and render
                obs, info = env.reset(seed=0)
                action = (
                    np.zeros_like(env.action_space.sample())
                    if hasattr(env.action_space, "sample")
                    else None
                )

                for step in range(3):
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break

                result = env.extract_render_data(obs)

                # Check if viewpoint is valid
                if "semantic_mask" in result:
                    is_valid = self._is_viewpoint_valid(result["semantic_mask"])

                    if is_valid or attempt == self.config.max_fallback_attempts:
                        # Either valid or this is our last attempt
                        if attempt > 0:
                            self.fallback_stats["fallback_used"] += 1
                        return result

                    # Try fallback viewpoint
                    fallback_viewpoint = self._generate_fallback_viewpoint(attempt + 1)
                    current_variation["viewpoint"] = fallback_viewpoint
                else:
                    # No segmentation mask, return as-is
                    return result

            except Exception as e:
                asset_id = (
                    asset_row.get("dir_name")
                    or asset_row.get("model_id")
                    or asset_row.name
                )
                if attempt == self.config.max_fallback_attempts:
                    print(
                        f"Error rendering asset {asset_id} after {attempt + 1} attempts: {e}"
                    )
                    raise e
                # Try again with fallback viewpoint
                fallback_viewpoint = self._generate_fallback_viewpoint(attempt + 1)
                current_variation["viewpoint"] = fallback_viewpoint
            finally:
                env.close()

        # Should not reach here, but just in case
        return result

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

        assets_with_extra = set()
        if remainder > 0:
            extra_indices = np.random.choice(
                len(assets_df), size=remainder, replace=False
            )
            assets_with_extra = set(extra_indices)

        total_images_generated = 0
        target_total = plan[f"total_{split}_images"]

        for asset_idx, (_, asset_row) in enumerate(assets_df.iterrows()):
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

    def create_dataset(self):
        """Create the balanced dataset"""
        total_images = 0
        for plan in self.balanced_plan.values():
            total_images += plan["total_train_images"] + plan["total_test_images"]

        print()
        print("=== CREATING DATASET ===")
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
                        writer,
                        plan["train_assets"],
                        plan,
                        "train",
                        class_name,
                    )

                    print(f"  Generating {plan['total_test_images']} test images...")
                    self._generate_images_for_class_split(
                        writer,
                        plan["test_assets"],
                        plan,
                        "test",
                        class_name,
                    )

        except Exception as e:
            print(f"Error during dataset creation: {e}")
            raise

        # Print fallback statistics
        total_renders = self.fallback_stats["total_renders"]
        fallback_used = self.fallback_stats["fallback_used"]
        fallback_rate = (
            (fallback_used / total_renders * 100) if total_renders > 0 else 0
        )

        print()
        print("=== CAMERA FALLBACK STATISTICS ===")
        print(f"Total renders: {total_renders}")
        print(f"Fallbacks used: {fallback_used} ({fallback_rate:.1f}%)")

        print()
        print("=== DATASET COMPLETE ===")
        print(f"File: {self.config.output_path}")


if __name__ == "__main__":
    # Example usage with mock data
    print("Please provide a PartNet Mobility DataFrame to create_test_dataset()")
    print("Expected columns: 'model_cat', 'model_id' (and optionally 'dir_name')")

    # Example with your existing code structure
    try:
        import mops_data.asset_manager.anno_handler as mops_ah

        df = mops_ah.load_annotations().partnet_mobility_df
        df = df.groupby("model_id").first().reset_index()

        # Create a small subset for testing
        # Sample30 random entries for quick testing
        test_df = df.sample(n=200, random_state=42).reset_index(drop=True)

        config = SingleObjectDatasetConfig(
            output_path="test.h5",
            target_train_images_per_class=30,
            target_test_images_per_class=15,
            min_assets_per_class=10,
            image_size=(512, 512),  # Small for testing
        )

        pipeline = BalancedDatasetPipeline(config, test_df)
        pipeline.create_dataset()

    except ImportError:
        print("mops_data.asset_manager.anno_handler not available for testing")
        print("Please provide your own DataFrame with the required columns")
