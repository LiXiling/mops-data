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

    # Generation parameters
    viewpoints: List[Dict] = None
    lighting_setups: List[Dict] = None
    backgrounds: List[Dict] = None

    def __post_init__(self):
        if self.viewpoints is None:
            # Diverse viewpoints for good coverage
            self.viewpoints = [
                {"elevation": elev, "azimuth": azim}
                for elev in [10, 20, 30, 45]
                for azim in range(0, 360, 45)  # Every 45 degrees
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
    Pipeline for generating a balanced single object dataset using Gymnasium API.

    This pipeline focuses on orchestrating the rendering process and delegates
    all file I/O operations to the HDF5Writer.
    """

    def __init__(self, config: SingleObjectDatasetConfig, partnet_mob_df: pd.DataFrame):
        """
        Initialize the balanced dataset creation pipeline

        Args:
            config: Dataset configuration
            partnet_mob_df: DataFrame with PartNet_Mobility asset information.
                           Should have columns: 'model_cat', 'model_id', and optionally 'dir_name'
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

        self._print_dataset_plan()

    def _print_dataset_plan(self):
        """Print the dataset creation plan"""
        print()
        print("=== BALANCED DATASET PLAN ===")
        print(
            f"Target: {self.config.target_train_images_per_class} train + {self.config.target_test_images_per_class} test images per class"
        )
        print(f"Classes: {len(self.balanced_plan)}")

        total_train = (
            len(self.balanced_plan) * self.config.target_train_images_per_class
        )
        total_test = len(self.balanced_plan) * self.config.target_test_images_per_class
        print(
            f"Total images: {total_train + total_test} ({total_train} train, {total_test} test)"
        )

        print()
        print("Per-class breakdown:")
        for class_name, plan in list(self.balanced_plan.items())[:5]:
            print(f"  {class_name}:")
            print(
                f"    Train: {plan['n_train_assets']} assets × {plan['train_images_per_asset']}(+{plan['train_remainder']}) = {plan['total_train_images']} images"
            )
            print(
                f"    Test:  {plan['n_test_assets']} assets × {plan['test_images_per_asset']}(+{plan['test_remainder']}) = {plan['total_test_images']} images"
            )

        if len(self.balanced_plan) > 5:
            print(f"  ... and {len(self.balanced_plan) - 5} more classes")

    def _filter_classes(self) -> pd.DataFrame:
        """Filter classes to those with enough assets for proper splits"""
        df = self.assets_df.copy()

        # Ensure we have the necessary columns
        if "model_cat" not in df.columns:
            raise ValueError(
                "DataFrame must have 'model_cat' column for object categories"
            )

        # Filter by class names if specified
        if self.config.classes_to_include:
            df = df[df["model_cat"].isin(self.config.classes_to_include)]

        # Filter classes by minimum asset count
        class_counts = df["model_cat"].value_counts()
        valid_classes = class_counts[
            class_counts >= self.config.min_assets_per_class
        ].index
        df = df[df["model_cat"].isin(valid_classes)]

        # Reset index to ensure we have clean indices for asset loading
        df = df.reset_index(drop=True)

        print(f"Filtered to {len(df)} assets across {len(valid_classes)} classes")
        return df

    def _create_balanced_plan(self) -> Dict[str, Dict]:
        """Create a balanced plan for each class"""
        plan = {}

        for class_name in self.filtered_df["model_cat"].unique():
            class_assets = self.filtered_df[self.filtered_df["model_cat"] == class_name]
            total_assets = len(class_assets)

            # Calculate train/test asset split
            n_test_assets = max(1, int(total_assets * self.config.test_asset_ratio))
            n_train_assets = total_assets - n_test_assets

            # Split assets into train/test
            train_assets, test_assets = train_test_split(
                class_assets,
                test_size=n_test_assets,
                random_state=self.config.random_seed,
            )

            # Calculate images per asset needed to reach target counts
            train_images_per_asset = max(
                1, self.config.target_train_images_per_class // n_train_assets
            )
            test_images_per_asset = max(
                1, self.config.target_test_images_per_class // n_test_assets
            )

            # Handle remainder by adding extra images to some assets
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
            # Sample without replacement
            indices = np.random.choice(
                len(self.base_variations), size=n_images, replace=False
            )
            return [self.base_variations[i] for i in indices]
        else:
            # Need to repeat some variations - add small random perturbations
            base_cycles = n_images // len(self.base_variations)
            remainder = n_images % len(self.base_variations)

            variations = []

            # Add full cycles of base variations
            for cycle in range(base_cycles):
                for base_var in self.base_variations:
                    var = base_var.copy()
                    # Add small random perturbations to avoid exact duplicates
                    if cycle > 0:
                        var["viewpoint"] = var["viewpoint"].copy()
                        var["viewpoint"]["azimuth"] += np.random.uniform(-5, 5)
                        var["lighting"] = var["lighting"].copy()
                        var["lighting"]["intensity"] *= np.random.uniform(0.95, 1.05)
                    variations.append(var)

            # Add remainder
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
        """
        Create a fresh render environment for each asset using Gymnasium API.
        This ensures clean loading without asset conflicts.
        """
        # Prepare environment kwargs
        env_kwargs = {
            "render_mode": "rgb_array",
            "num_envs": 1,
            "obs_mode": self.config.obs_mode,
            # Image configuration
            "image_size": self.config.image_size,
            "camera_distance": self.config.camera_distance,
            "camera_elevation": variation["viewpoint"]["elevation"],
            "camera_azimuth": variation["viewpoint"]["azimuth"],
            # Lighting configuration
            "lighting_type": variation["lighting"]["type"],
            "lighting_intensity": variation["lighting"]["intensity"],
            # Background configuration
            "background_type": variation["background"]["type"],
            "background_texture": variation["background"]["texture"],
        }

        # Add asset specification - prefer dir_name, fallback to model_id, then index
        if "dir_name" in asset_row and pd.notna(asset_row["dir_name"]):
            env_kwargs["dir_name"] = asset_row["dir_name"]
        elif "model_id" in asset_row and pd.notna(asset_row["model_id"]):
            env_kwargs["asset_id"] = str(asset_row["model_id"])
        else:
            env_kwargs["asset_index"] = asset_row.name

        # Add model_cat if available
        if "model_cat" in asset_row and pd.notna(asset_row["model_cat"]):
            env_kwargs["model_cat"] = asset_row["model_cat"]

        # Create environment using Gymnasium
        env = gym.make("DatasetRenderEnv-v1", **env_kwargs)

        return env

    def _generate_images_for_class_split(
        self,
        writer: HDF5Writer,
        assets_df: pd.DataFrame,
        plan: Dict,
        split: str,  # "train" or "test"
        class_name: str,
    ):
        """
        Generate images for a specific class and split (train/test)

        Args:
            writer: HDF5Writer instance to handle I/O
            assets_df: DataFrame containing assets for this split
            plan: Plan dictionary for this class
            split: "train" or "test"
            class_name: Name of the class
        """
        images_per_asset = plan[f"{split}_images_per_asset"]
        remainder = plan[f"{split}_remainder"]

        # Randomly select which assets get the extra images
        assets_with_extra = set()
        if remainder > 0:
            extra_indices = np.random.choice(
                len(assets_df), size=remainder, replace=False
            )
            assets_with_extra = set(extra_indices)

        total_images_generated = 0
        target_total = plan[f"total_{split}_images"]

        for asset_idx, (_, asset_row) in enumerate(assets_df.iterrows()):
            # Determine how many images to generate for this asset
            n_images = images_per_asset
            if asset_idx in assets_with_extra:
                n_images += 1

            # Safety check: don't exceed target total
            if total_images_generated + n_images > target_total:
                n_images = target_total - total_images_generated
                if n_images <= 0:
                    break

            # Get variations for this asset
            variations = self._sample_variations_for_asset(n_images)

            # Generate images for each variation
            for variation in variations:
                # Safety check: don't exceed target total
                if total_images_generated >= target_total:
                    break

                # Render the asset with this variation
                render_data = self._render_asset(asset_row, variation)

                # Prepare metadata
                render_params = {
                    "split": split,
                    "variation": variation,
                    "image_size": self.config.image_size,
                    "dataset_name": self.config.dataset_name,
                }

                # Get asset ID for metadata
                asset_id = str(
                    asset_row.get(
                        "model_id", asset_row.get("dir_name", f"asset_{asset_idx}")
                    )
                )

                # Let the HDF5Writer handle all the I/O
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

    def _render_asset(
        self, asset_row: pd.Series, variation: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Render a single asset using a fresh DatasetRenderEnv via Gymnasium API.
        Creates new environment for each asset to ensure clean loading.

        Args:
            asset_row: Row from assets DataFrame containing asset info
            variation: Dictionary with rendering parameters (viewpoint, lighting, background)

        Returns:
            Dictionary containing rendered data:
            - 'image': RGB image array (H, W, 3)
            - 'semantic_mask': Semantic mask array (H, W)
            - 'part_mask': Optional part mask array (H, W)
            - 'instance_mask': Optional instance mask array (H, W)
            - 'affordance_mask': Optional affordance mask array (H, W)
            - 'depth': Optional depth map (H, W)
            - 'normal': Optional normal map (H, W, 3)
        """
        # Create fresh environment for this specific asset and variation
        env = self._create_render_env(asset_row, variation)

        try:
            # Reset environment to load the object and get fresh state
            obs, info = env.reset(seed=0)

            # Step a few times to let physics/lighting settle
            # Use zero action (no action needed for static rendering)
            action = (
                np.zeros_like(env.action_space.sample())
                if hasattr(env.action_space, "sample")
                else None
            )

            for step in range(3):  # Match your original step count
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            # Extract rendered data from observations using the environment's method
            result = env.extract_render_data(obs)

            return result

        except Exception as e:
            asset_identifier = (
                asset_row.get("dir_name") or asset_row.get("model_id") or asset_row.name
            )
            print(f"Error rendering asset {asset_identifier}: {e}")
            raise e
        finally:
            # Always clean up the environment
            env.close()

    def create_dataset(self):
        """
        Create the balanced dataset

        This method orchestrates the entire dataset creation process,
        using the HDF5Writer to handle all file I/O operations.
        """
        # Calculate actual total images based on the plan
        total_images = 0
        for plan in self.balanced_plan.values():
            total_images += plan["total_train_images"] + plan["total_test_images"]

        print()
        print("=== CREATING DATASET ===")
        print(f"Output: {self.config.output_path}")
        print(f"Estimated total images: {total_images}")

        try:
            # Use HDF5Writer as context manager to handle file I/O
            # Add some buffer to the estimate in case of variations in actual generation
            estimated_images = max(total_images, int(total_images * 1.2))

            with HDF5Writer(
                self.config.output_path,
                list(self.balanced_plan.keys()),
                estimated_images,
            ) as writer:

                # Process each class
                for class_name in tqdm(self.balanced_plan.keys(), desc="Classes"):
                    plan = self.balanced_plan[class_name]

                    print(f"\nProcessing class: {class_name}")

                    # Generate train images
                    print(f"  Generating {plan['total_train_images']} train images...")
                    self._generate_images_for_class_split(
                        writer,
                        plan["train_assets"],
                        plan,
                        "train",
                        class_name,
                    )

                    # Generate test images
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

        print()
        print("=== DATASET COMPLETE ===")
        print(f"File: {self.config.output_path}")


# Example usage function
def create_test_dataset(partnet_df: pd.DataFrame):
    """
    Create a test dataset with provided DataFrame

    Args:
        partnet_df: DataFrame with PartNet Mobility asset information
                   Expected columns: 'model_cat', 'model_id', optionally 'dir_name'
    """
    config = SingleObjectDatasetConfig(
        output_path="test_dataset.h5",
        target_train_images_per_class=10,
        target_test_images_per_class=5,
        min_assets_per_class=3,
        image_size=(256, 256),  # Smaller for faster testing
        obs_mode="rgb+depth+segmentation+normal",
        # Minimal variations for quick testing
        viewpoints=[
            {"elevation": 20, "azimuth": 0},
            {"elevation": 20, "azimuth": 120},
            {"elevation": 20, "azimuth": 240},
        ],
        lighting_setups=[
            {"type": "studio", "intensity": 1.0},
        ],
        backgrounds=[
            {"type": "white", "texture": None},
        ],
    )

    pipeline = BalancedDatasetPipeline(config, partnet_df)
    pipeline.create_dataset()


if __name__ == "__main__":
    # Example usage with mock data
    print("Please provide a PartNet Mobility DataFrame to create_test_dataset()")
    print("Expected columns: 'model_cat', 'model_id' (and optionally 'dir_name')")

    # Example with your existing code structure
    try:
        import mops_data.asset_manager.anno_handler as mops_ah

        df = mops_ah.load_annotations().partnet_mobility_df

        # Create a small subset for testing
        subset_classes = ["Chair", "Table"]
        test_df = (
            df[df["model_cat"].isin(subset_classes)].head(20).reset_index(drop=True)
        )

        config = SingleObjectDatasetConfig(
            output_path="test.h5",
            target_train_images_per_class=2,
            target_test_images_per_class=1,
            min_assets_per_class=1,
            image_size=(128, 128),  # Small for testing
            obs_mode="rgb+segmentation",  # Minimal for faster testing
        )

        pipeline = BalancedDatasetPipeline(config, test_df)
        pipeline.create_dataset()

    except ImportError:
        print("mops_data.asset_manager.anno_handler not available for testing")
        print("Please provide your own DataFrame with the required columns")
