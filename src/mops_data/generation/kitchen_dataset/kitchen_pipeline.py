from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mops_data.generation.base_pipeline import BaseDatasetPipeline
from mops_data.generation.hdf_writer import HDF5Writer
from mops_data.generation.subprocess_renderer import (
    SPLIT_SEED_OFFSETS,
    render_batch_parallel,
)

ENV_ID = "KitchenRenderEnv-v1"
ENV_MODULE = "mops_data.envs.dataset_envs"

BATCH_SIZE = 100
NUM_WORKERS = 1


class KitchenDatasetPipeline(BaseDatasetPipeline):
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

    def _build_env_kwargs(self, asset_df: pd.DataFrame, variation: Dict) -> Dict:
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
                "asset_df": asset_df,
            }.items()
            if v is not None
        }

    def _prepare_render_job(
        self,
        assets: pd.DataFrame,
        split: str,
        attempt_index: int,
    ) -> dict:
        """Prepare a single render job with deterministic seeding.

        Each job gets its own seed derived from the split offset and
        attempt index, keeping results independent of batch layout.
        """
        split_offset = SPLIT_SEED_OFFSETS[split]
        image_seed = self.config.random_seed + split_offset + attempt_index
        rng = np.random.RandomState(image_seed)

        random_set = assets.sample(40, random_state=rng)

        # Seed global RNG for variation sampling
        np.random.seed(image_seed)
        variations = self._sample_variations_for_asset(
            self.config.max_resampling_attempts
        )

        attempts = [
            {
                "env_kwargs": self._build_env_kwargs(random_set, var),
                "seed": image_seed + attempt_idx,
                "num_steps": 2,
                "min_segments": self.config.min_segments_threshold,
            }
            for attempt_idx, var in enumerate(variations)
        ]

        return {
            "job_id": attempt_index,
            "attempts": attempts,
            "variations": variations,
        }

    def _generate_images_for_class_split(
        self,
        writer: HDF5Writer,
        assets: pd.DataFrame,
        target_count: int,
        split: str,
        class_name: str,
    ):
        """Generate and save images for a specific class and split."""
        if target_count <= 0:
            return

        pbar = tqdm(
            total=target_count, desc=f"  {split.capitalize():<5} images", unit="img"
        )

        generated_count = 0
        attempt_index = 0

        while generated_count < target_count:
            # Prepare a batch of render jobs with deterministic seeds
            n_jobs = BATCH_SIZE * NUM_WORKERS
            jobs = {}
            for _ in range(n_jobs):
                job = self._prepare_render_job(assets, split, attempt_index)
                jobs[job["job_id"]] = job
                attempt_index += 1

            # Split into per-worker batches and render in parallel
            job_list = list(jobs.values())
            batches = [
                job_list[i : i + BATCH_SIZE]
                for i in range(0, len(job_list), BATCH_SIZE)
            ]
            results = render_batch_parallel(ENV_ID, ENV_MODULE, batches)

            # Process results in job_id order for deterministic writes
            for result in sorted(results, key=lambda r: r["job_id"]):
                if generated_count >= target_count:
                    break
                if result["data"] is None:
                    continue

                variation = jobs[result["job_id"]]["variations"][
                    result["attempt_idx"]
                ]
                render_params = {
                    "split": split,
                    "variation": variation,
                    "image_size": self.config.image_size,
                }

                writer.add_image(
                    class_name=class_name,
                    render_params=render_params,
                    **result["data"],
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

        print("\n=== DATASET CREATION COMPLETE ===")
        print(f"File saved to: {self.config.output_path}")
