from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mops_data.generation.base_pipeline import BaseDatasetPipeline
from mops_data.generation.subprocess_renderer import (
    SPLIT_SEED_OFFSETS,
    render_batch_parallel,
)

ENV_ID = "SingleObjectRenderEnv-v1"
ENV_MODULE = "mops_data.envs.dataset_envs"

BATCH_SIZE = 500
NUM_WORKERS = 1
# How many viewpoints to render from each scene before rebuilding it.
# Scene rebuild (gym.make) is expensive (SAPIEN init + URDF loads);
# amortise that cost by rendering multiple camera angles per scene.
N_VIEWPOINTS_PER_SCENE = 4


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

    def _prepare_render_job(
        self,
        assets: List[Dict],
        split: str,
        attempt_index: int,
    ) -> dict:
        """Prepare a single render job with deterministic seeding.

        Jobs are grouped into scenes: every N_VIEWPOINTS_PER_SCENE consecutive
        jobs share the same asset (scene_key) so the subprocess worker can reuse
        the same gym env across them.  Within a scene, each job uses a different
        viewpoint variation so the renders are distinct.
        """
        split_offset = SPLIT_SEED_OFFSETS[split]

        # Scene-level: same asset for all viewpoints within one scene
        scene_idx = attempt_index // N_VIEWPOINTS_PER_SCENE
        asset_info = assets[scene_idx % len(assets)]

        # Viewpoint-level seed: unique per job for camera/lighting variation
        image_seed = self.config.random_seed + split_offset + attempt_index
        np.random.seed(image_seed)
        variations = self._sample_variations_for_asset(
            self.config.max_resampling_attempts
        )

        attempts = [
            {
                "env_kwargs": self._build_env_kwargs(asset_info, var),
                "seed": image_seed + attempt_idx,
                "num_steps": 3,
                "min_segments": self.config.min_segments_threshold,
            }
            for attempt_idx, var in enumerate(variations)
        ]

        return {
            "job_id": attempt_index,
            "scene_key": scene_idx,  # worker closes/recreates env when this changes
            "attempts": attempts,
            "variations": variations,
            "asset_id": asset_info["dir_name"],
        }

    def _generate_images_for_class_split(
        self,
        writer: Any,
        assets: List[Dict],
        target_count: int,
        split: str,
        class_name: str,
    ):
        """Generate and save images for a specific class and split."""
        if target_count <= 0:
            return

        pbar = tqdm(
            total=target_count,
            desc=f"  {split.capitalize():<5} images",
            unit="img",
            dynamic_ncols=True,
        )

        generated_count = 0
        attempt_index = 0
        total_attempts = 0

        while generated_count < target_count:
            n_jobs = BATCH_SIZE * NUM_WORKERS

            # Prepare a batch of render jobs with deterministic seeds
            jobs = {}
            for _ in range(n_jobs):
                job = self._prepare_render_job(assets, split, attempt_index)
                jobs[job["job_id"]] = job
                attempt_index += 1

            job_list = list(jobs.values())
            batches = [
                job_list[i : i + BATCH_SIZE]
                for i in range(0, len(job_list), BATCH_SIZE)
            ]

            # render_batch_parallel is a generator: yields one result per render.
            # This allows tqdm to update after every individual image rather than
            # waiting for the entire batch to complete.
            for result in render_batch_parallel(ENV_ID, ENV_MODULE, batches):
                if generated_count >= target_count:
                    break

                total_attempts += 1
                if result["data"] is None:
                    continue

                variation = jobs[result["job_id"]]["variations"][result["attempt_idx"]]
                render_params = {
                    "split": split,
                    "variation": variation,
                    "image_size": self.config.image_size,
                }
                asset_id = jobs[result["job_id"]]["asset_id"]

                writer.add_image(
                    class_name=class_name,
                    asset_id=asset_id,
                    render_params=render_params,
                    **result["data"],
                )
                generated_count += 1
                pbar.update(1)
                pbar.set_postfix(
                    hit_rate=f"{generated_count / total_attempts:.0%}",
                    refresh=False,
                )

        pbar.close()

    def create_dataset(self):
        """Create the balanced dataset by rendering assets and writing output."""
        total_images = sum(
            plan["target_train"] + plan["target_test"] for plan in self.plan.values()
        )
        print(f"\n=== CREATING DATASET: {self.config.output_path} ===")
        print(f"Estimated total images: {total_images}")

        try:
            with self._open_writer(
                max_images_estimate=total_images,
                class_names=list(self.plan.keys()),
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
