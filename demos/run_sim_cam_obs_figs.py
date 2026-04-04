"""Render one kitchen scene and save all observation modalities as PNG files.

Outputs to ``./rm_figs/``: rgb.png, depth.png, segm.png, class_segm.png,
aff.png, normal.png.
"""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import mops_data  # noqa: F401 — registers MOPS environments
import mops_data.asset_manager.anno_handler as mops_ah

OUTPUT_DIR = os.path.join(".", "rm_figs")


def _create_env():
    df = mops_ah.load_annotations().partnet_mobility_df
    return gym.make(
        "KitchenRenderEnv-v1",
        render_mode="rgb_array",
        obs_mode="rgb+depth+segmentation+normal",
        image_size=(640, 360),
        camera_distance=0.5,
        sensor_configs=dict(shader_pack="rt"),
        asset_df=df,
    )


def save_figs(obs, output_dir: str) -> None:
    """Save all observation modalities from a single step to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    cam = obs["sensor_data"]["base_camera"]

    # RGB
    rgb = cam["rgb"].cpu()[0].numpy().astype(np.uint8)
    plt.imsave(os.path.join(output_dir, "rgb.png"), rgb)

    # Depth
    depth = cam["depth"].cpu()[0].squeeze().numpy()
    plt.imsave(os.path.join(output_dir, "depth.png"), depth, cmap="viridis")

    if "segmentation" in cam:

        def _remap(tensor):
            """Remap arbitrary IDs to 0-N for colormap indexing."""
            out = tensor.clone()
            for i, val in enumerate(torch.unique(tensor)):
                out[tensor == val] = i
            return out

        # Part segmentation
        segm = _remap(cam["segmentation"].cpu()[0]).numpy().squeeze()
        plt.imsave(os.path.join(output_dir, "segm.png"), segm, cmap="nipy_spectral")

        # Class segmentation
        class_segm = _remap(cam["class_segmentation"].cpu()[0]).numpy().squeeze()
        plt.imsave(
            os.path.join(output_dir, "class_segm.png"), class_segm, cmap="nipy_spectral"
        )

        # Affordance segmentation (argmax over affordance channels)
        aff = cam["affordance_segmentation"].cpu()[0].argmax(dim=-1, keepdim=True)
        aff = _remap(aff).numpy().squeeze()
        plt.imsave(os.path.join(output_dir, "aff.png"), aff, cmap="nipy_spectral")

    if "normal" in cam:
        normal = cam["normal"].cpu()[0].numpy()
        normal = (normal + 1) / 2  # [-1, 1] → [0, 1]
        plt.imsave(os.path.join(output_dir, "normal.png"), normal)


if __name__ == "__main__":
    env = _create_env()
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        obs, _, _, _, _ = env.step(None)

    save_figs(obs, OUTPUT_DIR)
    print(f"Saved observation figures to {OUTPUT_DIR}/")
    env.close()
