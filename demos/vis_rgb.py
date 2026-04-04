"""Quick visualization of a few train and test RGB images from an HDF5 dataset."""

import argparse
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np


def vis_rgb(h5_path: str, n: int = 4, seed: int = 0) -> None:
    with h5py.File(h5_path, "r") as f:
        splits = f["metadata/splits"][:]  # bool array, True = train
        images_group = f["images"]
        image_keys = sorted(images_group.keys())  # image_000000, ...

        train_keys = [k for k, s in zip(image_keys, splits) if s]
        test_keys = [k for k, s in zip(image_keys, splits) if not s]

        rng = np.random.default_rng(seed)
        selected_train = rng.choice(
            train_keys, size=min(n, len(train_keys)), replace=False
        )
        selected_test = rng.choice(
            test_keys, size=min(n, len(test_keys)), replace=False
        )

        train_imgs = [images_group[k][:] for k in selected_train]
        test_imgs = [images_group[k][:] for k in selected_test]

        # Try to grab a class/asset label for each image
        image_info = f["metadata/image_info"][:]

        def get_label(key):
            idx = int(key.split("_")[1])
            meta = json.loads(image_info[idx])
            return meta.get("class_name") or meta.get("asset_id", key)

        train_labels = [get_label(k) for k in selected_train]
        test_labels = [get_label(k) for k in selected_test]

    n_cols = max(len(train_imgs), len(test_imgs))
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col, (img, label) in enumerate(zip(train_imgs, train_labels)):
        axes[0, col].imshow(img)
        axes[0, col].set_title(label, fontsize=8)
        axes[0, col].axis("off")

    for col, (img, label) in enumerate(zip(test_imgs, test_labels)):
        axes[1, col].imshow(img)
        axes[1, col].set_title(label, fontsize=8)
        axes[1, col].axis("off")

    # Hide any unused axes
    for col in range(len(train_imgs), n_cols):
        axes[0, col].axis("off")
    for col in range(len(test_imgs), n_cols):
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Train", fontsize=11, labelpad=8)
    axes[1, 0].set_ylabel("Test", fontsize=11, labelpad=8)

    fig.suptitle(h5_path, fontsize=9)
    plt.tight_layout()
    # plt.show()
    plt.savefig("vis_rgb.png", dpi=300)
    print(f"Saved visualization to vis_rgb.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize train/test RGB images from an HDF5 dataset."
    )
    parser.add_argument(
        "h5_path",
        nargs="?",
        default="data/mops_data/mops_kitchen_dataset.h5",
        help="Path to the HDF5 dataset file.",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=4,
        help="Number of images to show per split (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for image selection (default: 0).",
    )
    args = parser.parse_args()

    vis_rgb(args.h5_path, n=args.n, seed=args.seed)
