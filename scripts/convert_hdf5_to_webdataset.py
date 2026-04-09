"""Convert an HDF5 dataset to WebDataset (sharded TAR) format.

Reads a dataset produced by HDF5Writer and writes it as sharded TAR archives
using WebDatasetWriter, suitable for publishing on Hugging Face Hub.

Usage
-----
# Default output (same stem with _wds suffix):
python scripts/convert_hdf5_to_webdataset.py data/mops_data/dataset.h5

# Custom output directory:
python scripts/convert_hdf5_to_webdataset.py data/mops_data/dataset.h5 -o data/dataset_wds

# Custom shard size (samples per TAR file):
python scripts/convert_hdf5_to_webdataset.py data/mops_data/dataset.h5 --shard-size 200
"""

import argparse
import json
from pathlib import Path

import h5py
from tqdm import tqdm

from mops_data.generation.webdataset_writer import WebDatasetWriter

MASK_GROUPS = [
    "semantic",
    "instance",
    "part",
    "affordance",
    "depth",
    "normal",
    "is_partnet",
    "bbox",
]


def convert(h5_path: str, output_dir: str, shard_size: int = 100):
    """Convert an HDF5 dataset file to WebDataset format.

    Args:
        h5_path: Path to the source HDF5 file.
        output_dir: Directory where the WebDataset will be written.
        shard_size: Number of samples per TAR shard.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(str(h5_path), "r") as f:
        image_keys = sorted(f["images"].keys())
        n_images = len(image_keys)
        print(f"Found {n_images} images in {h5_path}")

        # Read metadata arrays.
        splits = f["metadata/splits"][:]
        image_info = f["metadata/image_info"][:]

        # Read class info if present.
        class_names = None
        class_labels = None
        if "labels" in f and "class_names" in f["labels"]:
            class_names = [n.decode("utf-8") for n in f["labels/class_names"][:]]
            class_labels = f["labels/class_labels"][:]
            print(f"Found {len(class_names)} classes")

        # Detect available mask groups.
        available_masks = []
        if "masks" in f:
            available_masks = [m for m in MASK_GROUPS if m in f["masks"]]
        print(f"Available mask groups: {available_masks}")

        with WebDatasetWriter(
            output_dir,
            max_images_estimate=n_images,
            class_names=class_names,
            shard_size=shard_size,
        ) as writer:
            for i, key in enumerate(tqdm(image_keys, desc="Converting", unit="img")):
                # Parse stored metadata for this image.
                meta = json.loads(image_info[i])
                render_params = meta.get("render_params", {})

                # Ensure split is set (should already be in render_params,
                # but fall back to the splits array for safety).
                if "split" not in render_params:
                    render_params["split"] = "train" if splits[i] else "test"

                kwargs: dict = {
                    "image": f["images"][key][:],
                    "asset_id": meta.get("asset_id", ""),
                    "render_params": render_params,
                }

                if class_names is not None and class_labels is not None:
                    kwargs["class_name"] = class_names[class_labels[i]]

                # Read all available masks.
                for mask_name in available_masks:
                    if key in f["masks"][mask_name]:
                        kwargs[mask_name] = f["masks"][mask_name][key][:]

                writer.add_image(**kwargs)

    print(f"\nConversion complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 dataset to WebDataset format."
    )
    parser.add_argument("h5_path", help="Path to the source HDF5 file.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: <h5_stem>_wds next to the H5 file).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100,
        help="Number of samples per TAR shard (default: 100).",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = str(Path(args.h5_path).with_suffix("")) + "_wds"

    convert(args.h5_path, output, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
