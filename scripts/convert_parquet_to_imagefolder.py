"""Convert a Parquet-format MOPS dataset to the HF ImageFolder format.

Usage::

    python scripts/convert_parquet_to_imagefolder.py <input_dir> <output_dir>

The input directory should contain ``train/`` and/or ``test/`` sub-directories
with ``.parquet`` shard files produced by the old ``ParquetWriter``.

The output follows the HF ImageFolder convention and can be loaded with::

    datasets.load_dataset("imagefolder", data_dir="<output_dir>")
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from mops_data.generation.imagefolder_writer import ImageFolderWriter

# Columns that were stored as .npy binary blobs in the old Parquet format.
_NPY_COLUMNS = ImageFolderWriter.MASK_COLUMNS | ImageFolderWriter.ARRAY_COLUMNS


def _iter_parquet_shards(split_dir: Path):
    """Yield (row_dict, shard_path) by streaming row groups from each shard file."""
    shard_files = sorted(split_dir.glob("*.parquet"))
    for shard_path in shard_files:
        pf = pq.ParquetFile(shard_path)
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx)
            for row_idx in range(table.num_rows):
                yield {col: table.column(col)[row_idx].as_py() for col in table.column_names}


def convert_parquet_to_imagefolder(input_dir: str, output_dir: str) -> None:
    """Stream through Parquet shards and re-write in ImageFolder format."""

    input_path = Path(input_dir)
    print(f"Scanning Parquet shards in {input_path} ...")

    # Discover splits and count rows (metadata only, no data loaded).
    splits: dict[str, list[Path]] = {}
    split_counts: dict[str, int] = {}
    for split in ("train", "test"):
        split_dir = input_path / split
        shards = sorted(split_dir.glob("*.parquet")) if split_dir.is_dir() else []
        if shards:
            splits[split] = shards
            total = sum(pq.read_metadata(s).num_rows for s in shards)
            split_counts[split] = total

    if not splits:
        raise FileNotFoundError(
            f"No train/ or test/ Parquet shards found in {input_path}"
        )

    # Detect class names from the old dataset_info.json if present.
    class_names = None
    info_path = input_path / "dataset_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        class_names = info.get("class_names")

    print(f"Output directory: {output_dir}")
    for s, n in split_counts.items():
        print(f"  {s}: {n} rows")
    if class_names:
        print(f"Class names: {class_names}")

    with ImageFolderWriter(output_dir, class_names=class_names) as writer:
        for split_name, shard_files in splits.items():
            split_dir = input_path / split_name
            n_rows = split_counts[split_name]
            print(f"\nConverting split '{split_name}' ({n_rows} rows) ...")

            for row in tqdm(
                _iter_parquet_shards(split_dir),
                total=n_rows,
                desc=split_name,
                unit="img",
            ):
                kwargs: dict = {}

                # RGB image — stored as {"bytes": <png>, "path": ...} dict.
                img_field = row.get("image")
                if isinstance(img_field, dict):
                    kwargs["image"] = np.array(
                        Image.open(io.BytesIO(img_field["bytes"]))
                    )
                elif isinstance(img_field, bytes):
                    kwargs["image"] = np.array(Image.open(io.BytesIO(img_field)))
                else:
                    kwargs["image"] = np.array(img_field)

                # render_params — stored as JSON string.
                render_params = json.loads(row["render_params"])
                render_params["split"] = split_name
                kwargs["render_params"] = render_params

                kwargs["asset_id"] = row.get("asset_id", "")

                if class_names and "class_name" in row:
                    kwargs["class_name"] = row["class_name"]

                # Decode .npy binary columns (masks + arrays).
                for col in _NPY_COLUMNS:
                    data = row.get(col)
                    if data is not None:
                        kwargs[col] = np.load(io.BytesIO(data))

                # bbox — stored as JSON string.
                bbox = row.get("bbox")
                if bbox is not None:
                    kwargs["bbox"] = json.loads(bbox)

                writer.add_image(**kwargs)

    print("\nConversion complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Parquet MOPS dataset to HF ImageFolder format."
    )
    parser.add_argument("input_dir", help="Path to existing Parquet dataset directory")
    parser.add_argument("output_dir", help="Path for the new ImageFolder dataset")
    args = parser.parse_args()

    convert_parquet_to_imagefolder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
