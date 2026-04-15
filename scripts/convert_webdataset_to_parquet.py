"""Convert a WebDataset (sharded TAR) dataset to Parquet shards.

Reads a dataset produced by :class:`WebDatasetWriter` (with ``.npz``-encoded
arrays for affordance/depth/normal that HF's webdataset loader cannot decode)
and rewrites it as Parquet shards with HF-friendly per-column encodings:

* RGB and integer masks remain PNG bytes.
* Affordance is re-encoded as per-channel COCO RLE (lossless, ~100x smaller
  than the dense array).
* Depth and normal are stored as raw float32 bytes (lossless, parquet zstd
  handles compression).

See :mod:`mops_data.generation.parquet_writer` for the output schema.

Usage
-----
# Default output (same parent dir, ``_parquet`` suffix):
python scripts/convert_webdataset_to_parquet.py data/mops_data/dataset_wds

# Custom output:
python scripts/convert_webdataset_to_parquet.py data/mops_data/dataset_wds -o data/mops_data/dataset_pq

# Custom shard size:
python scripts/convert_webdataset_to_parquet.py data/mops_data/dataset_wds --shard-size 200
"""

import argparse
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image
from tqdm import tqdm

from mops_data.generation.parquet_writer import ParquetWriter


# Component names whose payload is a single-channel integer PNG mask.
INT_MASK_KEYS = {"semantic", "instance", "part", "is_partnet"}
# Component names whose payload is a compressed numpy array (.npz).
NPZ_KEYS = {"affordance", "depth", "normal"}


def _decode_npz(payload: bytes) -> np.ndarray:
    with np.load(io.BytesIO(payload)) as f:
        return f["data"]


def _decode_png(payload: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(payload)))


def _parse_member_name(name: str) -> tuple[str, str]:
    """Split a TAR member name like ``000123.depth.npz`` into (key, suffix).

    The key is everything before the first dot; the suffix is the remainder
    (e.g. ``"depth.npz"`` or ``"png"`` for the bare image).
    """
    base = name.rsplit("/", 1)[-1]
    head, _, tail = base.partition(".")
    return head, tail


def _iter_samples(tar_path: Path) -> Iterator[dict[str, bytes]]:
    """Yield grouped sample dicts ``{suffix: bytes}`` from one TAR shard.

    Members are emitted in tar order; we rely on the writer grouping all
    files for a given key contiguously, but flush whenever we see a key
    transition.
    """
    current_key = None
    bucket: dict[str, bytes] = {}
    with tarfile.open(str(tar_path), "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, suffix = _parse_member_name(member.name)
            if current_key is not None and key != current_key:
                yield bucket
                bucket = {}
            current_key = key
            f = tar.extractfile(member)
            if f is None:
                continue
            bucket[suffix] = f.read()
        if bucket:
            yield bucket


def _sample_to_kwargs(
    sample: dict[str, bytes], split: str
) -> dict[str, Any] | None:
    """Convert a raw sample dict to ``ParquetWriter.add_image`` kwargs.

    Returns ``None`` if the sample is missing the RGB image (incomplete).
    """
    if "png" not in sample:
        return None

    meta = json.loads(sample["json"]) if "json" in sample else {}
    render_params = meta.get("render_params", {}) or {}
    if "split" not in render_params:
        render_params["split"] = split

    kwargs: dict[str, Any] = {
        "image": _decode_png(sample["png"]),
        "asset_id": meta.get("asset_id", ""),
        "render_params": render_params,
    }
    if "class_name" in meta:
        kwargs["class_name"] = meta["class_name"]

    for k in INT_MASK_KEYS:
        suffix = f"{k}.png"
        if suffix in sample:
            kwargs[k] = _decode_png(sample[suffix])

    for k in NPZ_KEYS:
        suffix = f"{k}.npz"
        if suffix in sample:
            kwargs[k] = _decode_npz(sample[suffix])

    if "bbox.json" in sample:
        kwargs["bbox"] = json.loads(sample["bbox.json"])

    return kwargs


def convert(input_dir: str, output_dir: str, shard_size: int = 100):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"WebDataset directory not found: {input_dir}")

    info_path = input_dir / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    class_names = info.get("class_names")

    # Collect tars per split + a count for the progress bar.
    shards_by_split: dict[str, list[Path]] = defaultdict(list)
    total_images = 0
    for split in ("train", "test"):
        split_dir = input_dir / split
        if not split_dir.exists():
            continue
        shards = sorted(split_dir.glob("*.tar"))
        shards_by_split[split] = shards
        total_images += info.get("splits", {}).get(split, {}).get("num_images", 0)

    pbar = tqdm(total=total_images or None, desc="Converting", unit="img")

    with ParquetWriter(
        output_dir,
        max_images_estimate=total_images,
        class_names=class_names,
        shard_size=shard_size,
    ) as writer:
        for split, shards in shards_by_split.items():
            for shard in shards:
                for sample in _iter_samples(shard):
                    kwargs = _sample_to_kwargs(sample, split=split)
                    if kwargs is None:
                        continue
                    writer.add_image(**kwargs)
                    pbar.update(1)

    pbar.close()
    print(f"\nConversion complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a WebDataset dataset to Parquet shards."
    )
    parser.add_argument(
        "input_dir", help="Path to the source WebDataset directory."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: <input_dir>_parquet).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500,
        help="Number of samples per Parquet shard (default: 500).",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = str(Path(args.input_dir)).rstrip("/") + "_parquet"

    convert(args.input_dir, output, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
