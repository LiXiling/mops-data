"""Parquet (HF-native) writer for Hugging Face Hub publishing.

Writes image datasets as sharded Parquet files.  Parquet is Hugging Face's
native storage format and plays well with ``datasets.load_dataset("parquet",
data_dir=...)`` without any of the webdataset/tar quirks (e.g. compressed
``.npz`` payloads that HF's webdataset loader cannot decode).

Per-column encoding (all lossless)::

    image           -- PNG bytes (uint8 RGB)                       [Image feature]
    semantic        -- PNG bytes (grayscale, uint8 or uint16)      [Image feature]
    instance        -- PNG bytes (grayscale, uint8 or uint16)      [Image feature]
    part            -- PNG bytes (grayscale, uint8 or uint16)      [Image feature]
    is_partnet      -- PNG bytes (binary, uint8)                   [Image feature]
    affordance      -- list[56] of COCO RLE dicts (binary multi-hot)
    depth           -- struct{data: bytes, shape: list[int], dtype: str}
                       (raw float32 little-endian bytes; parquet zstd
                        compresses smooth depth well)
    normal          -- struct{data: bytes, shape: list[int], dtype: str}
                       (raw float32 little-endian bytes)
    bbox            -- list[list[float]]  [[x, y, w, h, class_id], ...]
    image_id        -- string
    asset_id        -- string
    render_params   -- JSON string (schema varies per pipeline)
    class_name      -- string   (only if class_names supplied)
    class_idx       -- int      (only if class_names supplied)

Load with::

    from datasets import load_dataset
    ds = load_dataset("parquet", data_dir="dataset_dir")

Decode float arrays with::

    import numpy as np
    cell = row["depth"]               # {"data": b"...", "shape": [...], "dtype": "float32"}
    arr  = np.frombuffer(cell["data"], dtype=cell["dtype"]).reshape(cell["shape"])

Decode affordance with::

    from pycocotools import mask as coco_mask
    rles = row["affordance"]          # list of 56 RLE dicts
    channels = np.stack([coco_mask.decode(r) for r in rles], axis=-1)  # (H,W,56)

Directory layout::

    dataset_dir/
    ├── train/
    │   ├── 00000.parquet
    │   └── ...
    ├── test/
    │   ├── 00000.parquet
    │   └── ...
    └── dataset_info.json
"""

import datetime
import io
import json
import queue
import threading
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as coco_mask


def encode_rgb_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def encode_mask_png(arr: np.ndarray) -> bytes:
    mask = np.squeeze(arr)
    buf = io.BytesIO()
    if mask.max() <= 255:
        Image.fromarray(mask.astype(np.uint8), mode="L").save(buf, format="PNG")
    else:
        Image.fromarray(mask.astype(np.uint16), mode="I;16").save(buf, format="PNG")
    return buf.getvalue()


def encode_affordance_rle(arr: np.ndarray) -> list[dict]:
    """Encode an (H,W,C) multi-hot binary array as a list of C COCO RLE dicts.

    Each channel is encoded independently with ``pycocotools.mask.encode``.
    The binary mask must be Fortran-order uint8 per pycocotools' contract.
    The ``counts`` field is decoded back to a UTF-8 ``str`` so pyarrow can
    store it in a string column (pycocotools returns raw bytes).
    """
    if arr.ndim != 3:
        raise ValueError(f"affordance must be (H,W,C); got shape {arr.shape}")
    rles = []
    for c in range(arr.shape[2]):
        channel = np.asfortranarray((arr[:, :, c] > 0).astype(np.uint8))
        rle = coco_mask.encode(channel)
        rles.append(
            {
                "size": list(rle["size"]),
                "counts": rle["counts"].decode("ascii"),
            }
        )
    return rles


def encode_float_array(arr: np.ndarray) -> dict:
    """Encode a float ndarray as a struct of raw little-endian bytes + shape.

    Parquet zstd compression handles the redundancy; raw bytes preserve full
    float32 precision losslessly.
    """
    arr = np.ascontiguousarray(arr)
    return {
        "data": arr.tobytes(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


# ----------------------------------------------------------------------
# Arrow schema
# ----------------------------------------------------------------------

_RLE_STRUCT = pa.struct(
    [
        pa.field("size", pa.list_(pa.int32())),
        pa.field("counts", pa.string()),
    ]
)

_FLOAT_ARRAY_STRUCT = pa.struct(
    [
        pa.field("data", pa.binary()),
        pa.field("shape", pa.list_(pa.int32())),
        pa.field("dtype", pa.string()),
    ]
)

# HF `datasets.Image` on-disk layout: struct<bytes: binary, path: string>.
# Matching this exactly (plus the schema metadata below) is what makes the
# Hub dataset viewer render these columns as images instead of "unknown".
_IMAGE_STRUCT = pa.struct(
    [
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ]
)

_IMAGE_COLUMNS = ("image", "semantic", "instance", "part", "is_partnet")


def _build_schema(has_classes: bool) -> pa.Schema:
    fields = [
        pa.field("image_id", pa.string()),
        pa.field("asset_id", pa.string()),
        pa.field("render_params", pa.string()),  # JSON-serialized
        pa.field("image", _IMAGE_STRUCT),
        pa.field("semantic", _IMAGE_STRUCT),
        pa.field("instance", _IMAGE_STRUCT),
        pa.field("part", _IMAGE_STRUCT),
        pa.field("is_partnet", _IMAGE_STRUCT),
        pa.field("affordance", pa.list_(_RLE_STRUCT)),
        pa.field("depth", _FLOAT_ARRAY_STRUCT),
        pa.field("normal", _FLOAT_ARRAY_STRUCT),
        pa.field("bbox", pa.list_(pa.list_(pa.float64()))),
    ]
    if has_classes:
        fields += [
            pa.field("class_name", pa.string()),
            pa.field("class_idx", pa.int32()),
        ]

    features: dict[str, Any] = {
        "image_id": {"dtype": "string", "_type": "Value"},
        "asset_id": {"dtype": "string", "_type": "Value"},
        "render_params": {"dtype": "string", "_type": "Value"},
        "image": {"_type": "Image"},
        "semantic": {"_type": "Image"},
        "instance": {"_type": "Image"},
        "part": {"_type": "Image"},
        "is_partnet": {"_type": "Image"},
        "affordance": {
            "feature": {
                "size": {
                    "feature": {"dtype": "int32", "_type": "Value"},
                    "_type": "Sequence",
                },
                "counts": {"dtype": "string", "_type": "Value"},
            },
            "_type": "Sequence",
        },
        "depth": {
            "data": {"dtype": "binary", "_type": "Value"},
            "shape": {
                "feature": {"dtype": "int32", "_type": "Value"},
                "_type": "Sequence",
            },
            "dtype": {"dtype": "string", "_type": "Value"},
        },
        "normal": {
            "data": {"dtype": "binary", "_type": "Value"},
            "shape": {
                "feature": {"dtype": "int32", "_type": "Value"},
                "_type": "Sequence",
            },
            "dtype": {"dtype": "string", "_type": "Value"},
        },
        "bbox": {
            "feature": {
                "feature": {"dtype": "float64", "_type": "Value"},
                "_type": "Sequence",
            },
            "_type": "Sequence",
        },
    }
    if has_classes:
        features["class_name"] = {"dtype": "string", "_type": "Value"}
        features["class_idx"] = {"dtype": "int32", "_type": "Value"}

    metadata = {
        b"huggingface": json.dumps({"info": {"features": features}}).encode("utf-8")
    }
    return pa.schema(fields, metadata=metadata)


class ParquetWriter:
    """Write image datasets as sharded Parquet files for HF Hub.

    Drop-in replacement for WebDatasetWriter / HDF5Writer -- same
    ``add_image()`` interface and context-manager pattern.
    """

    def __init__(
        self,
        output_dir: str,
        max_images_estimate: int = 10000,
        class_names: Optional[List[str]] = None,
        shard_size: int = 100,
        compression: str = "zstd",
        row_group_size: int = 32,
    ):
        """
        Args:
            output_dir: Root directory for the Parquet output.
            max_images_estimate: Unused (API compat with other writers).
            class_names: Optional class names; enables per-image class columns.
            shard_size: Number of samples per Parquet shard.
            compression: Parquet compression codec (zstd recommended).
            row_group_size: Rows per parquet row group. Kept small so a
                single row group stays well under HF's 300 MB scan limit
                for random access (depth+normal are ~2 MB/sample raw).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_written = 0
        self._images_flushed = 0
        self.has_classes = class_names is not None
        self.class_names = class_names or []
        self.class_to_idx = (
            {name: idx for idx, name in enumerate(class_names)} if class_names else {}
        )
        self.shard_size = shard_size
        self.compression = compression
        self.row_group_size = row_group_size
        self.schema = _build_schema(self.has_classes)

        for split in ("train", "test"):
            (self.output_dir / split).mkdir(exist_ok=True)

        self._split_totals: dict[str, int] = {"train": 0, "test": 0}
        self._shard_counts: dict[str, int] = {"train": 0, "test": 0}
        self._buffers: dict[str, list[dict]] = {"train": [], "test": []}

        self._write_queue: queue.Queue = queue.Queue(maxsize=200)
        self._write_thread = threading.Thread(
            target=self._write_worker, daemon=True, name="parquet-writer"
        )
        self._write_thread.start()

    # ------------------------------------------------------------------
    # Shard management
    # ------------------------------------------------------------------

    def _flush_shard(self, split: str):
        buf = self._buffers[split]
        if not buf:
            return
        shard_path = (
            self.output_dir / split / f"{self._shard_counts[split]:05d}.parquet"
        )
        table = pa.Table.from_pylist(buf, schema=self.schema)
        pq.write_table(
            table,
            str(shard_path),
            compression=self.compression,
            row_group_size=self.row_group_size,
            write_page_index=True,
        )
        self._shard_counts[split] += 1
        self._buffers[split] = []

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------

    def _build_row(self, image_idx: int, kwargs: dict) -> tuple[str, dict]:
        image_id = f"{image_idx:06d}"
        split = kwargs["render_params"]["split"]

        asset_id = kwargs.get("asset_id", "")
        if isinstance(asset_id, np.ndarray):
            asset_id = asset_id.tolist()
        if not isinstance(asset_id, str):
            asset_id = json.dumps(asset_id)

        bbox = kwargs.get("bbox")
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        if bbox is None:
            bbox = []

        affordance = kwargs.get("affordance")
        if affordance is not None:
            affordance = encode_affordance_rle(affordance)

        depth = kwargs.get("depth")
        normal = kwargs.get("normal")

        def _img(b: Optional[bytes]) -> Optional[dict]:
            return None if b is None else {"bytes": b, "path": None}

        row: dict[str, Any] = {
            "image_id": f"image_{image_id}",
            "asset_id": asset_id,
            "render_params": json.dumps(kwargs["render_params"]),
            "image": _img(encode_rgb_png(kwargs["image"])),
            "semantic": _img(encode_mask_png(kwargs["semantic"]))
            if kwargs.get("semantic") is not None
            else None,
            "instance": _img(encode_mask_png(kwargs["instance"]))
            if kwargs.get("instance") is not None
            else None,
            "part": _img(encode_mask_png(kwargs["part"]))
            if kwargs.get("part") is not None
            else None,
            "is_partnet": _img(encode_mask_png(kwargs["is_partnet"]))
            if kwargs.get("is_partnet") is not None
            else None,
            "affordance": affordance,
            "depth": encode_float_array(depth) if depth is not None else None,
            "normal": encode_float_array(normal) if normal is not None else None,
            "bbox": bbox,
        }
        if self.has_classes:
            row["class_name"] = kwargs["class_name"]
            row["class_idx"] = self.class_to_idx[kwargs["class_name"]]
        return split, row

    def _write_one(self, image_idx: int, kwargs: dict):
        split, row = self._build_row(image_idx, kwargs)
        self._buffers[split].append(row)
        self._split_totals[split] += 1

        if len(self._buffers[split]) >= self.shard_size:
            self._flush_shard(split)

        self._images_flushed += 1
        if self._images_flushed % 100 == 0:
            print(f"Written {self._images_flushed} images...")

    def _write_worker(self):
        while True:
            item = self._write_queue.get()
            if item is None:
                self._write_queue.task_done()
                break
            image_idx, kwargs = item
            try:
                self._write_one(image_idx, kwargs)
            except Exception as e:
                print(f"Write error for image_{image_idx:06d}: {e}")
            finally:
                self._write_queue.task_done()

    # ------------------------------------------------------------------
    # Public API (mirrors WebDatasetWriter / HDF5Writer)
    # ------------------------------------------------------------------

    def add_image(self, **kwargs: Any) -> str:
        image_idx = self.images_written
        self.images_written += 1
        self._write_queue.put((image_idx, kwargs))
        return f"image_{image_idx:06d}"

    def finalize(self):
        self._write_queue.put(None)
        self._write_thread.join()

        if self.images_written == 0:
            print("Warning: No images were written.")
            return

        for split in ("train", "test"):
            self._flush_shard(split)

        info: dict[str, Any] = {
            "total_images": self.images_written,
            "creation_date": datetime.datetime.now().isoformat(),
            "version": "3.0",
            "format": "parquet",
            "compression": self.compression,
            "encodings": {
                "image": "PNG bytes (RGB uint8)",
                "semantic|instance|part|is_partnet": "PNG bytes (grayscale uint8/uint16)",
                "affordance": "list of COCO RLE dicts (one per channel)",
                "depth|normal": "raw float32 bytes + shape + dtype struct",
            },
            "splits": {
                split: {
                    "num_images": self._split_totals[split],
                    "num_shards": self._shard_counts[split],
                }
                for split in ("train", "test")
                if self._shard_counts[split] > 0
            },
        }
        if self.has_classes:
            info["class_names"] = self.class_names
            info["num_classes"] = len(self.class_names)

        (self.output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))

        print(f"\nFinalized dataset with {self.images_written} images.")
        for split in ("train", "test"):
            n = self._split_totals[split]
            if n:
                print(f"  {split}: {n} images ({self._shard_counts[split]} shard(s))")

    def close(self):
        for split in ("train", "test"):
            self._buffers[split] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        self.close()

    @property
    def total_images_written(self) -> int:
        return self.images_written
