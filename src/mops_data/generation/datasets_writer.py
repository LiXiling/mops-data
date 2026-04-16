"""HF Datasets-native Parquet writer for Hugging Face Hub publishing.

Built directly on the ``datasets`` library to guarantee full compatibility
with HF Hub Data Studio, streaming ``load_dataset()``, and dataset cards.

Column encodings (all lossless)::

    image           -- PNG bytes (RGB uint8)                [datasets.Image]
    semantic        -- PNG bytes (grayscale uint8/uint16)    [datasets.Image]
    instance        -- PNG bytes (grayscale uint8/uint16)    [datasets.Image]
    part            -- PNG bytes (grayscale uint8/uint16)    [datasets.Image]
    is_partnet      -- PNG bytes (binary mask)               [datasets.Image]
    affordance      -- compressed .npz bytes (H,W,56)        [binary]
    depth           -- compressed .npz bytes (H,W,1)         [binary]
    normal          -- compressed .npz bytes (H,W,3)         [binary]
    bbox            -- JSON string                           [string]
    image_id        -- string
    asset_id        -- string
    render_params   -- JSON string
    class_name      -- string   (only if class_names supplied)
    class_idx       -- int32    (only if class_names supplied)

Image columns use ``datasets.Image`` and render in Data Studio.  Array
columns use compressed ``.npz`` which preserves shape and dtype losslessly
and gets excellent compression on sparse affordance masks (~100x).  No
extra dependencies needed for decoding (just numpy).

Load with::

    from datasets import load_dataset
    ds = load_dataset("parquet", data_dir="dataset_dir")

Decode array columns::

    import numpy as np, io
    arr = np.load(io.BytesIO(row["depth"]))["data"]

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
from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image


# -----------------------------------------------------------------------
# Encoding helpers
# -----------------------------------------------------------------------


def _encode_npz(arr: np.ndarray) -> bytes:
    """Encode a numpy array as compressed .npz bytes (deflate)."""
    buf = io.BytesIO()
    np.savez_compressed(buf, data=arr)
    return buf.getvalue()


def _pil_to_hf_cell(img: Image.Image) -> dict:
    """Convert a PIL Image to the ``{"bytes": ..., "path": ...}`` dict
    expected by ``datasets.Image``."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}


def _make_rgb_cell(arr: np.ndarray) -> dict:
    """RGB uint8 array → HF Image cell."""
    return _pil_to_hf_cell(Image.fromarray(arr))


def _make_mask_cell(arr: np.ndarray) -> dict:
    """Single-channel integer mask → HF Image cell (lossless grayscale PNG)."""
    mask = np.squeeze(arr)
    if mask.max() <= 255:
        pil = Image.fromarray(mask.astype(np.uint8), mode="L")
    else:
        pil = Image.fromarray(mask.astype(np.uint16), mode="I;16")
    return _pil_to_hf_cell(pil)


# -----------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------


def _build_features(has_classes: bool) -> Features:
    """Build the HF ``Features`` spec for the dataset."""
    feat: dict[str, Any] = {
        "image_id": Value("string"),
        "asset_id": Value("string"),
        "render_params": Value("string"),
        "image": HFImage(),
        "semantic": HFImage(),
        "instance": HFImage(),
        "part": HFImage(),
        "is_partnet": HFImage(),
        "affordance": Value("binary"),
        "depth": Value("binary"),
        "normal": Value("binary"),
        "bbox": Value("string"),
    }
    if has_classes:
        feat["class_name"] = Value("string")
        feat["class_idx"] = Value("int32")
    return Features(feat)


# -----------------------------------------------------------------------
# Writer
# -----------------------------------------------------------------------


class DatasetsWriter:
    """Write image datasets as sharded Parquet via the ``datasets`` library.

    Drop-in replacement for WebDatasetWriter / ParquetWriter -- same
    ``add_image()`` interface and context-manager pattern.
    """

    def __init__(
        self,
        output_dir: str,
        max_images_estimate: int = 10000,
        class_names: Optional[List[str]] = None,
        shard_size: int = 500,
    ):
        """
        Args:
            output_dir: Root directory for the dataset output.
            max_images_estimate: Unused (API compat with other writers).
            class_names: Optional class names; enables class columns.
            shard_size: Number of samples per Parquet shard.
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
        self.features = _build_features(self.has_classes)

        for split in ("train", "test"):
            (self.output_dir / split).mkdir(exist_ok=True)

        self._split_totals: dict[str, int] = {"train": 0, "test": 0}
        self._shard_counts: dict[str, int] = {"train": 0, "test": 0}
        self._buffers: dict[str, list[dict]] = {"train": [], "test": []}

        # Background write thread (same pattern as other writers).
        self._write_queue: queue.Queue = queue.Queue(maxsize=200)
        self._write_thread = threading.Thread(
            target=self._write_worker, daemon=True, name="datasets-writer"
        )
        self._write_thread.start()

    # ------------------------------------------------------------------
    # Shard management
    # ------------------------------------------------------------------

    def _flush_shard(self, split: str):
        """Write buffered rows to a Parquet shard via ``datasets``."""
        buf = self._buffers[split]
        if not buf:
            return

        # Transpose list-of-dicts → dict-of-lists for Dataset.from_dict.
        columns: dict[str, list] = {k: [] for k in buf[0]}
        for row in buf:
            for k, v in row.items():
                columns[k].append(v)

        ds = Dataset.from_dict(columns, features=self.features)
        shard_path = (
            self.output_dir / split / f"{self._shard_counts[split]:05d}.parquet"
        )
        ds.to_parquet(str(shard_path))

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

        row: dict[str, Any] = {
            "image_id": f"image_{image_id}",
            "asset_id": asset_id,
            "render_params": json.dumps(kwargs["render_params"]),
            "image": _make_rgb_cell(kwargs["image"]),
            "semantic": _make_mask_cell(kwargs["semantic"])
            if kwargs.get("semantic") is not None
            else None,
            "instance": _make_mask_cell(kwargs["instance"])
            if kwargs.get("instance") is not None
            else None,
            "part": _make_mask_cell(kwargs["part"])
            if kwargs.get("part") is not None
            else None,
            "is_partnet": _make_mask_cell(kwargs["is_partnet"])
            if kwargs.get("is_partnet") is not None
            else None,
            "affordance": _encode_npz(kwargs["affordance"])
            if kwargs.get("affordance") is not None
            else None,
            "depth": _encode_npz(kwargs["depth"])
            if kwargs.get("depth") is not None
            else None,
            "normal": _encode_npz(kwargs["normal"])
            if kwargs.get("normal") is not None
            else None,
            "bbox": json.dumps(bbox if bbox is not None else []),
        }
        if self.has_classes:
            row["class_name"] = kwargs["class_name"]
            row["class_idx"] = self.class_to_idx[kwargs["class_name"]]
        return split, row

    # ------------------------------------------------------------------
    # Background write thread
    # ------------------------------------------------------------------

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
    # Public API (mirrors other writers)
    # ------------------------------------------------------------------

    def add_image(self, **kwargs: Any) -> str:
        """Enqueue a single sample for writing.  Returns immediately.

        Same interface as ``WebDatasetWriter.add_image()`` /
        ``ParquetWriter.add_image()``.
        """
        image_idx = self.images_written
        self.images_written += 1
        self._write_queue.put((image_idx, kwargs))
        return f"image_{image_idx:06d}"

    def finalize(self):
        """Drain the write queue, flush remaining shards, write metadata."""
        self._write_queue.put(None)  # poison pill
        self._write_thread.join()

        if self.images_written == 0:
            print("Warning: No images were written.")
            return

        for split in ("train", "test"):
            self._flush_shard(split)

        info: dict[str, Any] = {
            "total_images": self.images_written,
            "creation_date": datetime.datetime.now().isoformat(),
            "version": "4.0",
            "format": "datasets_parquet",
            "encodings": {
                "image": "PNG (RGB uint8) — datasets.Image feature",
                "semantic|instance|part|is_partnet": (
                    "PNG (grayscale) — datasets.Image feature"
                ),
                "affordance|depth|normal": (
                    "compressed .npz bytes — np.load(BytesIO(v))['data']"
                ),
                "bbox": "JSON string",
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
                print(
                    f"  {split}: {n} images ({self._shard_counts[split]} shard(s))"
                )

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
