import datetime
import io
import json
import queue
import threading
from pathlib import Path
from typing import Any, List, Optional

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


class ParquetWriter:
    """
    Write image-based datasets to Hugging Face Parquet format.

    Drop-in replacement for HDF5Writer — same add_image() interface and context
    manager pattern.  Output is a directory of Parquet shards per split, loadable
    with ``datasets.load_dataset("parquet", data_dir=...)``.

    The RGB image is stored as a ``datasets.Image()`` feature (PNG encoded).
    All other array columns (masks, depth, normals) are stored as raw ``.npy``
    bytes (``Value("binary")``) to preserve original dtypes losslessly.
    Decode any array column with::

        np.load(io.BytesIO(row["depth"]))

    Rows are written incrementally to Parquet via PyArrow's streaming writer,
    so memory usage is bounded by ``row_group_size`` (not ``shard_size``).
    """

    # Mirrors HDF5Writer.DATA_SPEC keys.
    DATA_SPEC = (
        "semantic",
        "instance",
        "part",
        "affordance",
        "depth",
        "normal",
        "is_partnet",
        "bbox",
    )

    # All array data stored as lossless npy bytes.
    ARRAY_COLUMNS = {
        "semantic",
        "instance",
        "part",
        "affordance",
        "depth",
        "normal",
        "is_partnet",
    }
    # JSON-serialised columns.
    JSON_COLUMNS = {"bbox"}

    def __init__(
        self,
        output_dir: str,
        max_images_estimate: int = 10000,
        class_names: Optional[List[str]] = None,
        shard_size: int = 1000,
        row_group_size: int = 50,
    ):
        """
        Args:
            output_dir: Directory where the Parquet dataset will be written.
            max_images_estimate: Unused (kept for API compat with HDF5Writer).
            class_names: Optional class names. Enables per-image class columns.
            shard_size: Number of images per Parquet shard file.
            row_group_size: Number of rows buffered before writing a row group
                to the open shard file. Controls peak memory usage.
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
        self._row_group_size = row_group_size

        # PyArrow schema (built lazily, cached).
        self._pa_schema: Optional[pa.Schema] = None

        # Per-split state.
        self._writers: dict[str, Optional[pq.ParquetWriter]] = {
            "train": None,
            "test": None,
        }
        self._row_buffers: dict[str, list[dict]] = {"train": [], "test": []}
        self._shard_rows: dict[str, int] = {"train": 0, "test": 0}
        self._shard_counts: dict[str, int] = {"train": 0, "test": 0}
        self._split_totals: dict[str, int] = {"train": 0, "test": 0}

        for split in ("train", "test"):
            (self.output_dir / split).mkdir(exist_ok=True)

        # Background write thread (same pattern as HDF5Writer).
        # maxsize caps queued raw samples to bound memory from the producer side.
        self._write_queue: queue.Queue = queue.Queue(maxsize=32)
        self._write_thread = threading.Thread(
            target=self._write_worker, daemon=True, name="parquet-writer"
        )
        self._write_thread.start()

    # ------------------------------------------------------------------
    # Features / schema
    # ------------------------------------------------------------------

    def _build_features(self) -> datasets.Features:
        """Build the HF Features schema (all DATA_SPEC columns included)."""
        feat: dict[str, Any] = {
            "image_id": datasets.Value("string"),
            "image": datasets.Image(),
            "asset_id": datasets.Value("string"),
            "render_params": datasets.Value("string"),
        }
        if self.has_classes:
            feat["class_name"] = datasets.Value("string")
            feat["class_idx"] = datasets.Value("int32")

        for name in self.DATA_SPEC:
            if name in self.JSON_COLUMNS:
                feat[name] = datasets.Value("string")
            else:
                feat[name] = datasets.Value("binary")

        return datasets.Features(feat)

    def _get_pa_schema(self) -> pa.Schema:
        """Return the Arrow schema (with HF metadata), cached after first call."""
        if self._pa_schema is None:
            self._pa_schema = self._build_features().arrow_schema
        return self._pa_schema

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_png(arr: np.ndarray) -> bytes:
        """Encode an RGB uint8 array as PNG bytes."""
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _encode_npy(arr: np.ndarray) -> bytes:
        """Encode a numpy array as .npy bytes (lossless)."""
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Row encoding
    # ------------------------------------------------------------------

    def _encode_row(self, image_idx: int, kwargs: dict) -> tuple[str, dict]:
        """Encode one sample into a row dict. Returns (split, row)."""
        image_id = f"image_{image_idx:06d}"
        split = kwargs["render_params"]["split"]

        asset_id = kwargs.get("asset_id", "")
        if isinstance(asset_id, (list, np.ndarray)):
            asset_id = json.dumps(
                asset_id.tolist() if isinstance(asset_id, np.ndarray) else asset_id
            )

        row: dict[str, Any] = {
            "image_id": image_id,
            "image": {"bytes": self._encode_png(kwargs["image"]), "path": None},
            "asset_id": asset_id,
            "render_params": json.dumps(kwargs["render_params"]),
        }

        if self.has_classes:
            row["class_name"] = kwargs["class_name"]
            row["class_idx"] = self.class_to_idx[kwargs["class_name"]]

        for name in self.DATA_SPEC:
            data = kwargs.get(name)
            if data is None:
                continue
            if name in self.JSON_COLUMNS:
                row[name] = json.dumps(
                    data.tolist() if isinstance(data, np.ndarray) else data
                )
            else:
                row[name] = self._encode_npy(data)

        return split, row

    # ------------------------------------------------------------------
    # Incremental Parquet writing
    # ------------------------------------------------------------------

    def _get_writer(self, split: str) -> pq.ParquetWriter:
        """Return the open ParquetWriter for *split*, creating one if needed."""
        if self._writers[split] is None:
            shard_path = (
                self.output_dir / split / f"{self._shard_counts[split]:05d}.parquet"
            )
            self._writers[split] = pq.ParquetWriter(
                str(shard_path), self._get_pa_schema()
            )
        return self._writers[split]

    def _flush_row_group(self, split: str):
        """Write buffered rows as a single row group to the current shard."""
        rows = self._row_buffers[split]
        if not rows:
            return

        schema = self._get_pa_schema()

        # Fill absent optional columns with None.
        field_names = [f.name for f in schema]
        for row in rows:
            for name in field_names:
                row.setdefault(name, None)

        columns = {name: [row[name] for row in rows] for name in field_names}
        table = pa.table(columns, schema=schema)

        writer = self._get_writer(split)
        writer.write_table(table)
        self._row_buffers[split] = []

    def _close_shard(self, split: str):
        """Flush remaining rows and close the current shard file."""
        self._flush_row_group(split)
        if self._writers[split] is not None:
            self._writers[split].close()
            self._writers[split] = None
            self._shard_counts[split] += 1
            self._shard_rows[split] = 0

    # ------------------------------------------------------------------
    # Background write thread
    # ------------------------------------------------------------------

    def _write_one(self, image_idx: int, kwargs: dict):
        split, row = self._encode_row(image_idx, kwargs)
        self._row_buffers[split].append(row)
        self._shard_rows[split] += 1
        self._split_totals[split] += 1

        if len(self._row_buffers[split]) >= self._row_group_size:
            self._flush_row_group(split)

        if self._shard_rows[split] >= self.shard_size:
            self._close_shard(split)

        self._images_flushed += 1
        if self._images_flushed % 100 == 0:
            print(f"Written {self._images_flushed} images...")

    def _write_worker(self):
        """Background thread: drains the write queue and calls _write_one."""
        while True:
            item = self._write_queue.get()
            if item is None:  # poison pill
                self._write_queue.task_done()
                break
            image_idx, kwargs = item
            try:
                self._write_one(image_idx, kwargs)
            except Exception as e:
                print(f"Parquet write error for image_{image_idx:06d}: {e}")
            finally:
                self._write_queue.task_done()

    # ------------------------------------------------------------------
    # Public API (mirrors HDF5Writer)
    # ------------------------------------------------------------------

    def add_image(self, **kwargs: Any) -> str:
        """
        Enqueue a single data entry for writing. Returns immediately.

        Accepts the same keyword arguments as ``HDF5Writer.add_image``:
        ``image``, ``asset_id``, ``render_params``, ``class_name``, and any
        DATA_SPEC keys (``semantic``, ``depth``, etc.).
        """
        image_idx = self.images_written
        self.images_written += 1
        self._write_queue.put((image_idx, kwargs))
        return f"image_{image_idx:06d}"

    def finalize(self):
        """Flush remaining data and write dataset metadata."""
        # Drain the write queue.
        self._write_queue.put(None)
        self._write_thread.join()

        if self.images_written == 0:
            print("Warning: No images were written.")
            return

        # Close any open shards (flushes remaining buffered rows).
        for split in ("train", "test"):
            self._close_shard(split)

        # Write dataset metadata.
        info: dict[str, Any] = {
            "total_images": self.images_written,
            "creation_date": datetime.datetime.now().isoformat(),
            "version": "2.0",
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
        """No-op (shard files are closed after each write)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        self.close()

    @property
    def total_images_written(self) -> int:
        return self.images_written
