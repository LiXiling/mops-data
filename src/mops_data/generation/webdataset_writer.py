"""WebDataset (TAR-based) writer for Hugging Face Hub publishing.

Writes image datasets as sharded TAR archives following the WebDataset
convention.  Each sample is a set of files sharing a common key (image ID)::

    {id}.png               -- RGB image  (PIL-decoded on load)
    {id}.semantic.png      -- semantic mask as lossless grayscale PNG
    {id}.affordance.npz    -- affordance multi-hot (H,W,56), compressed
    {id}.depth.npz         -- depth map (H,W,1) float32, compressed
    {id}.normal.npz        -- surface normal (H,W,3) float32, compressed
    {id}.bbox.json         -- bounding boxes
    {id}.json              -- metadata (asset_id, render_params, class info)

Single-integer masks (semantic, instance, part, is_partnet) are stored as
lossless grayscale PNG.  Multi-channel / float arrays (affordance, depth,
normal) are stored as compressed ``.npz`` (numpy deflate) to preserve shape
and dtype while keeping file size small -- especially for sparse arrays like
affordance (99%+ zeros, ~100x compression).

Load with::

    datasets.load_dataset("webdataset", data_dir="dataset_dir")

Decode array columns with::

    np.load(io.BytesIO(row["depth.npz"]))["data"]

Directory layout::

    dataset_dir/
    ├── train/
    │   ├── 00000.tar
    │   ├── 00001.tar
    │   └── ...
    ├── test/
    │   ├── 00000.tar
    │   └── ...
    └── dataset_info.json
"""

import datetime
import io
import json
import queue
import tarfile
import threading
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image


class WebDatasetWriter:
    """Write image datasets as sharded TAR archives (WebDataset format).

    Drop-in replacement for HDF5Writer -- same ``add_image()`` interface
    and context-manager pattern.
    """

    # Same keys as HDF5Writer.DATA_SPEC.
    DATA_SPEC = {
        "semantic": "png",
        "instance": "png",
        "part": "png",
        "affordance": "npz",
        "depth": "npz",
        "normal": "npz",
        "is_partnet": "png",
        "bbox": "json",
    }

    def __init__(
        self,
        output_dir: str,
        max_images_estimate: int = 10000,
        class_names: Optional[List[str]] = None,
        shard_size: int = 100,
    ):
        """
        Args:
            output_dir: Root directory for the WebDataset output.
            max_images_estimate: Unused (API compat with HDF5Writer).
            class_names: Optional class names; enables per-image class columns.
            shard_size: Number of samples per TAR shard.
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

        for split in ("train", "test"):
            (self.output_dir / split).mkdir(exist_ok=True)

        self._split_totals: dict[str, int] = {"train": 0, "test": 0}
        self._shard_counts: dict[str, int] = {"train": 0, "test": 0}
        self._shard_rows: dict[str, int] = {"train": 0, "test": 0}
        self._tar_files: dict[str, Optional[tarfile.TarFile]] = {
            "train": None,
            "test": None,
        }

        # Background write thread (same pattern as HDF5Writer).
        self._write_queue: queue.Queue = queue.Queue(maxsize=200)
        self._write_thread = threading.Thread(
            target=self._write_worker, daemon=True, name="webdataset-writer"
        )
        self._write_thread.start()

    # ------------------------------------------------------------------
    # TAR shard management
    # ------------------------------------------------------------------

    def _get_tar(self, split: str) -> tarfile.TarFile:
        """Return the open TAR for *split*, creating one if needed."""
        if self._tar_files[split] is None:
            shard_path = (
                self.output_dir / split / f"{self._shard_counts[split]:05d}.tar"
            )
            self._tar_files[split] = tarfile.open(str(shard_path), "w")
        return self._tar_files[split]

    def _close_shard(self, split: str):
        """Close the current TAR shard and advance the shard counter."""
        if self._tar_files[split] is not None:
            self._tar_files[split].close()
            self._tar_files[split] = None
            self._shard_counts[split] += 1
            self._shard_rows[split] = 0

    @staticmethod
    def _add_to_tar(tar: tarfile.TarFile, name: str, data: bytes):
        """Add *data* as a file named *name* to the TAR archive."""
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

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
    def _encode_mask_png(arr: np.ndarray) -> bytes:
        """Encode a single-channel integer mask as a lossless grayscale PNG."""
        mask = np.squeeze(arr)
        buf = io.BytesIO()
        if mask.max() <= 255:
            Image.fromarray(mask.astype(np.uint8), mode="L").save(buf, format="PNG")
        else:
            Image.fromarray(mask.astype(np.uint16), mode="I;16").save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _encode_npz(arr: np.ndarray) -> bytes:
        """Encode a numpy array as compressed .npz bytes (deflate)."""
        buf = io.BytesIO()
        np.savez_compressed(buf, data=arr)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Core write logic (runs on background thread)
    # ------------------------------------------------------------------

    def _write_one(self, image_idx: int, kwargs: dict):
        image_id = f"{image_idx:06d}"
        split = kwargs["render_params"]["split"]
        tar = self._get_tar(split)

        # --- RGB image ---
        self._add_to_tar(tar, f"{image_id}.png", self._encode_png(kwargs["image"]))

        # --- Data columns ---
        for name, fmt in self.DATA_SPEC.items():
            data = kwargs.get(name)
            if data is None:
                continue
            if fmt == "png":
                self._add_to_tar(
                    tar, f"{image_id}.{name}.png", self._encode_mask_png(data)
                )
            elif fmt == "npz":
                self._add_to_tar(tar, f"{image_id}.{name}.npz", self._encode_npz(data))
            elif fmt == "json":
                val = data.tolist() if isinstance(data, np.ndarray) else data
                self._add_to_tar(
                    tar, f"{image_id}.{name}.json", json.dumps(val).encode()
                )

        # --- Metadata JSON ---
        asset_id = kwargs.get("asset_id", "")
        if isinstance(asset_id, (list, np.ndarray)):
            asset_id = json.dumps(
                asset_id.tolist() if isinstance(asset_id, np.ndarray) else asset_id
            )

        metadata: dict[str, Any] = {
            "image_id": f"image_{image_id}",
            "asset_id": asset_id,
            "render_params": kwargs["render_params"],
        }
        if self.has_classes:
            metadata["class_name"] = kwargs["class_name"]
            metadata["class_idx"] = self.class_to_idx[kwargs["class_name"]]

        self._add_to_tar(tar, f"{image_id}.json", json.dumps(metadata).encode())

        self._split_totals[split] += 1
        self._shard_rows[split] += 1

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
                print(f"Write error for image_{image_idx:06d}: {e}")
            finally:
                self._write_queue.task_done()

    # ------------------------------------------------------------------
    # Public API (mirrors HDF5Writer)
    # ------------------------------------------------------------------

    def add_image(self, **kwargs: Any) -> str:
        """
        Enqueues a single data entry for writing.  Returns immediately;
        the actual TAR writes happen on the background write thread.

        Args (passed as keyword arguments):
            image (np.ndarray): The main RGB image array.
            asset_id (str): The ID of the primary asset or scene.
            render_params (dict): Rendering parameters, must contain 'split'.
            class_name (str): The object class name (required for class-based
                datasets).
            **other_data: Other data arrays, keys must match DATA_SPEC
                          (e.g., semantic=..., depth=...).
        """
        image_idx = self.images_written
        self.images_written += 1
        self._write_queue.put((image_idx, kwargs))
        return f"image_{image_idx:06d}"

    def finalize(self):
        """Finalizes the dataset by closing open shards and writing metadata."""
        # Drain the write queue before touching any state.
        self._write_queue.put(None)  # poison pill
        self._write_thread.join()

        if self.images_written == 0:
            print("Warning: No images were written.")
            return

        # Close any remaining open shards.
        for split in ("train", "test"):
            self._close_shard(split)

        info: dict[str, Any] = {
            "total_images": self.images_written,
            "creation_date": datetime.datetime.now().isoformat(),
            "version": "2.0",
            "format": "webdataset",
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

        split_counts = {
            "train": self._split_totals["train"],
            "test": self._split_totals["test"],
        }

        (self.output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))

        print(f"\nFinalized dataset with {self.images_written} images.")
        print(f"Split distribution: {split_counts}")
        for split in ("train", "test"):
            n = self._split_totals[split]
            if n:
                print(f"  {split}: {n} images ({self._shard_counts[split]} shard(s))")

    def close(self):
        """Closes any open TAR files."""
        for split in ("train", "test"):
            if self._tar_files[split] is not None:
                self._tar_files[split].close()
                self._tar_files[split] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        self.close()

    @property
    def total_images_written(self) -> int:
        """Get the total number of images written so far."""
        return self.images_written
