"""HF ImageFolder dataset writer.

Writes image datasets in the Hugging Face ImageFolder convention:
images stored as files on disk with a ``metadata.jsonl`` sidecar per split.

Load with::

    datasets.load_dataset("imagefolder", data_dir="dataset_dir")

Directory layout::

    dataset_dir/
    ├── train/
    │   ├── metadata.jsonl
    │   ├── images/
    │   │   └── 000000.png
    │   ├── semantic/         # masks as 16-bit grayscale PNG
    │   │   └── 000000.png
    │   ├── depth/            # float arrays as .npy
    │   │   └── 000000.npy
    │   └── …
    ├── test/
    │   └── …
    └── dataset_info.json

Mask columns (semantic, instance, part, is_partnet) are saved as lossless
grayscale PNG images and referenced via ``*_file_name`` in metadata so that
``datasets`` auto-loads them as ``Image()`` features.

Float / multi-channel arrays (depth, normal, affordance) are saved as ``.npy``
files and referenced as string paths in metadata.
"""

import datetime
import json
import queue
import threading
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from PIL import Image


class ImageFolderWriter:
    """Write image-based datasets in Hugging Face ImageFolder format.

    Drop-in replacement for the old ``ParquetWriter`` — same ``add_image()``
    interface and context-manager pattern.
    """

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

    # Saved as grayscale PNG; HF auto-loads via *_file_name → Image().
    MASK_COLUMNS = {"semantic", "instance", "part", "is_partnet"}
    # Saved as .npy; referenced as string paths in metadata.
    ARRAY_COLUMNS = {"affordance", "depth", "normal"}
    # JSON-serialised strings in metadata.
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
            output_dir: Root directory for the ImageFolder dataset.
            max_images_estimate: Unused (API compat with HDF5Writer).
            class_names: Optional list of class names; enables class columns.
            shard_size: Unused (API compat with ParquetWriter).
            row_group_size: Unused (API compat with ParquetWriter).
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

        # Create per-split directory trees.
        for split in ("train", "test"):
            split_dir = self.output_dir / split
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            for col in self.MASK_COLUMNS:
                (split_dir / col).mkdir(exist_ok=True)
            for col in self.ARRAY_COLUMNS:
                (split_dir / col).mkdir(exist_ok=True)

        self._split_totals: dict[str, int] = {"train": 0, "test": 0}

        # Metadata JSONL handles (one per split, opened lazily).
        self._metadata_files: dict[str, Any] = {"train": None, "test": None}

        # Background write thread (bounds memory via maxsize).
        self._write_queue: queue.Queue = queue.Queue(maxsize=32)
        self._write_thread = threading.Thread(
            target=self._write_worker, daemon=True, name="imagefolder-writer"
        )
        self._write_thread.start()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_metadata_file(self, split: str):
        if self._metadata_files[split] is None:
            path = self.output_dir / split / "metadata.jsonl"
            self._metadata_files[split] = open(path, "w")  # noqa: SIM115
        return self._metadata_files[split]

    @staticmethod
    def _save_mask_png(arr: np.ndarray, path: Path):
        """Save an integer mask as a lossless grayscale PNG."""
        mask = np.squeeze(arr)
        if mask.max() <= 255:
            Image.fromarray(mask.astype(np.uint8), mode="L").save(
                str(path), format="PNG"
            )
        else:
            Image.fromarray(mask.astype(np.uint16), mode="I;16").save(
                str(path), format="PNG"
            )

    # ------------------------------------------------------------------
    # Core write logic (runs on background thread)
    # ------------------------------------------------------------------

    def _write_one(self, image_idx: int, kwargs: dict):
        image_id = f"image_{image_idx:06d}"
        split = kwargs["render_params"]["split"]
        split_dir = self.output_dir / split

        # --- RGB image ---------------------------------------------------
        Image.fromarray(kwargs["image"]).save(
            str(split_dir / "images" / f"{image_id}.png"), format="PNG"
        )

        # --- Metadata row ------------------------------------------------
        asset_id = kwargs.get("asset_id", "")
        if isinstance(asset_id, (list, np.ndarray)):
            asset_id = json.dumps(
                asset_id.tolist() if isinstance(asset_id, np.ndarray) else asset_id
            )

        row: dict[str, Any] = {
            "file_name": f"images/{image_id}.png",
            "image_id": image_id,
            "asset_id": asset_id,
            "render_params": json.dumps(kwargs["render_params"]),
        }

        if self.has_classes:
            row["class_name"] = kwargs["class_name"]
            row["class_idx"] = self.class_to_idx[kwargs["class_name"]]

        # --- Mask columns → grayscale PNG --------------------------------
        for col in self.MASK_COLUMNS:
            data = kwargs.get(col)
            if data is None:
                continue
            mask_path = split_dir / col / f"{image_id}.png"
            self._save_mask_png(data, mask_path)
            row[f"{col}_file_name"] = f"{col}/{image_id}.png"

        # --- Array columns → .npy ----------------------------------------
        for col in self.ARRAY_COLUMNS:
            data = kwargs.get(col)
            if data is None:
                continue
            npy_path = split_dir / col / f"{image_id}.npy"
            np.save(str(npy_path), data)
            row[f"{col}_path"] = f"{col}/{image_id}.npy"

        # --- JSON columns ------------------------------------------------
        for col in self.JSON_COLUMNS:
            data = kwargs.get(col)
            if data is None:
                continue
            row[col] = json.dumps(
                data.tolist() if isinstance(data, np.ndarray) else data
            )

        # --- Append to metadata.jsonl ------------------------------------
        f = self._get_metadata_file(split)
        f.write(json.dumps(row) + "\n")

        self._split_totals[split] += 1
        self._images_flushed += 1
        if self._images_flushed % 100 == 0:
            print(f"Written {self._images_flushed} images...")

    # ------------------------------------------------------------------
    # Background write thread
    # ------------------------------------------------------------------

    def _write_worker(self):
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
    # Public API (mirrors ParquetWriter / HDF5Writer)
    # ------------------------------------------------------------------

    def add_image(self, **kwargs: Any) -> str:
        """Enqueue a single data entry for writing. Returns immediately.

        Accepts the same keyword arguments as ``ParquetWriter.add_image``:
        ``image``, ``asset_id``, ``render_params``, ``class_name``, and any
        DATA_SPEC keys (``semantic``, ``depth``, etc.).
        """
        image_idx = self.images_written
        self.images_written += 1
        self._write_queue.put((image_idx, kwargs))
        return f"image_{image_idx:06d}"

    def finalize(self):
        """Flush remaining data and write dataset metadata."""
        # Drain write queue.
        self._write_queue.put(None)
        self._write_thread.join()

        if self.images_written == 0:
            print("Warning: No images were written.")
            return

        # Close metadata files.
        for split in ("train", "test"):
            if self._metadata_files[split] is not None:
                self._metadata_files[split].close()

        # Write dataset_info.json.
        info: dict[str, Any] = {
            "total_images": self.images_written,
            "creation_date": datetime.datetime.now().isoformat(),
            "version": "3.0",
            "format": "imagefolder",
            "splits": {
                split: {"num_images": self._split_totals[split]}
                for split in ("train", "test")
                if self._split_totals[split] > 0
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
                print(f"  {split}: {n} images")

    def close(self):
        """No-op (kept for API compat)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        self.close()

    @property
    def total_images_written(self) -> int:
        return self.images_written
