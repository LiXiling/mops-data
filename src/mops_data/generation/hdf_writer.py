import datetime
import json
import queue
import threading
from typing import Any, List, Optional

import h5py
import numpy as np


class HDF5Writer:
    """
    A flexible class to write image-based datasets to HDF5 files.

    It supports multiple data streams (e.g., images, masks, metadata) and can be
    configured for datasets with or without top-level class labels. It is
    designed to be used as a context manager.
    """

    # Specification for all supported data types.
    # The key is the group name in HDF5 and the kwarg name in `add_image`.
    # The value is the GZIP compression level (4 = good ratio, ~3× faster than 9).
    DATA_SPEC = {
        # Standard masks and maps
        "semantic": 4,
        "instance": 4,
        "part": 4,
        "affordance": 4,
        "depth": 4,
        "normal": 4,
        "is_partnet": 4,
        # Data for multi-object scenes
        "bbox": 4,  # Bounding Box with format [x, y, w, h, class_id]
    }

    def __init__(
        self,
        file_path: str,
        max_images_estimate: int = 10000,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initializes HDF5Writer and creates the dataset file structure.

        Args:
            file_path: Path where the HDF5 file will be created.
            max_images_estimate: An estimate of total images for pre-allocation.
            class_names: Optional list of class names. If provided, enables
                         per-image class tracking.
        """
        self.h5_file = h5py.File(file_path, "w")
        self.images_written = 0  # enqueued count (main thread only)
        self._images_flushed = 0  # written-to-disk count (write thread only)
        self.has_classes = class_names is not None

        # Create main groups
        self.images_group = self.h5_file.create_group("images")
        self.masks_group = self.h5_file.create_group("masks")
        self.data_groups = {
            name: self.masks_group.create_group(name) for name in self.DATA_SPEC
        }
        self.metadata_group = self.h5_file.create_group("metadata")

        # Setup for class-based datasets
        if self.has_classes:
            self.class_names = class_names
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            self.labels_group = self.h5_file.create_group("labels")
            self._create_metadata_dataset(
                self.labels_group,
                "class_names",
                [n.encode("utf-8") for n in self.class_names],
            )

        # Pre-allocate resizable datasets for efficiency
        self.preallocated_datasets = {
            "splits": self._create_resizable_dataset(
                "splits", max_images_estimate, np.bool_
            ),
            "image_info": self._create_resizable_dataset(
                "image_info", max_images_estimate, h5py.special_dtype(vlen=str)
            ),
        }
        if self.has_classes:
            self.preallocated_datasets["class_labels"] = self._create_resizable_dataset(
                "class_labels", max_images_estimate, np.int32
            )

        # Background write thread: decouples gzip compression from rendering.
        # maxsize creates backpressure so the main loop doesn't race too far ahead.
        self._write_queue: queue.Queue = queue.Queue(maxsize=200)
        self._write_thread = threading.Thread(
            target=self._write_worker, daemon=True, name="hdf5-writer"
        )
        self._write_thread.start()

    def _create_resizable_dataset(self, name, shape_est, dtype):
        """Helper to create a 1D resizable dataset in the metadata group."""
        group = self.labels_group if name == "class_labels" else self.metadata_group
        return group.create_dataset(
            name, shape=(shape_est,), maxshape=(None,), dtype=dtype, chunks=True
        )

    def _create_compressed_dataset(self, group, name, data, comp_level):
        """Helper to create a dataset with gzip compression."""
        group.create_dataset(
            name,
            data=data,
            compression="gzip",
            compression_opts=comp_level,
            chunks=True,
        )

    def _create_metadata_dataset(self, group, name, data):
        """Helper to create a simple dataset, often for JSON metadata."""
        if isinstance(data, dict):
            data = json.dumps(data).encode("utf-8")
        group.create_dataset(name, data=data)

    def _write_one(self, image_idx: int, kwargs: dict):
        """Synchronous write executed on the write thread."""
        image_id = f"image_{image_idx:06d}"

        self._create_compressed_dataset(
            self.images_group, image_id, kwargs["image"], comp_level=4
        )

        metadata = {
            "image_id": image_id,
            "asset_id": kwargs["asset_id"],
            "render_params": kwargs["render_params"],
            "image_shape": kwargs["image"].shape,
        }
        if self.has_classes:
            class_name = kwargs["class_name"]
            metadata.update(
                {
                    "class_name": class_name,
                    "class_idx": self.class_to_idx[class_name],
                }
            )

        for name, comp_level in self.DATA_SPEC.items():
            data = kwargs.get(name)
            metadata[f"has_{name}"] = data is not None
            if data is not None:
                self._create_compressed_dataset(
                    self.data_groups[name], image_id, data, comp_level
                )

        self.preallocated_datasets["image_info"][image_idx] = json.dumps(metadata)
        self.preallocated_datasets["splits"][image_idx] = (
            kwargs["render_params"]["split"] == "train"
        )
        if self.has_classes:
            self.preallocated_datasets["class_labels"][image_idx] = metadata[
                "class_idx"
            ]

        self._images_flushed += 1
        if self._images_flushed % 100 == 0:
            print(f"Written {self._images_flushed} images...")

    def _write_worker(self):
        """Background thread: drains the write queue and calls _write_one."""
        while True:
            item = self._write_queue.get()
            if item is None:  # poison pill — shut down
                self._write_queue.task_done()
                break
            image_idx, kwargs = item
            try:
                self._write_one(image_idx, kwargs)
            except Exception as e:
                print(f"HDF5 write error for image_{image_idx:06d}: {e}")
            finally:
                self._write_queue.task_done()

    def add_image(self, **kwargs: Any) -> str:
        """
        Enqueues a single data entry for writing.  Returns immediately;
        the actual h5py writes happen on the background write thread.

        Args (passed as keyword arguments):
            image (np.ndarray): The main RGB image array.
            asset_id (str): The ID of the primary asset or scene.
            render_params (dict): Rendering parameters, must contain 'split'.
            class_name (str): The object class name (required for class-based datasets).
            **other_data: Other data arrays, keys must match DATA_SPEC
                          (e.g., semantic=..., depth=...).
        """
        image_idx = self.images_written
        self.images_written += 1
        self._write_queue.put((image_idx, kwargs))
        return f"image_{image_idx:06d}"

    def finalize(self):
        """Finalizes the file by trimming datasets and writing summary metadata."""
        # Drain the write queue before touching any datasets
        self._write_queue.put(None)  # poison pill
        self._write_thread.join()

        if self.images_written == 0:
            print("Warning: No images were written.")
            return

        for ds in self.preallocated_datasets.values():
            ds.resize((self.images_written,))

        self.metadata_group.attrs.update(
            {
                "total_images": self.images_written,
                "creation_date": datetime.datetime.now().isoformat(),
                "version": "2.0",
            }
        )

        # Calculate and store split statistics
        splits = self.preallocated_datasets["splits"][:]
        train_count = int(np.sum(splits))
        split_counts = {"train": train_count, "test": len(splits) - train_count}
        self._create_metadata_dataset(self.metadata_group, "split_counts", split_counts)

        # Calculate class statistics only if applicable
        if self.has_classes:
            self.metadata_group.attrs["num_classes"] = len(self.class_names)
            class_labels = self.preallocated_datasets["class_labels"][:]
            unique, counts = np.unique(class_labels, return_counts=True)
            class_counts = {self.class_names[i]: int(c) for i, c in zip(unique, counts)}
            self._create_metadata_dataset(
                self.metadata_group, "class_counts", class_counts
            )
            print(f"Class distribution: {class_counts}")

        print(f"\nFinalized dataset with {self.images_written} images.")
        print(f"Split distribution: {split_counts}")

    def close(self):
        """Closes the HDF5 file."""
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        self.close()

    @property
    def total_images_written(self) -> int:
        """Get the total number of images written so far"""
        return self.images_written
