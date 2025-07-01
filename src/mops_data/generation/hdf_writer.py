import datetime
import json
from typing import Any, Dict, List, Optional

import h5py
import numpy as np


class HDF5Writer:
    def __init__(
        self, file_path: str, class_names: List[str], max_images_estimate: int = 10000
    ):
        """
        Initialize HDF5Writer and create the dataset file

        Args:
            file_path: Path where the HDF5 file will be created
            class_names: List of class names
            max_images_estimate: Rough estimate of total images for preallocation
        """
        self.file_path = file_path
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.images_written = 0

        # Open the HDF5 file and create structure
        self.h5_file = h5py.File(file_path, "w")

        # Create main groups
        self.images_group = self.h5_file.create_group("images")
        self.masks_group = self.h5_file.create_group("masks")
        self.labels_group = self.h5_file.create_group("labels")
        self.metadata_group = self.h5_file.create_group("metadata")

        # Create subgroups for different mask types using a helper
        self.optional_groups = {
            "semantic": self.masks_group.create_group("semantic"),
            "instance": self.masks_group.create_group("instance"),
            "parts": self.masks_group.create_group("parts"),
            "affordance": self.masks_group.create_group("affordance"),
            "depth": self.masks_group.create_group("depth"),
            "normal": self.masks_group.create_group("normal"),
        }

        # Pre-allocate resizable datasets
        self.preallocated_datasets = {
            "class_labels": self.labels_group.create_dataset(
                "class_labels",
                shape=(max_images_estimate,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=True,
            ),
            "splits": self.labels_group.create_dataset(
                "splits",
                shape=(max_images_estimate,),
                maxshape=(None,),
                dtype=np.bool_,
                chunks=True,
            ),
            "image_info": self.metadata_group.create_dataset(
                "image_info",
                shape=(max_images_estimate,),
                maxshape=(None,),
                dtype=h5py.special_dtype(vlen=str),
                chunks=True,
            ),
        }

        # Store class names
        self.labels_group.create_dataset(
            "class_names", data=[name.encode("utf-8") for name in self.class_names]
        )

    def _add_optional_dataset(
        self, group: h5py.Group, name: str, data: np.ndarray, **kwargs
    ):
        """Helper to create a dataset with gzip compression."""
        group.create_dataset(name, data=data, compression="gzip", **kwargs)

    def add_image(self, **kwargs: Any) -> str:
        """
        Add a single image and its annotations to the dataset.

        Args (passed as keyword arguments):
            image: RGB image array (H, W, 3)
            semantic_mask: Semantic segmentation mask (H, W)
            class_name: Name of the object class
            asset_id: ID of the 3D asset used
            render_params: Dictionary with rendering parameters (must contain 'split')
            part_mask: Optional part segmentation mask (H, W)
            instance_mask: Optional instance segmentation mask (H, W)
            affordance_mask: Optional affordance segmentation mask (H, W)
            depth: Optional depth map (H, W)
            normal: Optional normal map (H, W, 3)

        Returns:
            image_id: String ID assigned to this image
        """
        image_idx = self.images_written
        image_id = f"image_{image_idx:06d}"

        # Store image
        self._add_optional_dataset(
            self.images_group,
            image_id,
            kwargs["image"],
            compression_opts=6,
            chunks=True,
        )

        # Store masks and other optional data
        optional_data_map = {
            "semantic": ("semantic_mask", 9),
            "instance": ("instance_mask", 9),
            "parts": ("part_mask", 9),
            "affordance": ("affordance_mask", 9),
            "depth": ("depth", 7),
            "normal": ("normal", 7),
        }

        metadata = {
            "image_id": image_id,
            "class_name": kwargs["class_name"],
            "class_idx": self.class_to_idx[kwargs["class_name"]],
            "asset_id": kwargs["asset_id"],
            "render_params": kwargs["render_params"],
            "image_shape": kwargs["image"].shape,
        }

        for group_name, (kwarg_key, comp_level) in optional_data_map.items():
            data = kwargs.get(kwarg_key)
            metadata[f"has_{kwarg_key}"] = data is not None
            if data is not None:
                if "mask" in kwarg_key:
                    metadata["mask_shape"] = data.shape
                self._add_optional_dataset(
                    self.optional_groups[group_name],
                    image_id,
                    data,
                    compression_opts=comp_level,
                    chunks=True,
                )

        # Store labels and metadata
        self.preallocated_datasets["class_labels"][image_idx] = metadata["class_idx"]
        self.preallocated_datasets["splits"][image_idx] = (
            kwargs["render_params"]["split"] == "train"
        )
        self.preallocated_datasets["image_info"][image_idx] = json.dumps(metadata)

        # Update counter
        self.images_written += 1
        if self.images_written % 100 == 0:
            print(f"Written {self.images_written} images...")

        return image_id

    def finalize(self):
        """
        Finalize the dataset by trimming arrays and adding final metadata
        """
        # Trim pre-allocated arrays to actual size
        for ds in self.preallocated_datasets.values():
            ds.resize((self.images_written,))

        # Add dataset-level metadata
        dataset_info = {
            "total_images": self.images_written,
            "num_classes": len(self.class_names),
            "creation_date": datetime.datetime.now().isoformat(),
            "version": "1.0",
        }
        self.metadata_group.attrs.update(dataset_info)

        # Calculate and store class statistics
        class_labels_arr = self.preallocated_datasets["class_labels"][:]
        unique_indices, counts = np.unique(class_labels_arr, return_counts=True)
        class_counts = {
            self.class_names[i]: int(c) for i, c in zip(unique_indices, counts)
        }
        self.metadata_group.create_dataset(
            "class_counts", data=json.dumps(class_counts).encode("utf-8")
        )

        # Calculate and store split statistics
        splits_array = self.preallocated_datasets["splits"][:]
        train_count = int(np.sum(splits_array))
        split_counts = {"train": train_count, "test": len(splits_array) - train_count}
        self.metadata_group.create_dataset(
            "split_counts", data=json.dumps(split_counts).encode("utf-8")
        )

        print(f"Finalized dataset with {self.images_written} images")
        print(f"Class distribution: {class_counts}")
        print(f"Split distribution: {split_counts}")

    def close(self):
        """Close the HDF5 file."""
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
