import json
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd


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
        self.current_image_id = 0

        # Open the HDF5 file and create structure
        self.h5_file = h5py.File(file_path, "w")
        self._create_dataset_structure(max_images_estimate)

    def _create_dataset_structure(self, max_images_estimate: int):
        """
        Create the HDF5 file structure

        Args:
            max_images_estimate: Rough estimate of total images for preallocation
        """
        # Create main groups
        self.images_group = self.h5_file.create_group("images")
        self.masks_group = self.h5_file.create_group("masks")
        self.labels_group = self.h5_file.create_group("labels")
        self.metadata_group = self.h5_file.create_group("metadata")

        # Create subgroups for different mask types
        self.semantic_masks_group = self.masks_group.create_group("semantic")
        self.instance_masks_group = self.masks_group.create_group("instance")
        self.part_masks_group = self.masks_group.create_group("parts")
        self.affordance_masks_group = self.masks_group.create_group("affordance")
        self.depth_group = self.masks_group.create_group("depth")
        self.normal_group = self.masks_group.create_group("normal")

        # Pre-allocate classification labels array
        self.class_labels = self.labels_group.create_dataset(
            "class_labels",
            shape=(max_images_estimate,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=True,
        )

        # Store class names
        self.labels_group.create_dataset(
            "class_names", data=[name.encode("utf-8") for name in self.class_names]
        )

        # Pre-allocate metadata arrays
        self.image_metadata = self.metadata_group.create_dataset(
            "image_info",
            shape=(max_images_estimate,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=str),
            chunks=True,
        )

    def add_image(
        self,
        image: np.ndarray,
        semantic_mask: np.ndarray,
        class_name: str,
        asset_id: str,
        render_params: Dict,
        part_mask: Optional[np.ndarray] = None,
        instance_mask: Optional[np.ndarray] = None,
        affordance_mask: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        normal: Optional[np.ndarray] = None,
    ) -> str:
        """
        Add a single image and its annotations to the dataset

        Args:
            image: RGB image array (H, W, 3)
            semantic_mask: Semantic segmentation mask (H, W)
            class_name: Name of the object class
            asset_id: ID of the 3D asset used
            render_params: Dictionary with rendering parameters
            part_mask: Optional part segmentation mask (H, W)
            instance_mask: Optional instance segmentation mask (H, W)
            affordance_mask: Optional affordance segmentation mask (H, W)
            depth: Optional depth map (H, W)
            normal: Optional normal map (H, W, 3)

        Returns:
            image_id: String ID assigned to this image
        """
        # Generate image ID
        image_id = f"image_{self.current_image_id:06d}"

        # Store image
        self.images_group.create_dataset(
            image_id, data=image, compression="gzip", compression_opts=6, chunks=True
        )

        # Store semantic mask
        self.semantic_masks_group.create_dataset(
            image_id,
            data=semantic_mask,
            compression="gzip",
            compression_opts=9,  # Higher compression for masks
            chunks=True,
        )

        # Store optional masks and data
        if part_mask is not None:
            self.part_masks_group.create_dataset(
                image_id,
                data=part_mask,
                compression="gzip",
                compression_opts=9,
                chunks=True,
            )

        if instance_mask is not None:
            self.instance_masks_group.create_dataset(
                image_id,
                data=instance_mask,
                compression="gzip",
                compression_opts=9,
                chunks=True,
            )

        if affordance_mask is not None:
            self.affordance_masks_group.create_dataset(
                image_id,
                data=affordance_mask,
                compression="gzip",
                compression_opts=9,
                chunks=True,
            )

        if depth is not None:
            self.depth_group.create_dataset(
                image_id,
                data=depth,
                compression="gzip",
                compression_opts=7,
                chunks=True,
            )

        if normal is not None:
            self.normal_group.create_dataset(
                image_id,
                data=normal,
                compression="gzip",
                compression_opts=7,
                chunks=True,
            )

        # Store class label
        class_idx = self.class_to_idx[class_name]
        self.class_labels[self.current_image_id] = class_idx

        # Store metadata
        metadata = {
            "image_id": image_id,
            "class_name": class_name,
            "class_idx": class_idx,
            "asset_id": asset_id,
            "render_params": render_params,
            "image_shape": image.shape,
            "mask_shape": semantic_mask.shape,
            "has_part_mask": part_mask is not None,
            "has_instance_mask": instance_mask is not None,
            "has_affordance_mask": affordance_mask is not None,
            "has_depth": depth is not None,
            "has_normal": normal is not None,
        }

        self.image_metadata[self.current_image_id] = json.dumps(metadata)

        # Update counters
        self.current_image_id += 1
        self.images_written += 1

        if self.images_written % 100 == 0:
            print(f"Written {self.images_written} images...")

        return image_id

    def finalize(self):
        """
        Finalize the dataset by trimming arrays and adding final metadata
        """
        # Trim pre-allocated arrays to actual size
        self.class_labels.resize((self.current_image_id,))
        self.image_metadata.resize((self.current_image_id,))

        # Add dataset-level metadata
        dataset_info = {
            "total_images": self.current_image_id,
            "num_classes": len(self.class_names),
            "creation_date": pd.Timestamp.now().isoformat(),
            "version": "1.0",
        }

        self.metadata_group.attrs.update(dataset_info)

        # Add class statistics
        class_counts = {}
        for i in range(len(self.class_names)):
            count = np.sum(self.class_labels[: self.current_image_id] == i)
            class_counts[self.class_names[i]] = int(count)

        self.metadata_group.create_dataset(
            "class_counts", data=json.dumps(class_counts).encode("utf-8")
        )

        print(f"Finalized dataset with {self.current_image_id} images")
        print(f"Class distribution: {class_counts}")

    def close(self):
        """Close the HDF5 file."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        self.close()

    # Properties for access to internal state
    @property
    def total_images_written(self) -> int:
        """Get the total number of images written so far"""
        return self.images_written
