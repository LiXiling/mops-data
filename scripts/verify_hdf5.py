"""Verify that an HDF5 dataset file is complete and internally consistent."""

import argparse
import json
import os
import sys

import h5py
import numpy as np

MASK_GROUPS = ["semantic", "instance", "part", "affordance", "depth", "normal", "is_partnet", "bbox"]


def verify_hdf5(h5_path: str, check_loadable: bool = False) -> bool:
    file_size_mb = os.path.getsize(h5_path) / (1024 ** 2)
    print(f"\n{'='*60}")
    print(f"File: {h5_path}")
    print(f"Size on disk: {file_size_mb:.2f} MB")
    print(f"{'='*60}\n")

    ok = True

    with h5py.File(h5_path, "r") as f:
        # --- Top-level groups ---
        print(f"Top-level groups: {list(f.keys())}")

        # --- Image count from metadata vs actual keys ---
        image_keys = sorted(f["images"].keys())
        n_images_actual = len(image_keys)
        n_images_meta = int(f["metadata"].attrs.get("total_images", -1))

        print(f"\n[Images]")
        print(f"  Keys in /images:          {n_images_actual}")
        print(f"  metadata.total_images:    {n_images_meta}")
        if n_images_meta != -1 and n_images_actual != n_images_meta:
            print(f"  MISMATCH: {n_images_actual} actual vs {n_images_meta} expected")
            ok = False
        else:
            print(f"  OK")

        # --- Splits array ---
        splits = f["metadata/splits"][:]
        print(f"\n[Splits]")
        print(f"  splits array length:      {len(splits)}")
        if len(splits) != n_images_actual:
            print(f"  MISMATCH: splits length {len(splits)} != images {n_images_actual}")
            ok = False
        train_count = int(np.sum(splits))
        test_count = len(splits) - train_count
        print(f"  Train: {train_count}  |  Test: {test_count}")

        # cross-check stored split_counts
        if "split_counts" in f["metadata"]:
            stored = json.loads(bytes(f["metadata/split_counts"][()]).decode())
            if stored.get("train") != train_count or stored.get("test") != test_count:
                print(f"  MISMATCH in stored split_counts: {stored}")
                ok = False
            else:
                print(f"  split_counts match: {stored}  OK")

        # --- image_info JSON ---
        print(f"\n[image_info]")
        image_info = f["metadata/image_info"][:]
        print(f"  image_info entries:       {len(image_info)}")
        if len(image_info) != n_images_actual:
            print(f"  MISMATCH: {len(image_info)} != {n_images_actual}")
            ok = False
        else:
            print(f"  OK")

        # --- Mask groups ---
        print(f"\n[Masks]")
        masks_group = f.get("masks")
        if masks_group is None:
            print("  /masks group not present")
        else:
            available = sorted(masks_group.keys())
            print(f"  Available mask groups: {available}")
            for name in available:
                n_mask = len(masks_group[name].keys())
                status = "OK" if n_mask == n_images_actual else f"MISMATCH ({n_mask} != {n_images_actual})"
                if n_mask != n_images_actual:
                    ok = False
                print(f"  {name:<14}: {n_mask:>6}  {status}")

        # --- Class labels (optional) ---
        if "labels" in f:
            print(f"\n[Labels]")
            class_names = [n.decode("utf-8") for n in f["labels/class_names"][:]]
            print(f"  Classes ({len(class_names)}): {class_names[:10]}{'...' if len(class_names) > 10 else ''}")
            class_labels = f["labels/class_labels"][:]
            print(f"  class_labels length: {len(class_labels)}")
            if len(class_labels) != n_images_actual:
                print(f"  MISMATCH: {len(class_labels)} != {n_images_actual}")
                ok = False
            unique, counts = np.unique(class_labels, return_counts=True)
            dist = {class_names[i]: int(c) for i, c in zip(unique, counts)}
            print(f"  Class distribution: {dist}")

        # --- Spot-check a few images can actually be loaded ---
        if check_loadable and n_images_actual > 0:
            print(f"\n[Loadability check — 5 random images]")
            rng = np.random.default_rng(42)
            sample_keys = rng.choice(image_keys, size=min(5, n_images_actual), replace=False)
            for k in sample_keys:
                try:
                    arr = f["images"][k][:]
                    print(f"  {k}: shape={arr.shape}  dtype={arr.dtype}  OK")
                except Exception as e:
                    print(f"  {k}: FAILED — {e}")
                    ok = False

        # --- Summary ---
        print(f"\n{'='*60}")
        if ok:
            print(f"RESULT: All checks passed. Dataset appears complete.")
        else:
            print(f"RESULT: One or more checks FAILED. See above for details.")
        print(f"{'='*60}\n")

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify an HDF5 dataset file.")
    parser.add_argument(
        "h5_path",
        nargs="?",
        default="data/debug_mops.h5",
        help="Path to the HDF5 file (default: data/debug_mops.h5).",
    )
    parser.add_argument(
        "--check-loadable",
        action="store_true",
        help="Also spot-check that a few images can actually be decoded.",
    )
    args = parser.parse_args()

    passed = verify_hdf5(args.h5_path, check_loadable=args.check_loadable)
    sys.exit(0 if passed else 1)
