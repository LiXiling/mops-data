import json
import os

import numpy as np
import sapien
import torch
from mani_skill.utils.geometry import rotation_conversions

from mops_data.asset_manager import anno_handler as mops_ah
from mops_data.asset_manager.object_annotation_registry import ObjectAnnotationRegistry


def parse_meta(model_dir: str) -> str:
    """Read ``meta.json`` and return the object's class name (``model_cat``)."""
    metadata_path = os.path.join(model_dir, "meta.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata["model_cat"]


def parse_semantics(model_dir: str) -> dict:
    """Parse ``semantics.txt`` and return a ``link_name → semantic_label`` dict."""
    semantics_path = os.path.join(model_dir, "semantics.txt")

    link_name_to_semantics = {}
    with open(semantics_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            link_name, joint_typ, semantics = line.strip().split(" ")
            link_name_to_semantics[link_name] = semantics
    return link_name_to_semantics


class PartNetMobilityLoader:
    """Loads PartNet-Mobility URDF objects into a ManiSkill scene.

    Handles URDF parsing, per-object scale sampling from annotation ranges, and
    wires each loaded articulation into the ``ObjectAnnotationRegistry`` so that
    segmentation IDs resolve to class and affordance labels.
    """

    def __init__(self, env, dir_path: str, registry: ObjectAnnotationRegistry):
        self.env = env
        self.dir_path = dir_path
        self.registry = registry

        self.anno_handler: mops_ah.AnnotationHandler = mops_ah.load_annotations()

        anno_df = self.anno_handler.partnet_mobility_df
        self.partnet_mob_annotations = anno_df
        self.partnet_small_mob_annotations = anno_df[~anno_df["is_large_object"]]

        self.segm_id_class_map = {}
        self.segm_id_afford_map = {}
        self.name_counter = 0

    def _load_urdf_builder(
        self, model_dir, pose, euler=None, scale=1.0, fix_root=False
    ):
        urdf_path = os.path.join(model_dir, "mobility.urdf")
        loader = self.env.scene.create_urdf_loader()
        loader.scale = scale
        loader.fix_root_link = fix_root

        if euler is None:
            euler = [0, 0, 0]

        quat = rotation_conversions.matrix_to_quaternion(
            rotation_conversions.euler_angles_to_matrix(torch.Tensor(euler), "XYZ")
        )

        maniskill_articulation_builders = loader.parse(urdf_path)[
            "articulation_builders"
        ]
        builder = maniskill_articulation_builders[0]
        builder.initial_pose = sapien.Pose(pose, quat)
        return builder

    def _get_obj_class_name(self, mob_id):
        obj_class_name = self.partnet_mob_annotations[
            self.partnet_mob_annotations["dir_name"] == mob_id
        ]["model_cat"]

        return obj_class_name.values[0]

    def _get_part_affordances(self, mob_id):
        linkname_to_affordances = self.partnet_mob_annotations[
            self.partnet_mob_annotations["dir_name"] == mob_id
        ][["link_name", "affordances"]]
        linkname_to_affordances = linkname_to_affordances.set_index("link_name")
        linkname_to_affordances = linkname_to_affordances["affordances"].to_dict()
        return linkname_to_affordances

    def _get_scale(self, mob_id):
        scale_range = self.partnet_mob_annotations[
            self.partnet_mob_annotations["dir_name"] == mob_id
        ]["scaling_factor_range"].values[0]

        scale = np.random.uniform(scale_range[0], scale_range[1])
        return scale

    def load(self, mob_id, pose, euler=None, scale=None, no_grav=False):
        """Load a PartNet-Mobility object into the simulation scene.

        Parses the URDF, samples a scale if none is provided, builds the
        SAPIEN articulation, and registers it with the annotation registry.

        Args:
            mob_id: PartNet directory name (int or str, e.g. ``"10235"``).
            pose: 3-element world position ``[x, y, z]``.
            euler: Optional ``[rx, ry, rz]`` rotation in radians.  Defaults to
                no rotation.
            scale: Object scale factor.  If ``None``, sampled uniformly from the
                per-object ``scaling_factor_range`` in the annotations.
            no_grav: If ``True``, the root link is fixed (no gravity effect).
        """
        # Load All Auxiliary Information
        model_dir = os.path.join(self.dir_path, f"{mob_id}")
        class_name = self._get_obj_class_name(int(mob_id))
        link_name_to_affords: dict = self._get_part_affordances(int(mob_id))

        if scale is None:
            scale = self._get_scale(int(mob_id))

        urdf_builder = self._load_urdf_builder(model_dir, pose, euler, scale, no_grav)

        # Create the Articulation Object
        try:
            art_obj = urdf_builder.build(
                name=f"obj_{self.name_counter}_{class_name}_{mob_id}"
            )
            self.name_counter += 1
        except Exception as e:
            print(f"Error loading object {mob_id}: {e}")
            return

        # Fill the Segmentation Maps after creating the Articulation Object
        self.registry.add_partnet_object(art_obj, class_name, link_name_to_affords)

    def load_by_index(self, idx: int, pose, scale=None, no_grav=False):
        """Load the *idx*-th object in the sorted PartNet-Mobility catalogue."""
        mob_id = sorted(self.partnet_mob_annotations["dir_name"].unique())[idx]
        self.load(mob_id, pose, scale, no_grav)

    def load_random_object(
        self, rng: np.random.Generator, pose, scale=None, no_grav=False
    ):
        """Load a uniformly random non-large PartNet-Mobility object."""
        mob_id = rng.choice(self.partnet_small_mob_annotations["dir_name"].unique())
        self.load(mob_id, pose, scale, no_grav)
