import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Link

from mops_data.asset_manager.object_annotation_registry import ObjectAnnotationRegistry


class AffordObsAugmentor:
    """Post-processes raw SAPIEN observations into affordance-annotated segmentation maps.

    Replaces rasterised RGB with ray-traced images (when a paired ``_rt`` camera
    is present) and converts the flat SAPIEN segmentation tensor into four
    per-pixel maps: instance, semantic class, PartNet-origin flag, and
    per-part multi-hot affordance labels.
    """

    def __init__(self, registry):
        self.registry: ObjectAnnotationRegistry = registry

    def augment(self, env: BaseEnv, obs):
        """Augment all cameras in *obs* in-place and return the modified dict.

        For each non-RT camera: swaps RGB with the ray-traced counterpart (if
        present) and replaces raw segmentation with the four derived maps from
        :meth:`augment_segmentations`.

        Args:
            env: Live ManiSkill environment, needed for ``segmentation_id_map``.
            obs: Raw observation dict from ``gym.reset()`` / ``gym.step()``.

        Returns:
            The same *obs* dict with ``sensor_data`` entries updated in-place.
        """
        for cam_name in obs["sensor_data"]:
            # Skip RT Augmentation
            if "_rt" in cam_name:
                continue

            # Replace RGB with raytrayced Image
            if (
                f"{cam_name}_rt" in obs["sensor_data"]
                and "rgb" in obs["sensor_data"][cam_name]
            ):
                obs["sensor_data"][cam_name]["rgb"] = obs["sensor_data"][
                    f"{cam_name}_rt"
                ]["rgb"]

            # Augment Segmentation Map
            camera = obs["sensor_data"][cam_name]

            if "segmentation" in camera:
                augmentations = self.augment_segmentations(env, camera["segmentation"])
                obs["sensor_data"][cam_name].update(augmentations)

        return obs

    def augment_segmentations(self, env: BaseEnv, camera_segmentations: torch.Tensor):
        """Convert a raw SAPIEN segmentation tensor to four annotation maps.

        Args:
            env: Live environment; provides ``segmentation_id_map``.
            camera_segmentations: Integer tensor ``(B, H, W, 1)`` with per-pixel
                SAPIEN object IDs.

        Returns:
            Dict with keys:
                ``instance_segmentation``   — link IDs collapsed to root object IDs.
                ``class_segmentation``      — semantic class IDs from AnnotationHandler.
                ``is_partnet``              — binary mask, 1 for PartNet-Mobility pixels.
                ``affordance_segmentation`` — multi-hot tensor ``(H, W, num_affordances)``.
        """
        img_shape = camera_segmentations.shape
        flat_segm = camera_segmentations.flatten()

        num_affords = self.registry.get_num_affords()

        # Instance Segmentation: Convert Linked Parts to Root ID
        instance_segm = flat_segm.clone()
        class_segm = flat_segm.clone()
        is_partnet = torch.zeros_like(camera_segmentations)
        afford_segm = torch.zeros_like(camera_segmentations)

        # Extend so that last dimension is max_afford_id
        afford_segm = afford_segm.expand(-1, -1, -1, num_affords).clone()

        for obj_tensor_id in flat_segm.unique():
            obj_id = obj_tensor_id.item()

            # Skip Background
            if obj_id == 0:
                continue

            obj = env.segmentation_id_map[obj_id]

            # For Link Parts, replace with Root ID (Instance Segmentation)
            if isinstance(obj, Link):
                root_id = obj.articulation.root._objs[0].entity.per_scene_id
                instance_segm[flat_segm == obj_id] = root_id

            # Binary mask for PartNet objects
            if self.registry.is_partnet(obj_id):
                is_partnet[camera_segmentations == obj_id] = 1

            class_id = self.registry.get_class_id(obj_id)
            class_segm[flat_segm == obj_id] = class_id

            affordances = torch.tensor(
                self.registry.get_affordance_list(obj_id),
                dtype=camera_segmentations.dtype,
                device=camera_segmentations.device,
            )

            mask = (camera_segmentations == obj_id).squeeze(-1)
            afford_segm[mask] = affordances.clone()

        res_dict = {}
        res_dict["instance_segmentation"] = instance_segm.reshape(img_shape)
        res_dict["class_segmentation"] = class_segm.reshape(img_shape)
        res_dict["is_partnet"] = is_partnet.reshape(img_shape)
        res_dict["affordance_segmentation"] = afford_segm

        return res_dict
