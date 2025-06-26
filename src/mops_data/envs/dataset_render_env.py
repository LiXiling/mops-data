from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

from mops_data.asset_manager.object_annotation_registry import ObjectAnnotationRegistry
from mops_data.asset_manager.partnet_mobility_loader import PartNetMobilityLoader
from mops_data.render.afford_obs_augmentor import AffordObsAugmentor
from mops_data.render.shader_config import RT_RGB_ONLY_CONFIG


@register_env("DatasetRenderEnv-v1", max_episode_steps=1)
class DatasetRenderEnv(BaseEnv):
    """
    Specialized rendering environment for dataset generation.

    This environment is designed to:
    1. Load a specific PartNet Mobility object
    2. Configure camera, lighting, and background based on parameters
    3. Render high-quality images with segmentation masks
    4. Be efficient for batch dataset generation
    """

    SUPPORTED_ROBOTS = ["none"]  # No robot needed for dataset generation

    def __init__(
        self,
        *args,
        # Asset specification - use DataFrame column names
        asset_index: Optional[int] = None,
        asset_id: Optional[str] = None,  # This will be 'model_id' from DataFrame
        dir_name: Optional[str] = None,  # This will be 'dir_name' from DataFrame
        model_cat: Optional[str] = None,  # This will be 'model_cat' from DataFrame
        # Rendering configuration
        image_size: Tuple[int, int] = (512, 512),
        camera_distance: float = 1.5,
        camera_elevation: float = 15.0,
        camera_azimuth: float = 0.0,
        # Lighting configuration
        lighting_type: str = "studio",
        lighting_intensity: float = 1.0,
        # Background configuration
        background_type: str = "white",
        background_texture: Optional[str] = None,
        # Object configuration
        object_scale: float = 0.8,
        object_position: Optional[np.ndarray] = None,
        # Observation mode
        **kwargs
    ):
        # Store rendering parameters
        self.asset_index = asset_index
        self.asset_id = asset_id  # model_id
        self.dir_name = dir_name
        self.model_cat = model_cat
        self.image_size = image_size
        self.camera_distance = camera_distance
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.lighting_type = lighting_type
        self.lighting_intensity = lighting_intensity
        self.background_type = background_type
        self.background_texture = background_texture
        self.object_scale = object_scale
        self.object_position = (
            object_position
            if object_position is not None
            else np.array([0.0, 0.0, 0.0])
        )

        # Initialize asset management
        self.object_annotation_registry = ObjectAnnotationRegistry()
        self.partnet_mobility_loader = PartNetMobilityLoader(
            env=self,
            dir_path="data/partnet_mobility",
            registry=self.object_annotation_registry,
        )
        self.afford_augmentor = AffordObsAugmentor(
            registry=self.object_annotation_registry
        )

        # Store loaded object for reference
        self.loaded_object = None

        # This calls self._load_scene() and creates all loaded objects
        super().__init__(*args, robot_uids="none", **kwargs)

    def _load_scene(self, options):
        """Load scene with the specified object"""

        # Load the specified object
        if self.dir_name is not None:
            # Use dir_name if available (most direct)
            self.loaded_object = self.partnet_mobility_loader.load(
                self.dir_name,
                self.object_position,
                no_grav=True,
                scale=self.object_scale,
            )
        elif self.asset_id is not None:
            # Use model_id
            self.loaded_object = self.partnet_mobility_loader.load(
                self.asset_id,
                self.object_position,
                no_grav=True,
                scale=self.object_scale,
            )
        elif self.asset_index is not None:
            # Use index-based loading
            self.loaded_object = self.partnet_mobility_loader.load_by_index(
                self.asset_index,
                self.object_position,
                no_grav=True,
                scale=self.object_scale,
            )
        else:
            raise ValueError(
                "Either asset_id, dir_name, or asset_index must be specified"
            )

        # Configure lighting
        self._setup_lighting()

        # Configure background
        self._setup_background()

        # Register objects for segmentation
        self.object_annotation_registry.register_missing_objects(self)

    def _setup_lighting(self):
        """Configure lighting based on lighting_type and intensity"""
        # Remove existing lights if needed
        # Note: Be careful about removing default scene lighting

        if self.lighting_type == "studio":
            # Studio lighting setup - multiple directional lights
            # Key light
            self.scene.add_directional_light(
                direction=[0.5, -1, -0.5],
                color=np.asarray([1, 1, 1]) * self.lighting_intensity,
            )
            # Fill light
            self.scene.add_directional_light(
                direction=[-0.5, -1, -0.5],
                color=np.asarray([0.6, 0.6, 0.8]) * self.lighting_intensity * 0.5,
            )
            # Rim light
            self.scene.add_directional_light(
                direction=[0, 1, -1],
                color=np.asarray([1, 0.9, 0.8]) * self.lighting_intensity * 0.3,
            )

        elif self.lighting_type == "natural":
            # Natural lighting - single strong directional light (sun)
            self.scene.add_directional_light(
                direction=[0.3, -1, -0.7],
                color=np.asarray([1, 0.95, 0.9]) * self.lighting_intensity,
            )
            # Ambient light
            self.scene.set_ambient_light(
                np.asarray([0.2, 0.2, 0.3]) * self.lighting_intensity * 0.3
            )

        elif self.lighting_type == "dramatic":
            # Dramatic lighting - strong single light source
            self.scene.add_directional_light(
                direction=[1, -1, -0.2],
                color=np.asarray([1, 0.9, 0.7]) * self.lighting_intensity,
            )
            # Minimal ambient
            self.scene.set_ambient_light(
                np.asarray([0.1, 0.1, 0.15]) * self.lighting_intensity * 0.2
            )

    def _setup_background(self):
        """Configure background based on background_type"""
        if self.background_type == "white":
            # Pure white background - often handled by renderer
            pass

        elif self.background_type == "wood_floor":
            # Wood floor texture
            if self.background_texture:
                # Load and apply wood texture
                pass  # Implement texture loading based on your asset system

        elif self.background_type == "concrete_floor":
            # Concrete floor texture
            if self.background_texture:
                # Load and apply concrete texture
                pass  # Implement texture loading based on your asset system

    def _get_obs_with_sensor_data(self, info, apply_texture_transforms=True):
        """Get observations with affordance augmentation"""
        obs = super()._get_obs_with_sensor_data(info, apply_texture_transforms)

        # Apply affordance augmentation
        augmented_obs = self.afford_augmentor.augment(self, obs)

        return augmented_obs

    def extract_render_data(self, obs: Dict) -> Dict[str, np.ndarray]:
        """
        Extract render data from observations in the format expected by HDF5Writer.

        Args:
            obs: Observation dictionary from environment step/reset

        Returns:
            Dictionary containing:
            - 'image': RGB image (H, W, 3)
            - 'semantic_mask': Semantic segmentation mask (H, W)
            - 'part_mask': Part segmentation mask (H, W) if available
            - 'affordance_mask': Affordance mask (H, W) if available
            - 'depth': Depth map (H, W) if available
            - 'normal': Normal map (H, W, 3) if available
        """
        result = {}

        # Extract camera observations - using standard camera name
        camera_obs = None
        if "sensor_data" in obs:
            # Look for common camera names
            for cam_name in ["base_camera", "fetch_head", "camera"]:
                if cam_name in obs["sensor_data"]:
                    camera_obs = obs["sensor_data"][cam_name]
                    break
        elif "base_camera" in obs:
            # Direct camera access
            camera_obs = obs["base_camera"]

        if camera_obs is None:
            raise ValueError("No camera observations found in obs")

        # Extract RGB image
        if "rgb" in camera_obs:
            rgb = camera_obs["rgb"]
            if hasattr(rgb, "cpu"):  # torch tensor
                rgb = rgb.cpu().numpy()
            if len(rgb.shape) == 4:  # batch dimension
                rgb = rgb[0]
            if rgb.dtype != np.uint8:
                rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
            result["image"] = rgb

        # Extract depth if available
        if "depth" in camera_obs:
            depth = camera_obs["depth"]
            if hasattr(depth, "cpu"):
                depth = depth.cpu().numpy()
            if len(depth.shape) == 3:  # batch dimension
                depth = depth[0]
            result["depth"] = depth.astype(np.float32)

        # Extract normal map if available
        if "normal" in camera_obs:
            normal = camera_obs["normal"]
            if hasattr(normal, "cpu"):
                normal = normal.cpu().numpy()
            if len(normal.shape) == 4:  # batch dimension
                normal = normal[0]
            # Normalize normal map from [-1, 1] to [0, 1] for storage
            normal = (normal + 1) / 2
            result["normal"] = normal.astype(np.float32)

        # Extract segmentation masks
        if "segmentation" in camera_obs:
            seg = camera_obs["segmentation"]
            if hasattr(seg, "cpu"):
                seg = seg.cpu().numpy()
            if len(seg.shape) == 3:  # batch dimension
                seg = seg[0]
            result["semantic_mask"] = seg.astype(np.uint8)

        # Extract class segmentation if available
        if "class_segmentation" in camera_obs:
            class_seg = camera_obs["class_segmentation"]
            if hasattr(class_seg, "cpu"):
                class_seg = class_seg.cpu().numpy()
            if len(class_seg.shape) == 3:  # batch dimension
                class_seg = class_seg[0]
            result["class_mask"] = class_seg.astype(np.uint8)

        # Extract affordance segmentation if available
        if "affordance_segmentation" in camera_obs:
            afford_seg = camera_obs["affordance_segmentation"]
            if hasattr(afford_seg, "cpu"):
                afford_seg = afford_seg.cpu().numpy()
            if len(afford_seg.shape) == 4:  # batch dimension
                afford_seg = afford_seg[0]
            # Handle multi-channel affordance masks
            if afford_seg.shape[-1] > 1:
                afford_seg = afford_seg.argmax(axis=-1)
            result["affordance_mask"] = afford_seg.astype(np.uint8)

        # Extract affordance masks from augmentor if available
        if "affordance_masks" in obs:
            afford_masks = obs["affordance_masks"]

            if "part_mask" in afford_masks:
                part_mask = afford_masks["part_mask"]
                if hasattr(part_mask, "cpu"):
                    part_mask = part_mask.cpu().numpy()
                if len(part_mask.shape) == 3:  # batch dimension
                    part_mask = part_mask[0]
                result["part_mask"] = part_mask.astype(np.uint8)

            if "affordance_mask" in afford_masks:
                afford_mask = afford_masks["affordance_mask"]
                if hasattr(afford_mask, "cpu"):
                    afford_mask = afford_mask.cpu().numpy()
                if len(afford_mask.shape) == 3:  # batch dimension
                    afford_mask = afford_mask[0]
                result["affordance_mask"] = afford_mask.astype(np.uint8)

        # Ensure we have at least an image and semantic mask
        if "image" not in result:
            raise ValueError("No RGB image found in observations")
        if "semantic_mask" not in result:
            # Create a dummy semantic mask if not available
            h, w = result["image"].shape[:2]
            result["semantic_mask"] = np.ones((h, w), dtype=np.uint8)

        return result

    @property
    def _default_sensor_configs(self):
        """Configure cameras based on current parameters"""
        # Calculate camera position from spherical coordinates
        elevation_rad = np.radians(self.camera_elevation)
        azimuth_rad = np.radians(self.camera_azimuth)

        x = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = self.camera_distance * np.sin(elevation_rad)

        camera_pos = self.object_position + np.array([x, y, z])

        # Camera looks at the object
        pose = sapien_utils.look_at(eye=camera_pos, target=self.object_position)

        configs = [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=self.image_size[0],
                height=self.image_size[1],
                fov=np.pi / 3,  # 60 degree FOV
            ),
        ]

        # Add RT camera if needed
        if "rt" in self.obs_mode.lower():
            configs.append(
                CameraConfig(
                    "base_camera_rt",
                    pose=pose,
                    width=self.image_size[0],
                    height=self.image_size[1],
                    fov=np.pi / 3,
                    shader_config=RT_RGB_ONLY_CONFIG,
                )
            )

        return configs

    @property
    def _default_human_render_camera_configs(self):
        """Default camera for human visualization"""
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        """No agent needed for dataset generation"""
        pass

    def _initialize_episode(self, env_idx, options):
        """Initialize episode - no special initialization needed"""
        pass

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """No reward needed for dataset generation"""
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """No reward needed for dataset generation"""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_obs_agent(self):
        """Get observations about the agent's state. By default it is proprioceptive observations which include qpos and qvel.
        Controller state is also included although most default controllers do not have any state.
        """
        return torch.zeros(self.num_envs, device=self.device)
