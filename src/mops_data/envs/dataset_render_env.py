from typing import Any, Dict, Optional, Tuple

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from mops_data.asset_manager.object_annotation_registry import ObjectAnnotationRegistry
from mops_data.asset_manager.partnet_mobility_loader import PartNetMobilityLoader
from mops_data.render.afford_obs_augmentor import AffordObsAugmentor
from mops_data.render.shader_config import RT_RGB_ONLY_CONFIG


@register_env("DatasetRenderEnv-v1", max_episode_steps=1)
class DatasetRenderEnv(BaseEnv):
    """
    Simplified rendering environment for dataset generation.

    Loads PartNet Mobility objects and renders them with configurable
    camera, lighting, and background settings.
    """

    SUPPORTED_ROBOTS = ["none"]

    def __init__(
        self,
        *args,
        # Asset specification
        asset_index: Optional[int] = None,
        asset_id: Optional[str] = None,
        dir_name: Optional[str] = None,
        model_cat: Optional[str] = None,
        # Rendering configuration
        image_size: Tuple[int, int] = (512, 512),
        camera_distance: float = 1.5,
        camera_elevation: float = 15.0,
        camera_azimuth: float = 0.0,
        # Lighting configuration
        lighting_type: str = "studio",
        lighting_intensity: float = 1.0,
        light_temperature: float = 5500.0,  # Kelvin (daylight ~5500K)
        # Background configuration
        background_type: str = "white",
        background_texture: Optional[str] = None,
        # Object configuration
        object_scale: float = 0.8,
        object_position: Optional[np.ndarray] = None,
        **kwargs
    ):
        # Store parameters
        self.asset_index = asset_index
        self.asset_id = asset_id
        self.dir_name = dir_name
        self.model_cat = model_cat
        self.image_size = image_size
        self.camera_distance = camera_distance
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.lighting_type = lighting_type
        self.lighting_intensity = lighting_intensity
        self.light_temperature = light_temperature
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

        self.loaded_object = None

        super().__init__(*args, robot_uids="none", **kwargs)

    def _load_scene(self, options):
        """Load scene with the specified object"""
        # Load object using available identifier
        if self.dir_name is not None:
            self.loaded_object = self.partnet_mobility_loader.load(
                self.dir_name,
                self.object_position,
                no_grav=True,
                scale=self.object_scale,
            )
        elif self.asset_id is not None:
            self.loaded_object = self.partnet_mobility_loader.load(
                self.asset_id,
                self.object_position,
                no_grav=True,
                scale=self.object_scale,
            )
        elif self.asset_index is not None:
            self.loaded_object = self.partnet_mobility_loader.load_by_index(
                self.asset_index,
                self.object_position,
                no_grav=True,
                scale=self.object_scale,
            )
        else:
            raise ValueError("Must specify one of: asset_id, dir_name, or asset_index")

        self.object_annotation_registry.register_missing_objects(self)

    def _kelvin_to_rgb(self, temperature: float) -> np.ndarray:
        """
        Convert color temperature in Kelvin to RGB values.
        Based on Tanner Helland's algorithm.

        Args:
            temperature: Color temperature in Kelvin (1000-40000)

        Returns:
            RGB array normalized to [0,1]
        """
        # Clamp temperature to reasonable range
        temp = np.clip(temperature, 1000, 40000) / 100.0

        # Calculate Red
        if temp <= 66:
            red = 255
        else:
            red = temp - 60
            red = 329.698727446 * (red**-0.1332047592)
            red = np.clip(red, 0, 255)

        # Calculate Green
        if temp <= 66:
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            green = temp - 60
            green = 288.1221695283 * (green**-0.0755148492)
        green = np.clip(green, 0, 255)

        # Calculate Blue
        if temp >= 66:
            blue = 255
        else:
            if temp <= 19:
                blue = 0
            else:
                blue = temp - 10
                blue = 138.5177312231 * np.log(blue) - 305.0447927307
                blue = np.clip(blue, 0, 255)

        return np.array([red, green, blue]) / 255.0

    def _load_lighting(self, options):
        """Simple enhanced lighting with two light sources"""
        # Get RGB color from temperature
        light_color = (
            self._kelvin_to_rgb(self.light_temperature) * self.lighting_intensity
        )

        # Slightly higher ambient light to avoid pure black shadows
        self.scene.ambient_light = [0.05, 0.05, 0.05]

        # Main light (your existing logic, just enhanced)
        if self.lighting_type == "studio":
            # Main key light
            self.scene.add_directional_light(
                [0.5, -1, -0.5], (light_color * 0.8).tolist(), shadow=True
            )
            # Simple fill light from opposite side
            self.scene.add_directional_light(
                [-0.3, -0.5, -0.3], (light_color * 0.3).tolist()
            )

        elif self.lighting_type == "natural":
            # Main sun light
            self.scene.add_directional_light(
                [0.3, -1, -0.7], (light_color * 0.9).tolist(), shadow=True
            )
            # Sky fill light
            self.scene.add_directional_light([0, 0, -1], (light_color * 0.4).tolist())

        elif self.lighting_type == "dramatic":
            # Strong side light
            self.scene.add_directional_light(
                [1, -1, -0.2], (light_color * 1.0).tolist(), shadow=True
            )
            # Subtle fill light
            self.scene.add_directional_light(
                [-0.5, -0.5, -0.5], (light_color * 0.2).tolist()
            )

    def _get_obs_with_sensor_data(self, info, apply_texture_transforms=True):
        """Get observations with affordance augmentation"""
        obs = super()._get_obs_with_sensor_data(info, apply_texture_transforms)
        return self.afford_augmentor.augment(self, obs)

    def extract_render_data(self, obs: Dict) -> Dict[str, np.ndarray]:
        """
        Extract render data from observations for HDF5Writer.

        Returns:
            Dictionary with keys: 'image', 'semantic_mask', 'part_mask',
            'instance_mask', 'affordance_mask', 'depth', 'normal'
        """
        result = {}
        camera_obs = obs["sensor_data"]["base_camera"]

        # Extract data, removing batch dimension [0]
        if "rgb" in camera_obs:
            result["image"] = camera_obs["rgb"].cpu()[0]
        if "depth" in camera_obs:
            result["depth"] = camera_obs["depth"].cpu()[0]
        if "normal" in camera_obs:
            result["normal"] = camera_obs["normal"].cpu()[0]
        if "segmentation" in camera_obs:
            result["part_mask"] = camera_obs["segmentation"].cpu()[0]
        if "class_segmentation" in camera_obs:
            result["semantic_mask"] = camera_obs["class_segmentation"].cpu()[0]
        if "instance_segmentation" in camera_obs:
            result["instance_mask"] = camera_obs["instance_segmentation"].cpu()[0]
        if "affordance_segmentation" in camera_obs:
            result["affordance_mask"] = camera_obs["affordance_segmentation"].cpu()[0]

        return result

    def is_valid_render(self, obs: Dict, min_segments: int = 3) -> bool:
        """
        Check if render is valid based on part segmentation complexity.

        Args:
            obs: Observation dictionary
            min_segments: Minimum number of unique segmentation values required

        Returns:
            True if render has sufficient segmentation detail
        """
        camera_obs = obs["sensor_data"]["base_camera"]
        if "segmentation" not in camera_obs:
            return True  # No segmentation to validate

        part_mask = camera_obs["segmentation"].cpu()[0]
        unique_values = np.unique(part_mask)
        return len(unique_values) >= min_segments

    @property
    def _default_sensor_configs(self):
        """Configure camera based on current parameters"""
        # Calculate camera position from spherical coordinates
        elevation_rad = np.radians(self.camera_elevation)
        azimuth_rad = np.radians(self.camera_azimuth)

        x = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = self.camera_distance * np.sin(elevation_rad)

        camera_pos = self.object_position + np.array([x, y, z])
        pose = sapien_utils.look_at(eye=camera_pos, target=self.object_position)

        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=self.image_size[0],
                height=self.image_size[1],
                fov=np.pi / 3,
            ),
            CameraConfig(
                "base_camera_rt",
                pose=pose,
                width=self.image_size[0],
                height=self.image_size[1],
                fov=np.pi / 3,
                shader_config=RT_RGB_ONLY_CONFIG,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        """Default camera for human visualization"""
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        """No agent needed"""
        pass

    def _initialize_episode(self, env_idx, options):
        """No special initialization needed"""
        pass

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """No reward needed"""
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """No reward needed"""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_obs_agent(self):
        """No agent observations needed"""
        return torch.zeros(self.num_envs, device=self.device)
