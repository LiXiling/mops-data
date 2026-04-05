import abc
from typing import Any, Dict, Tuple

import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from mops_data.asset_manager.object_annotation_registry import ObjectAnnotationRegistry
from mops_data.asset_manager.partnet_mobility_loader import PartNetMobilityLoader
from mops_data.render.afford_obs_augmentor import AffordObsAugmentor


class DatasetRenderEnv(BaseEnv, abc.ABC):
    """
    Simplified rendering environment for dataset generation.

    Loads PartNet Mobility objects and renders them with configurable
    camera, lighting, and background settings.
    """

    SUPPORTED_ROBOTS = ["none"]

    def __init__(
        self,
        *args,
        # Rendering configuration
        image_size: Tuple[int, int] = (512, 512),
        camera_distance: float = 1.5,
        camera_elevation: float = 15.0,
        camera_azimuth: float = 0.0,
        # Lighting configuration
        lighting_type: str = "studio",
        lighting_intensity: float = 1.0,
        light_temperature: float = 5500.0,  # Kelvin (daylight ~5500K)
        **kwargs,
    ):
        # Store parameters
        self.image_size = image_size
        self.camera_distance = camera_distance
        self.camera_elevation = camera_elevation
        self.camera_azimuth = camera_azimuth
        self.lighting_type = lighting_type
        self.lighting_intensity = lighting_intensity
        self.light_temperature = light_temperature

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

        super().__init__(*args, robot_uids="none", **kwargs)

    @abc.abstractmethod
    def _load_objects(self, options: Dict[str, Any]):
        """
        Load objects into the scene based on provided options.
        This method should be implemented by subclasses to handle specific object loading logic.
        """
        pass

    def _update_sensor_pose(self):
        """Push current camera_elevation/azimuth to the live SAPIEN camera sensor.

        Called after updating render params so that the next reset() renders from
        the new viewpoint without triggering a full scene rebuild (reconfigure).
        Subclasses with non-spherical cameras (e.g. kitchen fixture-relative)
        should override this.
        """
        if not (hasattr(self, "_sensors") and "base_camera" in self._sensors):
            return
        elevation_rad = np.radians(self.camera_elevation)
        azimuth_rad = np.radians(self.camera_azimuth)
        x = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = self.camera_distance * np.sin(elevation_rad)
        new_pose = sapien_utils.look_at(
            eye=np.array([x, y, z]), target=np.asarray([0.0, 0.0, 0.0])
        )
        pose_sp = new_pose.sp if hasattr(new_pose, "sp") else new_pose
        self._sensors["base_camera"].camera.set_local_pose(pose_sp)

    def update_render_params(
        self,
        camera_elevation: float = None,
        camera_azimuth: float = None,
        lighting_type: str = None,
        lighting_intensity: float = None,
        light_temperature: float = None,
        **_ignored,
    ):
        """Update viewpoint/lighting params and push camera pose to the live sensor.

        Because ManiSkill uses reconfiguration_freq=0 by default, _load_scene and
        _setup_sensors are only called once at gym.make() time.  Subsequent reset()
        calls skip reconfiguration, so attribute changes must be accompanied by a
        direct sensor update (_update_sensor_pose) to take effect.
        """
        if camera_elevation is not None:
            self.camera_elevation = camera_elevation
        if camera_azimuth is not None:
            self.camera_azimuth = camera_azimuth
        if lighting_type is not None:
            self.lighting_type = lighting_type
        if lighting_intensity is not None:
            self.lighting_intensity = lighting_intensity
        if light_temperature is not None:
            self.light_temperature = light_temperature
        self._update_sensor_pose()

    def _load_scene(self, options):
        self._load_objects(options)
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

    def build_render_data(self, obs: Dict) -> Dict[str, np.ndarray]:
        """
        Extract render data from observations for the dataset writer.
        """
        camera_obs = obs["sensor_data"]["base_camera"]
        key_map = {
            "rgb": "image",
            "depth": "depth",
            "normal": "normal",
            "segmentation": "part",
            "class_segmentation": "semantic",
            "instance_segmentation": "instance",
            "affordance_segmentation": "affordance",
            "is_partnet": "is_partnet",
        }

        return {
            target_key: camera_obs[source_key].cpu()[0]
            for source_key, target_key in key_map.items()
            if source_key in camera_obs
        }

    @abc.abstractmethod
    def is_valid_render(self, obs: Dict, min_segments: int = 3) -> bool:
        """
        Check if render is valid based on part segmentation complexity.

        Args:
            obs: Raw Observation dictionary, unextracted
            min_segments: Minimum number of unique segmentation values required

        Returns:
            True if render has sufficient segmentation detail
        """
        pass

    @property
    def _default_sensor_configs(self):
        """Configure camera based on current parameters"""
        # Calculate camera position from spherical coordinates
        elevation_rad = np.radians(self.camera_elevation)
        azimuth_rad = np.radians(self.camera_azimuth)

        x = self.camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = self.camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = self.camera_distance * np.sin(elevation_rad)

        camera_pos = np.array([x, y, z])
        pose = sapien_utils.look_at(eye=camera_pos, target=np.asarray([0.0, 0.0, 0.0]))

        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=self.image_size[0],
                height=self.image_size[1],
                fov=np.pi / 3,
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

    def get_state_dict(self) -> Dict[str, Any]:
        """Get environment state dict (empty for rendering env)"""
        return {}
