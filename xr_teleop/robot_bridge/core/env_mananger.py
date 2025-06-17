#!/usr/bin/env python3
"""
Environment Manager - Handles simulation environment setup and visualization
"""
import gymnasium as gym
import sapien.core as sapien
from config import logger


class EnvironmentManager:
    """
    Manages the simulation environment, visualization, and resets
    """

    def __init__(
        self, env_id="PickCube-v1", obs_mode="state", control_mode="pd_ee_pose"
    ):
        """
        Initialize the environment manager

        Args:
            env_id: ManiSkill environment ID to use
            obs_mode: Observation mode for the environment
            control_mode: Control mode for the robot
        """
        self.env_id = env_id
        self.obs_mode = obs_mode
        self.control_mode = control_mode

        # Environment and visualization objects
        self.env = None
        self.viewer = None
        self.target_viz = None
        self.robot_base_pose = None

        # Current observation
        self.current_obs = None

        # Setup the environment
        self.setup_environment()

    def setup_environment(self):
        """Initialize the simulation environment and visualization"""
        logger.info(f"Setting up environment: {self.env_id}")

        # Create environment with specified parameters
        self.env = gym.make(
            self.env_id,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            render_mode="rgb_array",
        )

        # Reset environment and get initial observation
        self.current_obs, _ = self.env.reset()

        # Initialize viewer for visualization
        self.viewer = self.env.render_human()

        # Create target visualization (green sphere for target position)
        self._create_visualization()

        # Store robot base pose for coordinate transformations
        self.robot_base_pose = self.env.unwrapped.agent.robot.get_pose()

        logger.info("Environment setup complete")
        return self.current_obs

    def _create_visualization(self):
        """Create visualization markers for the simulation"""
        # Create target position visualization (green sphere)
        scene = self.env.unwrapped.scene
        builder = scene.create_actor_builder()
        builder.add_sphere_visual(
            radius=0.02,
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
        )
        self.target_viz = builder.build(name="target_viz")

        # Could add more visualizations here (e.g., coordinate axes, workspace bounds)

    def reset(self):
        """Reset the environment to initial state"""
        logger.info("Resetting environment...")
        self.current_obs, _ = self.env.reset()

        # Update robot base pose as it might change after reset
        self.robot_base_pose = self.env.unwrapped.agent.robot.get_pose()

        logger.info("Environment reset successful")
        return self.current_obs

    def step(self, action):
        """
        Take a step in the environment

        Args:
            action: Action vector to apply

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        result = self.env.step(action)
        self.current_obs = result[0]

        # Check if episode is done
        if result[2]:  # terminated
            logger.info("Episode terminated")

        return result

    def render(self):
        """Render the current state of the environment"""
        return self.env.render_human()

    def update_target_visualization(self, target_pose):
        """
        Update the target visualization position

        Args:
            target_pose: Sapien Pose object with target position
        """
        if self.target_viz:
            self.target_viz.set_pose(target_pose)

    def get_tcp_pose(self):
        """
        Get the current Tool Center Point (TCP) pose

        Returns:
            Current TCP pose as a Sapien Pose object
        """
        return self.env.unwrapped.agent.tcp.pose

    def get_robot_pose(self):
        """
        Get the current robot base pose

        Returns:
            Current robot base pose as a Sapien Pose object
        """
        return self.env.unwrapped.agent.robot.get_pose()

    def is_quit_requested(self):
        """
        Check if user requested to quit the simulation

        Returns:
            True if quit requested, False otherwise
        """
        return self.viewer.window.key_press("q")

    def close(self):
        """Clean up resources and close the environment"""
        if self.env:
            self.env.close()
            logger.info("Environment closed")
