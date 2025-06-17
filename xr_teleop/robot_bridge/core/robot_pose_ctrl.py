#!/usr/bin/env python3
"""
Robot Pose Controller - Calculates and applies robot poses from controller input
"""
import numpy as np
import sapien.core as sapien
from config import logger
from utils.pose_utils import quaternion_to_euler, webxr_to_sapien_quaternion


class RobotPoseController:
    """
    Manages robot positioning, orientation, and movement based on controller input
    """

    def __init__(self):
        """Initialize the robot pose controller"""
        # Current robot pose
        self.current_pose = None
        self.target_pose = None

        # Last applied action
        self.last_action = None

        logger.info("Robot pose controller initialized")

    def update_target_pose(self, commands, initial_robot_pose):
        """
        Calculate target pose based on controller commands

        Args:
            commands: Dictionary of controller commands
            initial_robot_pose: Initial robot pose from when control started

        Returns:
            Sapien Pose object with target position and orientation
        """
        # Skip if not actively controlling
        if not commands["grip_active"]:
            return None

        # Get delta position from commands
        delta_pos = commands["delta_position"]

        # Get the initial robot position
        initial_pos = initial_robot_pose.p

        # Extract position values
        try:
            if hasattr(initial_pos, "shape") and len(initial_pos.shape) > 1:
                initial_pos_array = [float(initial_pos[0][i]) for i in range(3)]
            else:
                initial_pos_array = [float(initial_pos[i]) for i in range(3)]
        except:
            logger.warning("Could not extract initial position, using zeros")
            initial_pos_array = [0.0, 0.0, 0.0]

        # Create the new position
        new_robot_pos = [
            initial_pos_array[0] + delta_pos[0],
            initial_pos_array[1] + delta_pos[1],
            initial_pos_array[2] + delta_pos[2],
        ]

        # Handle rotation
        # Convert WebXR quaternion to Sapien format
        initial_quat = webxr_to_sapien_quaternion(commands["initial_rotation"])
        current_quat = webxr_to_sapien_quaternion(commands["current_rotation"])

        # Calculate relative rotation
        initial_pose = sapien.Pose(q=initial_quat)
        current_pose = sapien.Pose(q=current_quat)
        relative_pose = current_pose * initial_pose.inv()

        # Get the initial robot orientation
        initial_quat = initial_robot_pose.q

        # Extract quaternion values
        try:
            if hasattr(initial_quat, "shape") and len(initial_quat.shape) > 1:
                proc_quat = [float(initial_quat[0][i]) for i in range(4)]
            else:
                proc_quat = [float(initial_quat[i]) for i in range(4)]
        except:
            logger.warning("Could not extract initial quaternion, using identity")
            proc_quat = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion

        # Apply rotation to robot's initial pose
        robot_rotation = relative_pose * sapien.Pose(q=proc_quat)
        new_robot_quat = robot_rotation.q

        # Create target pose
        self.target_pose = sapien.Pose(p=new_robot_pos, q=new_robot_quat)
        return self.target_pose

    def calculate_action(self, target_pose, robot_base_pose, gripper_action):
        """
        Calculate robot action vector from target pose

        Args:
            target_pose: Target pose in world coordinates
            robot_base_pose: Robot base pose for coordinate transformation
            gripper_action: Gripper action value (0.0 = open, 1.0 = closed)

        Returns:
            Action vector for the robot
        """
        if target_pose is None:
            return None

        # Extract position and orientation
        pos = target_pose.p
        quat = target_pose.q

        # Convert to robot base frame
        pos_local = self._world_to_robot_frame(pos, robot_base_pose)

        # Convert quaternion to Euler angles for robot control
        euler = quaternion_to_euler(quat).detach().cpu().numpy()

        # Create action vector with position, orientation and gripper
        gripper_action = (
            2.0 * gripper_action - 1.0
        ) * -1  # Scale to [-1, 1] and invert
        gripper_val = np.array([gripper_action], dtype=np.float32)
        action = np.concatenate([pos_local, euler, gripper_val])

        self.last_action = action
        return action

    def _world_to_robot_frame(self, world_pos, robot_base_pose):
        """
        Convert a world-space position to robot-local coordinates

        Args:
            world_pos: Position in world coordinates
            robot_base_pose: Robot base pose

        Returns:
            Position in robot-local coordinates
        """
        # Create world pose with given position
        world_pose = sapien.Pose(p=world_pos)

        # Transform to robot-local frame
        local_pose = robot_base_pose.inv() * world_pose

        # Extract the position
        local_pos = local_pose.p

        # Convert to numpy array
        try:
            if hasattr(local_pos, "shape") and len(local_pos.shape) > 1:
                return np.array(
                    [float(local_pos[0][i]) for i in range(3)], dtype=np.float32
                )
            else:
                return np.array(local_pos, dtype=np.float32)
        except:
            logger.warning("Error converting local position, using zeros")
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_current_target_pose(self):
        """
        Get the current target pose

        Returns:
            Current target pose or None if not set
        """
        return self.target_pose

    def get_last_action(self):
        """
        Get the last calculated action

        Returns:
            Last action vector or None if not calculated
        """
        return self.last_action
