#!/usr/bin/env python3
"""
Simplified utilities for handling poses and transformations
between WebXR and robot coordinate systems
"""
import numpy as np
import torch
from config import logger
from mani_skill.utils.geometry import rotation_conversions
from scipy.spatial.transform import Rotation


def get_pose(obj):
    """
    Get pose from an object with error handling

    Args:
        obj: Object with pose information

    Returns:
        Pose object or None if not available
    """
    if obj is None:
        return None

    try:
        # Handle different object types that might have pose info
        if hasattr(obj, "get_pose"):
            return obj.get_pose()
        elif hasattr(obj, "pose"):
            return obj.pose
        else:
            return None
    except Exception as e:
        logger.error(f"Error getting pose: {e}")
        return None


def set_pose(obj, pose):
    """
    Set pose on an object with error handling

    Args:
        obj: Object to set pose on
        pose: Pose to set

    Returns:
        bool: Success or failure
    """
    if obj is None or pose is None:
        return False

    try:
        if hasattr(obj, "set_pose"):
            obj.set_pose(pose)
            return True
        elif hasattr(obj, "pose"):
            obj.pose = pose
            return True
        return False
    except Exception as e:
        logger.error(f"Error setting pose: {e}")
        return False


def webxr_to_sapien_quaternion(webxr_quat):
    """
    Convert WebXR quaternion [x,y,z,w] to Sapien quaternion [w,x,y,z]

    Args:
        webxr_quat: WebXR quaternion [x,y,z,w]

    Returns:
        list: Sapien quaternion [w,x,y,z]
    """
    if len(webxr_quat) != 4:
        logger.error(f"Invalid quaternion: {webxr_quat}")
        return [1, 0, 0, 0]  # Default identity quaternion

    return [
        webxr_quat[3],  # w
        webxr_quat[0],  # x
        webxr_quat[1],  # y
        webxr_quat[2],  # z
    ]


def quaternion_to_euler(quat):
    quat = torch.tensor(quat)
    return rotation_conversions.matrix_to_euler_angles(
        rotation_conversions.quaternion_to_matrix(quat), "XYZ"
    )


def transform_coordinates(position, rotation=None):
    """
    Transform coordinates from WebXR space to robot space

    WebXR: +X right, +Y up, +Z backward
    Robot: +X forward, +Y left, +Z up

    Args:
        position: [x,y,z] position in WebXR space
        rotation: Optional [x,y,z,w] quaternion in WebXR space

    Returns:
        tuple: (robot_position, robot_quaternion) or just robot_position if rotation is None
    """
    # Skip empty inputs
    if position is None:
        return None

    # Transform matrix: WebXR -> Robot space
    transform = np.array(
        [
            [0, 0, -1],  # WebXR +Z → Robot -X
            [-1, 0, 0],  # WebXR +X → Robot -Y
            [0, 1, 0],  # WebXR +Y → Robot +Z
        ]
    )

    # Transform position
    webxr_pos = np.array(position)
    robot_pos = transform @ webxr_pos

    # If no rotation provided, return just position
    if rotation is None:
        return robot_pos

    # Handle rotation transformation
    try:
        # Convert quaternion to rotation matrix
        webxr_quat = np.array([rotation[0], rotation[1], rotation[2], rotation[3]])
        webxr_rot_matrix = Rotation.from_quat(webxr_quat).as_matrix()

        # Apply coordinate transformation
        robot_rot_matrix = transform @ webxr_rot_matrix @ transform.T

        # Add downward orientation correction (for end effector)
        down_correction = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        robot_rot_matrix = robot_rot_matrix @ down_correction

        # Convert back to quaternion [w,x,y,z] for Sapien
        robot_quat = Rotation.from_matrix(robot_rot_matrix).as_quat()
        robot_quat_sapien = [robot_quat[3], robot_quat[0], robot_quat[1], robot_quat[2]]

        return robot_pos, robot_quat_sapien

    except Exception as e:
        logger.error(f"Error transforming rotation: {e}")
        # Return position and identity quaternion as fallback
        return robot_pos, [1, 0, 0, 0]
