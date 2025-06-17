#!/usr/bin/env python3
"""
Configuration for the robot bridge
"""
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bridge_robot_control")

# Default controller data structure
controller_data = {
    "right": {
        "position": [0, 0, 0],
        "rotation": [0, 0, 0, 1],  # [x, y, z, w] quaternion
        "buttons": {
            "grip_pressed": False,  # Hold to control robot
            "trigger_value": 0.0,  # For gripper control (0.0-1.0)
            "primary_pressed": False,  # Primary button (A/B or X/Y)
            "secondary_pressed": False,  # Secondary button (B/Y)
        },
    },
    "left": {
        "position": [0, 0, 0],
        "rotation": [0, 0, 0, 1],  # [x, y, z, w] quaternion
        "buttons": {
            "grip_pressed": False,
            "trigger_value": 0.0,
            "primary_pressed": False,
            "secondary_pressed": False,
        },
    },
}

# Global state flags
running = True  # Main program running flag
toggle_gripper = False  # Signal to toggle gripper state
previewer_following = False  # Whether preview visualization follows controller

# Recording settings
record_enabled = True  # Whether recording functionality is enabled
recording_active = False  # Whether a recording is currently in progress
recording_counter = 0  # Counter for recordings
recording_feedback = False  # Signal to send recording status back to client

# Robot control settings
robot_settings = {
    # Position scaling (sensitivity adjustment)
    "position_scale": 1.0,
    # Smoothing factor (0.0-1.0, higher = more smoothing)
    "smoothing": 0.3,
    # Control update rate in Hz
    "update_rate": 30,
    # Workspace limits (robot coordinate system)
    "workspace_limits": {
        "min": [-0.5, -0.5, 0.0],  # Min XYZ
        "max": [0.5, 0.5, 0.5],  # Max XYZ
    },
}

# Recording settings
recording_settings = {
    # Base directory for demonstrations
    "demos_dir": "vr_demos",
    # Whether to record controller data
    "record_controller_data": True,
    # Whether to include robot state
    "record_robot_state": True,
    # Whether to save metadata with each demo
    "save_metadata": True,
}
