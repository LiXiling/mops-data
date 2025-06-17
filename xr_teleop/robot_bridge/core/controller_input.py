#!/usr/bin/env python3
"""
Controller Input Processor - Handles VR controller input for robot control
"""
import numpy as np
from config import controller_data, logger, toggle_gripper


class ControllerInputProcessor:
    """
    Processes VR controller input data and translates to robot commands
    """

    def __init__(self, smoothing=0.3, position_scale=1.0):
        """
        Initialize the controller input processor

        Args:
            smoothing: Position smoothing factor (0-1, higher = more smoothing)
            position_scale: Position scaling factor for mapping controller to robot movement
        """
        # Configuration
        self.smoothing = smoothing
        self.position_scale = position_scale

        # Controller state tracking
        self.grip_active = False
        self.grip_pressed_prev = False
        self.trigger_value_prev = 0.0
        self.gripper_open = True
        self.secondary_pressed_prev = False
        self.primary_pressed_prev = False

        # Position tracking
        self.initial_controller_pos = None
        self.initial_controller_rot = None
        self.initial_robot_pose = None
        self.current_controller_pos = None
        self.current_controller_rot = None

        # Command outputs
        self.reset_requested = False
        self.primary_pressed = False
        self.gripper_action = 0.0  # 0.0 = open, 1.0 = closed
        self.delta_position = [0.0, 0.0, 0.0]
        self.relative_rotation = None

        logger.info("Controller input processor initialized")

    def start_control(self, controller, robot_pose):
        """
        Start controlling the robot with VR controller

        Args:
            controller: Controller data dictionary
            robot_pose: Initial robot pose to use as reference
        """
        # Store initial positions and orientations
        self.initial_controller_pos = controller["position"].copy()
        self.initial_controller_rot = controller["rotation"].copy()
        self.current_controller_pos = self.initial_controller_pos.copy()
        self.current_controller_rot = self.initial_controller_rot.copy()

        # Store robot's pose
        self.initial_robot_pose = robot_pose

        # Activate control
        self.grip_active = True
        logger.info("Started robot control")

    def stop_control(self):
        """Stop controlling the robot"""
        self.grip_active = False
        logger.info("Stopped robot control")

    def process_input(self):
        """
        Process controller input and generate robot commands

        Returns:
            Dictionary with control commands for the robot
        """
        global controller_data, toggle_gripper

        # Reset command flags
        self.reset_requested = False
        self.primary_pressed = False

        # Get right controller data (primary control)
        controller = controller_data["right"]

        # Check for A button press (primary button) - triggers saving recording
        primary_pressed = controller["buttons"].get("primary_pressed", False)
        if primary_pressed and not self.primary_pressed_prev:
            logger.info("A button pressed - Save recording request")
            self.primary_pressed = True
        self.primary_pressed_prev = primary_pressed

        # Check for B button press (secondary button) - triggers environment reset
        secondary_pressed = controller["buttons"].get("secondary_pressed", False)
        if secondary_pressed and not self.secondary_pressed_prev:
            logger.info("B button pressed - Requesting environment reset")
            self.secondary_pressed_prev = True
            self.reset_requested = True

            # Clear control state when resetting
            self.grip_active = False
            return self._build_command_dict()

        self.secondary_pressed_prev = secondary_pressed

        # Handle grip button (start/stop control)
        grip_pressed = controller["buttons"].get("grip_pressed", False)
        if grip_pressed != self.grip_pressed_prev:
            self.grip_pressed_prev = grip_pressed
            return self._build_command_dict()

        # Update position and orientation if control is active
        if self.grip_active:
            self._update_position_rotation(controller)

        # Handle trigger for gripper control
        trigger_value = controller["buttons"].get("trigger_value", 0.0)

        # Update gripper based on trigger value with small threshold to prevent jitter
        if abs(trigger_value - self.trigger_value_prev) > 0.05:
            self.gripper_action = max(0.0, min(1.0, trigger_value))

        self.trigger_value_prev = trigger_value

        return self._build_command_dict()

    def _update_position_rotation(self, controller):
        """
        Update position and rotation based on controller movement

        Args:
            controller: Controller data dictionary
        """
        if not self.grip_active or not self.initial_controller_pos:
            return

        # Get current controller state
        current_pos = controller["position"]
        current_rot = controller["rotation"]

        # Apply smoothing for stable movement
        smoothed_pos = [
            self.current_controller_pos[i] * self.smoothing
            + current_pos[i] * (1 - self.smoothing)
            for i in range(3)
        ]

        # Apply smoothing to rotation (quaternion)
        smoothed_rot = [
            self.current_controller_rot[i] * self.smoothing
            + current_rot[i] * (1 - self.smoothing)
            for i in range(4)
        ]

        # Normalize quaternion to prevent drift
        magnitude = np.sqrt(sum(x * x for x in smoothed_rot))
        if magnitude > 0:
            smoothed_rot = [x / magnitude for x in smoothed_rot]

        # Save current state for next frame
        self.current_controller_pos = smoothed_pos.copy()
        self.current_controller_rot = smoothed_rot.copy()

        # Calculate position delta from initial position
        delta_pos = [
            (smoothed_pos[i] - self.initial_controller_pos[i]) * self.position_scale
            for i in range(3)
        ]

        # Transform WebXR coordinates to robot coordinates
        # WebXR: +X right, +Y up, +Z backward
        # Robot: +X forward, +Y left, +Z up
        self.delta_position = [
            -delta_pos[2],  # WebXR Z → Robot -X
            -delta_pos[0],  # WebXR X → Robot -Y
            delta_pos[1],  # WebXR Y → Robot Z
        ]

        # Store the current rotation for quaternion calculations by robot pose controller
        self.current_rotation = smoothed_rot

    def _build_command_dict(self):
        """
        Build the command dictionary to return

        Returns:
            Dictionary with command parameters
        """
        return {
            "grip_active": self.grip_active,
            "grip_pressed": self.grip_pressed_prev,
            "gripper_action": self.gripper_action,
            "delta_position": self.delta_position,
            "current_rotation": (
                self.current_rotation if hasattr(self, "current_rotation") else None
            ),
            "initial_rotation": self.initial_controller_rot,
            "reset_requested": self.reset_requested,
            "primary_pressed": self.primary_pressed,
        }

    def is_controlling(self):
        """
        Check if the controller is actively controlling the robot

        Returns:
            True if actively controlling, False otherwise
        """
        return self.grip_active

    def has_initial_state(self):
        """
        Check if the controller has initial position and rotation set

        Returns:
            True if initial state is set, False otherwise
        """
        return (
            self.initial_controller_pos is not None
            and self.initial_controller_rot is not None
        )
