#!/usr/bin/env python3
"""
Gripper Controller - Manages robot gripper state and actions
"""
from config import logger


class GripperController:
    """
    Controls the robot gripper state and actions
    """

    def __init__(self):
        """Initialize the gripper controller"""
        # Gripper state
        self.gripper_open = True
        self.gripper_action = 0.0  # 0.0 = fully open, 1.0 = fully closed
        self.last_target_value = 0.0

        # Action configuration
        self.grip_speed = 0.1  # Speed of gripper movement (0.0-1.0)
        self.threshold = 0.05  # Threshold for detecting gripper action changes

        logger.info("Gripper controller initialized")

    def update_gripper(self, target_value):
        """
        Update gripper state based on target value

        Args:
            target_value: Target gripper value (0.0 = open, 1.0 = closed)

        Returns:
            Current gripper action value after update
        """
        # Skip if no significant change
        if abs(target_value - self.last_target_value) <= self.threshold:
            return self.gripper_action

        # Update the last target value
        self.last_target_value = target_value

        # Move gripper toward target with speed constraint
        if target_value > self.gripper_action:
            # Closing
            self.gripper_action = min(
                self.gripper_action + self.grip_speed, target_value
            )
        else:
            # Opening
            self.gripper_action = max(
                self.gripper_action - self.grip_speed, target_value
            )

        # Update open/closed state
        self.gripper_open = self.gripper_action < 0.5

        return self.gripper_action

    def toggle_gripper(self):
        """
        Toggle gripper between open and closed states

        Returns:
            New gripper action value
        """
        self.gripper_open = not self.gripper_open
        self.gripper_action = 0.0 if self.gripper_open else 1.0
        self.last_target_value = self.gripper_action

        logger.info(f"Gripper {'opened' if self.gripper_open else 'closed'}")
        return self.gripper_action

    def set_gripper(self, value):
        """
        Set gripper to specific value immediately

        Args:
            value: Gripper value (0.0 = open, 1.0 = closed)

        Returns:
            Set gripper action value
        """
        value = max(0.0, min(1.0, value))  # Clamp to valid range
        self.gripper_action = value
        self.last_target_value = value
        self.gripper_open = value < 0.5

        return self.gripper_action

    def get_gripper_action(self):
        """
        Get current gripper action value

        Returns:
            Current gripper action value
        """
        return self.gripper_action

    def is_gripper_open(self):
        """
        Check if gripper is currently open

        Returns:
            True if gripper is open, False otherwise
        """
        return self.gripper_open
