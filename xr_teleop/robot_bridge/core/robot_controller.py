#!/usr/bin/env python3
"""
Main Robot Controller - Integrates all components for VR-controlled robot manipulation
"""
import time

from config import (
    controller_data,
    logger,
    recording_active,
    recording_counter,
    recording_feedback,
    running,
    toggle_gripper,
)
from core.controller_input import ControllerInputProcessor
from core.env_mananger import EnvironmentManager
from core.gripper_ctrl import GripperController
from core.robot_pose_ctrl import RobotPoseController
from utils.record_utils import DemonstrationRecorder


class RobotController:
    """
    Main controller class integrating all components for robot control
    """

    def __init__(self, env_id="PickCube-v1"):
        """
        Initialize the robot controller

        Args:
            env_id: ManiSkill environment ID
        """
        # Configuration
        self.env_id = env_id
        self.update_rate = 30  # Control frequency in Hz

        # Initialize components
        logger.info(f"Initializing robot controller with environment: {env_id}")
        self.env_manager = EnvironmentManager(env_id=env_id, control_mode="pd_ee_pose")
        self.input_processor = ControllerInputProcessor(
            smoothing=0.3, position_scale=1.0
        )
        self.pose_controller = RobotPoseController()
        self.gripper_controller = GripperController()

        # Initialize recording functionality
        self.recorder = DemonstrationRecorder()
        self.recording = False

        # State tracking
        self.initial_robot_pose = None

    def process_controller(self):
        """
        Process controller input and update robot
        """
        global controller_data, toggle_gripper, recording_active, recording_counter, recording_feedback

        # Process controller input to get commands
        commands = self.input_processor.process_input()

        # Check for reset request (B button)
        if commands["reset_requested"]:
            # Stop existing recording if active
            if self.recording:
                logger.info("Discarding current recording")
                self.recorder.stop_recording(save=False)
                self.recording = False
                recording_active = False
                recording_feedback = True  # Signal to update clients

            # Reset environment
            self.env_manager.reset()
            self.initial_robot_pose = None

            # Start a new recording
            env_state = self.env_manager.env.get_state_dict()
            self.recorder.start_recording(env_state)
            self.recording = True
            recording_active = True
            recording_counter += 1
            recording_feedback = True  # Signal to update clients
            logger.info(f"Started new recording #{recording_counter}")
            return

        # Check for primary button (A button) - save recording
        if commands.get("primary_pressed", False):
            if self.recording:
                logger.info("Saving current recording")
                demo_path = self.recorder.stop_recording(save=True)
                if demo_path:
                    logger.info(f"Demo saved to {demo_path}")
                self.recording = False
                recording_active = False
                recording_feedback = True  # Signal to update clients
            return

        # Check for grip state change (start/stop control)
        if commands["grip_pressed"] and not self.input_processor.is_controlling():
            # Starting control - store current robot pose
            self.initial_robot_pose = self.env_manager.get_tcp_pose()
            self.input_processor.start_control(
                controller_data["right"], self.initial_robot_pose
            )
        elif not commands["grip_pressed"] and self.input_processor.is_controlling():
            # Stopping control
            self.input_processor.stop_control()

        # Update gripper based on trigger value
        gripper_action = self.gripper_controller.set_gripper(commands["gripper_action"])

        # Skip pose update if not controlling
        if not self.input_processor.is_controlling() or self.initial_robot_pose is None:
            return

        # Calculate target pose from controller movement
        target_pose = self.pose_controller.update_target_pose(
            commands, self.initial_robot_pose
        )

        # Update target visualization
        if target_pose:
            self.env_manager.update_target_visualization(target_pose)

        # Calculate and apply robot action
        if target_pose:
            action = self.pose_controller.calculate_action(
                target_pose, self.env_manager.robot_base_pose, gripper_action
            )

            if action is not None:
                # Apply action to robot
                obs, reward, terminated, truncated, info = self.env_manager.step(action)

                # Record action if recording is active
                if self.recording:
                    env_state = self.env_manager.env.get_state_dict()
                    controller_state = controller_data.copy()
                    self.recorder.record_action(action, env_state, controller_state)

    def run_simulation(self):
        """Main simulation loop"""
        logger.info("Starting robot simulation")

        # For FPS tracking
        frame_count = 0
        last_time = time.time()

        try:
            # Main control loop
            while running:
                frame_count += 1
                current_time = time.time()

                # Track FPS
                if frame_count % 60 == 0:
                    fps = 60.0 / (current_time - last_time)
                    last_time = current_time
                    logger.info(f"FPS: {fps:.1f}")

                # Process controller input and update robot
                self.process_controller()

                # Render scene (every other frame)
                if frame_count % 2 == 0:
                    self.env_manager.render()

                # Check for quit
                if self.env_manager.is_quit_requested():
                    logger.info("Quit key pressed")
                    break

                # Maintain control frequency
                elapsed = time.time() - current_time
                target_frame_time = 1.0 / self.update_rate

                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            # Clean up resources
            if self.recording:
                logger.info("Stopping recording before exit")
                self.recorder.stop_recording(save=True)

            self.env_manager.close()
            logger.info("Simulation ended")
