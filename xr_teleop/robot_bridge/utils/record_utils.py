#!/usr/bin/env python3
"""
Recording utilities for robot demonstrations
"""
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
from config import logger


class DemonstrationRecorder:
    """Records robot demonstrations including actions, states, and metadata"""

    def __init__(self, output_dir="demos"):
        """Initialize the demonstration recorder

        Args:
            output_dir: Directory to save demonstrations
        """
        self.output_dir = Path(output_dir)
        self.recording = False
        self.demo_count = 0
        self.reset_buffer()

        # Create output directory structure
        self.setup_directories()

    def setup_directories(self):
        """Create necessary directories for storing demonstrations"""
        # Create base output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create metadata file if it doesn't exist
        self.metadata_path = self.output_dir / "metadata.json"
        if not self.metadata_path.exists():
            with open(self.metadata_path, "w") as f:
                json.dump({"episodes": [], "version": "1.0"}, f)

    def reset_buffer(self):
        """Reset recording buffer"""
        self.buffer = {
            "actions": [],
            "env_states": [],
            "timestamps": [],
            "controller_data": [],
        }
        self.start_time = None

    def start_recording(self, env_state):
        """Start recording a new demonstration

        Args:
            env_state: Initial environment state
        """
        if self.recording:
            logger.warning("Already recording, stopping current recording first")
            self.stop_recording(save=False)

        logger.info("Starting new demonstration recording")
        self.recording = True
        self.reset_buffer()
        self.start_time = time.time()

        # Record initial state
        self.buffer["env_states"].append(env_state)
        self.buffer["timestamps"].append(0.0)

    def record_action(self, action, env_state, controller_data=None):
        """Record an action and the resulting state

        Args:
            action: Robot action vector
            env_state: Environment state after action
            controller_data: Optional controller data for debugging
        """
        if not self.recording:
            return

        # Calculate timestamp relative to start
        current_time = time.time()
        relative_time = current_time - self.start_time if self.start_time else 0.0

        # Add to buffer
        self.buffer["actions"].append(action)
        self.buffer["env_states"].append(env_state)
        self.buffer["timestamps"].append(relative_time)

        if controller_data:
            self.buffer["controller_data"].append(controller_data)

    def stop_recording(self, save=True):
        """Stop current recording and optionally save it

        Args:
            save: Whether to save the recording to disk

        Returns:
            str: Path to saved file or None if not saved
        """
        if not self.recording:
            logger.warning("Not currently recording")
            return None

        self.recording = False

        if save and len(self.buffer["actions"]) > 0:
            return self._save_demonstration()
        else:
            logger.info("Recording stopped without saving")
            self.reset_buffer()
            return None

    def _save_demonstration(self):
        """Save the current demonstration buffer to disk

        Returns:
            str: Path to saved file
        """
        # Increment demo count
        self.demo_count += 1

        # Create filename based on timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"demo_{timestamp}_{self.demo_count}.h5"
        filepath = self.output_dir / filename

        # Convert lists to numpy arrays
        actions = np.array(self.buffer["actions"], dtype=np.float32)

        # Save to HDF5 file
        try:
            with h5py.File(filepath, "w") as f:
                # Create a group for this trajectory
                traj_id = f"traj_{self.demo_count}"
                traj_group = f.create_group(traj_id)

                # Store actions
                traj_group.create_dataset("actions", data=actions)

                # Store timestamps
                timestamps = np.array(self.buffer["timestamps"], dtype=np.float32)
                traj_group.create_dataset("timestamps", data=timestamps)

                # Store states as a nested dictionary structure
                env_states_group = traj_group.create_group("env_states")
                for i, state in enumerate(self.buffer["env_states"]):
                    state_group = env_states_group.create_group(str(i))

                    # Store each component of the state
                    for key, value in state.items():
                        if isinstance(value, np.ndarray):
                            state_group.create_dataset(key, data=value)
                        elif isinstance(value, (int, float, bool)):
                            state_group.attrs[key] = value
                        elif isinstance(value, dict):
                            # Handle nested dictionaries recursively
                            nested_group = state_group.create_group(key)
                            for nested_key, nested_value in value.items():
                                if isinstance(nested_value, np.ndarray):
                                    nested_group.create_dataset(
                                        nested_key, data=nested_value
                                    )
                                else:
                                    nested_group.attrs[nested_key] = nested_value

            # Update metadata file
            self._update_metadata(filepath.name, len(actions))

            logger.info(f"Saved demonstration to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error saving demonstration: {e}")
            return None
        finally:
            self.reset_buffer()

    def _update_metadata(self, filename, num_actions):
        """Update metadata file with information about this demonstration

        Args:
            filename: Name of the h5 file
            num_actions: Number of actions in the demonstration
        """
        try:
            # Read existing metadata
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)

            # Add new episode entry
            episode_id = self.demo_count
            episode_info = {
                "episode_id": episode_id,
                "filename": filename,
                "num_actions": num_actions,
                "timestamp": time.time(),
                "reset_kwargs": {"seed": episode_id},
            }

            metadata["episodes"].append(episode_info)

            # Write updated metadata
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
