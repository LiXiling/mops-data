#!/usr/bin/env python3
"""
Utilities for processing WebXR controller data
"""
from config import controller_data, logger


def process_controller_message(data):
    """
    Process controller data from WebXR message

    Args:
        data: JSON data from WebXR WebSocket message
    """
    # Skip empty data
    if not data:
        return

    # Process each hand controller if present
    for hand in ["left", "right"]:
        if hand not in data:
            continue

        # Get data for the current hand
        hand_data = data[hand]

        # Update position if available
        if "position" in hand_data:
            controller_data[hand]["position"] = [
                float(pos) for pos in hand_data["position"]
            ]

        # Update rotation/orientation if available
        # Support both "rotation" and "orientation" keys for compatibility
        if "rotation" in hand_data:
            controller_data[hand]["rotation"] = [
                float(rot) for rot in hand_data["rotation"]
            ]
        elif "orientation" in hand_data:
            controller_data[hand]["rotation"] = [
                float(rot) for rot in hand_data["orientation"]
            ]

        # Process buttons
        process_buttons(hand, hand_data)

        # Also check debug info if available (for compatibility with some clients)
        process_debug_data(hand, hand_data)


def process_buttons(hand, hand_data):
    """
    Process button data from controller

    Args:
        hand: 'left' or 'right'
        hand_data: Controller data for the hand
    """
    # Get target dictionary for this hand
    controller_buttons = controller_data[hand]["buttons"]

    # Skip if no button data
    if "buttons" not in hand_data:
        return

    buttons = hand_data["buttons"]

    # Process grip button (usually index 1)
    extract_button_state(buttons, "button_1", controller_buttons, "grip_pressed")

    # Process trigger (usually index 0)
    extract_button_value(buttons, "button_0", controller_buttons, "trigger_value")

    # Process primary button (A/X) (try multiple possible indices)
    extract_button_state(buttons, "button_4", controller_buttons, "primary_pressed")

    # Process secondary button (B/Y) (try multiple possible indices)
    extract_button_state(buttons, "button_5", controller_buttons, "secondary_pressed")

    # Direct access to primary/secondary pressed from the buttons object (added in controllers.js)
    if "primary_pressed" in buttons:
        controller_buttons["primary_pressed"] = buttons["primary_pressed"]

    if "secondary_pressed" in buttons:
        controller_buttons["secondary_pressed"] = buttons["secondary_pressed"]


def process_debug_data(hand, hand_data):
    """
    Process debug data often included in WebXR messages

    Args:
        hand: 'left' or 'right'
        hand_data: Controller data for the hand
    """
    # Skip if no debug data
    if "debug" not in hand_data:
        return

    debug = hand_data["debug"]
    controller_buttons = controller_data[hand]["buttons"]

    # Get grip state
    if "gripValue" in debug and debug["gripValue"] > 0.5:
        controller_buttons["grip_pressed"] = True

    # Get trigger value
    if "triggerValue" in debug:
        controller_buttons["trigger_value"] = float(debug["triggerValue"])

    # Get primary button state (A/X)
    if "primaryButtonPressed" in debug:
        controller_buttons["primary_pressed"] = debug["primaryButtonPressed"]

    # Get secondary button state (B/Y)
    if "secondaryButtonPressed" in debug:
        controller_buttons["secondary_pressed"] = debug["secondaryButtonPressed"]


def extract_button_state(buttons, button_key, target_dict, target_key):
    """
    Extract button pressed state

    Args:
        buttons: Dictionary of buttons
        button_key: Key of the button to extract
        target_dict: Dictionary to store result in
        target_key: Key to store result under

    Returns:
        bool: True if button was found and extracted
    """
    if button_key not in buttons:
        return False

    button_data = buttons[button_key]

    # Handle different button data formats
    if isinstance(button_data, dict):
        target_dict[target_key] = button_data.get("pressed", False)
    elif isinstance(button_data, (int, float, bool)):
        target_dict[target_key] = bool(button_data)

    return True


def extract_button_value(buttons, button_key, target_dict, target_key):
    """
    Extract button value (for triggers and analog inputs)

    Args:
        buttons: Dictionary of buttons
        button_key: Key of the button to extract
        target_dict: Dictionary to store result in
        target_key: Key to store result under

    Returns:
        bool: True if button was found and extracted
    """
    if button_key not in buttons:
        return False

    button_data = buttons[button_key]

    # Handle different button data formats
    if isinstance(button_data, dict):
        target_dict[target_key] = float(button_data.get("value", 0.0))
    elif isinstance(button_data, (int, float)):
        target_dict[target_key] = float(button_data)

    return True
