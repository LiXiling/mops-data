/**
 * Controller tracking and data management with enhanced debugging
 */
import { debugLog } from './utils.js';

// Controller references and data
export let leftController = null;
export let rightController = null;

// Controller data structure (matches Python client expectations)
export let controllerData = {
    left: {
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        buttons: {},
        axes: {},
        debug: {
            triggerValue: 0,
            gripValue: 0,
            primaryButtonPressed: false,  // A/X button (usually button 4)
            secondaryButtonPressed: false  // B/Y button (usually button 5)
        }
    },
    right: {
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        buttons: {},
        axes: {},
        debug: {
            triggerValue: 0,
            gripValue: 0,
            primaryButtonPressed: false,  // A/X button (usually button 4)
            secondaryButtonPressed: false  // B/Y button (usually button 5)
        }
    }
};

/**
 * Handle input sources changing (controllers connected/disconnected)
 * @param {XRInputSourcesChangeEvent} event - The input sources change event
 */
export function onInputSourcesChange(event) {
    // Added input sources
    event.added.forEach(inputSource => {
        if (inputSource.handedness === 'left') {
            debugLog("Left controller added");
            leftController = inputSource;
        } else if (inputSource.handedness === 'right') {
            debugLog("Right controller added");
            rightController = inputSource;
        }
    });

    // Removed input sources
    event.removed.forEach(inputSource => {
        if (inputSource === leftController) {
            debugLog("Left controller removed");
            leftController = null;
        } else if (inputSource === rightController) {
            debugLog("Right controller removed");
            rightController = null;
        }
    });
}

/**
 * Update controller data from the current frame
 * @param {string} hand - 'left' or 'right'
 * @param {XRInputSource} controller - The controller input source
 * @param {XRFrame} frame - The current XR frame
 * @param {XRReferenceSpace} referenceSpace - The reference space
 */
export function updateControllerData(hand, controller, frame, referenceSpace) {
    const pose = frame.getPose(controller.gripSpace, referenceSpace);

    if (pose) {
        // Position
        const position = pose.transform.position;
        controllerData[hand].position = [
            parseFloat(position.x.toFixed(4)),
            parseFloat(position.y.toFixed(4)),
            parseFloat(position.z.toFixed(4))
        ];

        // Orientation (quaternion) - using 'rotation' key to match Python client
        const orientation = pose.transform.orientation;
        controllerData[hand].rotation = [
            parseFloat(orientation.x.toFixed(4)),
            parseFloat(orientation.y.toFixed(4)),
            parseFloat(orientation.z.toFixed(4)),
            parseFloat(orientation.w.toFixed(4))
        ];
    }

    // Gamepad data (buttons, triggers, thumbsticks, etc.)
    if (controller.gamepad) {
        // Buttons
        controller.gamepad.buttons.forEach((button, index) => {
            controllerData[hand].buttons[`button_${index}`] = {
                pressed: button.pressed,
                touched: button.touched,
                value: parseFloat(button.value.toFixed(4))
            };

            // Enhanced debugging - track common buttons specifically
            // Standard mapping: 0 = trigger, 1 = grip, 2-3 = joystick/touchpad buttons, 4 = A/X, 5 = B/Y buttons
            if (index === 0) {
                // Trigger
                controllerData[hand].debug.triggerValue = button.value;
            } else if (index === 1) {
                // Grip
                controllerData[hand].debug.gripValue = button.value;
            } else if (index === 4) {
                // Primary button (A/X)
                controllerData[hand].debug.primaryButtonPressed = button.pressed;
                // Add an easier access key in the main buttons object for the Python code
                controllerData[hand].buttons.primary_pressed = button.pressed;
            } else if (index === 5) {
                // Secondary button (B/Y)
                controllerData[hand].debug.secondaryButtonPressed = button.pressed;
                // Add an easier access key in the main buttons object for the Python code
                controllerData[hand].buttons.secondary_pressed = button.pressed;
            }
        });

        // Axes (thumbsticks, etc.)
        controller.gamepad.axes.forEach((axis, index) => {
            controllerData[hand].axes[`axis_${index}`] = parseFloat(axis.toFixed(4));
        });
    }
}

/**
 * Update the controller display in the UI
 */
export function updateControllerDisplay(elements) {
    // Left controller
    if (leftController) {
        elements.leftControllerElement.textContent =
            `Position: [${controllerData.left.position.join(', ')}]\n` +
            `Rotation: [${controllerData.left.rotation.join(', ')}]\n` +
            `Trigger: ${(controllerData.left.debug.triggerValue * 100).toFixed(1)}%\n` +
            `Grip: ${(controllerData.left.debug.gripValue * 100).toFixed(1)}%\n` +
            `X Button: ${controllerData.left.debug.primaryButtonPressed ? 'PRESSED' : 'released'}\n` +
            `Y Button: ${controllerData.left.debug.secondaryButtonPressed ? 'PRESSED' : 'released'}`;
    } else {
        elements.leftControllerElement.textContent = 'Left controller not detected';
    }

    // Right controller
    if (rightController) {
        elements.rightControllerElement.textContent =
            `Position: [${controllerData.right.position.join(', ')}]\n` +
            `Rotation: [${controllerData.right.rotation.join(', ')}]\n` +
            `Trigger: ${(controllerData.right.debug.triggerValue * 100).toFixed(1)}%\n` +
            `Grip: ${(controllerData.right.debug.gripValue * 100).toFixed(1)}%\n` +
            `A Button: ${controllerData.right.debug.primaryButtonPressed ? 'PRESSED' : 'released'}\n` +
            `B Button: ${controllerData.right.debug.secondaryButtonPressed ? 'PRESSED' : 'released'}`;
    } else {
        elements.rightControllerElement.textContent = 'Right controller not detected';
    }

    // Raw data
    elements.rawDataElement.textContent = JSON.stringify(controllerData, null, 2);
}

/**
 * Get the current controller data
 * @returns {Object} The controller data
 */
export function getControllerData() {
    return controllerData;
}
