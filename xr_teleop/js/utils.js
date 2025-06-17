/**
 * Utility functions for the WebXR controller tracker
 */

// Global UI elements
export const elements = {
    vrButton: document.getElementById('vr-button'),
    statusElement: document.getElementById('status'),
    leftControllerElement: document.getElementById('left-controller'),
    rightControllerElement: document.getElementById('right-controller'),
    rawDataElement: document.getElementById('raw-data'),
    canvas: document.getElementById('canvas'),
    debugLog: document.getElementById('debug-log')
};

/**
 * Debug logging function - logs to console and UI
 * @param {string} message - The message to log
 */
export function debugLog(message) {
    console.log(message);

    // Also show in the UI for debugging on Quest
    if (elements.debugLog) {
        const timestamp = new Date().toLocaleTimeString();
        elements.debugLog.innerHTML += `<div>${timestamp}: ${message}</div>`;

        // Keep only the last 10 messages
        const messages = elements.debugLog.children;
        if (messages.length > 10) {
            elements.debugLog.removeChild(messages[0]);
        }

        // Scroll to bottom
        elements.debugLog.scrollTop = elements.debugLog.scrollHeight;
    }
}

/**
 * Update the status message
 * @param {string} message - The status message
 */
export function updateStatus(message) {
    if (elements.statusElement) {
        elements.statusElement.textContent = message;
    }
    debugLog(message);
}
