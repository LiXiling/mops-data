/**
 * WebXR session management with passthrough support
 */
import { debugLog, updateStatus, elements } from './utils.js';
import { scene } from './renderer.js';
import { setupWebSocket } from './websocket.js';
import { leftController, rightController, onInputSourcesChange, updateControllerData, updateControllerDisplay, getControllerData } from './controllers.js';
import { updateControllerVisualization } from './renderer.js';
import { sendControllerData } from './websocket.js';

// XR session and reference space
let xrSession = null;
let xrReferenceSpace = null;
let usePassthrough = true; // Set to true to enable passthrough
let controllerSendInterval = null; // Interval for sending controller data

// Recording status display
let recordingActive = false;
let recordingCount = 0;
let recordingStartTime = null;

/**
 * Initialize WebXR for VR/AR with passthrough
 */
export function initWebXR() {
    debugLog("Initializing WebXR with passthrough support...");

    // Check if WebXR is available
    if (!navigator.xr) {
        debugLog("WebXR not available in this browser");
        updateStatus('WebXR not available in your browser');
        elements.vrButton.disabled = true;
        return;
    }

    // Add a toggle for passthrough mode
    const passthroughToggle = document.createElement('button');
    passthroughToggle.textContent = "Toggle Passthrough";
    passthroughToggle.style.marginLeft = "10px";
    passthroughToggle.addEventListener('click', () => {
        usePassthrough = !usePassthrough;
        debugLog(`Passthrough mode ${usePassthrough ? 'enabled' : 'disabled'}`);
        passthroughToggle.textContent = usePassthrough ? "Disable Passthrough" : "Enable Passthrough";
    });
    elements.vrButton.parentNode.insertBefore(passthroughToggle, elements.vrButton.nextSibling);

    // Create recording status display
    const recordingStatus = document.createElement('div');
    recordingStatus.id = 'recording-status';
    recordingStatus.style.marginTop = '10px';
    recordingStatus.style.padding = '8px';
    recordingStatus.style.borderRadius = '4px';
    recordingStatus.style.backgroundColor = '#333';
    recordingStatus.style.color = 'white';
    recordingStatus.style.display = 'none';
    recordingStatus.textContent = 'Recording: Not Active';
    elements.vrButton.parentNode.insertBefore(recordingStatus, passthroughToggle.nextSibling);
    elements.recordingStatus = recordingStatus;

    // Set up custom VR button handling
    elements.vrButton.addEventListener('click', () => {
        debugLog("VR button clicked");
        startXRSession();
    });

    // Check VR and passthrough support
    checkSessionSupport();
}

/**
 * Check support for different XR modes
 */
function checkSessionSupport() {
    // Check immersive-vr support (fallback option)
    navigator.xr.isSessionSupported('immersive-vr')
        .then((supported) => {
            if (supported) {
                debugLog("VR is supported on this device");
                elements.vrButton.disabled = false;

                // Now check for passthrough support
                navigator.xr.isSessionSupported('immersive-ar')
                    .then((arSupported) => {
                        if (arSupported) {
                            debugLog("AR mode is supported");
                            // Try to detect passthrough capability
                            elements.vrButton.textContent = "Start Mixed Reality";
                            updateStatus('Mixed Reality ready - press button to begin');
                        } else {
                            debugLog("AR not supported, using VR mode only");
                            elements.vrButton.textContent = "Enter VR";
                            updateStatus('VR ready - press Enter VR to begin');
                        }
                    })
                    .catch(error => {
                        debugLog("Error checking AR support: " + error.message);
                        elements.vrButton.textContent = "Enter VR";
                    });
            } else {
                debugLog("VR is not supported on this device");
                updateStatus('VR not supported on this device');
                elements.vrButton.disabled = true;
            }
        })
        .catch(error => {
            debugLog("Error checking XR support: " + error.message);
            updateStatus('Error checking WebXR support');
            elements.vrButton.disabled = true;
        });
}

/**
 * Request passthrough session for Meta Quest 3
 * This uses the Meta Quest specific method if available
 */
function requestQuestPassthrough() {
    debugLog("Attempting to start Quest-specific passthrough...");

    // First, check if we have the Oculus browser-specific API
    if (window.OculusXRSession) {
        debugLog("Found Oculus specific XR session API");

        // Oculus Browser specific approach for better passthrough
        const sessionInit = {
            requiredFeatures: ['local-floor'],
            optionalFeatures: ['hand-tracking'],
            domOverlay: { root: document.getElementById('overlay-content') },
            // Quest 3 specific passthrough
            environmentBlendMode: 'alpha-blend'
        };

        try {
            // This checks if we're in an Oculus Browser
            window.OculusXRSession.requestSession('immersive-ar', sessionInit)
                .then(onSessionStarted)
                .catch(error => {
                    debugLog(`Oculus passthrough error: ${error.name}, ${error.message}`);
                    tryStandardPassthrough();
                });
        } catch (err) {
            debugLog("Oculus API available but failed, trying standard approach");
            tryStandardPassthrough();
        }
    } else {
        // No Oculus specific API, try standard approach
        tryStandardPassthrough();
    }
}

/**
 * Try standard WebXR passthrough approach
 */
function tryStandardPassthrough() {
    debugLog("Trying standard WebXR passthrough...");

    const sessionInit = {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hand-tracking', 'plane-detection'],
        domOverlay: { root: document.getElementById('overlay-content') }
    };

    navigator.xr.requestSession('immersive-ar', sessionInit)
        .then(onSessionStarted)
        .catch(error => {
            debugLog(`Standard passthrough failed: ${error.name}, ${error.message}`);
            startVRFallback();
        });
}

/**
 * Start VR or AR Session with passthrough if available
 */
function startXRSession() {
    // Try passthrough AR first if enabled
    if (usePassthrough) {
        debugLog("Attempting to start passthrough session...");

        // Check if we're on Quest device (approximate detection)
        const userAgent = navigator.userAgent;
        const isQuestBrowser = userAgent.includes('Quest') || userAgent.includes('Oculus');

        if (isQuestBrowser) {
            debugLog("Quest browser detected, using specific passthrough approach");
            requestQuestPassthrough();
        } else {
            debugLog("Non-Quest browser, using standard approach");
            tryStandardPassthrough();
        }
    } else {
        // Use regular VR as requested
        startVRFallback();
    }
}

/**
 * Fall back to regular VR mode
 */
function startVRFallback() {
    debugLog("Starting regular VR session...");

    navigator.xr.requestSession('immersive-vr', {
        requiredFeatures: ['local-floor'],
        optionalFeatures: ['hand-tracking']
    })
        .then(onSessionStarted)
        .catch(error => {
            debugLog(`Error starting VR session: ${error.message}`);
            updateStatus(`Error starting VR: ${error.message}`);
        });
}

/**
 * On XR Session Started (either VR or AR with passthrough)
 * @param {XRSession} session - The XR session
 */
function onSessionStarted(session) {
    debugLog(`XR session started successfully (${session.mode})`);
    xrSession = session;

    if (session.mode === 'immersive-ar') {
        updateStatus("Passthrough session active");
        // For passthrough, we need a transparent background
        scene.background = null;
    } else {
        updateStatus("VR session active");
        // For VR, keep the solid background
        scene.background = null; // Use transparent for better passthrough support
    }

    // Show recording status
    if (elements.recordingStatus) {
        elements.recordingStatus.style.display = 'block';
        updateRecordingStatus(false);
    }

    // Setup WebSocket when session starts
    setupWebSocket();

    // Setup input sources
    debugLog("Setting up input sources...");
    session.addEventListener('inputsourceschange', onInputSourcesChange);

    // Set up XR rendering
    debugLog("Setting up XR rendering...");
    if (window.renderer) {
        window.renderer.xr.enabled = true;
        window.renderer.xr.setSession(session);
    }

    // Get reference space for tracking
    debugLog("Requesting reference space...");
    session.requestReferenceSpace('local')
        .then((referenceSpace) => {
            debugLog("Reference space acquired");
            xrReferenceSpace = referenceSpace;

            // Start sending controller data at a consistent rate
            // This ensures our robot control doesn't get overwhelmed with updates
            if (controllerSendInterval) {
                clearInterval(controllerSendInterval);
            }

            // Send controller data every 100ms (10 times per second)
            controllerSendInterval = setInterval(() => {
                sendControllerData(getControllerData());
            }, 100);

        })
        .catch(error => {
            debugLog("Error getting reference space: " + error.message);
        });

    session.addEventListener('end', onSessionEnded);
}

/**
 * Updates the recording status display
 * @param {boolean} isRecording - Whether recording is active
 */
export function updateRecordingStatus(isRecording, count = 0) {
    if (!elements.recordingStatus) return;

    recordingActive = isRecording;

    if (isRecording) {
        if (!recordingStartTime) {
            recordingStartTime = Date.now();
        }

        recordingCount = count;
        const elapsedSeconds = Math.floor((Date.now() - recordingStartTime) / 1000);
        elements.recordingStatus.textContent = `Recording #${count}: ${elapsedSeconds}s`;
        elements.recordingStatus.style.backgroundColor = '#cc0000';

        // Update DOM overlay in VR
        if (document.getElementById('overlay-status')) {
            document.getElementById('overlay-status').textContent = `Recording #${count}: ${elapsedSeconds}s`;
            document.getElementById('overlay-status').style.backgroundColor = 'rgba(204, 0, 0, 0.7)';
        }
    } else {
        recordingStartTime = null;
        elements.recordingStatus.textContent = 'Recording: Not Active';
        elements.recordingStatus.style.backgroundColor = '#333';

        // Update DOM overlay in VR
        if (document.getElementById('overlay-status')) {
            document.getElementById('overlay-status').textContent = 'Ready (B=Start, A=Save)';
            document.getElementById('overlay-status').style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        }
    }
}

/**
 * On XR Session Ended
 */
function onSessionEnded() {
    debugLog("XR session ended");
    xrSession = null;
    updateStatus("XR session ended");

    // Hide recording status
    if (elements.recordingStatus) {
        elements.recordingStatus.style.display = 'none';
    }

    // Stop sending controller data
    if (controllerSendInterval) {
        clearInterval(controllerSendInterval);
        controllerSendInterval = null;
    }
}

/**
 * Handle XR Frame Updates
 * @param {number} time - The current time
 * @param {XRFrame} frame - The current XR frame
 */
export function onXRFrame(time, frame) {
    const session = frame.session;
    const referenceSpace = xrReferenceSpace;

    if (!referenceSpace) {
        return;
    }

    // Update controller visualization
    updateControllerVisualization(frame, referenceSpace);

    // Update controller data
    if (leftController) {
        updateControllerData('left', leftController, frame, referenceSpace);
    }

    if (rightController) {
        updateControllerData('right', rightController, frame, referenceSpace);
    }

    // Update UI display
    updateControllerDisplay(elements);

    // Update recording duration if recording is active
    if (recordingActive && recordingStartTime && frame.frameCount % 60 === 0) {
        const elapsedSeconds = Math.floor((Date.now() - recordingStartTime) / 1000);
        if (elements.recordingStatus) {
            elements.recordingStatus.textContent = `Recording #${recordingCount}: ${elapsedSeconds}s`;
        }

        // Update DOM overlay in VR
        if (document.getElementById('overlay-status')) {
            document.getElementById('overlay-status').textContent = `Recording #${recordingCount}: ${elapsedSeconds}s`;
        }
    }

    // We don't need to send data here anymore, as it's handled by the interval
    // This prevents flooding the WebSocket with too much data
    // sendControllerData(getControllerData());
}
