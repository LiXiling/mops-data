/**
 * Main application entry point (VR Only)
 */
import { elements, debugLog } from './utils.js';
import { initRenderer, startAnimationLoop } from './renderer.js';
import { initWebXR, onXRFrame } from './webxr.js';

// Make renderer globally accessible for legacy code
window.renderer = null;

// Initialize application when page loads
window.addEventListener('load', () => {
    debugLog("Page loaded, initializing VR application...");

    // Initialize Three.js renderer
    window.renderer = initRenderer(elements);

    // No longer adding Three.js VRButton - using our custom button instead

    // Initialize WebXR
    initWebXR();

    // Start animation loop
    startAnimationLoop(onXRFrame);

    debugLog("VR application initialized successfully");
});
