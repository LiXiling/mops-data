/**
 * WebSocket connection management for VR controller tracking
 */
import { debugLog, updateStatus } from './utils.js';
import { updateRecordingStatus } from './webxr.js';

// WebSocket connection to send data to Python
let socket = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 3000; // 3 seconds
let manuallyDisconnected = false;

/**
 * Setup WebSocket connection to the Python server
 */
export function setupWebSocket() {
    // Don't try to reconnect if we've manually closed
    if (manuallyDisconnected) {
        debugLog('WebSocket reconnection skipped - manually disconnected');
        return;
    }

    // Check reconnection attempts
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        debugLog(`Maximum reconnection attempts (${MAX_RECONNECT_ATTEMPTS}) reached. Giving up.`);
        updateStatus('WebSocket connection failed - reload page to try again');
        return;
    }

    try {
        // Always use wss:// for secure WebSockets when your page is served over HTTPS
        const wsProtocol = 'wss:';
        // Use the explicit IP address - update this to your server's IP
        const wsUrl = `${wsProtocol}//141.3.53.34:8765`;

        debugLog(`Attempting to connect to WebSocket at: ${wsUrl} (attempt ${reconnectAttempts + 1}/${MAX_RECONNECT_ATTEMPTS})`);
        updateStatus(`VR session active | Connecting to WebSocket (attempt ${reconnectAttempts + 1})...`);

        // Close existing socket if open
        if (socket && socket.readyState !== WebSocket.CLOSED) {
            debugLog('Closing existing socket connection before reconnecting');
            socket.close();
        }

        socket = new WebSocket(wsUrl);

        // Set a connection timeout
        const connectionTimeout = setTimeout(() => {
            if (socket && socket.readyState !== WebSocket.OPEN) {
                debugLog('WebSocket connection timeout');
                socket.close();
            }
        }, 10000); // 10 second timeout

        socket.onopen = () => {
            debugLog('WebSocket connection established');
            updateStatus('VR session active | WebSocket connected');
            reconnectAttempts = 0; // Reset reconnect attempts on successful connection
            clearTimeout(connectionTimeout);

            // Send a test message to verify connection is working
            sendPing();
        };

        socket.onclose = (event) => {
            clearTimeout(connectionTimeout);
            debugLog(`WebSocket connection closed: Code ${event.code} - ${event.reason || 'No reason provided'}`);
            updateStatus('VR session active | WebSocket disconnected');

            // Try to reconnect after a delay, with increasing backoff
            if (!manuallyDisconnected) {
                reconnectAttempts++;
                const delay = RECONNECT_DELAY * reconnectAttempts;
                debugLog(`Will attempt to reconnect in ${delay / 1000} seconds`);
                setTimeout(setupWebSocket, delay);
            }
        };

        socket.onerror = (error) => {
            debugLog(`WebSocket error: ${error}`);
            // Add detailed error information to debug log
            if (error.message) {
                debugLog(`Error message: ${error.message}`);
            }
            updateStatus('VR session active | WebSocket error');
            // Don't need to do anything here - the onclose handler will be called
        };

        socket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);

                // Handle server messages
                if (message.type === 'ping') {
                    sendMessage({ type: 'pong', timestamp: Date.now() });
                } else if (message.type === 'pong') {
                    debugLog('Received pong from server');
                } else if (message.type === 'recording_status') {
                    // Handle recording status update message
                    if (message.recording_active !== undefined) {
                        debugLog(`Received recording status: ${message.recording_active ? 'active' : 'inactive'}`);
                        updateRecordingStatus(message.recording_active, message.recording_count || 0);
                    }
                }
            } catch (e) {
                debugLog(`Error processing message from server: ${e.message}`);
            }
        };
    } catch (error) {
        debugLog(`Error setting up WebSocket: ${error.message}`);
        updateStatus('Error setting up WebSocket');

        // Try to reconnect after a delay
        if (!manuallyDisconnected) {
            reconnectAttempts++;
            const delay = RECONNECT_DELAY * reconnectAttempts;
            debugLog(`Will attempt to reconnect in ${delay / 1000} seconds`);
            setTimeout(setupWebSocket, delay);
        }
    }
}

/**
 * Send a ping message to the server to check connection
 */
function sendPing() {
    sendMessage({ type: 'ping', timestamp: Date.now() });
}

/**
 * Send a message via WebSocket
 * @param {Object} data - Data to send
 */
export function sendMessage(data) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(data));
    }
}

/**
 * Send controller data via WebSocket
 * @param {Object} data - Controller data to send
 */
export function sendControllerData(data) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        // Add a timestamp to the data
        const dataWithTimestamp = {
            ...data,
            timestamp: Date.now()
        };
        socket.send(JSON.stringify(dataWithTimestamp));
    }
}

/**
 * Manually disconnect WebSocket and prevent automatic reconnection
 */
export function disconnectWebSocket() {
    manuallyDisconnected = true;
    if (socket) {
        debugLog('Manually disconnecting WebSocket');
        socket.close();
    }
}

/**
 * Reset the reconnection state and allow reconnections
 */
export function resetWebSocketState() {
    manuallyDisconnected = false;
    reconnectAttempts = 0;
}
