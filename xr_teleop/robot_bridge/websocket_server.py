#!/usr/bin/env python3
"""
WebSocket server for receiving WebXR controller data
"""
import asyncio
import json
import ssl
from pathlib import Path

from config import logger, recording_active, recording_counter, recording_feedback
from utils.controller_utils import process_controller_message
from websockets.exceptions import ConnectionClosed, InvalidMessage

# Global WebSocket connections
active_connections = set()


async def websocket_handler(websocket):
    """
    Handle WebSocket connections from the WebXR browser

    Args:
        websocket: WebSocket connection
    """
    message_count = 0

    # Add connection to global set
    active_connections.add(websocket)

    client = (
        websocket.remote_address
        if hasattr(websocket, "remote_address")
        else ("Unknown", 0)
    )
    logger.info(f"Client connected from {client[0]}:{client[1]}")

    try:
        # Initial status update
        await send_recording_status(websocket)

        async for message in websocket:
            try:
                message_count += 1
                data = json.loads(message)

                # Log data occasionally to avoid console spam
                if message_count % 100 == 0:
                    logger.info(f"Processed {message_count} messages")

                # Process controller data for both hands
                process_controller_message(data)

                # Check if we need to send recording status updates
                global recording_feedback
                if recording_feedback:
                    recording_feedback = False
                    await send_recording_status_to_all()

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {message[:100]}...")
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
    except ConnectionClosed:
        logger.info(f"Client {client[0]}:{client[1]} disconnected")
    except InvalidMessage:
        logger.warning(f"Invalid WebSocket message from {client[0]}:{client[1]}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Remove connection from set when done
        active_connections.remove(websocket)


async def send_recording_status(websocket):
    """
    Send recording status to a specific WebSocket client

    Args:
        websocket: WebSocket connection to send to
    """
    try:
        global recording_active, recording_counter
        message = {
            "type": "recording_status",
            "recording_active": recording_active,
            "recording_count": recording_counter,
        }
        await websocket.send(json.dumps(message))
    except Exception as e:
        logger.error(f"Error sending recording status: {str(e)}")


async def send_recording_status_to_all():
    """
    Send recording status to all connected WebSocket clients
    """
    global recording_active, recording_counter, active_connections

    if not active_connections:
        return

    message = {
        "type": "recording_status",
        "recording_active": recording_active,
        "recording_count": recording_counter,
    }

    json_message = json.dumps(message)

    # Create tasks for each connection
    send_tasks = []
    for websocket in active_connections:
        try:
            send_tasks.append(asyncio.create_task(websocket.send(json_message)))
        except Exception as e:
            logger.error(f"Error creating send task: {str(e)}")

    # Wait for all tasks to complete
    if send_tasks:
        try:
            await asyncio.gather(*send_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error sending status to clients: {str(e)}")


async def start_server(
    host="0.0.0.0", port=8765, use_ssl=False, cert_file=None, key_file=None
):
    """
    Start the WebSocket server

    Args:
        host: Host address to bind to
        port: Port to bind to
        use_ssl: Whether to use SSL
        cert_file: SSL certificate file path
        key_file: SSL key file path
    """
    import websockets

    ssl_context = None

    if use_ssl:
        if not cert_file or not key_file:
            logger.error("SSL enabled but certificate or key file not provided")
            return

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
            logger.info(f"SSL enabled with cert: {cert_file}, key: {key_file}")
        except Exception as e:
            logger.error(f"Error loading SSL certificates: {str(e)}")
            return

    try:
        server = await websockets.serve(
            websocket_handler,
            host,
            port,
            ssl=ssl_context,
            ping_interval=30,
            ping_timeout=10,
            max_size=10_485_760,
            close_timeout=10,
        )

        protocol = "wss" if use_ssl else "ws"
        logger.info(f"WebSocket server running at {protocol}://{host}:{port}")
        logger.info("Waiting for connections...")

        # Keep the server running
        await asyncio.Future()  # Run forever
    except OSError as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(
            "Check if the port is already in use or if you have permission to bind to this address"
        )
    except Exception as e:
        logger.error(f"Unexpected error starting server: {str(e)}")


def find_certificates():
    """
    Find certificates from the HTTPS server

    Returns:
        tuple: (cert_file, key_file) or (None, None) if not found
    """
    cert_dir = Path("certificates")
    cert_file = cert_dir / "server.crt"
    key_file = cert_dir / "server.key"

    if cert_file.exists() and key_file.exists():
        logger.info(f"Found certificates at {cert_dir}")
        return str(cert_file), str(key_file)

    return None, None
