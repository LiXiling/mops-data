#!/usr/bin/env python3
"""
Simplified Robot Bridge - Main entry point
Connects WebXR controller data to robot control
"""
import argparse
import asyncio
import logging
import threading

import mani_skill  # Import to load registered envs
from config import logger, running
from core.robot_controller import RobotController
from websocket_server import find_certificates, start_server

import mops_data  # Import to load registered envs


def run_robot_controller(env_id):
    """
    Run the robot controller in a separate thread

    Args:
        env_id: ManiSkill environment ID
    """
    try:
        controller = RobotController(env_id=env_id)
        controller.run_simulation()
    except Exception as e:
        logger.error(f"Error in robot controller: {e}")
        global running
        running = False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Robot Bridge")

    # Environment settings
    parser.add_argument(
        "--env-id", default="OpenCabinetDrawer-v2", help="ManiSkill environment ID"
    )

    # Connection settings
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket host address")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")

    # SSL options
    parser.add_argument("--ssl", action="store_true", help="Enable SSL/TLS (WSS)")
    parser.add_argument("--cert", help="SSL certificate file path")
    parser.add_argument("--key", help="SSL key file path")

    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


async def main_async(args):
    """Async main function to run the WebSocket server"""
    await start_server(
        host=args.host,
        port=args.port,
        use_ssl=args.ssl,
        cert_file=args.cert,
        key_file=args.key,
    )


def main():
    """Main application entry point"""
    # Parse command line arguments
    args = parse_arguments()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Handle SSL certificate discovery if needed
    if args.ssl and not (args.cert and args.key):
        cert_file, key_file = find_certificates()
        if cert_file and key_file:
            args.cert = cert_file
            args.key = key_file
            logger.info(f"Using discovered certificates: {cert_file}, {key_file}")
        else:
            logger.error("SSL enabled but no certificates found or provided")
            return

    # Start robot controller in a separate thread
    logger.info(f"Starting robot controller with environment: {args.env_id}")
    robot_thread = threading.Thread(target=run_robot_controller, args=(args.env_id,))
    robot_thread.daemon = True
    robot_thread.start()

    # Start WebSocket server in the main thread
    try:
        logger.info(f"Starting WebSocket server on {args.host}:{args.port}")
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        global running
        running = False
    except Exception as e:
        logger.error(f"Error running server: {e}")
        running = False

    # Wait for robot thread to finish
    robot_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
