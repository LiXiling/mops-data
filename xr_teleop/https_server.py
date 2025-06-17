import http.server
import os
import socketserver
import ssl
import subprocess
import webbrowser
from pathlib import Path

# Configuration
HOST = "141.3.53.34"  # Your server IP
PORT = 443  # HTTPS port
CERT_DIR = Path("certificates")  # Directory to store certificates


def generate_self_signed_cert():
    """Generate self-signed certificates for HTTPS."""
    print("Generating self-signed certificates...")

    # Create certificates directory if it doesn't exist
    CERT_DIR.mkdir(exist_ok=True)

    key_file = CERT_DIR / "server.key"
    cert_file = CERT_DIR / "server.crt"

    # Check if certificates already exist
    if key_file.exists() and cert_file.exists():
        print("Certificates already exist. Using existing certificates.")
        return key_file, cert_file

    # Generate key and certificate
    try:
        # Generate private key
        subprocess.run(["openssl", "genrsa", "-out", str(key_file), "2048"], check=True)

        # Generate self-signed certificate
        subprocess.run(
            [
                "openssl",
                "req",
                "-new",
                "-x509",
                "-key",
                str(key_file),
                "-out",
                str(cert_file),
                "-days",
                "365",
                "-subj",
                f"/CN={HOST}",
            ],
            check=True,
        )

        print(f"Certificates generated at {CERT_DIR}")
        return key_file, cert_file

    except subprocess.CalledProcessError as e:
        print(f"Error generating certificates: {e}")
        print("Try running this script with administrator privileges.")
        raise


def run_https_server(key_file, cert_file):
    """Run an HTTPS server with the generated certificates."""

    # Create SSL context
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    # Create HTTPS server
    handler = http.server.SimpleHTTPRequestHandler

    print(f"Starting HTTPS server at https://{HOST}:{PORT}")
    print("Note: You'll need to accept the security warning in your browser")
    print("since we're using a self-signed certificate.")
    print("Press Ctrl+C to stop the server.")

    # Create server with SSL context
    with socketserver.TCPServer((HOST, PORT), handler) as httpd:
        httpd.socket = ssl_context.wrap_socket(httpd.socket, server_side=True)
        try:
            # Open browser (optional)
            # webbrowser.open(f"https://{HOST}:{PORT}")
            # Start server
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    # Create simple index.html if it doesn't exist
    if not Path("index.html").exists():
        print("WebXR page not found. Please save your WebXR page as 'index.html'.")
        exit(1)

    # Generate certificates and run server
    try:
        key_file, cert_file = generate_self_signed_cert()
        run_https_server(key_file, cert_file)
    except Exception as e:
        print(f"Error: {e}")
