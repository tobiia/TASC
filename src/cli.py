import argparse
import os
import sys
import time
import signal
import socket
import webbrowser
from colorama import init, Fore

init(autoreset=True)

from src.config import PORT_BACKEND, PORT_FRONTEND, PROJECT_ROOT, FRONTEND
from src.process import popen


def wait_for_port(port, timeout=300, interval=0.25):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return True
        except OSError:
            time.sleep(interval)

    return False


def start_backend(dev=False):
    print(Fore.CYAN + "\nStarting backend...")
    env = os.environ.copy()
    if dev:
        env["TASC_DEV"] = "1"
    return popen(
        "backend (uvicorn)",
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.backend.app.main:app",
            "--port",
            str(PORT_BACKEND),
        ],
        cwd=PROJECT_ROOT,
        env=env,
    )


def start_frontend():
    print(Fore.CYAN + "\nStarting frontend...")
    npm = "npm.cmd" if sys.platform == "win32" else "npm"

    return popen(
        "frontend (vite/npm)",
        [npm, "run", "dev"],
        cwd=FRONTEND,
        shell=(sys.platform == "win32"),
    )


def main():

    print(Fore.CYAN + "*" * 60)
    print(Fore.CYAN + "TASC")
    print(Fore.CYAN + "*" * 60)

    parser = argparse.ArgumentParser(
        prog="tasc", description="TASC - Topic-Aware Semantic Change"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run", help="Start production server (backend only)")
    subparsers.add_parser("dev", help="Start full dev environment (frontend + backend)")

    args = parser.parse_args()

    # QUIT HERE IF INCORRECT OR W/E
    def shutdown(sig=None, frame=None):
        print(Fore.CYAN + "\nShutting down...")

        if frontend:
            frontend.kill()

        backend.kill()

        sys.exit(0)

    dev = args.command == "dev"
    backend = start_backend(dev)

    if not wait_for_port(PORT_BACKEND):
        print(Fore.YELLOW + "\nBackend did not start in time.")
        shutdown()

    frontend = None

    if args.command == "dev":
        frontend = start_frontend()

        url = f"http://localhost:{PORT_FRONTEND}"
        print(Fore.GREEN + f"\nBackend: http://localhost:{PORT_BACKEND}")
        print(Fore.GREEN + f"\nFrontend: http://localhost:{PORT_FRONTEND}")
        print(Fore.CYAN + "\nPress Ctrl+C to stop both.")
        webbrowser.open_new_tab(url)

    elif args.command == "run":
        url = f"http://localhost:{PORT_BACKEND}"
        print(Fore.GREEN + f"\nTASC running at {url}")
        print(Fore.CYAN + "\nPress Ctrl+C to stop.\n")
        webbrowser.open_new_tab(url)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    # TODO - detect when browser closes/process killed
    # and kill server
    try:
        while True:
            if backend and not backend.is_alive():
                print(Fore.YELLOW + "\nBackend exited.")
                break

            if frontend and not frontend.is_alive():
                print(Fore.YELLOW + "\nFrontend exited.")
                break

            time.sleep(1)

    finally:
        shutdown()


if __name__ == "__main__":
    main()
