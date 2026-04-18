"""Start the backend (uvicorn) and frontend (npm dev) together.

Run from the project root:
    python -m topical_semantic_change.dev
Or, if installed via pip:
    tsc-dev
"""

import subprocess
import sys
import signal
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "topical_semantic_change" / "frontend"


def main():
    npm = "npm.cmd" if sys.platform == "win32" else "npm"

    backend = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "topical_semantic_change.backend.app.main:app",
        ],
        cwd=PROJECT_ROOT,
    )
    frontend = subprocess.Popen(
        [npm, "run", "dev"],
        cwd=FRONTEND_DIR,
    )

    print("Backend:  http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("Press Ctrl+C to stop both.\n")

    def _shutdown(sig, frame):
        print("\nShutting down...")
        frontend.terminate()
        backend.terminate()
        frontend.wait()
        backend.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Poll until either process exits, then shut down the other
    while True:
        if backend.poll() is not None:
            print("\nBackend exited — stopping frontend.")
            frontend.terminate()
            break
        if frontend.poll() is not None:
            print("\nFrontend exited — stopping backend.")
            backend.terminate()
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
