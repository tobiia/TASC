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


def _popen(label, *args, **kwargs):
    try:
        return subprocess.Popen(*args, **kwargs)
    except FileNotFoundError as e:
        print(f"Error: could not start {label} — {e}")
        print(f"  Make sure the required tool is installed and on your PATH.")
        sys.exit(1)


def _stop(proc, name, timeout=5):
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  {name} did not stop in time, killing it.")
        proc.kill()
        proc.wait()


def main():
    npm = "npm.cmd" if sys.platform == "win32" else "npm"

    backend = _popen(
        "uvicorn (is it installed? try: pip install uvicorn)",
        [sys.executable, "-m", "uvicorn", "topical_semantic_change.backend.app.main:app"],
        cwd=PROJECT_ROOT,
    )
    frontend = _popen(
        "npm (is Node.js installed? https://nodejs.org)",
        [npm, "run", "dev"],
        cwd=FRONTEND_DIR,
    )

    print("Backend:  http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("Press Ctrl+C to stop both.\n")

    def _shutdown(sig, frame):
        print("\nShutting down...")
        _stop(frontend, "frontend")
        _stop(backend, "backend")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Poll until either process exits, then shut down the other
    while True:
        if backend.poll() is not None:
            print("\nBackend exited — stopping frontend.")
            _stop(frontend, "frontend")
            break
        if frontend.poll() is not None:
            print("\nFrontend exited — stopping backend.")
            _stop(backend, "backend")
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
