"""Start the backend (uvicorn) and frontend (npm dev) together.

Run from the project root:
    python -m topical_semantic_change.dev
Or, if installed via pip:
    tasc
"""

import socket
import subprocess
import sys
import signal
import time
import os
from pathlib import Path
from .config import PORT_BACKEND, PORT_FRONTEND, PROJECT_ROOT, FRONTEND


def _pid_on_port(port):
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["netstat", "-ano"], text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if f":{port} " in line and "LISTENING" in line:
                    return line.split()[-1]
        else:
            out = subprocess.check_output(
                ["lsof", "-ti", f":{port}"], text=True, stderr=subprocess.DEVNULL
            )
            return out.strip().split()[0]
    except Exception:
        pass
    return None


def _port_in_use(port):
    for host in ("127.0.0.1", "::1"):
        try:
            with socket.socket(
                socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM
            ) as s:
                s.settimeout(0.3)
                if s.connect_ex((host, port)) == 0:
                    return True
        except Exception:
            continue
    return False


def _check_ports():
    blocked = []
    for port in [PORT_BACKEND, PORT_FRONTEND]:
        if _port_in_use(port):
            blocked.append((port, _pid_on_port(port)))

    if not blocked:
        return

    for port, pid in blocked:
        line = f"Port {port} is already in use"
        if pid:
            if sys.platform == "win32":
                kill_cmd = f"taskkill /F /T /PID {pid}"
            else:
                kill_cmd = f"kill -TERM -{pid}"
            line += f" by PID {pid} — kill it with: {kill_cmd}"
        print(line)

    sys.exit(1)


def _popen(label, *args, **kwargs):
    try:
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["preexec_fn"] = os.setsid

        return subprocess.Popen(*args, **kwargs)
    except FileNotFoundError as e:
        print(f"Error: could not start {label} — {e}")
        print("  Make sure the required tool is installed and on your PATH.")
        sys.exit(1)


def _kill_tree_windows(pid):
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _kill_tree_unix(pid):
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        pass


def _stop(proc, name, timeout=5):
    if proc.poll() is not None:
        return  # already stopped

    print(f"Stopping {name}...")

    try:
        if sys.platform == "win32":
            _kill_tree_windows(proc.pid)
        else:
            _kill_tree_unix(proc.pid)
    except Exception as e:
        print(f"  Failed to terminate {name} cleanly: {e}")

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  {name} did not stop in time, killing forcefully.")
        try:
            proc.kill()
        except Exception:
            pass


def main():
    _check_ports()

    npm = "npm.cmd" if sys.platform == "win32" else "npm"

    backend = _popen(
        "uvicorn (try: pip install uvicorn)",
        [
            sys.executable,
            "-m",
            "uvicorn",
            "topical_semantic_change.backend.app.main:app",
        ],
        cwd=PROJECT_ROOT,
    )

    frontend = _popen(
        "npm (https://nodejs.org)",
        [npm, "run", "dev"],
        cwd=FRONTEND,
        shell=(sys.platform == "win32"),
    )

    print("Backend:  http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("Press Ctrl+C to stop both.\n")

    def _shutdown(sig=None, frame=None):
        print("\nShutting down...")
        _stop(frontend, "frontend")
        _stop(backend, "backend")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    try:
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
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
