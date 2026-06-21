import subprocess
import sys
import os
import signal


class Process:
    def __init__(self, label, popen_obj):
        self.label = label
        self.proc = popen_obj

    def is_alive(self):
        return self.proc.poll() is None

    def kill(self, timeout=5):
        if not self.is_alive():
            return

        print(f"Stopping {self.label}...")

        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except Exception:
            pass

        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def popen(label, args, **kwargs):
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid

    return Process(label, subprocess.Popen(args, **kwargs))
