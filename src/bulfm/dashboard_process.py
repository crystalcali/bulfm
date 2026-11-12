"""Launch the local dashboard and report success only after its identity check passes."""
import http.client
import json
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

DEFAULT_PORT = 7861


class PortBusy(RuntimeError):
    pass


def health(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        if response.status == 200:
            result = json.loads(response.read(16384))
            if isinstance(result, dict) and result.get("app") == "bulfm":
                return result
    except (OSError, ValueError, http.client.HTTPException):
        pass
    finally:
        connection.close()
    return None


def start_dashboard(root, port=None):
    if port is None:
        for candidate in range(DEFAULT_PORT, DEFAULT_PORT + 10):
            try:
                return start_dashboard(root, candidate)
            except PortBusy:
                continue
        raise PortBusy(f"No free dashboard port in {DEFAULT_PORT}–{DEFAULT_PORT + 9}")
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535")
    root = Path(root).resolve()
    url = f"http://127.0.0.1:{port}"
    identity = health(port)
    if identity and identity.get("runs_dir") == str(root):
        return url
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise PortBusy(f"Port {port} is already in use by another service. "
                           f"Try /dashboard {port + 1}") from exc
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "dashboard.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "bulfm.cli", "--runs", str(root),
             "dashboard", "--port", str(port)], stdout=log, stderr=log,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            with log_path.open(errors="replace") as log:
                detail = "".join(deque(log, maxlen=15))
            raise RuntimeError(f"Dashboard failed to start. Log: {log_path}\n{detail}")
        identity = health(port)
        if identity and identity.get("runs_dir") == str(root):
            return url
        time.sleep(0.1)
    raise RuntimeError(f"Dashboard did not become ready within 10 seconds. Check {log_path}")
