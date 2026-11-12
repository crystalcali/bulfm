import json
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from bulfm import dashboard_process as dp


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_refuses_port_used_by_other_service(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        with pytest.raises(RuntimeError, match="already in use by another service"):
            dp.start_dashboard(tmp_path, sock.getsockname()[1])


def test_reuses_verified_dashboard(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"app": "bulfm", "runs_dir": str(tmp_path)}).encode())
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_port
        assert dp.start_dashboard(tmp_path, port) == f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_reports_child_startup_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "health", lambda _: None)
    monkeypatch.setattr(dp.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(poll=lambda: 1))
    with pytest.raises(RuntimeError, match="failed to start"):
        dp.start_dashboard(tmp_path, free_port())


def test_real_server_start_and_identity(tmp_path, monkeypatch):
    original = subprocess.Popen
    processes = []
    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(dp.subprocess, "Popen", launch)
    port = free_port()
    try:
        assert dp.start_dashboard(tmp_path, port) == f"http://127.0.0.1:{port}"
        assert dp.health(port)["runs_dir"] == str(tmp_path)
        assert dp.start_dashboard(tmp_path, port) == f"http://127.0.0.1:{port}"
        assert len(processes) == 1
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=10)
