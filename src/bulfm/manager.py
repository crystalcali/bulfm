import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .config import Config
from .storage import alive, atomic_json, read_json, safe_run, snapshot


def start(root, config, resume=None, adapter=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if resume:
        run = safe_run(root, resume)
        state = read_json(run / "state.json")
        if alive(state):
            raise ValueError("This run is already running")
        if not (run / "latest.json").exists():
            raise ValueError("No checkpoint yet. Start a new run after fixing the error")
        (run / "STOP").unlink(missing_ok=True)
        Config.load(run / "config.yaml")
    else:
        c = Config.load(config)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        run = root / run_id
        run.mkdir()
        c.save(run / "config.yaml")
        state = {"created": time.time(), "step": 0, "model": c.model, "task": c.task}
        if adapter:
            adapter = Path(adapter).resolve()
            if not (adapter / "adapter_config.json").is_file():
                raise ValueError("Evaluation requires a LoRA adapter checkpoint")
            state["evaluation_adapter"] = str(adapter)
    atomic_json(run / "state.json", {**state, "status": "starting"})
    command = [sys.executable, "-u", "-m", "bulfm.worker", "--run", str(run)]
    if resume:
        command.append("--resume")
    if adapter:
        command.append("--evaluate-only")
    # A detached worker survives terminal closure / SSH disconnection.
    with (run / "worker.log").open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
    # Worker imports the ML stack before its first update; publish PID during that interval.
    current = read_json(run / "state.json")
    current.update(pid=process.pid, process_started=psutil.Process(process.pid).create_time())
    atomic_json(run / "state.json", current)
    return run


def stop(root, run_id):
    run = safe_run(Path(root), run_id)
    if not alive(read_json(run / "state.json")):
        raise ValueError("Run is not active")
    (run / "STOP").touch()
    return "Stop requested. The worker saves after the current optimizer step."


def list_runs(root):
    root = Path(root)
    if not root.exists():
        return []
    return [snapshot(p) for p in sorted(root.iterdir(), reverse=True)
            if p.is_dir() and (p / "config.yaml").is_file()]
