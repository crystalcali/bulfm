import json
import os
import re
import time
from pathlib import Path

import psutil

RUN_ID = re.compile(r"^[a-zA-Z0-9_-]+$")


def atomic_json(path: Path, value):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default if default is not None else {}


def append_event(run: Path, event, **values):
    with (run / "metrics.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.time(), "event": event, **values}, allow_nan=False) + "\n")


def tail_events(run, limit=500):
    from collections import deque
    try:
        with (Path(run) / "metrics.jsonl").open() as f:
            lines = deque(f, maxlen=limit)
        result = []
        for line in lines:
            try:
                result.append(json.loads(line))
            except ValueError:
                pass  # writer may still be appending its final line
        return result
    except OSError:
        return []


def alive(state):
    try:
        process = psutil.Process(state["pid"])
        return (abs(process.create_time() - state["process_started"]) < 0.1
                and process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except (KeyError, psutil.Error):
        return False


def snapshot(run):
    run = Path(run)
    state = read_json(run / "state.json")
    if state.get("status") in {"starting", "running", "evaluating", "stopping"} and not alive(state):
        state["status"] = "interrupted"
    return {"id": run.name, **state, "metrics": tail_events(run)}


def safe_run(root: Path, run_id: str):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    run = (root / run_id).resolve()
    if run.parent != root.resolve() or not run.is_dir():
        raise ValueError("Unknown run")
    return run
