import argparse
import importlib.util
import json
import shlex
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Config
from .dashboard_process import DEFAULT_PORT, start_dashboard
from .manager import list_runs, start, stop
from .storage import safe_run, snapshot

console = Console()
HELP = """/train [configs/spark.yaml]   Start a detached GRPO run
/resume RUN_ID               Resume model + optimizer + RNG
/stop RUN_ID                 Save and stop after current optimizer step
/status [RUN_ID]             Runs, phase, rewards and progress
/logs RUN_ID                 Last 50 worker log lines
/config [FILE]               Show validated configuration
/eval RUN_ID                 Evaluate saved LoRA on fixed held-out seeds
/dashboard [PORT]            Start dashboard (default port: 7861)
/benchmark [CONFIG]          Compare 1/4/8 environments (foreground)
/doctor                      Check CUDA and browser dependencies
/env-check                   Verify a real click and reward without loading LFM3
/help                        Show commands
/exit                        Close terminal (training continues)"""


def status(root, run_id=None):
    rows = [snapshot(safe_run(Path(root), run_id))] if run_id else list_runs(root)
    table = Table(title="BULFM runs", expand=True)
    for label in ("Run", "State / phase", "Step", "Reward", "Success", "tok/s", "Progress"):
        table.add_column(label)
    for row in rows[:20]:
        m = row.get("latest_metrics", {})
        table.add_row(row["id"], f"{row.get('status', '?')} / {row.get('phase', '…')}",
                      f"{row.get('step', 0)}/{row.get('total_steps', '?')}",
                      f"{m['reward']:.3f}" if "reward" in m else "—",
                      f"{m['success_rate']:.0%}" if "success_rate" in m else "—",
                      f"{m['tokens_per_second']:.1f}" if "tokens_per_second" in m else "—",
                      f"eval {row.get('eval_completed', 0)}/{row.get('eval_total', '?')}"
                      if row.get("status") == "evaluating" else
                      f"envs {row.get('active_envs', 0)}/{row.get('num_envs', 1)}")
    console.print(table)
    if run_id and rows[0].get("error"):
        console.print(rows[0]["error"], style="red", markup=False)


def doctor():
    import platform
    console.print(f"BULFM {__version__} · Python {platform.python_version()} · {platform.machine()}")
    okay = True
    for module in ("torch", "transformers", "peft", "browsergym.miniwob", "playwright"):
        try:
            found = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            found = False
        console.print(f"{'OK' if found else 'MISSING'}  {module}")
        okay &= found
    try:
        from .diagnostics import training_imports
        training_imports()
        console.print("OK  Transformers model classes and PEFT imports")
    except Exception as exc:
        from .diagnostics import error_message
        console.print(error_message(exc), style="red", markup=False)
        okay = False
    if importlib.util.find_spec("torch"):
        import torch
        console.print(f"PyTorch {torch.__version__}; CUDA build {torch.version.cuda}; "
                      f"CUDA available {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            console.print(f"GPU: {torch.cuda.get_device_name()}; bf16: {torch.cuda.is_bf16_supported()}")
            # Allocation + matmul checks that the wheel actually supports GB10, not just discovery.
            a = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
            (a @ a).sum().item()
            console.print("OK  CUDA bf16 matmul")
        else:
            okay = False
    page = Path("vendor/miniwob-plusplus/miniwob/html/miniwob/click-test.html")
    import os
    pages = page.exists() or bool(os.environ.get("MINIWOB_URL"))
    console.print(f"{'OK' if pages else 'MISSING'}  MiniWoB pages / MINIWOB_URL")
    okay &= pages
    if importlib.util.find_spec("playwright"):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            executable = Path(p.chromium.executable_path)
            console.print(f"{'OK' if executable.exists() else 'MISSING'}  Chromium: {executable}")
            okay &= executable.exists()
    return okay


def env_check():
    import re

    from .environment import Browser
    env = Browser(Config())
    try:
        obs = env.reset(23)
        buttons = re.findall(r"\[(\d+)\] button", obs["tree"])
        if not buttons:
            raise RuntimeError("click-test did not expose its button in the accessibility tree")
        _, reward, done = env.step(f"click('{buttons[0]}')")
        if reward <= 0 or not done:
            raise RuntimeError(f"click-test failed: reward={reward}, done={done}")
        console.print(f"OK  Real BrowserGym click-test: reward={reward}, done={done}")
    finally:
        env.close()


def execute(words, root):
    command, *args = words
    command = command.lstrip("/")
    if command in {"stop", "resume", "logs", "eval"} and len(args) != 1:
        raise ValueError(f"Usage: /{command} RUN_ID. Use /status to list runs.")
    if command == "train":
        run = start(root, args[0] if args else "configs/spark.yaml")
        console.print(f"Started {run.name}\n/status {run.name}\n/logs {run.name}")
    elif command == "resume":
        console.print(f"Resumed {start(root, None, resume=args[0]).name}")
    elif command == "stop":
        console.print(stop(root, args[0]))
    elif command == "status":
        status(root, args[0] if args else None)
    elif command == "logs":
        from collections import deque
        run = safe_run(Path(root), args[0])
        with (run / "worker.log").open(errors="replace") as f:
            console.print("".join(deque(f, maxlen=50)), markup=False)
    elif command == "config":
        console.print_json(json.dumps(Config.load(args[0] if args else "configs/spark.yaml").model_dump()))
    elif command == "eval":
        from .checkpoint import latest
        source = safe_run(Path(root), args[0])
        c = Config.load(source / "config.yaml")
        if not c.lora:
            raise ValueError("Manual evaluation command currently accepts LoRA runs only")
        run = start(root, source / "config.yaml", adapter=latest(source))
        console.print(f"Evaluation started: {run.name}")
    elif command == "benchmark":
        from .benchmark import run_benchmark
        run_benchmark(args[0] if args else "configs/spark-book-flight-parallel.yaml", root)
    elif command == "doctor":
        if not doctor():
            raise ValueError("Environment is not ready for Spark training. See setup instructions")
    elif command == "env-check":
        env_check()
    elif command == "dashboard":
        url = start_dashboard(root, int(args[0]) if args else None)
        console.print(f"Dashboard ready: {url}")
    elif command == "help":
        console.print(HELP, markup=False)
    elif command in {"exit", "quit"}:
        return False
    else:
        raise ValueError("Unknown command. Type /help")
    return True


def shell(root):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    Path(root).mkdir(parents=True, exist_ok=True)
    session = PromptSession(history=FileHistory(str(Path(root) / ".history")),
                            completer=WordCompleter(["/" + x for x in
                                ["train", "resume", "stop", "status", "logs", "config", "eval",
                                 "dashboard", "benchmark", "doctor", "env-check", "help", "exit"]]))
    console.print(Panel("[bold cyan]BULFM[/bold cyan] · LFM3 Browser RL\n"
                        "GRPO • DGX Spark • /help\nTraining continues when this terminal closes.",
                        border_style="cyan"))
    def toolbar():
        rows = list_runs(root)
        if not rows:
            return " /train configs/spark-smoke.yaml  •  /dashboard  •  /help "
        row = rows[0]
        return (f" {row['id']} | {row.get('status', 'starting')} | "
                f"{row.get('phase', '')} | step {row.get('step', 0)} ")
    while True:
        try:
            with patch_stdout():
                line = session.prompt("❯ ", bottom_toolbar=toolbar, refresh_interval=2)
            words = shlex.split(line)
            if words and not execute(words, root):
                return
        except (EOFError, KeyboardInterrupt):
            console.print("Terminal closed. Detached workers continue.")
            return
        except Exception as exc:
            console.print(str(exc), style="red", markup=False)


def main():
    parser = argparse.ArgumentParser(description="Local LFM3 browser GRPO training")
    parser.add_argument("--runs", default="runs", help="Run storage directory")
    parser.add_argument("command", nargs="?", default="chat")
    parser.add_argument("args", nargs="*")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    opts = parser.parse_args()
    if opts.command == "chat":
        shell(opts.runs)
    elif opts.command == "dashboard":
        import uvicorn

        from .dashboard import create_app
        uvicorn.run(create_app(opts.runs), host=opts.host, port=opts.port)
    else:
        try:
            execute([opts.command, *opts.args], opts.runs)
        except Exception as exc:
            console.print(str(exc), style="red", markup=False)
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
