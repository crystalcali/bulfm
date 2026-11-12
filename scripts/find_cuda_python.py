"""Bounded discovery of existing training environments, without scanning datasets or weights."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts/check_cuda.py"


def candidates(root=ROOT, home=None):
    home = Path.home() if home is None else Path(home)
    paths = [root / ".venv/bin/python", Path(sys.executable)]
    for name in ("python3.12", "python3", "python"):
        if executable := shutil.which(name):
            paths.append(Path(executable))
    if os.environ.get("VIRTUAL_ENV"):
        paths.append(Path(os.environ["VIRTUAL_ENV"]) / "bin/python")
    if os.environ.get("CONDA_PREFIX"):
        paths.append(Path(os.environ["CONDA_PREFIX"]) / "bin/python")
    # Fixed-depth globbing: never descend into large model/data directories.
    for base in (home / "work", home / "projects"):
        for pattern in (".venv/bin/python", "*/.venv/bin/python", "*/*/.venv/bin/python",
                        "*/venv/bin/python", "*/*/venv/bin/python"):
            paths.extend(sorted(base.glob(pattern)))
    for pattern in (".venvs/*/bin/python", "venvs/*/bin/python",
                    "miniconda3/envs/*/bin/python", "anaconda3/envs/*/bin/python"):
        paths.extend(sorted(home.glob(pattern)))
    paths.extend([Path("/opt/venv/bin/python"), Path("/opt/pytorch/bin/python")])
    seen = set()
    for path in paths:
        # Do NOT resolve symlinks: separate venvs may share the same underlying executable.
        path = Path(os.path.abspath(path))
        if path not in seen and path.is_file() and os.access(path, os.X_OK):
            seen.add(path)
            yield path


def find(paths=None, probe=None):
    if paths is None:
        paths = candidates()
    if probe is None:
        probe = subprocess.run
    for index, executable in enumerate(paths):
        if index >= 64:
            break
        print(f"Checking Python: {executable}", file=sys.stderr, flush=True)
        try:
            result = probe([str(executable), str(CHECK)], capture_output=True, text=True,
                           timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"  skipped: {exc}", file=sys.stderr)
            continue
        if result.returncode == 0:
            print(result.stdout.strip(), file=sys.stderr)
            return str(executable)
        print("  " + (result.stderr.strip() or "CUDA check failed"), file=sys.stderr)
    return None


if __name__ == "__main__":
    selected = find()
    if selected:
        print(selected)
    else:
        raise SystemExit(1)
