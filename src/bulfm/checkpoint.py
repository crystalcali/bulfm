import random
import shutil
from pathlib import Path

import numpy as np
import torch

from .storage import atomic_json, read_json


def save(run, model, tok, optimizer, step, keep):
    root = Path(run) / "checkpoints"
    root.mkdir(exist_ok=True)
    dest = root / f"step-{step:06d}"
    temp = root / f".step-{step:06d}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    model.save_pretrained(temp)
    tok.save_pretrained(temp)
    torch.save({"step": step, "optimizer": optimizer.state_dict(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               temp / "training.pt")
    # A previously saved completed step is already a valid checkpoint.
    if dest.exists():
        shutil.rmtree(temp)
    else:
        temp.rename(dest)
    atomic_json(Path(run) / "latest.json", {"path": str(dest.resolve()), "step": step})
    for old in sorted(root.glob("step-*"))[:-keep]:
        shutil.rmtree(old)
    return dest


def latest(run):
    data = read_json(Path(run) / "latest.json")
    path = Path(data.get("path", "/nonexistent"))
    if not (path / "training.pt").is_file():
        raise ValueError("No complete checkpoint available")
    return path


def restore(path, optimizer):
    # Only load trusted local checkpoints written by this application.
    data = torch.load(Path(path) / "training.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(data["optimizer"])
    random.setstate(data["python_rng"])
    np.random.set_state(data["numpy_rng"])
    torch.set_rng_state(data["torch_rng"])
    if data["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(data["cuda_rng"])
    return data["step"]
