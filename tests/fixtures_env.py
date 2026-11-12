"""Picklable CPU-only fixtures for spawned environment workers."""
import os
import time
from pathlib import Path


class FakeBrowser:
    def __init__(self, config):
        self.directory = config.miniwob_url

    def reset(self, seed):
        self.seed = seed
        return {"goal": str(seed), "tree": "[13] button", "error": "", "pid": os.getpid()}

    def step(self, action):
        if action == "crash":
            raise ValueError("fixture browser crash")
        if action == "barrier":
            root = Path(self.directory)
            (root / str(os.getpid())).touch()
            deadline = time.monotonic() + 5
            while len(list(root.iterdir())) < 2:
                if time.monotonic() > deadline:
                    raise TimeoutError("Browser steps were not dispatched concurrently")
                time.sleep(.01)
        return self.reset(self.seed), 1.0, True

    def close(self):
        pass
