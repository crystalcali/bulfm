"""CPU-only browser subprocesses; spawn avoids inheriting the trainer's CUDA context."""
import multiprocessing as mp
import os
import signal
import time
import traceback

from .config import Config
from .environment import Browser


def _browser_worker(connection, config, factory=Browser):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # Clean up browsers if the trainer is terminated unexpectedly (Linux/Spark).
    if os.name == "posix":
        def terminate(*_):
            raise SystemExit
        signal.signal(signal.SIGTERM, terminate)
        try:
            import ctypes
            parent = os.getppid()
            ctypes.CDLL(None).prctl(1, signal.SIGTERM)
            if os.getppid() != parent:
                raise SystemExit
        except (OSError, AttributeError):
            pass
    env = None
    try:
        env = factory(Config.model_validate(config))
        connection.send(("ok", None))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                result = env.reset(payload)
            elif command == "step":
                result = env.step(payload)
            else:
                raise ValueError(f"Unknown environment command: {command}")
            connection.send(("ok", result))
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class BrowserPool:
    def __init__(self, config, factory=Browser):
        self.size = config.num_envs
        self.timeout = config.env_timeout_seconds
        self.connections, self.processes = [], []
        self.closed = False
        context = mp.get_context("spawn")
        try:
            for _ in range(self.size):
                parent, child = context.Pipe()
                process = context.Process(target=_browser_worker,
                                          args=(child, config.model_dump(), factory), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for index in range(self.size):
                self._receive(index)
        except BaseException:
            self.close()
            raise

    def _receive(self, index):
        connection = self.connections[index]
        deadline = time.monotonic() + self.timeout
        while not connection.poll(min(0.1, max(0, deadline - time.monotonic()))):
            if not self.processes[index].is_alive():
                raise RuntimeError(f"Browser worker {index} exited unexpectedly")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Browser worker {index} did not respond within {self.timeout}s")
        try:
            status, result = connection.recv()
        except EOFError as exc:
            raise RuntimeError(f"Browser worker {index} closed its connection") from exc
        if status != "ok":
            raise RuntimeError(f"Browser worker {index} failed:\n{result}")
        return result

    def reset_many(self, indices, seeds):
        if len(indices) != len(seeds) or len(set(indices)) != len(indices):
            raise ValueError("Each reset needs one seed and a unique environment index")
        for index, seed in zip(indices, seeds):
            self.connections[index].send(("reset", seed))
        return [self._receive(index) for index in indices]

    def send_steps(self, indices, actions):
        if len(indices) != len(actions) or len(set(indices)) != len(indices):
            raise ValueError("Each step needs one action and a unique environment index")
        for index, action in zip(indices, actions):
            self.connections[index].send(("step", action))

    def receive_steps(self, indices):
        return [self._receive(index) for index in indices]

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (OSError, EOFError, BrokenPipeError):
                pass
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        for connection in self.connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
