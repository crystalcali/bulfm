"""End-to-end worker check with real spawned browsers and a deterministic tiny action policy."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bulfm.config import Config
from bulfm.storage import snapshot


class ActionTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, *args, **kwargs):
        return [2, 3]

    def decode(self, *args, **kwargs):
        return "click('13')"

    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text("{}")


class ActionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)
        self.config = SimpleNamespace(_commit_hash="test-action-policy")
        self.batches = []
        self.forwards = 0

    def generate(self, input_ids, **kwargs):
        self.batches.append(input_ids.shape[0])
        action = torch.tensor([[4, 1]], device=input_ids.device).expand(input_ids.shape[0], -1)
        return torch.cat((input_ids, action), dim=1)

    def forward(self, input_ids, **kwargs):
        self.forwards += 1
        return SimpleNamespace(logits=self.embedding(input_ids))

    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path) / "model.pt")


@pytest.mark.skipif(os.environ.get("BULFM_REAL_BROWSER") != "1", reason="Real browser opt-in")
def test_worker_parallel_browser_evaluate_train_checkpoint(tmp_path, monkeypatch):
    import importlib.metadata

    import bulfm.policy
    from bulfm.worker import Worker
    model = ActionPolicy()
    monkeypatch.setattr(bulfm.policy, "load_policy", lambda *a, **kw: (model, ActionTokenizer()))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test-runtime")
    c = Config(steps=1, num_envs=2, group_size=2, generation_batch_size=2,
               beta=0, device="cpu", precision="fp32", eval_episodes=2, episode_steps=2)
    c.save(tmp_path / "config.yaml")
    Worker(tmp_path).execute()
    state = snapshot(tmp_path)
    assert state["status"] == "completed"
    assert state["step"] == 1
    assert state["latest_metrics"]["success_rate"] == 1
    assert model.batches == [2, 2, 2]  # baseline, one training group, final evaluation
    assert model.forwards == 3  # one batched old-policy forward + two backward forwards, no eval scores
    assert (tmp_path / "checkpoints/step-000001/training.pt").exists()
    assert json.loads((tmp_path / "evaluation-before.json").read_text())["complete"]
    assert json.loads((tmp_path / "evaluation-after.json").read_text())["success_rate"] == 1


def test_parallel_resource_benchmark_reports_one_four_eight(tmp_path, monkeypatch):
    import bulfm.env_pool
    import bulfm.policy
    from bulfm.benchmark import run_benchmark
    model = ActionPolicy()
    monkeypatch.setattr(bulfm.policy, "load_policy", lambda *a, **kw: (model, ActionTokenizer()))

    class Pool:
        def __init__(self, c):
            self.size = c.num_envs
        def reset_many(self, indices, seeds):
            return [{"goal": "Click", "tree": "button", "error": ""} for _ in indices]
        def send_steps(self, indices, actions):
            pass
        def receive_steps(self, indices):
            return [({"goal": "Click", "tree": "", "error": ""}, 1.0, True) for _ in indices]
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
    monkeypatch.setattr(bulfm.env_pool, "BrowserPool", Pool)
    config = tmp_path / "config.yaml"
    Config(beta=0, device="cpu", precision="fp32", episode_steps=1).save(config)
    result = json.loads(run_benchmark(config, tmp_path).read_text())
    assert [r["num_envs"] for r in result["results"]] == [1, 4, 8]
    assert all(r["episodes_per_minute"] > 0 for r in result["results"])
    assert 8 in model.batches and 4 in model.batches and 1 in model.batches
    assert result["fastest_measured_num_envs"] in [1, 4, 8]


def test_stop_discards_unfinished_grpo_step_and_closes_pool(tmp_path, monkeypatch):
    import importlib.metadata

    import bulfm.env_pool
    import bulfm.policy
    import bulfm.rollout
    from bulfm.grpo import token_logps
    from bulfm.rollout import Episode, RolloutCancelled, Transition
    from bulfm.worker import Worker
    model = ActionPolicy()
    initial = model.embedding.weight.detach().clone()
    monkeypatch.setattr(bulfm.policy, "load_policy", lambda *a, **kw: (model, ActionTokenizer()))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test-runtime")
    closed = []
    class Pool:
        def __init__(self, c):
            pass
        def close(self):
            closed.append(True)
    monkeypatch.setattr(bulfm.env_pool, "BrowserPool", Pool)
    c = Config(steps=1, num_envs=2, group_size=2, groups_per_step=2, beta=0,
               device="cpu", precision="fp32", baseline_eval=False)
    c.save(tmp_path / "config.yaml")
    worker = Worker(tmp_path)
    calls = []
    def collect(*a, **kw):
        calls.append(True)
        if len(calls) == 2:
            worker.stop = True
            raise RolloutCancelled
        return [Episode([Transition([2, 3], [token, 1],
                                    token_logps(model, [2, 3], [token, 1]).detach(), None)],
                        reward=float(i)) for i, token in enumerate([4, 5])]
    monkeypatch.setattr(bulfm.rollout, "collect_many", collect)
    worker.execute()
    assert snapshot(tmp_path)["status"] == "stopped"
    assert snapshot(tmp_path)["step"] == 0
    assert torch.equal(initial, model.embedding.weight)
    assert (tmp_path / "checkpoints/step-000000/training.pt").exists()
    assert closed == [True]
