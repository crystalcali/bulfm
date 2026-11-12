import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from bulfm import checkpoint
from bulfm.config import Config
from bulfm.dashboard import create_app
from bulfm.environment import parse_action
from bulfm.grpo import advantages, surrogate, token_logps
from bulfm.policy import prompt_ids
from bulfm.storage import atomic_json, safe_run, snapshot, tail_events


def test_group_advantages():
    a = advantages([0, 0, 1, 1])
    assert a.mean().abs() < 1e-7
    assert a[0] < 0 < a[3]
    assert torch.equal(advantages([1, 1]), torch.zeros(2))


def test_clipped_gradient_and_reference():
    lp = torch.tensor([-1.0, -1.0], requires_grad=True)
    loss = surrogate(lp, lp.detach(), torch.tensor(1.0), .2).mean()
    loss.backward()
    assert (lp.grad < 0).all()  # gradient descent increases successful action probability
    lp = torch.tensor([-1.0], requires_grad=True)
    ref = torch.tensor([-2.0])
    kl_loss = surrogate(lp, lp.detach(), torch.tensor(0.0), .2, ref, .04).sum()
    kl_loss.backward()
    assert lp.grad > 0  # KL pushes overly probable actions towards original policy
    clipped = torch.tensor([0.0], requires_grad=True)
    surrogate(clipped, torch.tensor([-1.0]), torch.tensor(1.0), .2).backward()
    assert clipped.grad == 0


class Tiny(torch.nn.Module):
    """Causal bigram LM for numerical training checks, no downloaded weights."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 16)
        self.config = SimpleNamespace(_commit_hash="tiny-test")

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.embedding(input_ids))

    def save_pretrained(self, path):
        torch.save(self.state_dict(), Path(path) / "model.pt")


def test_only_action_tokens_and_shift():
    model = Tiny()
    got = token_logps(model, [2, 3, 4], [5, 6])
    expected = model.embedding.weight[[4, 5]].log_softmax(-1)[[0, 1], [5, 6]]
    assert torch.allclose(got, expected)
    assert got.shape == (2,)
    (-got.sum()).backward()
    active = model.embedding.weight.grad.abs().sum(-1).nonzero().flatten().tolist()
    assert active == [4, 5]  # no loss on prompt tokens [2,3,4]


@pytest.mark.parametrize("text,valid", [
    ("click('13')", True), ("```python\nfill('2', 'hello')\n```", True),
    ("scroll(0, 500)", True), ("noop()", True),
    ("click('1'); click('2')", False), ("__import__('os').system('id')", False),
    ("click(str(13))", False), ("click('1', extra=2)", False),
    ("scroll('up')", False), ("click(13)", False),
])
def test_action_parser(text, valid):
    action, actual = parse_action(text)
    assert actual == valid
    if not valid:
        assert action == "noop()"


def test_config_rejects_typos_and_seed_overlap():
    with pytest.raises(ValueError):
        Config(lern_rate=1e-5)
    with pytest.raises(ValueError):
        Config(lora=False, beta=.04)
    with pytest.raises(ValueError):
        Config(context_tokens=256, completion_tokens=200)
    with pytest.raises(ValueError):
        Config(seed=1_000_000_000, steps=100)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return list(range(len(messages[0]["content"]) + len(messages[1]["content"])))

    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text("{}")


def test_context_keeps_goal_and_instructions():
    class Recorder(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return super().apply_chat_template(messages, **kwargs)
    tok = Recorder()
    c = Config(context_tokens=1024)
    ids = prompt_ids(tok, c, {"goal": "CLICK CORRECT", "tree": "x" * 5000, "error": ""}, ["y" * 2000])
    assert len(ids) <= c.context_tokens - c.completion_tokens
    assert tok.messages[0]["content"] == c.system_prompt
    assert "CLICK CORRECT" in tok.messages[1]["content"]


def test_checkpoint_optimizer_rng_and_retention(tmp_path):
    model, tok = Tiny(), Tokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model.embedding.weight.sum().backward()
    optimizer.step()
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    checkpoint.save(tmp_path, model, tok, optimizer, 1, 2)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=.9)
    assert checkpoint.restore(checkpoint.latest(tmp_path), optimizer2) == 1
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert optimizer2.param_groups[0]["lr"] == .01
    assert len(optimizer2.state) > 0
    checkpoint.save(tmp_path, model, tok, optimizer, 2, 2)
    checkpoint.save(tmp_path, model, tok, optimizer, 3, 2)
    assert not (tmp_path / "checkpoints/step-000001").exists()
    assert checkpoint.latest(tmp_path).name == "step-000003"


def test_dashboard_live_api_and_path_validation(tmp_path):
    run = tmp_path / "test-run"
    run.mkdir()
    Config().save(run / "config.yaml")
    atomic_json(run / "state.json", {"status": "completed", "step": 3})
    (run / "metrics.jsonl").write_text(json.dumps({"event": "train", "reward": .5}) + '\n{"partial"')
    client = TestClient(create_app(tmp_path))
    assert "BULFM" in client.get("/").text
    assert client.get("/api/runs").json()[0]["id"] == "test-run"
    assert client.get("/api/runs/test-run").json()["metrics"][0]["reward"] == .5
    assert client.get("/api/runs/unknown").status_code == 404
    with pytest.raises(ValueError):
        safe_run(tmp_path, "../escape")
    assert len(tail_events(run)) == 1
    atomic_json(run / "state.json", {"status": "running", "pid": 999999999})
    assert snapshot(run)["status"] == "interrupted"


def test_worker_full_cycle_and_resume(tmp_path, monkeypatch):
    """Real backward/optimizer/checkpoint path, using tiny model and deterministic fake browser."""
    import importlib.metadata

    import bulfm.environment
    import bulfm.policy
    import bulfm.rollout
    from bulfm.rollout import Episode, Transition
    from bulfm.worker import Worker

    class FakeBrowser:
        def __init__(self, c):
            pass
        def close(self):
            pass
    model = Tiny()
    def load(c, adapter=None):
        if adapter:
            model.load_state_dict(torch.load(Path(adapter) / "model.pt", weights_only=True))
        return model, Tokenizer()
    count = 0
    seeds = []
    def collect(model, tok, env, c, seed, **kwargs):
        nonlocal count
        count += 1
        seeds.append(seed)
        completion = [4, 5] if count % 2 else [6, 7]
        lp = token_logps(model, [2, 3], completion).detach()
        return Episode([Transition([2, 3], completion, lp, None)], reward=float(count % 2),
                       success=bool(count % 2))
    monkeypatch.setattr(bulfm.environment, "Browser", FakeBrowser)
    monkeypatch.setattr(bulfm.policy, "load_policy", load)
    monkeypatch.setattr(bulfm.rollout, "collect", collect)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test-runtime")
    run = tmp_path / "run"
    run.mkdir()
    c = Config(steps=2, group_size=2, beta=0, baseline_eval=False, eval_episodes=1,
               eval_every=0, save_every=1, device="cpu", precision="fp32")
    c.save(run / "config.yaml")
    original = model.embedding.weight.detach().clone()
    worker = Worker(run)
    worker.execute()
    assert snapshot(run)["status"] == "completed"
    assert snapshot(run)["step"] == 2
    assert not torch.equal(original, model.embedding.weight)
    assert seeds[:4] == [23, 23, 24, 24]
    assert (run / "evaluation-after.json").exists()
    c.steps = 3
    c.save(run / "config.yaml")
    Worker(run).execute(resume=True)
    assert snapshot(run)["step"] == 3
    assert [m["step"] for m in tail_events(run) if m["event"] == "train"] == [1, 2, 3]


def test_manager_start_and_stop(tmp_path, monkeypatch):
    import os

    import psutil

    from bulfm import manager
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(manager.subprocess, "Popen", popen)
    config = tmp_path / "spark.yaml"
    Config().save(config)
    run = manager.start(tmp_path / "runs", config)
    assert snapshot(run)["status"] == "starting"
    assert calls[0][1]["start_new_session"]
    assert calls[0][0][-2:] == ["--run", str(run)]
    assert psutil.Process().create_time() == snapshot(run)["process_started"]
    assert "Stop requested" in manager.stop(tmp_path / "runs", run.name)
    assert (run / "STOP").exists()
    assert "already requested" in manager.stop(tmp_path / "runs", run.name)
    with pytest.raises(ValueError, match="already running"):
        manager.start(tmp_path / "runs", None, resume=run.name)


def test_import_error_reports_original_dependency_failure():
    from bulfm.diagnostics import error_message
    try:
        try:
            raise AttributeError("module 'lib' has no attribute 'GEN_EMAIL'")
        except AttributeError as root:
            raise ModuleNotFoundError("Could not import module AutoModel") from root
    except ModuleNotFoundError as exc:
        message = error_message(exc)
    assert "AutoModel" in message
    assert "Root cause: AttributeError" in message
    assert "GEN_EMAIL" in message


def test_error_without_chain_stays_short():
    from bulfm.diagnostics import error_message
    assert error_message(ValueError("bad config")) == "ValueError: bad config"


def test_prompt_ids_with_transformers_default_batch_encoding():
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    backend = BackendTokenizer(WordLevel({"[UNK]": 0, "user": 1, "assistant": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tok.chat_template = ("{% for message in messages %}{{ message.role }} {{ message.content }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant{% endif %}")
    messages = [{"role": "user", "content": "Click the button"}]
    default = tok.apply_chat_template(messages, tokenize=True)
    assert "input_ids" in default  # Transformers 5 returns BatchEncoding by default.
    ids = prompt_ids(tok, Config(), {"goal": "Click", "tree": "[13] button", "error": ""}, [])
    assert isinstance(ids, list) and all(type(i) is int for i in ids)
    tensor = torch.tensor([ids], dtype=torch.long)
    assert tensor.ndim == 2 and tensor.shape[1] == len(ids)
