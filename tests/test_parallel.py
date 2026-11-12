from contextlib import nullcontext

import pytest
import torch
from fixtures_env import FakeBrowser

from bulfm.config import Config
from bulfm.env_pool import BrowserPool
from bulfm.grpo import batch_token_logps, token_logps
from bulfm.policy import generate_batch
from bulfm.rollout import RolloutCancelled, collect_many


def test_pool_uses_distinct_processes_and_concurrent_steps(tmp_path):
    with BrowserPool(Config(num_envs=2, miniwob_url=str(tmp_path)), factory=FakeBrowser) as pool:
        observations = pool.reset_many([0, 1], [23, 23])
        assert len({obs["pid"] for obs in observations}) == 2
        assert [obs["goal"] for obs in observations] == ["23", "23"]
        pool.send_steps([0, 1], ["barrier", "barrier"])
        assert all(result[1:] == (1.0, True) for result in pool.receive_steps([0, 1]))
    assert all(not process.is_alive() for process in pool.processes)


def test_pool_propagates_errors_and_closes(tmp_path):
    with BrowserPool(Config(num_envs=2), factory=FakeBrowser) as pool:
        pool.reset_many([0], [23])
        pool.send_steps([0], ["crash"])
        with pytest.raises(RuntimeError, match="fixture browser crash"):
            pool.receive_steps([0])
    assert all(not process.is_alive() for process in pool.processes)


class MemoryPool:
    size = 2

    def __init__(self):
        self.seeds, self.turns, self.pending = {}, {}, {}

    def reset_many(self, indices, seeds):
        for i, seed in zip(indices, seeds):
            self.seeds[i], self.turns[i] = seed, 0
        return [{"goal": str(seed), "tree": "button", "error": ""} for seed in seeds]

    def send_steps(self, indices, actions):
        for i in indices:
            self.turns[i] += 1
            done = self.turns[i] >= self.seeds[i] % 3 + 1
            self.pending[i] = ({"goal": str(self.seeds[i]), "tree": "button", "error": ""},
                               float(self.seeds[i]) if done else 0.0, done)

    def receive_steps(self, indices):
        return [self.pending.pop(i) for i in indices]


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, *a, **kw):
        return "click('13')"

    def apply_chat_template(self, *a, **kw):
        return [2, 3]


def test_scheduler_refills_slots_and_preserves_episode_order(monkeypatch):
    import bulfm.rollout
    batches, scored, traced, progress = [], [], [], []
    def generate(model, tok, prompts, c, sample, score):
        batches.append(len(prompts))
        scored.append(score)
        return [([4, 1], torch.zeros(2) if score else None, None) for _ in prompts]
    monkeypatch.setattr(bulfm.rollout, "generate_batch", generate)
    seeds = [0, 1, 2, 3, 4]
    episodes = collect_many(None, Tokenizer(), MemoryPool(), Config(episode_steps=5), seeds,
                            trace=traced.append, heartbeat=lambda **v: progress.append(v))
    assert [e.reward for e in episodes] == list(map(float, seeds))
    assert [len(e.transitions) for e in episodes] == [1, 2, 3, 1, 2]
    assert 2 in batches and all(scored)
    assert all(event["valid"] for event in traced)
    assert progress[-1]["episodes_completed"] == 5
    assert progress[-1]["active_envs"] == 0
    episodes = collect_many(None, Tokenizer(), MemoryPool(), Config(), [23, 23], sample=False)
    assert all(t.old_logps is None for e in episodes for t in e.transitions)
    assert scored[-1] is False


def test_scheduler_stop_does_not_return_partial_training_group():
    with pytest.raises(RolloutCancelled):
        collect_many(None, None, MemoryPool(), Config(), [23, 23], should_stop=lambda: True)


class StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.forwards = 0

    def generate(self, input_ids, attention_mask, **kwargs):
        self.inputs, self.mask = input_ids, attention_mask
        self.kwargs = kwargs
        result = torch.tensor([[4, 1, 0], [5, 6, 1]], device=input_ids.device)
        return torch.cat((input_ids, result), dim=1)

    def forward(self, **kw):
        self.forwards += 1
        raise AssertionError("Scoring must not run during evaluation")

    def disable_adapter(self):
        raise AssertionError("Reference forward must not run during evaluation")


def test_batched_evaluation_padding_eos_and_no_scoring():
    model = StubModel()
    results = generate_batch(model, Tokenizer(), [[2, 3], [2, 3, 4]],
                             Config(generation_use_cache=True), sample=False, score=False)
    assert model.inputs.tolist() == [[0, 2, 3], [2, 3, 4]]
    assert model.mask.tolist() == [[0, 1, 1], [1, 1, 1]]
    assert [r[0] for r in results] == [[4, 1], [5, 6, 1]]
    assert all(r[1:] == (None, None) for r in results)
    assert model.kwargs["use_cache"] is True
    assert model.forwards == 0


def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    config = GPT2Config(vocab_size=32, n_positions=128, n_embd=32, n_layer=1, n_head=2,
                        bos_token_id=2, eos_token_id=1, pad_token_id=0,
                        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    return GPT2LMHeadModel(config).eval()


def test_batched_scores_match_individual_causal_scores():
    torch.manual_seed(23)
    model = tiny_model()
    prompts, completions = [[2, 3], [2, 4, 5, 6]], [[8, 1], [9, 10, 1]]
    scores = batch_token_logps(model, prompts, completions, 0)
    for p, c, batch in zip(prompts, completions, scores):
        assert torch.allclose(batch, token_logps(model, p, c), atol=1e-6)
    (-sum(s.sum() for s in scores)).backward()
    assert model.transformer.wte.weight.grad is not None


def test_real_model_batched_generation_and_cache_agree():
    torch.manual_seed(7)
    model = tiny_model()
    model.disable_adapter = nullcontext
    prompts = [[2, 3], [2, 4, 5, 6]]
    uncached = generate_batch(model, Tokenizer(), prompts, Config(completion_tokens=4, beta=0),
                              sample=False, score=False)
    cached = generate_batch(model, Tokenizer(), prompts,
                            Config(completion_tokens=4, beta=0, generation_use_cache=True),
                            sample=False, score=False)
    assert [r[0] for r in cached] == [r[0] for r in uncached]
    for ids, result in zip(prompts, cached):
        alone = generate_batch(model, Tokenizer(), [ids], Config(completion_tokens=4, beta=0),
                               sample=False, score=False)
        assert result[0] == alone[0][0]
