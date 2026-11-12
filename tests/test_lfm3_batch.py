"""Validate the custom LFM3 hybrid/cache API with tiny random weights; no full weights download."""
import os

import pytest
import torch

from bulfm.config import LORA_TARGETS, Config
from bulfm.grpo import batch_token_logps, token_logps
from bulfm.policy import generate_batch


@pytest.mark.skipif(os.environ.get("BULFM_LFM3_TEST") != "1", reason="LFM3 source download opt-in")
def test_lfm3_hybrid_cache_batched_scores_and_lora_backward():
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(23)
    config = AutoConfig.from_pretrained("LiquidAI/LFM3-1.7B", trust_remote_code=True)
    for name, value in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            layer_types=["conv", "sliding_attention", "full_attention"],
                            vocab_size=128, pad_token_id=0, eos_token_id=1, bos_token_id=2,
                            max_position_embeddings=512, sliding_window=16).items():
        setattr(config, name, value)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                            attn_implementation="eager").float()
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=LORA_TARGETS,
                                            task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
    prompts = [[2, 3, 4], [2, 5, 6, 7, 8, 9, 10]]
    c = Config(device="cpu", precision="fp32", completion_tokens=8, beta=0)
    uncached = generate_batch(model, Tokenizer(), prompts, c, sample=False, score=False)
    cached = generate_batch(model, Tokenizer(), prompts, c.model_copy(
        update={"generation_use_cache": True}), sample=False, score=False)
    assert [r[0] for r in uncached] == [r[0] for r in cached]
    completions = [[11, 1], [12, 13, 14, 1]]
    batched = batch_token_logps(model, prompts, completions, 0)
    for prompt, completion, batch in zip(prompts, completions, batched):
        assert torch.allclose(batch, token_logps(model, prompt, completion), atol=1e-5)
    model.train()
    scores = batch_token_logps(model, prompts, completions, 0)
    (-sum(s.mean() for s in scores)).backward()
    assert any(p.grad is not None for name, p in model.named_parameters() if "lora_" in name)
