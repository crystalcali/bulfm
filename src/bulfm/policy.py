import random

import numpy as np
import torch

from .config import Config
from .grpo import token_logps


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_policy(c: Config, adapter=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if c.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check your Spark PyTorch installation with bulfm doctor")
    if c.device == "cuda" and c.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bf16")
    dtype = torch.bfloat16 if c.precision == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(c.model, revision=c.model_revision,
                                       trust_remote_code=c.trust_remote_code, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        c.model, revision=c.model_revision, trust_remote_code=c.trust_remote_code,
        dtype=dtype, attn_implementation="eager",
    ).to(c.device)
    if c.lora:
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(
                r=c.lora_rank, lora_alpha=c.lora_alpha, lora_dropout=0.0,
                target_modules=c.lora_targets, task_type="CAUSAL_LM", bias="none",
            ))
    # Equal policy in rollout eval() and optimization train(); checkpointing can stay enabled.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if c.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model, tok


def prompt_ids(tok, c, obs, history):
    # Shrink only observation/history, never left-truncate away the system instructions or goal.
    budget = c.context_tokens - c.completion_tokens
    tree, past = obs["tree"], "\n".join(history[-10:])
    for _ in range(40):
        messages = [{"role": "system", "content": c.system_prompt}, {"role": "user", "content":
            f"Goal: {obs['goal']}\nPrevious actions: {past}\n"
            f"Last error: {obs['error']}\nCurrent page:\n{tree}\nNext action:"}]
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False)
        if len(ids) <= budget:
            return ids
        if past:
            past = past[len(past)//2 + 1:]
        elif tree:
            tree = tree[:len(tree)//2]
        else:
            raise ValueError("System prompt and goal exceed context_tokens; increase context")
    raise ValueError("Could not fit prompt in context")


@torch.no_grad()
def generate(model, tok, ids, c, sample=True):
    model.eval()
    inputs = torch.tensor([ids], device=next(model.parameters()).device)
    kwargs = {"do_sample": sample}
    if sample:
        kwargs.update(temperature=1.0, top_p=1.0, top_k=0)
    # Uncached decoding works for hybrid operators without relying on vLLM/third-party kernels.
    out = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                         max_new_tokens=c.completion_tokens, pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id, use_cache=False,
                         repetition_penalty=1.0, **kwargs)
    completion = out[0, len(ids):].tolist()
    if not completion:
        raise RuntimeError("Model generated an empty completion")
    old = token_logps(model, ids, completion).detach().cpu()
    reference = None
    if c.beta:
        with model.disable_adapter():
            reference = token_logps(model, ids, completion).detach().cpu()
    return completion, old, reference
