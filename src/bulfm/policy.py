import random

import numpy as np
import torch

from .config import Config
from .grpo import batch_token_logps


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
    if c.native_tools:
        from .tools_protocol import native_prompt_ids
        return native_prompt_ids(tok, c, obs, history)
    # Shrink only observation/history, never left-truncate away the system instructions or goal.
    budget = c.context_tokens - c.completion_tokens
    tree, past = obs["tree"], "\n".join(h["action"] if isinstance(h, dict) else h for h in history[-10:])
    for _ in range(40):
        messages = [{"role": "system", "content": c.system_prompt}, {"role": "user", "content":
            f"Goal: {obs['goal']}\nPrevious actions: {past}\n"
            f"Last error: {obs['error']}\nCurrent page:\n{tree}\nNext action:"}]
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False, return_dict=False)
        if not isinstance(ids, list) or not ids or not all(type(i) is int for i in ids):
            raise TypeError("Chat template must return a non-empty list of integer token IDs")
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
def generate_batch(model, tok, prompts, c, sample=True, score=True):
    if not prompts:
        return []
    model.eval()
    device = next(model.parameters()).device
    width = max(map(len, prompts))
    inputs = torch.full((len(prompts), width), tok.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    for row, ids in enumerate(prompts):
        if not ids:
            raise ValueError("Empty generation prompt")
        inputs[row, width-len(ids):] = torch.tensor(ids, device=device)
        mask[row, width-len(ids):] = 1
    kwargs = {"do_sample": sample}
    if sample:
        kwargs.update(temperature=1.0, top_p=1.0, top_k=0)
    out = model.generate(input_ids=inputs, attention_mask=mask,
                         max_new_tokens=c.completion_tokens, pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id, use_cache=c.generation_use_cache,
                         repetition_penalty=1.0, **kwargs)
    completions = []
    eos = tok.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    for row in range(len(prompts)):
        completion = out[row, width:].tolist()
        for index, token in enumerate(completion):
            if token in eos:
                completion = completion[:index + 1]
                break
        if not completion:
            raise RuntimeError("Model generated an empty completion")
        completions.append(completion)
    # Greedy evaluation only needs actions/rewards, not old-policy or reference forwards.
    old, reference = [None] * len(prompts), [None] * len(prompts)
    if score:
        old = [p.detach().cpu() for p in batch_token_logps(model, prompts, completions,
                                                         tok.pad_token_id)]
        if c.beta:
            with model.disable_adapter():
                reference = [p.detach().cpu() for p in batch_token_logps(
                    model, prompts, completions, tok.pad_token_id)]
    return list(zip(completions, old, reference))


@torch.no_grad()
def generate(model, tok, ids, c, sample=True):
    return generate_batch(model, tok, [ids], c, sample=sample, score=sample)[0]
