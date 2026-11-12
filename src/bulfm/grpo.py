"""GRPO math, isolated from environment/model loading for numerical tests."""
import torch


def advantages(rewards):
    r = torch.as_tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std(unbiased=False) + 1e-4)


def token_logps(model, prompt, completion):
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + completion], device=device)
    # Shift by one: each action token is predicted from preceding context.
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    logits = logits[0, len(prompt) - 1:-1].float()
    labels = ids[0, len(prompt):]
    return logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)


def surrogate(logps, old_logps, advantage, epsilon, reference=None, beta=0.0):
    ratio = torch.exp(logps - old_logps)
    obj = torch.minimum(ratio * advantage, ratio.clamp(1-epsilon, 1+epsilon) * advantage)
    if reference is not None and beta:
        delta = reference - logps
        kl = torch.exp(delta) - delta - 1
        obj = obj - beta * kl
    return -obj


def batch_token_logps(model, prompts, completions, pad_token_id):
    """Score unpadded trajectories together; prompt/padding tokens never enter the loss."""
    if not prompts or len(prompts) != len(completions):
        raise ValueError("Expected matching non-empty prompt/completion batches")
    device = next(model.parameters()).device
    lengths = [len(p) + len(c) for p, c in zip(prompts, completions)]
    ids = torch.full((len(prompts), max(lengths)), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, (prompt, completion) in enumerate(zip(prompts, completions)):
        if not prompt or not completion:
            raise ValueError("Empty prompt/completion in scoring batch")
        ids[row, :lengths[row]] = torch.tensor(prompt + completion, device=device)
        mask[row, :lengths[row]] = 1
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    result = []
    for row, (prompt, completion) in enumerate(zip(prompts, completions)):
        start = len(prompt) - 1
        action_logits = logits[row, start:start + len(completion)].float()
        labels = ids[row, len(prompt):lengths[row]]
        result.append(action_logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1))
    return result
