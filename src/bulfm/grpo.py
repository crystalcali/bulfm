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
