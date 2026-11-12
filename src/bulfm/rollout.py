from dataclasses import dataclass, field

import torch

from .environment import parse_action
from .policy import generate, prompt_ids


@dataclass
class Transition:
    prompt: list[int]
    completion: list[int]
    old_logps: torch.Tensor
    reference: torch.Tensor | None


@dataclass
class Episode:
    transitions: list[Transition] = field(default_factory=list)
    reward: float = 0.0
    success: bool = False
    invalid_actions: int = 0


def collect(model, tok, env, c, seed, trace=None, sample=True, heartbeat=None):
    # Every member of one GRPO group resets to the SAME task seed.
    obs, history = env.reset(seed), []
    episode = Episode()
    for turn in range(c.episode_steps):
        ids = prompt_ids(tok, c, obs, history)
        completion, old, reference = generate(model, tok, ids, c, sample=sample)
        text = tok.decode(completion, skip_special_tokens=True)
        action, valid = parse_action(text)
        next_obs, reward, done = env.step(action)
        episode.transitions.append(Transition(ids, completion, old, reference))
        episode.reward += reward
        episode.success |= reward > 0
        episode.invalid_actions += not valid
        history.append(action)
        if trace:
            trace({"seed": seed, "turn": turn, "goal": obs["goal"], "page": obs["tree"],
                   "response": text, "action": action, "valid": valid, "reward": reward,
                   "done": done, "error": next_obs["error"]})
        if heartbeat:
            heartbeat(turn=turn + 1, action=action)
        obs = next_obs
        if done:
            break
    return episode
