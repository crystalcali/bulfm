from dataclasses import dataclass, field

import torch

from .environment import parse_action
from .policy import generate, generate_batch, prompt_ids


@dataclass
class Transition:
    prompt: list[int]
    completion: list[int]
    old_logps: torch.Tensor | None
    reference: torch.Tensor | None


@dataclass
class Episode:
    transitions: list[Transition] = field(default_factory=list)
    reward: float = 0.0
    success: bool = False
    invalid_actions: int = 0


class RolloutCancelled(Exception):
    """Discard an unfinished on-policy group when stopping."""


def collect(model, tok, env, c, seed, trace=None, sample=True, heartbeat=None, should_stop=None):
    # Every member of one GRPO group resets to the SAME task seed.
    obs, history = env.reset(seed), []
    episode = Episode()
    for turn in range(c.episode_steps):
        if should_stop and should_stop():
            raise RolloutCancelled
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
            heartbeat(turn=turn + 1, action=action, valid_action=valid, last_response=text,
                      last_reward=reward, active_envs=1)
        obs = next_obs
        if done:
            break
    return episode



def collect_many(model, tok, pool, c, seeds, trace=None, sample=True, heartbeat=None,
                 should_stop=None):
    """Collect independent episodes with shared generation; results retain seed/member order."""
    if not seeds:
        return []
    episodes = [Episode() for _ in seeds]
    active, cursor, completed = {}, 0, 0
    while active or cursor < len(seeds):
        if should_stop and should_stop():
            raise RolloutCancelled
        free = [i for i in range(pool.size) if i not in active]
        count = min(len(free), len(seeds) - cursor)
        if count:
            indices = free[:count]
            observations = pool.reset_many(indices, seeds[cursor:cursor + count])
            for slot, obs in zip(indices, observations):
                active[slot] = {"member": cursor, "obs": obs, "history": [], "turn": 0}
                cursor += 1
        slots = sorted(active)
        generated = {}
        # Send browser work immediately after each model batch, overlapping it with later batches.
        for start in range(0, len(slots), c.generation_batch_size):
            if should_stop and should_stop():
                raise RolloutCancelled
            subset = slots[start:start + c.generation_batch_size]
            prompts = [prompt_ids(tok, c, active[i]["obs"], active[i]["history"]) for i in subset]
            if heartbeat:
                heartbeat(active_envs=len(slots), generation_batch=len(subset),
                          episodes_completed=completed, episodes_total=len(seeds))
            outputs = generate_batch(model, tok, prompts, c, sample=sample, score=sample)
            actions = []
            for slot, ids, (completion, old, reference) in zip(subset, prompts, outputs):
                text = tok.decode(completion, skip_special_tokens=True)
                action, valid = parse_action(text)
                generated[slot] = (Transition(ids, completion, old, reference), text, action, valid)
                actions.append(action)
            pool.send_steps(subset, actions)
        results = pool.receive_steps(slots)
        for slot, (obs, reward, done) in zip(slots, results):
            current = active[slot]
            transition, text, action, valid = generated[slot]
            member = current["member"]
            ep = episodes[member]
            ep.transitions.append(transition)
            ep.reward += reward
            ep.success |= reward > 0
            ep.invalid_actions += not valid
            if trace:
                trace({"seed": seeds[member], "member": member, "env": slot,
                       "turn": current["turn"], "goal": current["obs"]["goal"],
                       "page": current["obs"]["tree"], "response": text, "action": action,
                       "valid": valid, "reward": reward, "done": done, "error": obs["error"]})
            current["turn"] += 1
            current["history"].append(action)
            current["obs"] = obs
            if heartbeat:
                heartbeat(turn=current["turn"], action=action, valid_action=valid,
                          last_response=text, last_reward=reward, env=slot)
            if done or current["turn"] >= c.episode_steps:
                completed += 1
                del active[slot]
        if heartbeat:
            heartbeat(episodes_completed=completed, episodes_total=len(seeds), active_envs=len(active))
    return episodes
