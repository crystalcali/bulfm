"""Detached training worker. All accelerator imports stay outside the terminal/dashboard."""
import argparse
import fcntl
import json
import os
import signal
import time
import traceback
from pathlib import Path

import psutil

from .config import Config
from .diagnostics import error_message
from .storage import append_event, atomic_json, read_json


class Worker:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.c = Config.load(self.run / "config.yaml")
        self.stop = False
        self.state = {**read_json(self.run / "state.json"), "pid": os.getpid(),
                      "process_started": psutil.Process().create_time()}
        self.last_heartbeat = 0
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "stop", True))

    def update(self, **values):
        self.state.update(values, updated=time.time())
        atomic_json(self.run / "state.json", self.state)

    def heartbeat(self, **values):
        self.update(**values)

    def stopping(self):
        return self.stop or (self.run / "STOP").exists()

    def trace(self, event):
        with (self.run / "episodes.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), **event}, ensure_ascii=False) + "\n")

    def switch_task(self, env, task_config):
        if getattr(self, "env_task", None) == task_config.task:
            return env
        if env is not None:
            env.close()
        from .env_pool import BrowserPool
        from .environment import Browser
        env = BrowserPool(task_config) if task_config.num_envs > 1 else Browser(task_config)
        self.env_task = task_config.task
        return env

    def evaluate(self, model, tok, env, step, phase):
        base = self.c
        tasks = self.curriculum.unlocked() if self.curriculum else [base]
        results = []
        try:
            for task in tasks:
                if self.stopping():
                    break
                count = base.curriculum.eval_episodes_per_task if self.curriculum else base.eval_episodes
                self.c = base.model_copy(update={"task": task.task, "episode_steps": task.episode_steps,
                                                 "eval_episodes": count})
                env = self.switch_task(env, self.c)
                self.update(task=task.task)
                result = self._evaluate_task(model, tok, env, step, phase)
                if result:
                    results.append(result)
            if not self.stopping():
                total = sum(r["episodes"] for r in results)
                atomic_json(self.run / f"evaluation-{phase}.json", {
                    "complete": len(results) == len(tasks) and all(r["complete"] for r in results),
                    "step": step, "episodes": total, "by_task": results,
                    "success_rate": sum(r["success_rate"] * r["episodes"] for r in results) / max(1, total)})
        finally:
            self.c = base
        return env

    def _evaluate_task(self, model, tok, env, step, phase):
        import numpy as np

        from .rollout import RolloutCancelled, collect, collect_many
        episodes = []
        start = time.monotonic()
        self.update(status="evaluating", phase=phase, eval_episode=0,
                    eval_completed=0, eval_total=self.c.eval_episodes)
        for offset in range(0, self.c.eval_episodes, self.c.num_envs):
            if self.stopping():
                break
            seeds = [self.c.eval_seed + i for i in
                     range(offset, min(offset + self.c.num_envs, self.c.eval_episodes))]
            self.update(eval_episode=offset + 1)
            def progress(**values):
                if "episodes_completed" in values:
                    values["eval_completed"] = offset + values["episodes_completed"]
                values["eval_seconds"] = time.monotonic() - start
                self.heartbeat(**values)
            def trace(event):
                self.trace({"evaluation": True, "phase": phase, "task": self.c.task, **event})
            try:
                if self.c.num_envs > 1:
                    batch = collect_many(model, tok, env, self.c, seeds, sample=False,
                                         trace=trace, heartbeat=progress, should_stop=self.stopping)
                else:
                    batch = [collect(model, tok, env, self.c, seeds[0], sample=False,
                                     trace=trace, heartbeat=progress, should_stop=self.stopping)]
            except RolloutCancelled:
                break
            episodes.extend(batch)
            self.update(eval_completed=len(episodes), eval_seconds=time.monotonic() - start)
            print(f"Evaluation {phase}: {len(episodes)}/{self.c.eval_episodes} episodes", flush=True)
        result = None
        if episodes:
            result = {"step": step, "phase": phase, "task": self.c.task, "episodes": len(episodes),
                      "complete": len(episodes) == self.c.eval_episodes,
                      "seconds": time.monotonic() - start,
                      "success_rate": float(np.mean([e.success for e in episodes])),
                      "mean_reward": float(np.mean([e.reward for e in episodes]))}
            append_event(self.run, "eval", **result)
            atomic_json(self.run / f"evaluation-{phase}-{self.c.task}.json", result)
            print(f"Evaluation {phase}: {result}", flush=True)
        self.update(status="running", phase="training", active_envs=0)
        return result

    def execute(self, resume=False, evaluate_only=False):
        import numpy as np
        import torch

        from . import checkpoint
        from .grpo import advantages, surrogate, token_logps
        from .policy import load_policy, seed_all
        from .rollout import RolloutCancelled, collect, collect_many

        c = self.c
        seed_all(c.seed)
        self.update(status="running", phase="loading", total_steps=c.steps, num_envs=c.num_envs,
                    generation_batch_size=c.generation_batch_size,
                    generation_use_cache=c.generation_use_cache)
        checkpoint_path = checkpoint.latest(self.run) if resume else None
        adapter = self.state.get("evaluation_adapter") or (checkpoint_path if c.lora else None)
        # A full-tuning checkpoint is a standalone model, not a PEFT adapter.
        load_c = c.model_copy(update={"model": str(checkpoint_path), "model_revision": "main"}) \
            if checkpoint_path and not c.lora else c
        model, tok = load_policy(load_c, adapter=adapter)
        resolved = getattr(model.config, "_commit_hash", None)
        if resolved and c.model_revision == "main" and not checkpoint_path:
            c.model_revision = resolved
            c.save(self.run / "config.yaml")
        # Record resolved model commit and actual runtime, even if the config used 'main'.
        from importlib.metadata import version
        atomic_json(self.run / "runtime.json", {
            "torch": torch.__version__, "transformers": version("transformers"),
            "peft": version("peft"), "device": str(next(model.parameters()).device),
            "gpu": torch.cuda.get_device_name() if c.device == "cuda" else None,
            "resolved_model_commit": getattr(model.config, "_commit_hash", None),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=c.learning_rate, weight_decay=0.0)
        step = checkpoint.restore(checkpoint_path, optimizer) if checkpoint_path else 0
        self.update(step=step)
        from .curriculum import Curriculum
        self.curriculum = Curriculum(c.curriculum) if c.curriculum else None
        if self.curriculum:
            events = []
            metrics_path = self.run / "metrics.jsonl"
            if metrics_path.exists():
                for line in metrics_path.read_text().splitlines():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            self.curriculum.restore(events, step)
        env = self.switch_task(None, c)
        try:
            if evaluate_only:
                env = self.evaluate(model, tok, env, step, "manual")
                self.update(status="stopped" if self.stopping() else "completed", phase="evaluation")
                return
            baseline = read_json(self.run / "evaluation-before.json")
            if step == 0 and c.baseline_eval and not baseline.get("complete", bool(baseline)):
                env = self.evaluate(model, tok, env, 0, "before")
            while step < c.steps and not self.stopping():
                start = time.monotonic()
                self.update(status="running", phase="rollouts", step=step, eval_completed=0, eval_total=0)
                optimizer.zero_grad(set_to_none=True)
                all_rewards, all_success, all_tokens = [], [], 0
                task_stats = {}
                if c.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                loss_value, zero_groups, invalid_actions, actions = 0.0, 0, 0, 0
                rollout_seconds, backward_seconds, interrupted = 0.0, 0.0, False
                for group in range(c.groups_per_step):
                    selected = self.curriculum.choose(c.seed + step * c.groups_per_step + group) if self.curriculum else c
                    group_c = c.model_copy(update={"task": selected.task, "episode_steps": selected.episode_steps})
                    env = self.switch_task(env, group_c)
                    self.update(group=group + 1, phase="rollouts", task=group_c.task,
                                curriculum=self.curriculum.snapshot() if self.curriculum else None)
                    task_seed = c.seed + step * c.groups_per_step + group
                    collect_started = time.monotonic()
                    try:
                        if c.num_envs > 1:
                            episodes = collect_many(
                                model, tok, env, group_c, [task_seed] * c.group_size,
                                trace=lambda e: self.trace({"task": group_c.task, "phase": "train", **e}),
                                heartbeat=self.heartbeat, should_stop=self.stopping)
                        else:
                            episodes = []
                            for member in range(c.group_size):
                                self.update(group=group + 1, member=member + 1)
                                episodes.append(collect(model, tok, env, group_c, task_seed,
                                                        trace=lambda e: self.trace({"task": group_c.task, "phase": "train", **e}),
                                                        heartbeat=self.heartbeat, should_stop=self.stopping))
                    except RolloutCancelled:
                        interrupted = True
                        break
                    rollout_seconds += time.monotonic() - collect_started
                    stats = task_stats.setdefault(group_c.task, {"successes": [], "valid_rates": [], "rewards": []})
                    stats["successes"].extend(ep.success for ep in episodes)
                    stats["valid_rates"].extend(1 - ep.invalid_actions / max(1, len(ep.transitions)) for ep in episodes)
                    stats["rewards"].extend(ep.reward for ep in episodes)
                    rewards = [ep.reward for ep in episodes]
                    advs = advantages(rewards)
                    zero_groups += float(np.std(rewards)) < 1e-8
                    all_rewards.extend(rewards)
                    all_success.extend(ep.success for ep in episodes)
                    self.update(phase="backward")
                    backward_started = time.monotonic()
                    model.train()
                    for ep, adv in zip(episodes, advs):
                        length = sum(len(t.completion) for t in ep.transitions)
                        all_tokens += length
                        invalid_actions += ep.invalid_actions
                        actions += len(ep.transitions)
                        # Per-trajectory token mean, then mean over the GRPO group.
                        denominator = length * c.group_size * c.groups_per_step
                        for tr in ep.transitions:
                            if tr.old_logps is None:
                                raise RuntimeError("Training transition missing old-policy scores")
                            lp = token_logps(model, tr.prompt, tr.completion)
                            ref = tr.reference.to(lp.device) if tr.reference is not None else None
                            loss = surrogate(lp, tr.old_logps.to(lp.device), adv.to(lp.device),
                                             c.clip_epsilon, ref, c.beta).sum() / denominator
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Non-finite GRPO loss; previous checkpoint preserved")
                            loss.backward()
                            loss_value += loss.detach().item()
                    model.eval()
                    backward_seconds += time.monotonic() - backward_started
                if interrupted:
                    optimizer.zero_grad(set_to_none=True)
                    break
                norm = torch.nn.utils.clip_grad_norm_(parameters, c.max_grad_norm,
                                                     error_if_nonfinite=True)
                lr = c.learning_rate * min(1.0, (step + 1) / max(1, c.warmup_steps))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()
                step += 1
                if self.curriculum:
                    for task, stats in task_stats.items():
                        self.curriculum.observe(task, stats["successes"], stats["valid_rates"])
                duration = time.monotonic() - start
                metrics = {
                    "task_stats": task_stats,
                    "curriculum": self.curriculum.snapshot() if self.curriculum else None,
                    "learning_signal": "no_reward_variation" if zero_groups == c.groups_per_step else "mixed_rewards",
                    "step": step, "loss": loss_value, "reward": float(np.mean(all_rewards)),
                    "success_rate": float(np.mean(all_success)), "lr": lr,
                    "grad_norm": float(norm), "step_seconds": duration,
                    "num_envs": c.num_envs, "generation_batch_size": c.generation_batch_size,
                    "rollout_seconds": rollout_seconds, "backward_seconds": backward_seconds,
                    "episodes_per_minute": c.group_size * c.groups_per_step * 60 / rollout_seconds,
                    "rollout_tokens_per_second": all_tokens / rollout_seconds,
                    "action_tokens": all_tokens, "tokens_per_second": all_tokens / duration,
                    "zero_variance_groups": zero_groups / c.groups_per_step,
                    "invalid_action_rate": invalid_actions / max(1, actions),
                    "gpu_peak_gb": torch.cuda.max_memory_reserved() / 1e9 if c.device == "cuda" else 0,
                    "ram_used_gb": psutil.virtual_memory().used / 1e9,
                }
                append_event(self.run, "train", **metrics)
                self.update(step=step, latest_metrics=metrics, curriculum=metrics["curriculum"])
                print(json.dumps(metrics), flush=True)
                if step % c.save_every == 0 or step == c.steps or self.stopping():
                    self.update(phase="saving")
                    checkpoint.save(self.run, model, tok, optimizer, step, c.keep_checkpoints)
                if c.eval_every and step % c.eval_every == 0 and step < c.steps:
                    env = self.evaluate(model, tok, env, step, f"step-{step}")
            # Includes stop-before-first-update: a valid step-0 checkpoint can be resumed.
            checkpoint.save(self.run, model, tok, optimizer, step, c.keep_checkpoints)
            if not self.stopping():
                env = self.evaluate(model, tok, env, step, "after")
            self.update(status="stopped" if self.stopping() else "completed", phase="idle", step=step)
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    with (run / ".worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This run already has a worker") from None
        worker = Worker(run)
        try:
            worker.execute(args.resume, args.evaluate_only)
        except Exception as exc:
            worker.update(status="failed", error=error_message(exc))
            traceback.print_exc()
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
