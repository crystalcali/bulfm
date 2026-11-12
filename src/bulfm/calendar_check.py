"""Compare prompts on a saved adapter, without training or changing the source run."""
import fcntl
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import READONLY_HINT, Config
from .storage import alive, atomic_json, read_json, safe_run

OLD_HINT = "A readonly textbox cannot be filled: interact with its visible picker controls instead."


def repaired_prompt(prompt):
    if READONLY_HINT in prompt:
        return prompt
    if OLD_HINT in prompt:
        return prompt.replace(OLD_HINT, READONLY_HINT)
    return prompt.rstrip() + "\n" + READONLY_HINT


@contextmanager
def stopped_run(root, run_id):
    source = safe_run(Path(root), run_id)
    if alive(read_json(source / "state.json")):
        raise ValueError("Stop this run first and wait for stopped: /stop " + run_id)
    with (source / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This run is busy; wait for its worker/check to finish") from None
        yield source


def apply_fix(root, run_id):
    with stopped_run(root, run_id) as source:
        c = Config.load(source / "config.yaml")
        new_prompt = repaired_prompt(c.system_prompt)
        if new_prompt == c.system_prompt and c.native_tools:
            print("Readonly instruction already updated; checkpoint unchanged.", flush=True)
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        backup = source / f"config.before-readonly-fix-{stamp}.yaml"
        backup.write_bytes((source / "config.yaml").read_bytes())
        c.system_prompt = new_prompt
        c.native_tools = True
        temporary = source / "config.readonly-fix.tmp"
        c.save(temporary)
        temporary.replace(source / "config.yaml")
        atomic_json(source / "readonly-fix.json", {
            "time": time.time(), "backup": backup.name, "system_prompt": new_prompt,
            "checkpoint": read_json(source / "latest.json"),
        })
        print(f"Updated {source / 'config.yaml'}\nBackup: {backup}\n"
              "Model, optimizer and curriculum history preserved. Worker remains stopped.", flush=True)


def run_check(root, run_id, episodes=2):
    if not 1 <= episodes <= 32:
        raise ValueError("Calendar check episodes must be between 1 and 32")
    from .checkpoint import latest
    from .env_pool import BrowserPool
    from .environment import Browser
    from .policy import load_policy, seed_all
    from .rollout import collect, collect_many

    with stopped_run(root, run_id) as source:
        original = Config.load(source / "config.yaml")
        if not original.lora:
            raise ValueError("Calendar check currently accepts LoRA checkpoints only")
        adapter = latest(source)
        # Always compare the pre-fix prompt when calendar-fix has already been applied.
        fix = read_json(source / "readonly-fix.json")
        if fix.get("backup"):
            original = Config.load(source / fix["backup"])
        n = min(original.num_envs, episodes, 4)
        c = original.model_copy(update={
            "curriculum": None, "task": "miniwob.choose-date-easy", "episode_steps": 8,
            "num_envs": n, "generation_batch_size": min(original.generation_batch_size, n),
            "gradient_checkpointing": False,
        })
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        dest = Path(root).resolve() / "checks" / stamp
        dest.mkdir(parents=True)
        c.save(dest / "config.yaml")
        print(f"Calendar check: {dest}\nAdapter: {adapter}\n"
              "Greedy original/native tool protocol, same held-out seeds. No training.", flush=True)
        seed_all(c.seed)
        model, tok = load_policy(c, adapter=adapter)
        env = BrowserPool(c) if n > 1 else Browser(c)
        results = {}
        try:
            for label, prompt in (("original", original.system_prompt),
                                  ("repaired", repaired_prompt(original.system_prompt))):
                trial = c.model_copy(update={"system_prompt": prompt,
                                             "native_tools": original.native_tools if label == "original" else True})
                trial.save(dest / f"config-{label}.yaml")
                seed_all(c.seed)
                traces = []
                start = time.monotonic()
                def trace(event):
                    event = {"task": c.task, "phase": label, **event}
                    traces.append(event)
                    with (dest / "episodes.jsonl").open("a") as f:
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")
                def progress(**values):
                    if "action" in values:
                        print(f"{label}: turn={values.get('turn')} action={values['action']} "
                              f"valid={values.get('valid_action')} reward={values.get('last_reward')}",
                              flush=True)
                seeds = [c.eval_seed + i for i in range(episodes)]
                if n > 1:
                    batch = collect_many(model, tok, env, trial, seeds, sample=False,
                                         trace=trace, heartbeat=progress)
                else:
                    batch = [collect(model, tok, env, trial, seed, sample=False,
                                     trace=trace, heartbeat=progress) for seed in seeds]
                count = max(1, len(traces))
                result = {"native_tools": trial.native_tools, "episodes": len(batch), "success_rate": sum(e.success for e in batch) / len(batch),
                          "mean_reward": sum(e.reward for e in batch) / len(batch),
                          "valid_action_rate": sum(e["valid"] for e in traces) / count,
                          "environment_error_rate": sum(bool(e["error"]) for e in traces) / count,
                          "seconds": time.monotonic() - start}
                results[label] = result
                print(label + ": " + json.dumps(result), flush=True)
                atomic_json(dest / "report.json", {"source_run": run_id, "adapter": str(adapter),
                                                    "seeds": seeds, "results": results})
        finally:
            env.close()
        good = results["repaired"]["success_rate"] > 0 and results["repaired"]["valid_action_rate"] >= .5
        print("Diagnosis: " + ("Some usable actions and successes; try the fix and monitor RL."
                               if good else "Prompt repair alone is insufficient in this check; inspect traces before more RL."),
              flush=True)
        print(f"Report: {dest / 'report.json'}\nSource run and checkpoint were not modified.", flush=True)
        return dest
