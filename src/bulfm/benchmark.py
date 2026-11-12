"""One model, short real rollouts and one backward/update at each environment count."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def run_benchmark(config_path, root="runs"):
    import psutil
    import torch

    from .config import Config
    from .env_pool import BrowserPool
    from .grpo import advantages, surrogate, token_logps
    from .policy import load_policy, seed_all
    from .rollout import collect_many
    from .storage import atomic_json

    c = Config.load(config_path)
    if not c.lora:
        raise ValueError("Resource benchmark requires LoRA to restore initial adapters between trials")
    c = c.model_copy(update={"episode_steps": min(3, c.episode_steps),
                             "completion_tokens": min(64, c.completion_tokens)})
    seed_all(c.seed)
    model, tok = load_policy(c)
    parameters = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().cpu().clone() for p in parameters]
    output = Path(root).resolve() / "benchmarks" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    report = {"model": c.model, "task": c.task, "episodes_per_trial": 8,
              "episode_steps": c.episode_steps, "completion_tokens": c.completion_tokens,
              "generation_use_cache": c.generation_use_cache,
              "note": "Short resource probe, not a quality benchmark or a full-context memory guarantee",
              "results": []}
    atomic_json(output / "report.json", report)
    print(f"Benchmark: {output}\nRun without other GPU workloads for comparable timings.", flush=True)
    for count in (1, 4, 8):
        config = c.model_copy(update={"num_envs": count, "generation_batch_size": count})
        with torch.no_grad():
            for parameter, saved in zip(parameters, initial):
                parameter.copy_(saved)
        seed_all(c.seed)
        model.zero_grad(set_to_none=True)
        if c.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW(parameters, lr=c.learning_rate, weight_decay=0.0)
        try:
            started = time.monotonic()
            with BrowserPool(config) as pool:
                startup = time.monotonic() - started
                started = time.monotonic()
                episodes = collect_many(model, tok, pool, config, [c.seed] * 8)
                rollout_seconds = time.monotonic() - started
                started = time.monotonic()
                advs = advantages([e.reward for e in episodes])
                model.train()
                for episode, adv in zip(episodes, advs):
                    length = sum(len(t.completion) for t in episode.transitions)
                    for tr in episode.transitions:
                        lp = token_logps(model, tr.prompt, tr.completion)
                        reference = tr.reference.to(lp.device) if tr.reference is not None else None
                        loss = surrogate(lp, tr.old_logps.to(lp.device), adv.to(lp.device),
                                         c.clip_epsilon, reference, c.beta).sum() / (length * 8)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite benchmark loss")
                        loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, c.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                if c.device == "cuda":
                    torch.cuda.synchronize()
                backward_seconds = time.monotonic() - started
                tokens = sum(len(t.completion) for ep in episodes for t in ep.transitions)
                row = {"num_envs": count, "startup_seconds": startup,
                       "rollout_seconds": rollout_seconds, "backward_seconds": backward_seconds,
                       "episodes_per_minute": 8 * 60 / rollout_seconds,
                       "rollout_tokens_per_second": tokens / rollout_seconds,
                       "cycle_seconds": rollout_seconds + backward_seconds,
                       "gpu_peak_gb": torch.cuda.max_memory_reserved() / 1e9 if c.device == "cuda" else 0,
                       "ram_available_gb": psutil.virtual_memory().available / 1e9}
        except torch.OutOfMemoryError as exc:
            row = {"num_envs": count, "error": str(exc), "status": "out_of_memory"}
        report["results"].append(row)
        atomic_json(output / "report.json", report)
        print(json.dumps(row, indent=2), flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        if row.get("status") == "out_of_memory":
            break
    successful = [r for r in report["results"] if "cycle_seconds" in r]
    if successful:
        report["fastest_measured_num_envs"] = min(successful, key=lambda r: r["cycle_seconds"])["num_envs"]
        atomic_json(output / "report.json", report)
        print(f"Fastest measured configuration: {report['fastest_measured_num_envs']} environments", flush=True)
    print(f"Saved: {output / 'report.json'}", flush=True)
    return output / "report.json"
