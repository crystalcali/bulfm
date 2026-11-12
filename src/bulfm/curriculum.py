"""Deterministic task selection and promotion from completed training groups only."""
import random
from collections import defaultdict, deque


class Curriculum:
    def __init__(self, config):
        self.config = config
        self.stage = 0
        self.history = defaultdict(lambda: deque(maxlen=config.window_episodes))

    def observe(self, task, successes, valid_rates):
        self.history[task].extend(zip(successes, valid_rates))
        while self.stage < len(self.config.stages) - 1:
            ready = True
            for item in self.config.stages[self.stage].tasks:
                stats = self.stats(item.task)
                ready &= (stats["episodes"] >= self.config.min_episodes
                          and stats["success_rate"] >= self.config.promote_success
                          and stats["valid_action_rate"] >= self.config.promote_valid)
            if not ready:
                break
            self.stage += 1

    def stats(self, task):
        rows = list(self.history[task])
        n = len(rows)
        return {"episodes": n,
                "success_rate": sum(x[0] for x in rows) / n if n else 0,
                "valid_action_rate": sum(x[1] for x in rows) / n if n else 0}

    def choose(self, seed):
        rng = random.Random(seed)
        replay = [t for stage in self.config.stages[:self.stage] for t in stage.tasks]
        tasks = replay if replay and rng.random() < self.config.replay_fraction else self.config.stages[self.stage].tasks
        # Prefer tasks with mixed outcomes, while retaining exploration of unsolved tasks.
        weights = [0.1 + (lambda p: p * (1 - p))(self.stats(t.task)["success_rate"]) for t in tasks]
        return rng.choices(tasks, weights=weights, k=1)[0]

    def unlocked(self):
        return [t for stage in self.config.stages[:self.stage + 1] for t in stage.tasks]

    def snapshot(self):
        return {"stage": self.stage, "name": self.config.stages[self.stage].name,
                "tasks": {t.task: self.stats(t.task) for t in self.unlocked()}}

    def restore(self, events, step):
        # Ignore events ahead of the checkpoint when resuming an interrupted run.
        completed = {event["step"]: event for event in events
                     if event.get("event") == "train" and event.get("step", 0) <= step}
        for _, event in sorted(completed.items()):
            for task, stats in event.get("task_stats", {}).items():
                self.observe(task, stats["successes"], stats["valid_rates"])
