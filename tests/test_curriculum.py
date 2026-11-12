import json

import pytest

from bulfm.config import Config, CurriculumConfig
from bulfm.curriculum import Curriculum


def schedule():
    return CurriculumConfig.model_validate({
        "min_episodes": 4, "window_episodes": 8,
        "stages": [{"name": "easy", "tasks": [{"task": "miniwob.click-test"}]},
                   {"name": "hard", "tasks": [{"task": "miniwob.enter-text"}]}]})


def test_promotion_requires_success_and_validity_and_keeps_replay():
    c = Curriculum(schedule())
    c.observe("miniwob.click-test", [True] * 4, [0.3] * 4)
    assert c.stage == 0
    c.observe("miniwob.click-test", [True] * 8, [1.] * 8)
    assert c.stage == 1
    tasks = [c.choose(seed).task for seed in range(100)]
    assert 15 < tasks.count("miniwob.click-test") < 45
    assert c.choose(42) == c.choose(42)


def test_no_promotion_from_equal_failures():
    c = Curriculum(schedule())
    c.observe("miniwob.click-test", [False] * 100, [1.] * 100)
    assert c.stage == 0
    assert c.stats("miniwob.click-test")["episodes"] == 8


def test_restore_only_checkpoint_history_and_deduplicate_steps():
    c = Curriculum(schedule())
    event = {"event": "train", "step": 1, "task_stats": {
        "miniwob.click-test": {"successes": [True] * 2, "valid_rates": [1.] * 2}}}
    c.restore([event, event, {**event, "step": 2}], step=1)
    assert c.stage == 0
    assert c.stats("miniwob.click-test")["episodes"] == 2
    d = Curriculum(schedule())
    d.restore([event, {**event, "step": 2}], step=2)
    assert d.stage == 1


def test_curriculum_config_validation_and_roundtrip(tmp_path):
    c = Config.load("configs/spark-curriculum.yaml")
    c.save(tmp_path / "config.yaml")
    assert Config.load(tmp_path / "config.yaml") == c
    assert c.curriculum.stages[-1].tasks[0].task == "miniwob.book-flight"
    with pytest.raises(ValueError):
        CurriculumConfig.model_validate({**schedule().model_dump(), "min_episodes": 20})
    duplicate = schedule().model_dump()
    duplicate["stages"].append(duplicate["stages"][0])
    with pytest.raises(ValueError):
        CurriculumConfig.model_validate(duplicate)


def test_worker_changes_tasks_records_metrics_and_resumes(tmp_path, monkeypatch):
    from test_real_parallel import ActionPolicy, ActionTokenizer

    import bulfm.environment
    import bulfm.policy
    from bulfm.storage import snapshot
    from bulfm.worker import Worker

    opened = []
    class Browser:
        def __init__(self, config):
            self.task = config.task
            opened.append(self.task)
        def reset(self, seed):
            return {"goal": self.task, "tree": "[13] button", "error": ""}
        def step(self, action):
            return self.reset(0), 1., True
        def close(self):
            pass
    monkeypatch.setattr(bulfm.environment, "Browser", Browser)
    monkeypatch.setattr(bulfm.policy, "load_policy", lambda *a, **kw: (ActionPolicy(), ActionTokenizer()))
    cur = schedule().model_copy(update={"min_episodes": 2})
    c = Config(steps=3, group_size=2, num_envs=1, beta=0, device="cpu", precision="fp32",
               baseline_eval=False, eval_every=0, save_every=1, curriculum=cur)
    c.save(tmp_path / "config.yaml")
    Worker(tmp_path).execute()
    state = snapshot(tmp_path)
    assert state["step"] == 3
    assert state["curriculum"]["stage"] == 1
    assert set(opened) == {"miniwob.click-test", "miniwob.enter-text"}
    assert state["latest_metrics"]["learning_signal"] == "no_reward_variation"
    traces = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert all("task" in t and "phase" in t for t in traces)
    training = [t for t in traces if not t.get("evaluation")]
    for group in range(3):
        assert len({t["task"] for t in training[group*2:group*2+2]}) == 1
        assert len({t["seed"] for t in training[group*2:group*2+2]}) == 1
    c.steps = 4
    c.save(tmp_path / "config.yaml")
    Worker(tmp_path).execute(resume=True)
    assert snapshot(tmp_path)["step"] == 4
    assert snapshot(tmp_path)["curriculum"]["stage"] == 1
    report = json.loads((tmp_path / "evaluation-after.json").read_text())
    assert len(report["by_task"]) == 2
    assert all(r["complete"] for r in report["by_task"])


@pytest.mark.skipif(__import__('os').environ.get('BULFM_REAL_BROWSER') != '1', reason='Real browser opt-in')
@pytest.mark.parametrize('task', ['miniwob.click-test', 'miniwob.click-button', 'miniwob.enter-text',
                                  'miniwob.choose-date-easy', 'miniwob.choose-date', 'miniwob.book-flight'])
def test_curriculum_real_browser_pages(task):
    from bulfm.environment import Browser
    env = Browser(Config(task=task))
    try:
        obs = env.reset(23)
        assert obs['goal'] and obs['tree']
        page = env.env.unwrapped.page
        if 'choose-date' in task or task.endswith('book-flight'):
            field = page.locator('#datepicker')
            bid = field.get_attribute('bid')
            assert f'[{bid}] textbox' in obs['tree']
            assert 'readonly=True' in next(line for line in obs['tree'].splitlines() if f'[{bid}]' in line)
            obs, _, _ = env.step(f"click({bid!r})")
            assert not obs['error']
            assert page.locator('.ui-datepicker-calendar').is_visible()
            if task.endswith('choose-date-easy'):
                day = str(int(obs['goal'].split('Select ')[1].split('/')[1]))
                link = page.get_by_role('link', name=day, exact=True)
                obs, _, _ = env.step(f"click({link.get_attribute('bid')!r})")
                assert not obs['error']
                submit = page.get_by_role('button', name='Submit', exact=True)
                _, reward, done = env.step(f"click({submit.get_attribute('bid')!r})")
                assert done and reward > 0
        elif task.endswith('enter-text'):
            text = obs['goal'].split('"')[1]
            bid = page.locator('#tt').get_attribute('bid')
            obs, _, _ = env.step(f"fill({bid!r}, {text!r})")
            assert not obs['error']
            bid = page.locator('#subbtn').get_attribute('bid')
            _, reward, done = env.step(f"click({bid!r})")
            assert done and reward > 0
    finally:
        env.close()


def test_parallel_curriculum_recreates_pool_and_preserves_group_identity(tmp_path, monkeypatch):
    from test_real_parallel import ActionPolicy, ActionTokenizer

    import bulfm.env_pool
    import bulfm.policy
    from bulfm.worker import Worker

    pools = []
    class Pool:
        def __init__(self, c):
            self.task, self.size = c.task, c.num_envs
            self.closed = False
            pools.append(self)
        def reset_many(self, indices, seeds):
            assert len(set(seeds)) == 1  # training group uses one seed; eval count set to one
            return [{"goal": self.task, "tree": "[13] button", "error": ""} for _ in indices]
        def send_steps(self, indices, actions):
            assert not self.closed
        def receive_steps(self, indices):
            return [({"goal": self.task, "tree": "", "error": ""}, 1., True) for _ in indices]
        def close(self):
            self.closed = True
    monkeypatch.setattr(bulfm.env_pool, "BrowserPool", Pool)
    monkeypatch.setattr(bulfm.policy, "load_policy", lambda *a, **kw: (ActionPolicy(), ActionTokenizer()))
    cur = schedule().model_copy(update={"min_episodes": 2, "eval_episodes_per_task": 1})
    Config(steps=3, group_size=2, num_envs=2, beta=0, device="cpu", precision="fp32",
           baseline_eval=False, eval_every=1, curriculum=cur).save(tmp_path / "config.yaml")
    Worker(tmp_path).execute()
    assert all(p.closed for p in pools)
    assert {p.task for p in pools} == {'miniwob.click-test', 'miniwob.enter-text'}
    events = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['event'] == 'train']) == 3
    assert all(e['complete'] for e in events if e['event'] == 'eval')
