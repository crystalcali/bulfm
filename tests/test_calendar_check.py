import hashlib
import json
from types import SimpleNamespace

import pytest

from bulfm.calendar_check import apply_fix, repaired_prompt, run_check
from bulfm.config import READONLY_HINT, Config
from bulfm.storage import atomic_json


def source(tmp_path):
    run = tmp_path / 'source'
    run.mkdir()
    Config(system_prompt='Custom policy.\nA readonly textbox cannot be filled: interact with its visible picker controls instead.',
           num_envs=2, device='cpu', precision='fp32', beta=0).save(run / 'config.yaml')
    checkpoint = run / 'checkpoints/step-000041'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'training.pt').write_bytes(b'optimizer RNG checkpoint')
    atomic_json(run / 'latest.json', {'path': str(checkpoint), 'step': 41})
    atomic_json(run / 'state.json', {'status': 'stopped', 'step': 41})
    return run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fix_preserves_checkpoint_custom_prompt_and_is_idempotent(tmp_path):
    run = source(tmp_path)
    old = (run / 'config.yaml').read_bytes()
    cp = run / 'checkpoints/step-000041/training.pt'
    before = digest(cp)
    apply_fix(tmp_path, 'source')
    config = Config.load(run / 'config.yaml')
    assert config.system_prompt.startswith('Custom policy.')
    assert READONLY_HINT in config.system_prompt
    backup = next(run.glob('config.before-readonly-fix-*.yaml'))
    assert backup.read_bytes() == old
    fixed = (run / 'config.yaml').read_bytes()
    apply_fix(tmp_path, 'source')
    assert (run / 'config.yaml').read_bytes() == fixed
    assert len(list(run.glob('config.before-readonly-fix-*.yaml'))) == 1
    assert digest(cp) == before
    assert json.loads((run / 'state.json').read_text())['step'] == 41
    assert repaired_prompt(Config().system_prompt) == Config().system_prompt


def test_reject_active_run_and_concurrent_check(tmp_path, monkeypatch):
    import fcntl

    import bulfm.calendar_check as module
    run = source(tmp_path)
    monkeypatch.setattr(module, 'alive', lambda _: True)
    with pytest.raises(ValueError, match='Stop this run'):
        apply_fix(tmp_path, 'source')
    monkeypatch.setattr(module, 'alive', lambda _: False)
    with (run / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='busy'):
            apply_fix(tmp_path, 'source')


def test_saved_adapter_check_same_seeds_no_update_and_persistent_report(tmp_path, monkeypatch):
    import bulfm.env_pool
    import bulfm.policy
    import bulfm.rollout
    run = source(tmp_path)
    old_config = (run / 'config.yaml').read_bytes()
    cp = run / 'checkpoints/step-000041/training.pt'
    before = digest(cp)
    seen = []
    loaded = []
    closed = []
    class Pool:
        def __init__(self, c):
            assert c.task == 'miniwob.choose-date-easy' and c.curriculum is None
        def close(self):
            closed.append(True)
    monkeypatch.setattr(bulfm.env_pool, 'BrowserPool', Pool)
    def load(c, adapter):
        loaded.append(adapter)
        return object(), object()
    monkeypatch.setattr(bulfm.policy, 'load_policy', load)
    def collect(model, tok, env, c, seeds, sample, trace, heartbeat):
        assert sample is False
        fixed = READONLY_HINT in c.system_prompt
        seen.append((c.system_prompt, list(seeds)))
        for seed in seeds:
            trace({'seed': seed, 'action': "click('17')" if fixed else 'noop()',
                   'valid': fixed, 'reward': float(fixed), 'error': '', 'response': 'test'})
            heartbeat(turn=1, action='test', valid_action=fixed, last_reward=float(fixed))
        return [SimpleNamespace(success=fixed, reward=float(fixed)) for _ in seeds]
    monkeypatch.setattr(bulfm.rollout, 'collect_many', collect)
    dest = run_check(tmp_path, 'source')
    report = json.loads((dest / 'report.json').read_text())
    assert len(loaded) == 1 and loaded[0].name == 'step-000041'
    assert seen[0][1] == seen[1][1]
    assert report['results']['original']['success_rate'] == 0
    assert report['results']['repaired']['success_rate'] == 1
    assert closed == [True]
    assert (run / 'config.yaml').read_bytes() == old_config
    assert digest(cp) == before
    # Applying before checking must still compare against the actual pre-fix prompt.
    apply_fix(tmp_path, 'source')
    seen.clear()
    run_check(tmp_path, 'source')
    assert READONLY_HINT not in seen[0][0] and READONLY_HINT in seen[1][0]
