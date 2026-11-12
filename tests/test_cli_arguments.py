import pytest

from bulfm.cli import execute


@pytest.mark.parametrize("command", ["stop", "resume", "logs", "eval"])
@pytest.mark.parametrize("arguments", [[], ["one", "two"]])
def test_run_commands_require_exactly_one_id(command, arguments, tmp_path):
    with pytest.raises(ValueError, match=f"Usage: /{command} RUN_ID"):
        execute([command, *arguments], tmp_path)
