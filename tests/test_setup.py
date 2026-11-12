"""Bootstrap regression checks use stub downloaders; no CUDA or network required."""
import os
import subprocess
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "scripts/ensure_uv.sh"


def run_shell(script, tmp_path):
    return subprocess.run(["bash", "-c", 'set -eu; source "$1"; ' + script,
                           "test", str(HELPER), str(tmp_path)],
                          env={**os.environ, "PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True)


def executable(path, content="#!/bin/sh\nexit 0\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def test_reuses_uv_from_path(tmp_path):
    executable(tmp_path / "bin/uv")
    result = run_shell('PATH="$2/bin:$PATH"; ensure_uv "$2/install"; '
                       'test "$BULFM_UV" = "$2/bin/uv"', tmp_path)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "install").exists()


def test_finds_uv_outside_path(tmp_path):
    executable(tmp_path / "user-bin/uv")
    result = run_shell('ensure_uv "$2/install" "$2/user-bin/uv"; '
                       'test "$BULFM_UV" = "$2/user-bin/uv"', tmp_path)
    assert result.returncode == 0, result.stderr


def test_bootstraps_uv_and_reuses_it(tmp_path):
    installer = tmp_path / "fake-installer"
    installer.write_text('set -eu\nmkdir -p "$UV_UNMANAGED_INSTALL"\n'
                         'printf "#!/bin/sh\\nexit 0\\n" > "$UV_UNMANAGED_INSTALL/uv"\n'
                         'chmod +x "$UV_UNMANAGED_INSTALL/uv"\n')
    result = run_shell('fixture="$2/fake-installer"; '
                       'curl() { cp "$fixture" "${@: -1}"; }; '
                       'ensure_uv "$2/install"; test -x "$BULFM_UV"; '
                       'curl() { return 99; }; ensure_uv "$2/install"', tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "install/uv").exists()


def test_download_failure_is_actionable(tmp_path):
    result = run_shell('curl() { return 22; }; ensure_uv "$2/install"', tmp_path)
    assert result.returncode != 0
    assert "Could not download uv" in result.stderr
    assert not (tmp_path / "install/uv").exists()


def load_discovery():
    import importlib.util
    path = HELPER.parent / "find_cuda_python.py"
    spec = importlib.util.spec_from_file_location("find_cuda_python", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovery_skips_system_without_torch(tmp_path):
    from types import SimpleNamespace
    module = load_discovery()
    checked = []
    def probe(command, **kwargs):
        checked.append(command[0])
        if command[0].endswith("working/bin/python"):
            return SimpleNamespace(returncode=0, stdout="CUDA works", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="No module named torch")
    paths = [tmp_path / "system/bin/python", tmp_path / "working/bin/python"]
    assert module.find(paths, probe) == str(paths[1])
    assert checked == list(map(str, paths))


def test_discovery_handles_bad_interpreter_and_timeout(tmp_path):
    module = load_discovery()
    def probe(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    assert module.find([tmp_path / "broken/python"], probe) is None


def test_discovery_keeps_distinct_venvs_with_shared_python(tmp_path):
    module = load_discovery()
    for project in ("one", "two"):
        target = tmp_path / "work" / project / ".venv/bin/python"
        target.parent.mkdir(parents=True)
        target.symlink_to(os.sys.executable)
    paths = list(module.candidates(root=tmp_path / "bulfm", home=tmp_path))
    assert tmp_path / "work/one/.venv/bin/python" in paths
    assert tmp_path / "work/two/.venv/bin/python" in paths


def test_setup_creates_cuda_environment_when_host_has_no_torch(tmp_path):
    """Run the complete installer with stub GPU/download/package commands."""
    import shutil
    project = tmp_path / "project"
    shutil.copytree(HELPER.parent, project / "scripts")
    binaries = tmp_path / "bin"
    executable(binaries / "python3", "#!/bin/sh\nexit 1\n")
    executable(binaries / "nvidia-smi", "#!/bin/sh\necho NVIDIA-GB10\n")
    executable(binaries / "git", "#!/bin/sh\necho 7fd85d71a4b60325c6585396ec4f48377d049838\n")
    python_stub = tmp_path / "venv-python"
    executable(python_stub, '''#!/bin/bash
set -eu
case "$1" in
  scripts/check_cuda.py) test -f .venv/.cuda-ready ;;
  -c) echo 'torch==2.14.1+cu130' ;;
  -m) test "$2" = playwright ;;
  *) exit 90 ;;
esac
''')
    executable(binaries / "uv", '''#!/bin/bash
set -eu
printf '%s\\n' "$*" >> "$BULFM_TEST_LOG"
if [[ "$1" == venv ]]; then
  mkdir -p .venv/bin
  cp "$BULFM_TEST_PYTHON" .venv/bin/python
  printf '#!/bin/sh\\nexit 0\\n' > .venv/bin/bulfm
  chmod +x .venv/bin/bulfm
elif [[ "$*" == *'torch=='* ]]; then
  touch .venv/.cuda-ready
fi
''')
    (project / "vendor/miniwob-plusplus/.git").mkdir(parents=True)
    env = {**os.environ, "PATH": f"{binaries}:/usr/bin:/bin",
           "BULFM_TEST_LOG": str(tmp_path / "commands.log"),
           "BULFM_TEST_PYTHON": str(python_stub)}
    env.pop("BULFM_PYTHON", None)
    result = subprocess.run(["bash", "scripts/setup_spark.sh"], cwd=project, env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = (tmp_path / "commands.log").read_text()
    assert "venv --python 3.12 .venv" in commands
    assert "torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130" in commands
    assert "-e .[train]" in commands
    assert "Ready." in result.stdout
