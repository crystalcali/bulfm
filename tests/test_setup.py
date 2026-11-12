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
