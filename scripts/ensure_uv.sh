#!/usr/bin/env bash
# Sourced by setup_spark.sh. Keep bootstrap independent of the CUDA Python environment.
ensure_uv() {
  local install_dir="$1"
  shift
  local candidate installer
  if command -v uv >/dev/null 2>&1; then
    BULFM_UV=$(command -v uv)
    return 0
  fi
  for candidate in "$install_dir/uv" "$@"; do
    if [[ -x "$candidate" ]]; then
      BULFM_UV="$candidate"
      return 0
    fi
  done
  installer=$(mktemp) || return 1
  echo 'uv not found; installing it locally for BULFM...'
  if command -v curl >/dev/null 2>&1; then
    if ! curl --fail --show-error --silent --location --retry 2 \
      https://astral.sh/uv/install.sh --output "$installer"; then
      rm -f "$installer"
      echo 'Could not download uv. Check network access to astral.sh and GitHub.' >&2
      return 1
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget -q https://astral.sh/uv/install.sh -O "$installer"; then
      rm -f "$installer"
      echo 'Could not download uv. Check network access to astral.sh and GitHub.' >&2
      return 1
    fi
  else
    rm -f "$installer"
    echo 'Install curl or wget, then run this setup again.' >&2
    return 1
  fi
  if ! UV_UNMANAGED_INSTALL="$install_dir" sh "$installer"; then
    rm -f "$installer"
    echo 'uv installation failed. Check access to GitHub release downloads.' >&2
    return 1
  fi
  rm -f "$installer"
  BULFM_UV="$install_dir/uv"
  if [[ ! -x "$BULFM_UV" ]]; then
    echo "uv installer did not create $BULFM_UV" >&2
    return 1
  fi
}
