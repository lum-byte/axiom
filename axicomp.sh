#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"

TARGET="${1:-linux}"
PYTHON="${PYTHON:-.venv/bin/python}"

case "$TARGET" in
  linux|Linux64)
    make build PYTHON="$PYTHON"
    ;;
  runtime-linux|runtime|so)
    make build-runtime PYTHON="$PYTHON"
    ;;
  test)
    make test PYTHON="$PYTHON"
    ;;
  clean)
    make clean PYTHON="$PYTHON"
    ;;
  all)
    make build PYTHON="$PYTHON"
    if command -v cmd.exe >/dev/null 2>&1; then
      cmd.exe /c axicomp.cmd
    else
      printf '%s\n' "Windows build skipped: cmd.exe not available."
    fi
    ;;
  *)
    printf '%s\n' "usage: ./axicomp.sh [linux|runtime-linux|test|clean|all]" >&2
    exit 2
    ;;
esac

