#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/.."

STRICT="${VERIFY_STRICT_VENV:-0}"

pick_python() {
  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
    return 0
  fi

  if [[ -x "../.venv/bin/python" ]]; then
    echo "../.venv/bin/python"
    return 0
  fi

  # Detect primary worktree venv (portable, no hardcoded paths)
  if command -v git >/dev/null 2>&1; then
    primary_wt="$(git worktree list --porcelain 2>/dev/null | awk '
      $1=="worktree"{wt=$2}
      $1=="branch" && ($2 ~ /refs\/heads\/(main|master)$/){print wt; exit}
    ')"
    if [[ -n "${primary_wt:-}" && -x "${primary_wt}/.venv/bin/python" ]]; then
      echo "${primary_wt}/.venv/bin/python"
      return 0
    fi
  fi

  if [[ -x "../../model-lab/.venv/bin/python" ]]; then
    echo "../../model-lab/.venv/bin/python"
    return 0
  fi

  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "${VIRTUAL_ENV}/bin/python"
    return 0
  fi

  if [[ "$STRICT" == "1" ]]; then
    echo "ERROR: No venv python found (.venv/bin/python, ../.venv/bin/python, or \$VIRTUAL_ENV). Set up a venv or unset VERIFY_STRICT_VENV." >&2
    exit 1
  fi

  echo "python3"
}

PY="$(pick_python)"
echo "verify_repo.sh: using python: $PY"
echo "verify_repo.sh: STRICT=${STRICT}"

PYTHONPATH=. "$PY" -m pytest -q tests/integration/test_backend_invariants.py
PYTHONPATH=. "$PY" -m pytest -q tests/api/test_artifact_download_security.py

pushd client >/dev/null
npm run build
popd >/dev/null

echo "verify_repo.sh: OK"
