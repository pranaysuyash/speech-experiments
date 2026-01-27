#!/usr/bin/env bash
set -euo pipefail

# No-reload backend for Mode B verification.
# Stable process, no hot-swap issues.

HOST="${MODEL_LAB_HOST:-127.0.0.1}"
PORT="${MODEL_LAB_PORT:-8000}"

# Resolve project root and activate venv from there
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate venv from project root if present
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/venv/bin/activate"
fi

# Ensure PYTHONPATH points at project root so imports like `server` resolve
export PYTHONPATH=${PYTHONPATH:-}:"$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# Prefer uvicorn from the project venv if available
if [ -x "$PROJECT_ROOT/.venv/bin/uvicorn" ]; then
  UV_BIN="$PROJECT_ROOT/.venv/bin/uvicorn"
else
  UV_BIN="$(command -v uvicorn || true)"
fi

if [ -z "${UV_BIN:-}" ]; then
  echo "ERROR: uvicorn not found. Activate your venv or install uvicorn." >&2
  exit 1
fi

echo "Starting backend (no reload) on $HOST:$PORT using $UV_BIN..."
exec "$UV_BIN" server.main:app --host "$HOST" --port "$PORT"
