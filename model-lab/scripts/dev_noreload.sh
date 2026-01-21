#!/usr/bin/env bash
set -euo pipefail

# No-reload backend for Mode B verification.
# Stable process, no hot-swap issues.

HOST="${MODEL_LAB_HOST:-127.0.0.1}"
PORT="${MODEL_LAB_PORT:-8000}"

# Activate venv
if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

export PYTHONPATH=${PYTHONPATH:-}:.
export PYTHONUNBUFFERED=1

echo "Starting backend (no reload) on $HOST:$PORT..."
exec uvicorn server.main:app --host "$HOST" --port "$PORT"
