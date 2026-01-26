#!/usr/bin/env bash
set -euo pipefail

# Starts both backend (uvicorn) and frontend (vite) and tails logs for a quick demo/recording.
# Usage: ./scripts/start_demo.sh   (run from anywhere)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs/demo"
mkdir -p "$LOG_DIR"

# Prefer project venv
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.venv/bin/activate"
  UV_BIN="$PROJECT_ROOT/.venv/bin/uvicorn"
else
  UV_BIN="$(command -v uvicorn || true)"
fi

if [ -z "${UV_BIN:-}" ]; then
  echo "ERROR: uvicorn not found. Activate your venv or install uvicorn." >&2
  exit 1
fi

# Backend
BACKEND_LOG="$LOG_DIR/backend.log"
echo "Starting backend (uvicorn) -> $BACKEND_LOG"
nohup "$UV_BIN" server.main:app --host 127.0.0.1 --port 8000 --reload > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

# Frontend (vite) - force port 5173 for reproducible recording
FRONTEND_LOG="$LOG_DIR/frontend.log"
echo "Starting frontend (npm run dev -- --port 5173) -> $FRONTEND_LOG"
cd "$PROJECT_ROOT/client"
nohup npm run dev -- --port 5173 > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Wait briefly for servers to start
sleep 1

echo
echo "Backend: http://127.0.0.1:8000  (health: /health)"
echo "Frontend: http://localhost:5173/"
echo "Backend PID: $BACKEND_PID, Frontend PID: $FRONTEND_PID"

echo "Tailing logs (press Ctrl-C to stop and kill servers)"

# Trap Ctrl-C to kill background processes
_cleanup() {
  echo "\nStopping demo servers..."
  pkill -P $BACKEND_PID 2>/dev/null || true
  pkill -P $FRONTEND_PID 2>/dev/null || true
  kill "$BACKEND_PID" 2>/dev/null || true
  kill "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap _cleanup SIGINT SIGTERM

# Tail both logs together
# Use tail -f on both logs so you can record the terminal while servers boot and show activity.
# Use --retry so tail waits for file recreation on some systems. If not supported, plain tail -f will work.

if tail --version >/dev/null 2>&1; then
  tail --retry -n +1 -f "$BACKEND_LOG" "$FRONTEND_LOG"
else
  tail -n +1 -f "$BACKEND_LOG" "$FRONTEND_LOG"
fi
