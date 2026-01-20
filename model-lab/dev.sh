#!/bin/bash
set -euo pipefail

BACKEND_PORT=8000
FRONTEND_PORT=5174

# Filter shown in terminal (files still get everything)
FILTER_BACKEND=${FILTER_BACKEND:-"(/api/|RUN|STEP|manifest|FAILED|COMPLETED|CANCEL|Exception|Traceback|ERROR)"}
FILTER_FRONTEND=${FILTER_FRONTEND:-"(http|ready|error|warn|HMR|compiled|transform)"}

kill_port() {
  local PORT=$1
  local PID
  PID=$(lsof -ti:$PORT || true)
  if [ -n "${PID:-}" ]; then
    echo "Stopping PID $PID on port $PORT..."
    kill -TERM "$PID" 2>/dev/null || true
    sleep 0.5
    kill -KILL "$PID" 2>/dev/null || true
  fi
}

echo "=================================================="
echo "Model Lab: Dev Server"
echo "=================================================="

echo "[1/4] Cleaning up ports..."
kill_port $BACKEND_PORT
kill_port $FRONTEND_PORT

echo "[2/4] Activating Python Environment..."
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Activated .venv"
elif [ -d "venv" ]; then
  source venv/bin/activate
  echo "Activated venv"
else
  echo "WARNING: No .venv or venv directory found!"
fi

export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1

echo "[3/4] Starting Backend (Port $BACKEND_PORT)..."
: > server.log
python -u server/main.py 2>&1 \
  | tee -a server.log \
  | sed -u 's/^/[backend] /' \
  | grep -E --line-buffered "$FILTER_BACKEND" &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

echo "[4/4] Starting Frontend (Port $FRONTEND_PORT)..."
: > frontend.log
(
  cd client
  npm run dev -- --port $FRONTEND_PORT 2>&1
) \
  | tee -a frontend.log \
  | sed -u 's/^/[frontend] /' \
  | grep -E --line-buffered "$FILTER_FRONTEND" &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo "=================================================="
echo "Backend:  http://localhost:$BACKEND_PORT"
echo "Frontend: http://localhost:$FRONTEND_PORT"
echo "=================================================="
echo "Ctrl+C to stop"

trap 'echo "Stopping..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; exit 0' INT TERM

wait
