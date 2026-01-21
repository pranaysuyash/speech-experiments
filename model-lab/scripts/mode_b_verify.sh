#!/usr/bin/env bash
# scripts/mode_b_verify.sh
# Canonical entry point for Mode B verification.
# Kills stale processes, starts clean backend, runs full harness, cleans up.
set -euo pipefail

API_BASE="http://127.0.0.1:8000"
LOG_FILE="/tmp/model_lab_mode_b.log"
PID_FILE="/tmp/model_lab_mode_b.pid"

cleanup() {
  if [ -f "$PID_FILE" ]; then
    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT

echo "============================================="
echo "🔒 Mode B: Full Verification (Blessed Command)"
echo "============================================="
echo ""

# 1. Kill Model Lab processes on 8000 (safe: only kills our uvicorn)
echo "[1/5] Checking for stale Model Lab listeners on :8000..."
REPO_PATH="$(pwd)"
PIDS="$(lsof -nP -tiTCP:8000 -sTCP:LISTEN || true)"
KILLED=""

for pid in $PIDS; do
  # Check if this is a Model Lab process (cwd or cmdline contains repo path or server.main)
  CMDLINE="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  CWD="$(lsof -p "$pid" -Fn 2>/dev/null | grep '^n.*model-lab' | head -1 || true)"
  
  if [[ "$CMDLINE" == *"server.main"* ]] || [[ "$CMDLINE" == *"model-lab"* ]] || [[ -n "$CWD" ]]; then
    echo "   Killing Model Lab process: $pid ($CMDLINE)"
    kill "$pid" 2>/dev/null || true
    KILLED="$KILLED $pid"
  else
    echo "   ⚠️ Skipping non-Model-Lab process: $pid ($CMDLINE)"
  fi
done

if [[ -n "$KILLED" ]]; then
  sleep 0.5
  for pid in $KILLED; do
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 0.2
fi

# Confirm port is free (fail if occupied by non-Model-Lab process)
PIDS="$(lsof -nP -tiTCP:8000 -sTCP:LISTEN || true)"
if [[ -n "${PIDS}" ]]; then
  CMDLINE="$(ps -p "$PIDS" -o args= 2>/dev/null || true)"
  echo "❌ Port 8000 occupied by non-Model-Lab process: $PIDS"
  echo "   Command: $CMDLINE"
  echo "   Manually kill it or use a different port."
  exit 1
fi
echo "   Port 8000 is free."

# 2. Start stable backend
echo "[2/5] Starting stable backend (dev_noreload.sh)..."
./scripts/dev_noreload.sh > "$LOG_FILE" 2>&1 &
API_PID=$!
echo "$API_PID" > "$PID_FILE"
sleep 2

# 3. Verify correct app mounted
echo "[3/5] Verifying backend mounts correct app..."
if ! curl -sS "${API_BASE}/openapi.json" | grep -q '"/api/runs"'; then
  echo "❌ OpenAPI missing /api/runs - wrong app mounted!"
  echo "   Backend log:"
  tail -n 50 "$LOG_FILE"
  exit 1
fi

if ! curl -sS -o /dev/null -w "%{http_code}" "${API_BASE}/api/runs" | grep -q "200"; then
  echo "❌ /api/runs did not return 200"
  curl -sS -D - -o /dev/null "${API_BASE}/api/runs" | head -n 10
  exit 1
fi
echo "   Backend verified: /api/runs returns 200."

# 4. Run harness
echo "[4/5] Running Mode B harness..."
./scripts/user_test_harness.sh
HARNESS_EXIT=$?

# 5. Report
echo ""
echo "[5/5] Cleanup and summary..."
if [ $HARNESS_EXIT -eq 0 ]; then
  echo "============================================="
  echo "🎉 Mode B Verification: PASSED"
  echo "============================================="
else
  echo "============================================="
  echo "❌ Mode B Verification: FAILED (Exit $HARNESS_EXIT)"
  echo "============================================="
  exit $HARNESS_EXIT
fi
