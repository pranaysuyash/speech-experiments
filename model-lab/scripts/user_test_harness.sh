#!/usr/bin/env bash
set -euo pipefail

# Mode B harness must be deterministic about the backend it talks to.
# Default to IPv4 loopback, not localhost (macOS often resolves localhost to ::1).
API_BASE="${MODEL_LAB_API_BASE:-http://127.0.0.1:8000}"

join_url() {
  local base="${1%/}"
  local path="$2"
  [[ "$path" == /* ]] || path="/$path"
  echo "${base}${path}"
}

curl_code() {
  local url="$1"
  local out="$2"
  curl -sS -o "$out" -w "%{http_code}" "$url" || true
}

wait_for_api() {
  local tries="${1:-40}"   # 40 * 0.25s = 10s
  local sleep_s="${2:-0.25}"
  local openapi_url; openapi_url="$(join_url "$API_BASE" "/openapi.json")"
  local runs_url; runs_url="$(join_url "$API_BASE" "/api/runs")"

  echo "API_BASE: $API_BASE"
  echo "Readiness: $openapi_url must contain /api/runs, then $runs_url must return 200"

  local i
  for i in $(seq 1 "$tries"); do
    local tmp; tmp="$(mktemp)"
    local code
    code="$(curl_code "$openapi_url" "$tmp")"
    if [[ "$code" == "200" ]] && grep -q '"/api/runs"' "$tmp"; then
      rm -f "$tmp"
      tmp="$(mktemp)"
      code="$(curl_code "$runs_url" "$tmp")"
      if [[ "$code" == "200" ]]; then
        rm -f "$tmp"
        echo "API ready."
        return 0
      fi
    fi
    rm -f "$tmp"
    sleep "$sleep_s"
  done

  echo "ERROR: API not ready or wrong app mounted at $API_BASE"
  echo "Listener on 8000 (if any):"
  lsof -nP -iTCP:8000 -sTCP:LISTEN || true
  echo "HEAD /api/runs:"
  curl -sS -D - -o /dev/null "$runs_url" | sed -n '1,20p' || true
  exit 1
}

echo "============================================="
echo "🛠  Model Lab: User Test Harness (Mode B)"
echo "============================================="

FAILED_STEP="alignment"

# 1. Environment Check
echo "[1/5] checking backend invariants..."
./scripts/check_backend.sh

# 2. Readiness gate with OpenAPI verification
echo "[2/5] Waiting for API readiness..."
wait_for_api
echo "✅ API is UP."

# 3. Scenario 1: Success Run (Golden Path)
echo ""
echo "[3/4] Running Scenario 1: Golden Path (Success)..."
echo "CMD: python scripts/run_session.py --input inputs/valid_test.wav --force"
OUT_SUCCESS=$(.venv/bin/python scripts/run_session.py --input inputs/valid_test.wav --force)
RUN_ID_SUCCESS=$(echo "$OUT_SUCCESS" | grep "RUN_SESSION_RESULT" | sed 's/.*"run_id":"\([^"]*\)".*/\1/')
echo "✅ Started Success Run: $RUN_ID_SUCCESS"

# 4. Scenario 2: Forced Failure (Deterministic)
echo ""
echo "[4/5] Running Scenario 2: Forced Failure (Step: $FAILED_STEP)..."
echo "CMD: MODEL_LAB_TESTING=1 MODEL_LAB_FAIL_STEP=$FAILED_STEP python scripts/run_session.py ..."
OUT_FAIL=$(MODEL_LAB_TESTING=1 MODEL_LAB_FAIL_STEP=$FAILED_STEP .venv/bin/python scripts/run_session.py --input inputs/valid_test.wav --force || true)
RUN_ID_FAIL=$(echo "$OUT_FAIL" | grep "RUN_SESSION_RESULT" | sed 's/.*"run_id":"\([^"]*\)".*/\1/')
echo "✅ Started Failure Run: $RUN_ID_FAIL"

# 5. Scenario 3: v2 Registry Missing (Schema Gating Test)
echo ""
echo "[5/5] Running Scenario 3: v2 Registry Failure (Schema Gap)..."
# First create a success run, then corrupt it
OUT_V2=$(.venv/bin/python scripts/run_session.py --input inputs/valid_test.wav --force)
RUN_ID_V2=$(echo "$OUT_V2" | grep "RUN_SESSION_RESULT" | sed 's/.*"run_id":"\([^"]*\)".*/\1/' | tr -d '\r' | xargs)
echo "DEBUG S5: RUN_ID_V2='$RUN_ID_V2'"

# Give RunsIndex time to debounce/refresh filesystem (race mitigation)
sleep 2

# Find the run directory robustly
RUN_DIR_V2=$(find runs -type d -name "$RUN_ID_V2" -print -quit || true)
if [ -z "$RUN_DIR_V2" ]; then
  echo "❌ Could not find run directory for $RUN_ID_V2 under ./runs"
  echo "   Fix: update this script to locate RUNS_ROOT or parse run_dir from RUN_SESSION_RESULT."
  exit 1
fi

MANIFEST_PATH="$RUN_DIR_V2/manifest.json"
echo "   Corrupting manifest at $MANIFEST_PATH ..."

# Use the SAME interpreter as the repo (avoid system python drift)
.venv/bin/python - <<PY
import json
from pathlib import Path

p = Path("$MANIFEST_PATH")
d = json.loads(p.read_text(encoding="utf-8"))
d["artifacts_by_type"] = {}
d["manifest_schema_version"] = 2
p.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
PY

echo "✅ Created v2 Registry Failure Run: $RUN_ID_V2"

echo "ℹ️  Probing Schema Gate via API (S5)..."
S5_URL="$(join_url "$API_BASE" "/api/runs/$RUN_ID_V2/transcript")"
echo "DEBUG S5: Probing URL '$S5_URL'"
S5_BODY="$(mktemp)"
S5_CODE="$(curl_code "$S5_URL" "$S5_BODY")"

if [ "$S5_CODE" != "400" ]; then
  echo "❌ S5 FAIL: expected HTTP 400, got $S5_CODE"
  echo "Body:"
  cat "$S5_BODY"
  exit 1
fi

if ! grep -q "E_ARTIFACT_REGISTRY_MISSING" "$S5_BODY"; then
  echo "❌ S5 FAIL: missing E_ARTIFACT_REGISTRY_MISSING in response body"
  echo "Body:"
  cat "$S5_BODY"
  exit 1
fi

echo "✅ S5 SUCCESS: API returned 400 with E_ARTIFACT_REGISTRY_MISSING"
rm -f "$S5_BODY"

# Summary
echo ""
echo "============================================="
echo "🎉 Harness Complete. Automated Contracts Verified."
echo "============================================="
echo ""
echo "🔍 Verify Scenario 1 (Success):"
echo "   URL: http://localhost:5174/runs/$RUN_ID_SUCCESS"
echo "   Goal: Confirm status is COMPLETED and Bundle is downloadable."
echo ""
echo "🔍 Verify Scenario 2 (Failure):"
echo "   URL: http://localhost:5174/runs/$RUN_ID_FAIL"
echo "   Goal: Confirm status is FAILED, failed_step is '$FAILED_STEP', and Error Surface shows 'Simulated Test Failure'."
echo ""
echo "🔍 Verify Scenario 3 (Registry Gap):"
echo "   URL: http://localhost:5174/runs/$RUN_ID_V2"
echo "   Goal: Open Detail View. It should show NO TRANSCRIPT (empty) or error if UI bubbles 400."
echo "         But the Harness already verified the API contract above via S5."
echo ""
echo "🛠  Debug Panel (Mode B):"
echo "   Ensure you opened the app with: VITE_DEBUG_UI=1 npm run dev"
echo "   Check that Fingerprint hash matches between Runs List and Run Detail."
echo ""
