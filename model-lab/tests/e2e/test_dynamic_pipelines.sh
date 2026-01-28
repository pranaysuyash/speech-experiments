#!/bin/bash
set -euo pipefail

SERVER="localhost:8000"
TEST_AUDIO="/tmp/test_audio.wav"
TIMEOUT=60

echo "=== E2E Test: Dynamic Pipeline Selection ==="

# 0) Generate test audio if needed
if [[ ! -f "$TEST_AUDIO" ]]; then
  echo "Generating 2-second silent WAV..."
  ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 2 -y "$TEST_AUDIO" 2>/dev/null
  echo "Created $TEST_AUDIO"
fi

# 1) Prove HEAD
if command -v git &> /dev/null; then
  git rev-parse HEAD
  git show --name-only --oneline -n 1
fi

# Helper: poll run status until terminal
poll_run() {
  local run_id="$1"
  local start_time=$(date +%s)
  
  while true; do
    local status
    status=$(curl -sS --max-time 5 "$SERVER/api/runs/$run_id/status" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    echo "  status=$status"
    
    if [[ "$status" =~ ^(COMPLETED|FAILED|STALE)$ ]]; then
      echo "$status"
      return 0
    fi
    
    local now=$(date +%s)
    if (( now - start_time > TIMEOUT )); then
      echo "TIMEOUT"
      return 1
    fi
    sleep 2
  done
}

# ============================================================================
# Test 1: Custom steps via direct step list
# ============================================================================
echo ""
echo "=== Test 1: Custom steps (ingest,asr) ==="

curl -sS --max-time 10 -X POST "$SERVER/api/workbench/runs" \
  -F "file=@$TEST_AUDIO" \
  -F "use_case_id=e2e_custom_steps" \
  -F "steps=ingest,asr" | tee /tmp/run1.json

RUN1=$(python3 -c "import json; print(json.load(open('/tmp/run1.json'))['run_id'])")
echo "RUN1=$RUN1"

echo "Polling for completion..."
STATUS1=$(poll_run "$RUN1")
if [[ "$STATUS1" != "COMPLETED" ]]; then
  echo "FAIL: Run 1 did not complete (status=$STATUS1)"
  exit 1
fi

# Verify resolved steps
echo "Verifying resolved steps..."
curl -sS --max-time 5 "$SERVER/api/runs/$RUN1/status" | tee /tmp/run1_status.json

python3 - <<'PY'
import json
import sys

data = json.load(open("/tmp/run1_status.json"))
steps_completed = set(data.get("steps_completed", []))
expected = {"ingest", "asr"}

print(f"steps_completed: {steps_completed}")
if not expected.issubset(steps_completed):
    print(f"FAIL: Expected {expected} but got {steps_completed}")
    sys.exit(1)
print("OK: Custom steps test passed")
PY

# ============================================================================
# Test 2: Preprocessing operators
# ============================================================================
echo ""
echo "=== Test 2: Preprocessing operators (trim_silence, normalize_loudness) ==="

curl -sS --max-time 10 -X POST "$SERVER/api/workbench/runs" \
  -F "file=@$TEST_AUDIO" \
  -F "use_case_id=e2e_preprocessing" \
  -F "steps=ingest,asr" \
  -F "preprocessing=trim_silence,normalize_loudness" | tee /tmp/run2.json

RUN2=$(python3 -c "import json; print(json.load(open('/tmp/run2.json'))['run_id'])")
echo "RUN2=$RUN2"

echo "Polling for completion..."
STATUS2=$(poll_run "$RUN2")
if [[ "$STATUS2" != "COMPLETED" ]]; then
  echo "FAIL: Run 2 did not complete (status=$STATUS2)"
  exit 1
fi

# Verify run_request.json contains preprocessing info
echo "Verifying preprocessing in run_request..."
RUN2_DIR=$(python3 -c "import json; print(json.load(open('/tmp/run2.json'))['run_dir'])")

python3 - <<PY
import json
import sys
from pathlib import Path

run_request_path = Path("$RUN2_DIR") / "run_request.json"
if not run_request_path.exists():
    print("FAIL: run_request.json not found")
    sys.exit(1)

data = json.loads(run_request_path.read_text())
preprocessing = data.get("preprocessing", [])
print(f"preprocessing: {preprocessing}")

expected = ["trim_silence", "normalize_loudness"]
if set(preprocessing) != set(expected):
    print(f"FAIL: Expected preprocessing {expected} but got {preprocessing}")
    sys.exit(1)
print("OK: Preprocessing operators test passed")
PY

# ============================================================================
# Test 3: Pipeline template (fast_asr)
# ============================================================================
echo ""
echo "=== Test 3: Pipeline template (fast_asr) ==="

curl -sS --max-time 10 -X POST "$SERVER/api/workbench/runs" \
  -F "file=@$TEST_AUDIO" \
  -F "use_case_id=e2e_template" \
  -F "pipeline_template=fast_asr" | tee /tmp/run3.json

RUN3=$(python3 -c "import json; print(json.load(open('/tmp/run3.json'))['run_id'])")
echo "RUN3=$RUN3"

echo "Polling for completion..."
STATUS3=$(poll_run "$RUN3")
if [[ "$STATUS3" != "COMPLETED" ]]; then
  echo "FAIL: Run 3 did not complete (status=$STATUS3)"
  exit 1
fi

# Verify template was applied
echo "Verifying pipeline template in run_request..."
RUN3_DIR=$(python3 -c "import json; print(json.load(open('/tmp/run3.json'))['run_dir'])")

python3 - <<PY
import json
import sys
from pathlib import Path

run_request_path = Path("$RUN3_DIR") / "run_request.json"
if not run_request_path.exists():
    print("FAIL: run_request.json not found")
    sys.exit(1)

data = json.loads(run_request_path.read_text())
template = data.get("pipeline_template")
steps = data.get("steps_requested", [])
pipeline_config = data.get("pipeline_config", {})

print(f"pipeline_template: {template}")
print(f"steps_requested: {steps}")
print(f"pipeline_config.steps: {pipeline_config.get('steps')}")

# fast_asr template should have ingest and asr
if template != "fast_asr":
    print(f"FAIL: Expected template 'fast_asr' but got '{template}'")
    sys.exit(1)

expected_steps = {"ingest", "asr"}
if not expected_steps.issubset(set(steps)):
    print(f"FAIL: Expected steps {expected_steps} but got {steps}")
    sys.exit(1)

print("OK: Pipeline template test passed")
PY

# Verify run completed and steps match
echo "Verifying completed steps..."
curl -sS --max-time 5 "$SERVER/api/runs/$RUN3/status" | tee /tmp/run3_status.json

python3 - <<'PY'
import json
import sys

data = json.load(open("/tmp/run3_status.json"))
steps_completed = set(data.get("steps_completed", []))
expected = {"ingest", "asr"}

print(f"steps_completed: {steps_completed}")
if not expected.issubset(steps_completed):
    print(f"FAIL: Expected {expected} in completed steps but got {steps_completed}")
    sys.exit(1)
print("OK: Pipeline template steps verified")
PY

echo ""
echo "=== ALL TESTS PASSED ==="
