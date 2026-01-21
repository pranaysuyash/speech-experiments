#!/usr/bin/env bash
# scripts/ci_mode_b_gate.sh
# Lightweight CI/pre-push gate for Mode B.
# Runs readiness check + S5 probe only (faster than full harness).
#
# Exit Codes:
#   0 = PASSED
#   1 = Server not reachable
#   2 = Wrong app mounted (OpenAPI missing /api/runs)
#   3 = S5 did not return expected 400 + E_ARTIFACT_REGISTRY_MISSING
set -euo pipefail

API_BASE="${MODEL_LAB_API_BASE:-http://127.0.0.1:8000}"

echo "🔒 Mode B CI Gate: Quick Verification"
echo "API_BASE: $API_BASE"

# 1. Server reachable check
echo -n "[1/3] Server reachable... "
if ! curl -sS --connect-timeout 2 "${API_BASE}/health" >/dev/null 2>&1; then
  echo "❌"
  echo "FAIL: Server not reachable at $API_BASE"
  exit 1
fi
echo "✅"

# 2. OpenAPI check (wrong app detection)
echo -n "[2/3] OpenAPI contains /api/runs... "
if ! curl -sS "${API_BASE}/openapi.json" 2>/dev/null | grep -q '"/api/runs"'; then
  echo "❌"
  echo "FAIL: openapi.json missing /api/runs route (wrong app mounted)"
  exit 2
fi
echo "✅"

# 3. S5 probe (create corrupted run, verify 400)
echo -n "[3/3] S5 schema gating probe... "

# Create a minimal corrupted manifest for testing
TEST_RUN_ID="ci_gate_test_$(date +%s)"
TEST_DIR="runs/sessions/ci_gate/${TEST_RUN_ID}"
mkdir -p "$TEST_DIR"

cat > "${TEST_DIR}/manifest.json" <<EOF
{
  "run_id": "$TEST_RUN_ID",
  "status": "COMPLETED",
  "manifest_schema_version": 2,
  "artifacts_by_type": {},
  "steps": {}
}
EOF

# Give index time to pick up
sleep 1

S5_BODY="$(mktemp)"
S5_CODE="$(curl -sS -o "$S5_BODY" -w "%{http_code}" "${API_BASE}/api/runs/${TEST_RUN_ID}/transcript" || true)"

# Cleanup test run
rm -rf "runs/sessions/ci_gate"

if [ "$S5_CODE" == "400" ] && grep -q "E_ARTIFACT_REGISTRY_MISSING" "$S5_BODY"; then
  rm -f "$S5_BODY"
  echo "✅"
else
  echo "❌ (expected 400 + E_ARTIFACT_REGISTRY_MISSING, got $S5_CODE)"
  cat "$S5_BODY"
  rm -f "$S5_BODY"
  exit 3
fi

echo ""
echo "🎉 Mode B CI Gate: PASSED"
exit 0
