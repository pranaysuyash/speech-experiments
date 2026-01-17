#!/bin/bash
set -euo pipefail

# 0) Prove HEAD and show the actual diff you claim
git rev-parse HEAD
git show --name-only --oneline -n 1
git diff --stat HEAD~1..HEAD

# 1) Create experiment with two FULL candidates
curl -sS -X POST localhost:8000/api/experiments \
  -F "file=@/tmp/test_audio.wav" \
  -F "use_case_id=asr_smoke" \
  -F "candidate_ids=asr_full_default,asr_full_default" | tee /tmp/exp.json

EXP_ID=$(python3 - <<'PY'
import json; print(json.load(open("/tmp/exp.json"))["experiment_id"])
PY
)
echo "EXP_ID=$EXP_ID"

# 2) Start A and B
curl -sS -X POST "localhost:8000/api/experiments/$EXP_ID/runs/start" \
  -H 'content-type: application/json' -d '{"candidate_id":"A"}' | tee /tmp/startA.json
curl -sS -X POST "localhost:8000/api/experiments/$EXP_ID/runs/start" \
  -H 'content-type: application/json' -d '{"candidate_id":"B"}' | tee /tmp/startB.json

RUN_A=$(python3 - <<'PY'
import json; print(json.load(open("/tmp/startA.json"))["run_id"])
PY
)
RUN_B=$(python3 - <<'PY'
import json; print(json.load(open("/tmp/startB.json"))["run_id"])
PY
)
echo "RUN_A=$RUN_A"
echo "RUN_B=$RUN_B"
test "$RUN_A" != "$RUN_B"

# 3) Prove non-blocking during execution
for i in {1..10}; do
  curl -sS -o /dev/null -w "health $i %{http_code} %{time_total}s\n" --max-time 1 localhost:8000/health || echo "health $i TIMEOUT"
  sleep 0.5
done

# 4) Poll until terminal
while true; do
  A=$(curl -sS "localhost:8000/api/experiments/$EXP_ID" | python3 -c "import sys,json; r=json.load(sys.stdin)['runs']; print(next((x['status'] for x in r if x['candidate_id']=='A'), 'UNKNOWN'))")
  B=$(curl -sS "localhost:8000/api/experiments/$EXP_ID" | python3 -c "import sys,json; r=json.load(sys.stdin)['runs']; print(next((x['status'] for x in r if x['candidate_id']=='B'), 'UNKNOWN'))")
  echo "A=$A B=$B"
  if [[ "$A" =~ ^(COMPLETED|FAILED)$ && "$B" =~ ^(COMPLETED|FAILED)$ ]]; then
    break
  fi
  sleep 2
done

# 5) Compare transcript A vs B
curl -sS "localhost:8000/api/experiments/$EXP_ID/compare?left=$RUN_A&right=$RUN_B&artifact=transcript" \
  | tee /tmp/compare.json

python3 - <<'PY'
import json
d=json.load(open("/tmp/compare.json"))
la=d["left"]["available"]; ra=d["right"]["available"]
lt=d["left"]["text"] or ""; rt=d["right"]["text"] or ""
print("left.available", la, "len", len(lt))
print("right.available", ra, "len", len(rt))
assert la and ra and len(lt)>0 and len(rt)>0
PY

echo "PROOF OK"
