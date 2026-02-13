#!/bin/bash
set -u

# GLM-TTS observation script
# Prints raw observations and a YAML receipt.
# No interpretive words, no icons, no content-based exits.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR="$REPO_ROOT/models/glm_tts"
REPO_DIR="$MODEL_DIR/repo"
VENV_DIR="$MODEL_DIR/venv"

EXPECTED_COMMIT="c5dc7aecc3b4032032d631b271e767893984f821"

OBS_VENV="missing"
OBS_REPO_DIR="missing"
OBS_REPO_GIT="missing"
OBS_REPO_COMMIT="unknown"
OBS_EXPECTED_COMMIT="$EXPECTED_COMMIT"

OBS_PATCH_DIR="missing"
OBS_PATCH_TOTAL="0"
OBS_PATCH_REVERSE_APPLY_OK="0"

OBS_WEIGHTS_RECEIPT="missing"
OBS_CKPT_DIR="missing"
OBS_CKPT_SIZE_KB="unknown"
OBS_CKPT_FILE_COUNT="unknown"

OBS_IMPORTS_ATTEMPTED="not_attempted"
OBS_IMPORTS_EXIT_CODE="unknown"
OBS_TORCH_VERSION="unknown"

OBS_BENCH_ATTEMPTED="not_attempted"
OBS_BENCH_EXIT_CODE="unknown"

echo "=== GLM-TTS Observation Log ==="

# Venv
if [ -x "$VENV_DIR/bin/python" ]; then
  OBS_VENV="present"
fi
echo "[OBSERVATION] venv_status: $OBS_VENV"

# Repo
if [ -d "$REPO_DIR" ]; then
  OBS_REPO_DIR="present"
fi
if [ -d "$REPO_DIR/.git" ]; then
  OBS_REPO_GIT="present"
  OBS_REPO_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")"
fi
echo "[OBSERVATION] repo_dir: $OBS_REPO_DIR"
echo "[OBSERVATION] repo_git: $OBS_REPO_GIT"
echo "[OBSERVATION] repo_commit: $OBS_REPO_COMMIT"
echo "[OBSERVATION] expected_commit: $OBS_EXPECTED_COMMIT"

# Patches: reverse-apply check, counts only
PATCH_DIR="$MODEL_DIR/patches"
if [ -d "$PATCH_DIR" ]; then
  OBS_PATCH_DIR="present"
  patch_total=0
  reverse_apply_ok=0

  for p in "$PATCH_DIR"/*.patch; do
    [ -f "$p" ] || continue
    patch_total=$((patch_total + 1))
    if [ "$OBS_REPO_GIT" = "present" ]; then
      if git -C "$REPO_DIR" apply --reverse --check "$p" >/dev/null 2>&1; then
        reverse_apply_ok=$((reverse_apply_ok + 1))
      fi
    fi
  done

  OBS_PATCH_TOTAL="$patch_total"
  OBS_PATCH_REVERSE_APPLY_OK="$reverse_apply_ok"
fi
echo "[OBSERVATION] patches_dir: $OBS_PATCH_DIR"
echo "[OBSERVATION] patches_total: $OBS_PATCH_TOTAL"
echo "[OBSERVATION] patches_reverse_apply_ok: $OBS_PATCH_REVERSE_APPLY_OK"

# Weights receipt + ckpt dir stats
if [ -f "$MODEL_DIR/ckpt/.source.json" ]; then
  OBS_WEIGHTS_RECEIPT="present"
fi
echo "[OBSERVATION] weights_receipt: $OBS_WEIGHTS_RECEIPT"

if [ -d "$MODEL_DIR/ckpt" ]; then
  OBS_CKPT_DIR="present"
  size_kb="$(du -sk "$MODEL_DIR/ckpt" 2>/dev/null | awk '{print $1}')"
  [ -n "$size_kb" ] && OBS_CKPT_SIZE_KB="$size_kb"
  file_count="$(find "$MODEL_DIR/ckpt" -type f 2>/dev/null | wc -l | tr -d ' ')"
  [ -n "$file_count" ] && OBS_CKPT_FILE_COUNT="$file_count"
fi
echo "[OBSERVATION] ckpt_dir: $OBS_CKPT_DIR"
echo "[OBSERVATION] ckpt_size_kb: $OBS_CKPT_SIZE_KB"
echo "[OBSERVATION] ckpt_file_count: $OBS_CKPT_FILE_COUNT"

# Imports (raw exit code only)
if [ "$OBS_VENV" = "present" ]; then
  PY="$VENV_DIR/bin/python"
  OBS_IMPORTS_ATTEMPTED="attempted"
  "$PY" -c "import torch, transformers, pynini, Cython, soxr" >/dev/null 2>&1
  OBS_IMPORTS_EXIT_CODE="$?"
  OBS_TORCH_VERSION="$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")"
fi
echo "[OBSERVATION] imports_attempted: $OBS_IMPORTS_ATTEMPTED"
echo "[OBSERVATION] imports_exit_code: $OBS_IMPORTS_EXIT_CODE"
echo "[OBSERVATION] torch_version: $OBS_TORCH_VERSION"

# Optional bench (raw exit code only)
if [ "${GLM_TTS_FULL_TEST:-}" = "1" ] && [ "$OBS_VENV" = "present" ]; then
  OBS_BENCH_ATTEMPTED="attempted"
  CMD="cd \"$REPO_ROOT\" && source \"$VENV_DIR/bin/activate\" && python scripts/run_bench.py tts --model glm_tts --text 'Hello world' --device cpu --output"
  bash -c "$CMD" >/dev/null 2>&1
  OBS_BENCH_EXIT_CODE="$?"
fi
echo "[OBSERVATION] bench_attempted: $OBS_BENCH_ATTEMPTED"
echo "[OBSERVATION] bench_exit_code: $OBS_BENCH_EXIT_CODE"

echo ""
echo "=== RECEIPT (paste into docs/run_receipts.md) ==="
echo '```yaml'
echo "glm_tts_observation:"
echo "  timestamp_utc: '$(date -u '+%Y-%m-%dT%H:%M:%SZ')'"
echo "  script_ran: true"
echo "  host_env:"
echo "    kernel: $(uname -s)"
echo "    arch: $(uname -m)"
echo "  observations:"
echo "    venv_status: $OBS_VENV"
echo "    repo_dir: $OBS_REPO_DIR"
echo "    repo_git: $OBS_REPO_GIT"
echo "    repo_commit: $OBS_REPO_COMMIT"
echo "    expected_commit: $OBS_EXPECTED_COMMIT"
echo "    patches_dir: $OBS_PATCH_DIR"
echo "    patches_total: $OBS_PATCH_TOTAL"
echo "    patches_reverse_apply_ok: $OBS_PATCH_REVERSE_APPLY_OK"
echo "    weights_receipt: $OBS_WEIGHTS_RECEIPT"
echo "    ckpt_dir: $OBS_CKPT_DIR"
echo "    ckpt_size_kb: $OBS_CKPT_SIZE_KB"
echo "    ckpt_file_count: $OBS_CKPT_FILE_COUNT"
echo "    imports_attempted: $OBS_IMPORTS_ATTEMPTED"
echo "    imports_exit_code: $OBS_IMPORTS_EXIT_CODE"
echo "    torch_version: $OBS_TORCH_VERSION"
echo "    bench_attempted: $OBS_BENCH_ATTEMPTED"
echo "    bench_exit_code: $OBS_BENCH_EXIT_CODE"
echo '```'

exit 0
