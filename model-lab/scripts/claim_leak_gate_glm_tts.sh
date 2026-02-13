#!/bin/bash
set -uo pipefail

rg -n "\b(success|failed|failure|timeout|verified|verify|verifiable|complete|incomplete)\b|✅|❌|✓|✗" \
  models/glm_tts/PINS.md models/glm_tts/install.sh scripts/verify_glm_tts.sh docs/lcs_log.md \
  2>/dev/null || true
