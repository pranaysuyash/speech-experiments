#!/usr/bin/env bash
set -euo pipefail

PY="$(pwd)/.venv/bin/python"
"$PY" -m pytest -q tests/integration/test_backend_invariants.py
"$PY" -m pytest -q tests/api/test_artifact_download_security.py

pushd client >/dev/null
npm run build
popd >/dev/null
