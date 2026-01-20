#!/usr/bin/env bash
set -euo pipefail

uv run pytest -q tests/integration/test_backend_invariants.py

pushd client >/dev/null
npm run build
popd >/dev/null
