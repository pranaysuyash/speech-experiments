#!/usr/bin/env bash
set -euo pipefail

uv sync --all-extras --dev
uv run pytest -q tests/integration/test_backend_invariants.py

pushd client >/dev/null
npm ci
npm run build
popd >/dev/null
