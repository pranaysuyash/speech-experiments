#!/usr/bin/env bash
set -euo pipefail

# Regression Check: Runs verification suite before commits
# Usage: ./scripts/regression_check.sh [--staged]
#
# Runs:
# 1. Backend invariant tests
# 2. Security tests
# 3. Frontend build
# 4. TypeScript check (if applicable)

STAGED_MODE=false
if [[ "${1:-}" == "--staged" ]]; then
    STAGED_MODE=true
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "=== Regression Check ==="

# Track failures
FAILED=0

# 1. Backend invariant tests
echo "→ Running backend invariant tests..."
if PYTHONPATH=. pytest -q tests/integration/test_backend_invariants.py 2>/dev/null; then
    echo "  ✅ Backend invariants passed"
else
    echo "  ❌ Backend invariants FAILED"
    FAILED=1
fi

# 2. Security tests
echo "→ Running security tests..."
if PYTHONPATH=. pytest -q tests/api/test_artifact_download_security.py 2>/dev/null; then
    echo "  ✅ Security tests passed"
else
    echo "  ❌ Security tests FAILED"
    FAILED=1
fi

# 3. Frontend build (if client exists)
if [[ -d "client" ]]; then
    echo "→ Building frontend..."
    if (cd client && npm run build) >/dev/null 2>&1; then
        echo "  ✅ Frontend build passed"
    else
        echo "  ❌ Frontend build FAILED"
        FAILED=1
    fi
fi

# 4. Python type check (optional, warn only)
echo "→ Running type check..."
if PYTHONPATH=. mypy server/ harness/ --ignore-missing-imports 2>/dev/null; then
    echo "  ✅ Type check passed"
else
    echo "  ⚠️  Type check has warnings (non-blocking)"
fi

# Summary
echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "=== ✅ All regression checks passed ==="
    exit 0
else
    echo "=== ❌ Some checks failed ==="
    exit 1
fi
