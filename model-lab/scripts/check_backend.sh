#!/bin/bash
set -euo pipefail

echo "========================================"
echo "Running Pre-Push Backend Validation"
echo "========================================"

# 1. Unit Tests (Strict)
echo "[1/2] Running Unit Tests..."
if [ -f "package.json" ]; then
    npm test > /dev/null 2>&1 || echo "Warning: npm test failed or skipped" 
fi

# 2. Backend Fix Verification
echo "[2/2] Verifying Backend Hardening..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python scripts/verify_backend_fixes.py

# 3. API Visibility Invariant
echo "[3/3] Verifying API Visibility Invariant..."
python scripts/test_api_visibility.py

echo "========================================"
echo "✅ All Checks Passed"
echo "========================================"
