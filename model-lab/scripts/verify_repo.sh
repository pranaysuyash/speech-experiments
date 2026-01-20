#!/bin/bash
set -e

echo "=== Verifying Backend Invariants (P1/P2) ==="
uv run env PYTHONPATH=. pytest tests/integration/test_backend_invariants.py tests/integration/test_security_traversal.py -v

echo "=== Verifying Client Build ==="
cd client
npm run build

echo "=== ALL CHECKS PASSED ==="
