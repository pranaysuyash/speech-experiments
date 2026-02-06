#!/bin/bash
# Final OSS Readiness Checklist
# Run this before launch (Week 4, Thursday)

set -e

echo "🔍 Running final OSS readiness checks..."
echo ""

PASSED=0
FAILED=0

# Helper functions
check_pass() {
    echo "✅ $1"
    ((PASSED++))
}

check_fail() {
    echo "❌ $1"
    ((FAILED++))
}

# 1. License
if [ -f LICENSE ]; then
    check_pass "LICENSE file exists"
else
    check_fail "LICENSE missing"
fi

# 2. Governance files
[ -f CODE_OF_CONDUCT.md ] && check_pass "Code of Conduct" || check_fail "Code of Conduct missing"
[ -f CONTRIBUTING.md ] && check_pass "Contributing guide" || check_fail "Contributing guide missing"
[ -f SECURITY.md ] && check_pass "Security policy" || check_fail "Security policy missing"

# 3. Hardcoded paths
PATHS=$(grep -r "/Users/pranay" . --exclude-dir=".venv" --exclude-dir=".git" --exclude-dir="__pycache__" --exclude="*.log" 2>/dev/null | wc -l | tr -d ' ')
if [ "$PATHS" -eq 0 ]; then
    check_pass "No hardcoded paths"
else
    check_fail "Found $PATHS hardcoded paths"
fi

# 4. Tests
echo -n "⏳ Running tests... "
if source .venv/bin/activate && pytest -m "not real_e2e" -q > /dev/null 2>&1; then
    check_pass "Tests pass"
else
    check_fail "Tests failing"
fi

# 5. Linting
echo -n "⏳ Checking linting... "
ERRORS=$(source .venv/bin/activate && ruff check . --statistics 2>&1 | grep "Found" | awk '{print $2}' || echo "999")
if [ "$ERRORS" -lt 20 ]; then
    check_pass "Linting clean (<20 errors, found $ERRORS)"
else
    check_fail "Too many linting errors ($ERRORS)"
fi

# 6. Documentation
[ -f docs/README.md ] && check_pass "Docs index exists" || check_fail "Docs index missing"
[ -f CHANGELOG.md ] && check_pass "Changelog exists" || check_fail "Changelog missing"

# 7. GitHub templates
[ -d .github/ISSUE_TEMPLATE ] && check_pass "Issue templates configured" || check_fail "Issue templates missing"
[ -f .github/PULL_REQUEST_TEMPLATE.md ] && check_pass "PR template exists" || check_fail "PR template missing"

# 8. CI/CD
[ -f .github/workflows/ci.yml ] && check_pass "CI workflow exists" || check_fail "CI workflow missing"

echo ""
echo "📊 Summary"
echo "=========="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ "$FAILED" -eq 0 ]; then
    echo "🎉 All checks passed! Ready for launch! 🚀"
    exit 0
else
    echo "⚠️  Please fix failing checks before launch"
    exit 1
fi
