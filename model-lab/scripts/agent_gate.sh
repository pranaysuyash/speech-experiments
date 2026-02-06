#!/usr/bin/env bash
set -euo pipefail

# Agent Gate: Enforces worklog discipline before commits
# Usage: ./scripts/agent_gate.sh [--staged]
#
# Checks:
# 1. Worklog ticket exists for significant changes
# 2. No prohibited file modifications (e.g., CHANGELOG.md)
# 3. Evidence of verification (test runs, etc.)

STAGED_MODE=false
if [[ "${1:-}" == "--staged" ]]; then
    STAGED_MODE=true
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "=== Agent Gate Check ==="

# 1. Check for prohibited file modifications
PROHIBITED_FILES=("docs/CHANGELOG.md")

for file in "${PROHIBITED_FILES[@]}"; do
    if $STAGED_MODE; then
        if git diff --cached --name-only | grep -q "^${file}$"; then
            echo "❌ BLOCKED: Cannot modify protected file: $file"
            echo "   Remove from staging: git reset HEAD $file"
            exit 1
        fi
    else
        if git diff --name-only | grep -q "^${file}$"; then
            echo "⚠️  WARNING: Protected file modified: $file"
        fi
    fi
done

# 2. Check worklog has recent entries (sanity check)
WORKLOG="docs/WORKLOG_TICKETS.md"
if [[ -f "$WORKLOG" ]]; then
    # Check if worklog was modified recently (in last 50 commits)
    WORKLOG_COMMITS=$(git log --oneline -50 -- "$WORKLOG" | wc -l | tr -d ' ')
    if [[ "$WORKLOG_COMMITS" -lt 1 ]]; then
        echo "⚠️  WARNING: Worklog not updated in last 50 commits"
        echo "   Consider adding a ticket for your work"
    fi
else
    echo "❌ BLOCKED: Worklog file missing: $WORKLOG"
    exit 1
fi

# 3. Check for large changes without ticket reference in commit message
if $STAGED_MODE; then
    STAGED_FILES=$(git diff --cached --name-only | wc -l | tr -d ' ')
    if [[ "$STAGED_FILES" -gt 5 ]]; then
        echo "ℹ️  INFO: $STAGED_FILES files staged - ensure worklog ticket exists"
    fi
fi

# 4. Quick sanity: no .env files staged
if $STAGED_MODE; then
    if git diff --cached --name-only | grep -E "^\.env$|\.env\.local$" | grep -v ".example"; then
        echo "❌ BLOCKED: Cannot commit .env files (secrets)"
        exit 1
    fi
fi

echo "✅ Agent gate passed"
