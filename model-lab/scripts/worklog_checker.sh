#!/usr/bin/env bash
# Worklog checker: Ensure recent commits reference tickets
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

echo "Checking recent commits for ticket references..."
git log --oneline -10 | grep -E "TCK-[0-9]{8}-[0-9]{3}" || echo "No ticket references in recent commits. Consider adding TCK-XXXX references."

echo "Checking worklog for open tickets..."
open_count=$(grep -c '\*\*OPEN\*\*' docs/WORKLOG_TICKETS.md 2>/dev/null || true)
echo "Open tickets: ${open_count}"
