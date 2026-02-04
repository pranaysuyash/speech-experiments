#!/usr/bin/env bash
# Simple audit review helper
# Prints audit files and suggests missing tickets
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

echo "Scanning docs/audit for findings..."
shopt -s nullglob
files=(docs/audit/*.md)
if [ ${#files[@]} -eq 0 ]; then
  echo "No audit files found in docs/audit/"
  exit 0
fi

for f in "${files[@]}"; do
  echo "- $f"
  # look for lines that likely represent findings (simple heuristic)
  grep -nE "(Finding|Issue|Recommendation|TODO)" "$f" || true
done

# Count tickets
tickets_count=$(grep -c "### TCK-" docs/WORKLOG_TICKETS.md || true)
echo "Worklog tickets count: ${tickets_count}"

echo "Quick check complete. If audits have findings not present in docs/WORKLOG_TICKETS.md, create tickets using prompts/workflow/worklog-v1.0.md"
