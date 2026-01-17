#!/usr/bin/env bash
set -euo pipefail

# Guard script to prevent tracking generated runs artifacts
# Add to CI workflow to enforce

bad=$(git ls-files | rg '^runs/' || true)

# Allow exactly these keepers
allowed_regex='^(runs/\.gitkeep)$'

if [[ -n "${bad}" ]]; then
  bad2=$(echo "${bad}" | rg -v "${allowed_regex}" || true)
  if [[ -n "${bad2}" ]]; then
    echo "ERROR: tracked files under runs/ detected:"
    echo "${bad2}"
    exit 1
  fi
fi

echo "OK: no tracked generated run artifacts."
