#!/usr/bin/env bash
# Configure repo to use the bundled .githooks directory
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

if [ -d .githooks ]; then
  git config core.hooksPath .githooks
  echo "Set git hooks path to .githooks"
else
  echo "No .githooks directory found. Create .githooks/ and add hooks or skip."
fi
