# Agent Instructions

## Worktrees
- All work happens in the primary worktree on master branch.
- No separate worktrees needed.

## Branch
- Use branch: master
- Keep diffs comprehensive but tested. Include regression checks.

## Non-negotiable verification before pushing to master
In the workspace:
- PYTHONPATH=. pytest -q tests/integration/test_backend_invariants.py
- PYTHONPATH=. pytest -q tests/api/test_artifact_download_security.py
- cd client && npm run build
- git status --porcelain must be empty (after commit)
- For commits with >10% LOC changes in existing files, review diff for improvements

## Regression Check Process
- git add -A to stage all changes
- Check git diff --cached --stat for LOC changes
- For files with significant changes (>10% of total lines), review diff to ensure better/comprehensive/updated
- If not satisfactory, update code before committing
- Commit with descriptive message
- Push if approved

## Do not change
- docs/CHANGELOG.md must not be modified or removed.
