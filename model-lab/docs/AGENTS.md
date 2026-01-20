# Agent Instructions

## Worktrees
- Gemini works in its own worktree (do not create or modify it).
- Codex works only in: ../model-lab-codex (relative to repo root).
- Never edit files directly in the primary worktree.

## Codex branch
- Use branch: codex/hardening-followups (rebased on origin/master).
- Keep diffs small. Tests first, then code, then tooling/CI.

## Non-negotiable verification before merging to master
From ../model-lab-codex/model-lab:
- PYTHONPATH=. pytest -q tests/integration/test_backend_invariants.py
- PYTHONPATH=. pytest -q tests/api/test_artifact_download_security.py
- cd client && npm run build
- git status --porcelain must be empty

## Do not change
- docs/CHANGELOG.md must not be modified or removed.
