# Repo Hygiene Sweep (repo-hygiene-sweep-v1.0)

**Purpose:** Ensure no stray files or untracked artifacts in the repo.

Steps:
1. Run `git status --porcelain` — should be clean after commits.
2. Check for orphaned docs: Ensure all docs in `docs/` are indexed in `docs/README.md` or referenced.
3. Check for unused prompts: Ensure all prompts in `prompts/` are listed in `prompts/README.md`.
4. Check for stray assets: No uncommitted files in root or unexpected directories.
5. If issues found, create tickets in `docs/WORKLOG_TICKETS.md` to clean up.

Evidence: Command outputs from `git status`, `find docs/ -name "*.md"`, etc.