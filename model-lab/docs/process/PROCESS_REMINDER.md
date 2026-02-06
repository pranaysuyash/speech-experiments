# Process Reminder (Quick Reference)

**Read this before starting any work.**

> **This is an open exploration lab.** Any AI model capability is in scope.  
> Don't limit to common tasks - explore music generation, audio separation, depth estimation, molecule design, pose tracking, or anything a model can do.

## 5-Second Checklist

1. ✅ Ticket exists in `docs/WORKLOG_TICKETS.md`?
2. ✅ Scope defined (in-scope, out-of-scope)?
3. ✅ Using correct prompt from `prompts/README.md`?
4. ✅ Working on master branch?
5. ✅ Will update worklog when done?

## Before You Code

```bash
# 1. Check current state
git status --porcelain
git branch --show-current

# 2. Find or create ticket
rg "TCK-" docs/WORKLOG_TICKETS.md | tail -5

# 3. Enable hooks (first time only)
git config core.hooksPath .githooks
```

## During Work

- Stay in scope (no "one more thing")
- Document evidence for claims
- Run tests frequently

## After Work

```bash
# 1. Run verification
PYTHONPATH=. pytest -q tests/integration/test_backend_invariants.py
cd client && npm run build

# 2. Stage all changes
git add -A

# 3. Commit with ticket reference
git commit -m "fix(scope): description

Refs: TCK-YYYYMMDD-###"

# 4. Update worklog ticket to COMPLETED
```

## Evidence Labels

Every claim needs one:

- **Observed**: Directly verified (test output, command result)
- **Inferred**: Logical conclusion from observed facts
- **Unknown**: Cannot determine, needs investigation

## Key Files

| File | Purpose |
|------|---------|
| `docs/WORKLOG_TICKETS.md` | Track all work |
| `docs/CLAIMS.md` | Model performance claims |
| `prompts/README.md` | Find the right prompt |
| `docs/AGENTS.md` | Full agent instructions |

## Don't Forget

- 🚫 Don't create branches
- 🚫 Don't delete unrecognized changes
- 🚫 Don't skip worklog updates
- 🚫 Don't modify CHANGELOG.md
- ✅ Do use `git add -A`
- ✅ Do document evidence
- ✅ Do run tests before commit
