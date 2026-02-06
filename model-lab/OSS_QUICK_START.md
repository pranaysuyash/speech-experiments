# 🚀 OSS Roadmap Quick Start

**For solo founders who want to execute immediately.**

## TL;DR
- **Time:** 80 hours (4 weeks × 20h/week = 4h/day)
- **Goal:** Legally safe, publicly ready, launch-worthy OSS repo
- **Detailed Plan:** See `OSS_IMPLEMENTATION_ROADMAP.md`

## Today (First 30 Minutes)

1. **Read the full roadmap:**
   ```bash
   cat OSS_IMPLEMENTATION_ROADMAP.md
   ```

2. **Set up tracking:**
   ```bash
   cp OSS_IMPLEMENTATION_ROADMAP.md OSS_DAILY_LOG.md
   # Check off tasks as you complete them
   ```

3. **Start Week 1, Task 1.1:**
   ```bash
   # Add LICENSE file (5 minutes)
   cat > LICENSE << 'EOF'
   MIT License
   
   Copyright (c) 2026 Model Lab Contributors
   
   Permission is hereby granted, free of charge, to any person obtaining a copy...
   EOF
   git add LICENSE
   git commit -m "feat: Add MIT LICENSE [0.5h]"
   ```

## Daily Routine

### Morning (9 AM - 9:15 AM)
```bash
cd model-lab
git pull
source .venv/bin/activate

# Quick health check
pytest -m "not real_e2e" -q
ruff check . --statistics
```

### Work Block (9:15 AM - 1:00 PM)
- Follow tasks for current day in roadmap
- Commit every 30-60 minutes
- Run tests after each change

### End of Day (5:45 PM - 6:00 PM)
```bash
# Validate
pytest -m "not real_e2e"
ruff check .

# Commit
git add -A
git commit -m "feat: [Week X Day Y] - [summary]"

# Update progress
# Check off completed tasks in OSS_DAILY_LOG.md
```

## Critical Path

**Must complete in order:**
1. Week 1 → LICENSE + governance (legal safety)
2. Week 2 → Path cleanup (privacy/portability)
3. Week 3 → Code quality (professional polish)
4. Week 4 → Launch prep (marketing ready)

**Can't skip:**
- Week 1, Task 1.1 (LICENSE) - legal blocker
- Week 2, Tasks 2.1-2.2 (path cleanup) - privacy blocker
- Week 4, Task 4.7 (secret scan) - security blocker

**Can defer:**
- Week 3, Task 3.9 (type hints) - nice-to-have
- Week 4, Task 4.3 (social assets) - can do post-launch

## Weekly Checkpoints

### End of Week 1 (Friday)
```bash
./scripts/oss_final_checklist.sh  # Should show LICENSE ✅
git tag -a v0.1.0-oss.week1 -m "Checkpoint: Legal foundation"
```

**Sign-off:** "Repo is legally safe to open-source"

### End of Week 2 (Friday)
```bash
./scripts/audit_paths.sh  # Should show 0 occurrences
git tag -a v0.1.0-oss.week2 -m "Checkpoint: Docs ready"
```

**Sign-off:** "Documentation is public-ready"

### End of Week 3 (Friday)
```bash
ruff check . --statistics  # Should show <20 errors
git tag -a v0.1.0-oss.week3 -m "Checkpoint: Code quality"
```

**Sign-off:** "Code quality is high"

### End of Week 4 (Friday)
```bash
./scripts/oss_final_checklist.sh  # All ✅
git tag -a v0.1.0 -m "Initial public release"
```

**Sign-off:** "Ready for HN/Reddit launch"

## Rollback Commands

**If something breaks:**
```bash
# Undo last commit
git reset --hard HEAD~1

# Or go back to checkpoint
git reset --hard v0.1.0-oss.week[N]

# Run tests
pytest -m "not real_e2e"
```

## Time Budget by Week

| Week | Focus | Hours | Critical? |
|------|-------|-------|-----------|
| Week 1 | Legal + governance | 18h | 🔴 YES |
| Week 2 | Docs + paths | 22h | 🔴 YES |
| Week 3 | Code quality | 20h | 🟡 MEDIUM |
| Week 4 | Launch prep | 20h | 🟢 POLISH |

**If pressed for time:** 
- Weeks 1-2 are mandatory (40h)
- Week 3 can be reduced (do critical fixes only, ~10h)
- Week 4 can be done post-launch (except secret scan)

## Success Criteria

**Must have:**
- [x] LICENSE (MIT)
- [ ] Zero hardcoded paths
- [ ] Zero secrets
- [ ] Tests passing
- [ ] CODE_OF_CONDUCT, CONTRIBUTING, SECURITY

**Nice to have:**
- [ ] Linting <20 errors
- [ ] CHANGELOG
- [ ] FAQ in README
- [ ] Social media posts drafted

## Launch Day Checklist

**Friday, Week 4:**
1. Run `./scripts/oss_final_checklist.sh` (all ✅)
2. Create GitHub Release (v0.1.0)
3. Post to HackerNews (Show HN)
4. Post to Reddit (r/MachineLearning)
5. Tweet/LinkedIn
6. Monitor comments for first 48h

## Emergency Contacts

**If stuck:**
1. Check `OSS_IMPLEMENTATION_ROADMAP.md` rollback section
2. Test in isolation (new branch)
3. Skip non-critical task and move on
4. Use `git reflog` to find safe state

**If secrets exposed:**
1. STOP everything
2. Rotate credentials immediately
3. Follow Week 4, Task 4.7 procedure
4. Use `git filter-repo` to rewrite history

## Tools You'll Need

**Install now:**
```bash
# Package manager (already have)
brew install uv

# Security scanning
brew install trufflehog gitleaks git-secrets

# Markdown linting (optional)
npm install -g markdownlint-cli
```

## Questions?

- Detailed roadmap: `OSS_IMPLEMENTATION_ROADMAP.md`
- Daily tracking: `OSS_DAILY_LOG.md` (create from roadmap)
- Validation: `./scripts/oss_final_checklist.sh`

**Start now:** Week 1, Task 1.1 (LICENSE) - takes 5 minutes! 🚀
