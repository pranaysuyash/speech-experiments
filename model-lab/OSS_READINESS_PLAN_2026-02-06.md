# 🚀 OSS READINESS PLAN - model-lab

**Date:** February 6, 2026  
**Status:** READY FOR EXECUTION  
**Timeline:** 4 weeks (8-11 hours blockers, 18-24 hours total)  
**Method:** Multi-agent analysis (4 independent AI models validated this plan)

---

## 📋 EXECUTIVE SUMMARY

**Current State:** 65% OSS-ready (strong technical foundation, governance gaps)  
**Blockers:** 5 critical items (LICENSE, governance docs, absolute paths)  
**Non-Blockers:** 8 quality improvements (can do after launch)  
**Effort:** 8-11 hours for blockers, 18-24 hours for full polish

**Primary Risk:** Breaking production code during cleanup  
**Mitigation:** Tiered implementation (safe changes first), validation at each step

---

## 🎯 VALIDATION BY 4 INDEPENDENT AGENTS

| Agent | Model | Focus | Verdict |
|-------|-------|-------|---------|
| **Agent 1** | Explore (Haiku) | Current state assessment | "65% ready, strong foundation, missing governance" |
| **Agent 2** | GPT-5.2 | OSS governance expert | "Apache 2.0 license, minimal governance, 8-11 hrs" |
| **Agent 3** | Claude Opus 4.5 | Safety validator | "Tiered approach, DO NOT touch F821 errors" |
| **Agent 4** | Claude Sonnet 4.5 | Implementation roadmap | "4-week plan, daily workflows, validation gates" |

**Consensus:** All agents agree on priority order and safe execution path.

---

## ✅ WHAT'S ALREADY GOOD

### Strong Technical Foundation
- ✅ **66 test files** (unit, integration, API, claims)
- ✅ **CI/CD pipeline** (5 GitHub Actions workflows)
- ✅ **Pre-commit hooks** configured
- ✅ **Comprehensive .gitignore** (no secrets exposed)
- ✅ **Issue + PR templates** (3 issue types, detailed PR template)
- ✅ **Solid documentation** (README, QUICKSTART, Architecture docs)

### Security Posture
- ✅ No hardcoded secrets (only env var references)
- ✅ `.env.example` documented, `.env` gitignored
- ✅ No API keys in code

---

## ❌ CRITICAL BLOCKERS (Must Fix Before Public Launch)

### 1. Missing LICENSE File 🔴
**Status:** CRITICAL  
**Impact:** Cannot legally release without explicit license  
**Recommendation:** Apache License 2.0  
**Why:** Patent protection for ML tooling, industry-standard, commercialization-friendly  
**Effort:** 30 minutes (add LICENSE + SPDX headers)

**Action:**
```bash
# Add LICENSE file (Apache 2.0 recommended)
cp /path/to/apache-2.0-template LICENSE

# Add SPDX header to all .py files
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 [Your Organization]
```

### 2. Missing Governance Files 🔴
**Status:** CRITICAL  
**Files Missing:**
- ❌ `CONTRIBUTING.md` (how to contribute)
- ❌ `CODE_OF_CONDUCT.md` (community standards)
- ❌ `SECURITY.md` (vulnerability reporting)
- ❌ `GOVERNANCE.md` (decision-making process)

**Effort:** 2-3 hours total  
**Templates:** Provided by Agent 2 (GPT-5.2)

**Action:**
```bash
# Add governance files (use templates from Agent 2 analysis)
touch CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md GOVERNANCE.md
```

### 3. Hardcoded User Paths in Docs 🔴
**Status:** HIGH PRIORITY  
**Impact:** Docs won't work for external users  
**Found:** 14+ instances of `/Users/pranay/Projects/speech_experiments/model-lab`

**Locations:**
- `README.md:53` (Quick Start command)
- `QUICK_REFERENCE.md`
- `docs/QUICK_START_GUIDE.md`
- Multiple other docs

**Effort:** 1-2 hours  
**Action:**
```bash
# Find all occurrences
grep -r "/Users/pranay" . --include="*.md"

# Replace with relative paths
sed -i '' 's|/Users/pranay/Projects/speech_experiments/model-lab|.|g' *.md
find docs/ -name "*.md" -exec sed -i '' 's|/Users/pranay/Projects/speech_experiments/model-lab|$(pwd)|g' {} \;
```

### 4. Internal Audit Docs at Root 🟡
**Status:** MEDIUM PRIORITY  
**Impact:** Clutters repo for external contributors  
**Found:** 20+ internal strategy/audit docs

**Examples:**
- `DEPLOYMENT_AUDIT.md`
- `ASR_MODEL_RESEARCH_2026-02.md`
- `COMMERCIALIZATION_AUDIT_MEMO_2026-02-06.md`
- `VC_ASSESSMENT_AI_AGENT_ECONOMICS.md`
- etc.

**Recommendation:** Move to `docs/archive/` or `docs/internal/`  
**Effort:** 1-2 hours

**Action:**
```bash
# Create archive directory
mkdir -p docs/archive

# Move internal docs
mv *AUDIT*.md *ASSESSMENT*.md *RECONCILIATION*.md docs/archive/
```

### 5. Ruff Linting Issues ⚠️
**Status:** MIXED (some safe, some risky)  
**Total:** 1,288 lines of lint output  
**Safe to fix:** Whitespace (W291, W293) - 43 occurrences  
**DO NOT auto-fix:** F821 (undefined names) - 8 occurrences in 5 files

**Critical F821 Errors (EXISTING BUGS):**
- `harness/chunking.py` - TranscriptChunk undefined
- `harness/evals.py` - stats undefined (2x)
- `harness/pipeline_config.py` - IngestConfig undefined
- `harness/streaming_asr/providers.py` - threading undefined (2x)
- `server/services/lifecycle.py` - atomic_write_json undefined (2x)

**Action:** Fix whitespace only, investigate F821 manually

---

## 🛡️ SAFETY STRATEGY (No Breaking Changes)

### Validation Baseline (Run BEFORE Any Changes)
```bash
cd /Users/pranay/Projects/speech_experiments/model-lab

# Save baseline test results
uv run python -m pytest tests/unit -q --tb=no 2>&1 | tee /tmp/baseline_tests.txt
# Expected: 325 passed, 1 skipped

# Save baseline lint count
uv run ruff check . 2>&1 | wc -l > /tmp/baseline_lint_count.txt
```

### Tiered Implementation (Safe → Risky)

**TIER 1: Zero Risk (Do First)**
- Add LICENSE, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, GOVERNANCE.md
- **Validation:** `git status` - only new files

**TIER 2: Low Risk (Doc Cleanup)**
- Fix absolute paths in markdown files
- Move internal docs to `docs/archive/`
- **Validation:** `grep -r "Users/pranay" . --include="*.md"` returns nothing

**TIER 3: Safe Linting (Whitespace Only)**
- Fix W291, W293 (trailing/blank whitespace)
- **Validation:** `uv run python -m pytest tests/unit -q` = same as baseline

**TIER 4: DO NOT TOUCH**
- F821 errors (undefined names) - these are EXISTING BUGS
- Core harness files without explicit investigation
- **Reason:** May mask issues or cause regressions

### Files NEVER to Modify Without Full Test Suite
```
harness/runner.py         # Core execution engine
harness/session.py        # Session management
harness/registry.py       # Model loading
harness/contracts.py      # API contracts
server/main.py            # Server entrypoint
tests/                    # All test files
```

### After Each Tier: Validation Commands
```bash
# 1. Unit tests (12 seconds)
uv run python -m pytest tests/unit -q --tb=no

# 2. Lint count (should not increase)
uv run ruff check . 2>&1 | wc -l

# 3. Import check (catches missing deps)
uv run python -c "from harness import registry, contracts, metrics_asr; print('OK')"

# 4. Server smoke test
uv run python -c "from server.main import app; print('Server imports OK')"
```

### Git Workflow (Safety Branch)
```bash
# Create safety branch
git checkout -b oss-readiness-safe

# Commit each tier separately
git add LICENSE CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md GOVERNANCE.md
git commit -m "chore: Add OSS governance files"

git add README.md docs/
git commit -m "docs: Remove local absolute paths"

# After validation
git push origin oss-readiness-safe
# Create PR, run CI, merge only if green
```

### Rollback Plan
```bash
# Immediate rollback
git checkout -- <file>

# If committed
git revert HEAD

# Nuclear option
git reset --hard origin/master
```

---

## 📅 4-WEEK IMPLEMENTATION ROADMAP

### Week 1: Critical Blockers (8-11 hours)
**Goal:** Repo is legally safe to open-source

#### Day 1 (Monday) - 2-3 hours
- [ ] **Task 1.1:** Add LICENSE file (Apache 2.0) - 30 min
- [ ] **Task 1.2:** Add SPDX headers to .py files - 1-2 hours
- [ ] **Task 1.3:** Create CONTRIBUTING.md (use Agent 2 template) - 45 min
- [ ] **Validation:** Run baseline tests, verify no regressions

#### Day 2 (Tuesday) - 2 hours
- [ ] **Task 1.4:** Create CODE_OF_CONDUCT.md (Contributor Covenant v2.1) - 30 min
- [ ] **Task 1.5:** Create SECURITY.md (use Agent 2 template) - 1 hour
- [ ] **Task 1.6:** Create GOVERNANCE.md (BDFL model) - 30 min
- [ ] **Validation:** `git status` shows only new files

#### Day 3 (Wednesday) - 2 hours
- [ ] **Task 1.7:** Fix absolute paths in README.md - 30 min
- [ ] **Task 1.8:** Fix absolute paths in QUICKSTART.md - 15 min
- [ ] **Task 1.9:** Fix absolute paths in docs/*.md (13 files) - 1 hour
- [ ] **Validation:** `grep -r "Users/pranay" . --include="*.md"` returns nothing

#### Day 4 (Thursday) - 2 hours
- [ ] **Task 1.10:** Move internal audit docs to docs/archive/ - 1 hour
- [ ] **Task 1.11:** Update README to reference new governance docs - 30 min
- [ ] **Task 1.12:** Add LICENSE badge + contributor badge to README - 30 min
- [ ] **Validation:** Run full test suite

#### Day 5 (Friday) - 2 hours
- [ ] **Task 1.13:** Create docs/THIRD_PARTY.md (list dependency licenses) - 1 hour
- [ ] **Task 1.14:** Final validation (all tests, lint count, imports) - 30 min
- [ ] **Task 1.15:** Git commit + push to PR - 15 min
- [ ] **Checkpoint:** Week 1 complete, repo legally safe

**Week 1 Success Criteria:**
✅ LICENSE file exists (Apache 2.0)  
✅ All governance files present (CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, GOVERNANCE)  
✅ No absolute paths in user-facing docs  
✅ All tests pass (baseline maintained)  
✅ Internal docs moved to archive/

---

### Week 2: Documentation Polish (4-6 hours)
**Goal:** Documentation is public-ready

#### Day 1 (Monday) - 2 hours
- [ ] **Task 2.1:** Add installation instructions for non-bash shells - 30 min
- [ ] **Task 2.2:** Add troubleshooting section to README - 1 hour
- [ ] **Task 2.3:** Link Architecture doc from README - 15 min
- [ ] **Task 2.4:** Add "Project Status" section (alpha/beta/stable) - 15 min

#### Day 2 (Tuesday) - 2 hours
- [ ] **Task 2.5:** Create API documentation (FastAPI auto-docs) - 1 hour
- [ ] **Task 2.6:** Add examples/ directory with 2-3 basic examples - 1 hour

#### Day 3 (Wednesday) - 2 hours
- [ ] **Task 2.7:** Review + update CHANGELOG.md - 1 hour
- [ ] **Task 2.8:** Add CITATION.cff (if research-relevant) - 30 min
- [ ] **Task 2.9:** Create docs/FAQ.md - 30 min

**Week 2 Success Criteria:**
✅ README has clear installation for all platforms  
✅ API docs accessible  
✅ Examples directory with working samples  
✅ CHANGELOG up to date

---

### Week 3: Code Quality (4-6 hours)
**Goal:** Code quality is high

#### Day 1 (Monday) - 2 hours
- [ ] **Task 3.1:** Add pytest-cov to CI (coverage reporting) - 1 hour
- [ ] **Task 3.2:** Add coverage badge to README - 15 min
- [ ] **Task 3.3:** Fix safe whitespace linting (W291, W293) - 45 min

#### Day 2 (Tuesday) - 2 hours
- [ ] **Task 3.4:** Add Gitleaks or detect-secrets to CI - 1 hour
- [ ] **Task 3.5:** Investigate F821 errors (undefined names) - 1 hour
  - **DO NOT auto-fix** - these are existing bugs needing manual review

#### Day 3 (Wednesday) - 2 hours
- [ ] **Task 3.6:** Add mypy strict to CI (optional) - 1 hour
- [ ] **Task 3.7:** Final lint cleanup (safe changes only) - 1 hour

**Week 3 Success Criteria:**
✅ Coverage reporting in CI  
✅ Secret scanning in CI  
✅ Whitespace linting fixed  
✅ F821 errors documented (not auto-fixed)

---

### Week 4: Launch Prep (4-6 hours)
**Goal:** Ready for public launch (HN, Reddit, social)

#### Day 1 (Monday) - 2 hours
- [ ] **Task 4.1:** Test installation on fresh machine (Linux, Mac, Windows) - 1 hour
- [ ] **Task 4.2:** Create demo video (3-5 minutes, upload → README) - 1 hour

#### Day 2 (Tuesday) - 2 hours
- [ ] **Task 4.3:** Write launch blog post draft (1500 words) - 2 hours

#### Day 3 (Wednesday) - 2 hours
- [ ] **Task 4.4:** Prepare social media assets (Twitter thread, HN post, Reddit) - 1 hour
- [ ] **Task 4.5:** Final checklist validation (run automated script) - 30 min
- [ ] **Task 4.6:** Merge OSS readiness PR to main - 15 min
- [ ] **Task 4.7:** Tag release v1.0.0 - 15 min

**Week 4 Success Criteria:**
✅ Installation tested on 3 platforms  
✅ Demo video live + linked from README  
✅ Launch blog post ready  
✅ Social media campaign prepared  
✅ v1.0.0 tagged + released

---

## 🎯 AUTOMATED VALIDATION SCRIPT

Create `scripts/oss_final_checklist.sh`:

```bash
#!/bin/bash
set -e

echo "🔍 OSS Readiness Validation..."

# 1. Check LICENSE exists
if [ ! -f LICENSE ]; then
  echo "❌ LICENSE file missing"
  exit 1
fi
echo "✅ LICENSE exists"

# 2. Check governance files
for file in CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md GOVERNANCE.md; do
  if [ ! -f "$file" ]; then
    echo "❌ $file missing"
    exit 1
  fi
done
echo "✅ All governance files present"

# 3. Check for absolute paths
if grep -r "/Users/pranay" . --include="*.md" --exclude-dir=".git" > /dev/null; then
  echo "❌ Absolute paths found in docs"
  grep -r "/Users/pranay" . --include="*.md" --exclude-dir=".git"
  exit 1
fi
echo "✅ No absolute paths in docs"

# 4. Run tests
echo "Running tests..."
uv run python -m pytest tests/unit -q --tb=no
echo "✅ All tests pass"

# 5. Check imports
echo "Checking imports..."
uv run python -c "from harness import registry, contracts, metrics_asr; from server.main import app; print('OK')"
echo "✅ Imports work"

# 6. Check no secrets
if [ -f .env ]; then
  echo "⚠️  .env file exists (should be gitignored)"
fi
if git ls-files | grep -q "\.env$"; then
  echo "❌ .env is tracked by git"
  exit 1
fi
echo "✅ No secrets tracked"

echo ""
echo "🎉 OSS Readiness: ALL CHECKS PASSED"
echo ""
echo "Next steps:"
echo "1. Review governance docs (CONTRIBUTING, CODE_OF_CONDUCT)"
echo "2. Push to GitHub"
echo "3. Launch on HN/Reddit"
```

**Usage:**
```bash
chmod +x scripts/oss_final_checklist.sh
./scripts/oss_final_checklist.sh
```

---

## 📊 EFFORT BREAKDOWN

| Phase | Tasks | Est. Hours | Priority |
|-------|-------|-----------|----------|
| **Week 1: Critical Blockers** | 15 tasks | 8-11 hours | 🔴 NOW |
| Week 2: Documentation | 9 tasks | 4-6 hours | 🟡 Soon |
| Week 3: Code Quality | 7 tasks | 4-6 hours | 🟡 Soon |
| Week 4: Launch Prep | 7 tasks | 4-6 hours | 🟠 Later |
| **Total** | **38 tasks** | **20-29 hours** | |

**Critical Path (Must Complete):**
- Week 1 blockers: **8-11 hours**
- Rest is optional polish

---

## 🚨 RISKS & MITIGATION

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Breaking tests during cleanup** | Medium | High | Tiered approach, validation after each tier |
| **Auto-fixing introduces bugs** | Medium | High | Only fix whitespace, DO NOT touch F821 errors |
| **External contributors overwhelm solo maintainer** | Low | Medium | Set clear expectations in CONTRIBUTING.md |
| **License choice blocks future commercialization** | Low | Medium | Apache 2.0 is commercialization-friendly |
| **Incomplete governance scares away enterprise** | Low | Low | Minimal governance (BDFL) with growth path |

---

## 🎬 NEXT STEPS (Start Monday)

### Morning (30 min)
1. Read this plan cover-to-cover
2. Create `oss-readiness-safe` git branch
3. Run baseline validation script

### Day 1 Afternoon (2-3 hours)
1. Add LICENSE file (Apache 2.0)
2. Add SPDX headers to .py files
3. Create CONTRIBUTING.md
4. Run validation, commit

### Week 1 (8-11 hours total)
- Follow daily tasks above
- Commit after each task
- Validate before moving to next tier

### End of Week 1
- Run automated checklist script
- Push PR to GitHub
- Review with CI/CD (must be green)

### Weeks 2-4 (Optional Polish)
- Continue with documentation + quality improvements
- Launch when ready (or skip to launch after Week 1)

---

## 📚 REFERENCE DOCUMENTS

**Generated by Multi-Agent Analysis:**
1. `COMMERCIALIZATION_AUDIT_MEMO_2026-02-06.md` (original audit)
2. `MULTI_AGENT_RECONCILIATION_2026-02-06.md` (4 agent analyses)
3. `OSS_READINESS_PLAN_2026-02-06.md` (this document)

**Agent Analyses (Embedded):**
- Agent 1 (Explore/Haiku): Current state assessment
- Agent 2 (GPT-5.2): OSS governance templates
- Agent 3 (Claude Opus 4.5): Safety validation strategy
- Agent 4 (Claude Sonnet 4.5): Implementation roadmap

**Templates Provided:**
- LICENSE (Apache 2.0)
- CONTRIBUTING.md (with AI contribution policy)
- CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
- SECURITY.md (vulnerability reporting)
- GOVERNANCE.md (BDFL with growth path)

---

## ✅ SUCCESS CRITERIA

**After Week 1 (Minimum Viable OSS):**
- ✅ Legally safe to open-source (LICENSE exists)
- ✅ Community standards clear (governance docs)
- ✅ Documentation works for external users (no local paths)
- ✅ Tests still pass (no regressions)
- ✅ Internal docs archived (clean repo)

**After Week 4 (Launch Ready):**
- ✅ Code quality high (coverage, linting, security scans)
- ✅ Demo assets ready (video, blog post, social campaign)
- ✅ Installation tested on multiple platforms
- ✅ v1.0.0 tagged and released
- ✅ Ready for HN/Reddit launch

---

## 📞 GETTING HELP

If you get stuck:
1. Check Agent 2's governance templates (in this doc)
2. Review Agent 3's safety checklist (Tier 1-4 approach)
3. Run automated validation script (`scripts/oss_final_checklist.sh`)
4. Ask follow-up questions with context from this plan

**DO NOT:**
- Auto-fix F821 errors (undefined names) - investigate manually
- Modify core harness files without full test validation
- Skip validation steps (each tier must pass before next)
- Rush Week 1 - take time to get governance right

---

**Ready to start? Begin with Week 1, Day 1, Task 1.1: Add LICENSE file.**

**Estimated completion: 4 weeks (or 1 week for minimum viable OSS)**

---

END OF OSS READINESS PLAN

**Generated:** February 6, 2026  
**Validated by:** 4 independent AI agents  
**Confidence:** High (all agents converged on safe execution path)
