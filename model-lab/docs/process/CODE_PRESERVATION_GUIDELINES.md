# Code Preservation & Reactivation Guidelines

**Version**: 1.0  
**Last Updated**: 2026-02-05  
**Applies To**: All AI agents working on code cleanup, refactoring, or issue remediation

---

## Philosophy: Implementation Over Deletion

> "Don't just delete unused code. Understand why it exists, see if it can make the app better, and implement functionality rather than delete."

This guideline establishes a **preservation-first, implementation-preferred** approach to handling unused, dead, or seemingly obsolete code.

---

## The Problem with "Cleanup"

Traditional cleanup approaches often:

- Delete code that was intended for a feature
- Lose institutional knowledge embedded in existing code
- Miss opportunities to complete partially-implemented features
- Create false confidence through line-count reduction

**Example:** A model adapter appears unused. Deletion removes 200 lines. But those 200 lines were a nearly-complete adapter for a new ASR model that just needed wiring. Deletion = lost value.

---

## Core Principles

### 1. Investigation Before Action

**When you find unused code, ask:**

1. **What was this supposed to do?**
   - Check commit history: `git log --all -- <file>`
   - Check related tickets/worklog entries
   - Look for TODO comments or feature flags

2. **Why was it abandoned?**
   - Was it deprioritized?
   - Was there a technical blocker?
   - Was it waiting for another feature?

3. **Is it complete or partial?**
   - Complete but unintegrated = high value to activate
   - Partial but functional core = medium value to complete
   - Stub/placeholder = evaluate effort vs. value

4. **Would it add value if activated?**
   - Check product roadmap
   - Consider user value
   - Evaluate technical debt impact

### 2. The Decision Matrix

| Condition | State | Recommended Action |
|-----------|-------|-------------------|
| Complete code, clear purpose, adds value | **Dormant Feature** | **ACTIVATE** - Wire it up, add tests, ship it |
| 70%+ complete, clear value, feasible completion | **Partial Implementation** | **COMPLETE** - Finish and integrate |
| Has tests, well-structured, unclear integration | **Orphaned Code** | **INVESTIGATE** - Find original intent, decide activate/archive |
| Broken/incomplete, no clear purpose, low value | **Technical Debt** | **ARCHIVE** - Move to archive/, document why |
| Obsolete (replaced by better implementation) | **Superseded** | **DELETE** - With explicit approval and documentation |
| Duplicate of existing functionality | **Redundant** | **MERGE** - Combine best aspects, then delete duplicate |

### 3. The Implementation-Deletion Spectrum

```
IMPLEMENTATION (Preferred)              DELETION (Last Resort)
        ↓                                       ↓
   [Activate] → [Complete] → [Merge] → [Archive] → [Delete]
   Dormant      Partial      Combine   Preserve    Remove
   features     features     features  history     forever
```

**Default position: Move left on the spectrum.**

---

## Practical Workflow

### Step 1: Discovery

You find code that appears unused:

```bash
# Example: This model adapter is not imported anywhere
harness/adapters/experimental_tts.py
```

### Step 2: Investigation (REQUIRED)

```bash
# 1. Git history - who created it and why?
git log --all --oneline -- harness/adapters/experimental_tts.py
git show <commit-hash> --stat

# 2. Search for references (even in comments)
rg -i "experimental.*tts" docs/WORKLOG_TICKETS.md
rg -r "experimental_tts" docs/

# 3. Check for feature flags
rg -i "experimental_tts|EXPERIMENTAL_TTS" harness/ models/

# 4. Look at the code itself
head -50 harness/adapters/experimental_tts.py
```

### Step 3: Decision Documentation

Create a brief analysis in the ticket:

```markdown
## Code Investigation: experimental_tts.py

**File**: harness/adapters/experimental_tts.py  
**Lines**: 150  
**Status**: Appears unused (no imports found)

### Investigation Findings

**Git History:**
- Created: 2026-01-15 by dev-X
- Last modified: 2026-01-20
- Commit message: "WIP: Experimental TTS adapter for new model"

**Worklog References:**
- TCK-20260115-042: "Add experimental TTS support"
- Status: Blocked on "model weights availability"

**Code State:**
- Adapter: Complete (inference, preprocessing)
- Logic: Partial (no streaming support)
- Tests: None

### Decision

**RECOMMENDATION: COMPLETE**

Rationale:
- Adapter is 80% complete
- Model weights now available
- Value: High (adds TTS comparison capability)

**Action:** Create ticket to complete integration
**Alternative:** Archive if deprioritized
```

---

## Quick Reference Card

### When You Find Unused Code:

```
1. INVESTIGATE (15-30 min)
   ├── Git history
   ├── Worklog search
   ├── Code review
   └── Product alignment

2. DECIDE
   ├── COMPLETE → If 70%+ done, clear value
   ├── ACTIVATE → If done but unintegrated
   ├── MERGE → If redundant
   ├── ARCHIVE → If unclear/unmaintainable
   └── DELETE → Only if superseded/obvious

3. DOCUMENT
   ├── Worklog ticket
   ├── Investigation findings
   └── Decision rationale

4. EXECUTE
   └── Prefer implementation over deletion
```

---

## Related Documents

- `docs/AGENTS.md` - Core agent principles (Preservation First)
- `docs/process/PROMPT_STYLE_GUIDE.md` - Prompt conventions
- `docs/WORKLOG_TICKETS.md` - Work tracking
