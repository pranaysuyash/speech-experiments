# Ownership Policy

This document defines ownership and responsibility for key areas of the model-lab codebase.

## Ownership Principles

1. **Every file has an implicit owner** — the last significant contributor or the person who created it.
2. **Ownership is not gatekeeping** — it's responsibility for quality and context.
3. **Consult before major changes** — if you're making significant changes to a file, check git history for the owner.
4. **Document ownership explicitly** — for critical paths, add CODEOWNERS or document in this file.

## Critical Paths (Require Extra Care)

| Path | Owner | Notes |
|------|-------|-------|
| `server/api/` | Core Team | API contracts must be stable |
| `harness/` | Model Team | Inference pipeline, model registry |
| `models/` | Model Team | Model configurations, adapters |
| `client/` | Frontend Team | React UI, must build cleanly |
| `tests/integration/` | QA | Backend invariants |
| `docs/CHANGELOG.md` | DO NOT MODIFY | Protected file |

## Decision Authority

| Decision Type | Authority | Process |
|---------------|-----------|---------|
| New model addition | Model Team | Add to registry, benchmark, document |
| API breaking change | Core Team + User | Requires deprecation notice |
| Dependency updates | Any | Must pass all tests |
| Architecture changes | Core Team | Requires ADR in docs/audit/ |
| Prompt library updates | Any | Update prompts/README.md |

## Ownership Transfer

When ownership needs to transfer:

1. Document in worklog ticket
2. Handoff notes with context
3. Update this file if it's a critical path

## Escalation

If you're unsure about ownership or authority:

1. Check git log for the file
2. Check this document
3. Ask in the ticket/PR
4. When in doubt, preserve existing behavior
