# Prompts Index

This folder contains reusable prompts that guide agents and contributors when performing work in this repository.

## Start Here (Any Agent)
1. Use: `prompts/workflow/agent-entrypoint-v1.0.md`
2. Update tracking: `docs/WORKLOG_TICKETS.md` (append-only)
3. Ensure local enforcement if needed: `git config core.hooksPath .githooks` (see `scripts/setup-githooks.sh`)

## Workflow / Tracking
- Agent entrypoint: `prompts/workflow/agent-entrypoint-v1.0.md`
- Worklog update helper: `prompts/workflow/worklog-v1.0.md`
- Pre-flight check: `prompts/workflow/pre-flight-check-v1.0.md`
- Project setup verification: `prompts/workflow/project-setup-verification-v1.0.md`
- Repo hygiene sweep: `prompts/workflow/repo-hygiene-sweep-v1.0.md`
- Prompt quality gate: `prompts/workflow/prompt-quality-gate-v1.0.md`

## Audit / Review
- Comprehensive product audit: `prompts/audit/comprehensive-audit-v1.0.md`

## Project Context (Read Before Making Changes)
- Overview: `docs/PROJECT_RULES.md`
- Architecture & implementation docs under `docs/`.
- Process reminders: `docs/process/PROCESS_REMINDER.md`
- Issues workflow: `docs/ISSUES_WORKFLOW.md`
- Prompt style guide: `docs/process/PROMPT_STYLE_GUIDE.md`
- Code preservation: `docs/process/CODE_PRESERVATION_GUIDELINES.md`

> Note: This is a repo-specific subset. Extend with role or domain prompts as needed.