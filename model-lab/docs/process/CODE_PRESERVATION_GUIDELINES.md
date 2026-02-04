# Code Preservation Guidelines

Principle: Investigate before deleting. Preserve contributor intent and history unless there is explicit approval to remove code.

When you find unused or suspicious code:
1. Search git history: `git log -p -- <file>` to understand origin and rationale.
2. Ask the author (if resolvable) or add a note in the audit file referencing the code section.
3. If removal is considered, prefer moving to `archive/` with a pointer note and an audit explaining why deletion is chosen.
4. If code is partially complete and useful, prefer activation (complete the work) rather than deleting.
5. Document every preservation decision in `docs/audit/<sanitized-file>.md` and create a worklog ticket.

No deletions without explicit approval recorded in a ticket.
