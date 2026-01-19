# Legacy Data Risks

## Violations
- **Missing `failure_step`**: 48 historical runs (pre-Jan 18, 2026) in `FAILED` state lack the specific failure step field.
- **Invalid Timestamp Format**: 4 historical runs use non-ISO/UTC timestamps (missing 'Z').

## Acceptability
- These runs represent pre-production developer tests.
- Current code handles missing fields gracefully (defaults to "Unknown" or standard parsing).
- No customer data is affected as this is a fresh launch.

## Non-Blocking Justification
- System stability is unaffected by malformed historical JSON.
- New runs function correctly and adhere to all invariants.

## Conclusion
**No migration required for MVP.**
