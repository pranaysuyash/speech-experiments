# Quarantined Runs

Runs moved here are excluded from arsenal/decisions generation but preserved for audit.

## Manifest

| Run Path | Reason | Quarantined | By |
|----------|--------|-------------|-----|
| faster_whisper/asr/2026-01-10_00-13-54.json | WER=0 from poisoned smoke GT (derived from model output) | 2026-01-11 | agent |

## Policy

- Never delete runs; quarantine them
- Arsenal generator ignores this directory
- Include reason for future debugging
