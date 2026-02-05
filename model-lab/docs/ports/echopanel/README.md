# EchoPanel ports

This folder contains **verbatim-ish** copies of EchoPanel docs that are relevant to Model-Lab’s audio research, kept here so the lab is the single place to iterate on evaluation methodology.

**Port date**: 2026-02-05  
**Source repo**: `../EchoPanel/docs/...` (local checkout)

## What’s here (ASR-relevant)

- `STREAMING_ASR_AUDIT_2026-02.md` — deep audit of the streaming architecture and edge cases
- `STREAMING_ASR_NLP_AUDIT.md` — streaming ASR + NLP concurrency, buffering, correctness risks, test ideas
- `WS_CONTRACT.md` — WebSocket contract details (PCM format, event types)
- `HARDWARE_AND_PERFORMANCE.md` — quick perf/hardware notes
- `STATUS_AND_ROADMAP.md` — feature status + roadmap items (includes ASR abstraction references)

## How to use in Model-Lab

- Treat these as **inputs** for lab experiments (chunk size, VAD, boundary artifacts, concurrency patterns).
- When you extract something into an actual lab contract or harness implementation, add a short pointer back to the originating doc (filename + section header).

