# WebSocket Contract: Live Listener v0.2

**Provenance**: Ported from `EchoPanel/docs/WS_CONTRACT.md` on 2026-02-05. Content below is carried over for reference; validate against current server behavior when used as a lab contract.

This document is the source of truth for the client/server WebSocket protocol between the macOS app and the backend for v0.1.

## Endpoint
- WebSocket path: `/ws/live-listener`
- Transport: WebSocket over TLS in production (`wss://`)

## Message types (overview)
- Client to server:
  - JSON control messages: `start`, `stop`
  - Binary audio messages: PCM frames, `pcm_s16le`, mono, 16 kHz
- Server to client:
  - ASR events: `asr_partial`, `asr_final`
  - Analysis events: `cards_update`, `entities_update`
  - Status events: `status`
  - Finalization: `final_summary`

## Binary framing (client to server)
- Each binary message is a single audio frame.
- Encoding: PCM signed 16-bit little-endian (`pcm_s16le`)
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Source tagging: Clients should ideally wrap audio in the JSON structure below, but raw binary is accepted as "system" source for backward compatibility.
- v0.2 Preferred Framing (JSON):
  ```json
  {
    "source": "system" | "mic",
    "pcm_base64": "..."
  }
  ```
  *(Note: The spec actually defines `audio_frame` as a JSON structure in the text below, but v0.1 used raw binary. v0.2 will support both or move to JSON for multi-source. For now, we document the raw frame behavior and note the v0.2 extension)*.

Actual v0.2 Implementation Plan defines `audio_frame` as JSON. Let's align with the Plan:
**Client to Server**:
- **audio_frame**:
  - `session_id`: string (UUID)
  - `source`: "system" | "mic" (default: "system")
  - `pcm16_base64`: string (Base64 encoded PCM16 data)
  - `sample_rate`: number (default: 16000)
  - `channels`: number (default: 1)
- Timing:
  - The backend should treat frame arrival order as the clock for streaming ASR.
  - The client should send frames at approximately real time cadence.

## JSON control messages (client to server)
All JSON messages are UTF-8 text frames.

### `start`
Schema:
```json
{
  "type": "start",
  "session_id": "uuid",
  "sample_rate": 16000,
  "format": "pcm_s16le",
  "channels": 1
}
```

### `audio` (Client to Server)
Schema:
```json
{
  "type": "audio",
  "source": "system" | "mic",
  "data": "base64_encoded_pcm16_samples"
}
```
*Note: Binary messages are still supported for backward compatibility (defaults to "system"), but JSON is preferred for multi-source.*

### `stop`
Schema:
```json
{
  "type": "stop",
  "session_id": "uuid"
}
```

## Server events (server to client)

### ASR: `asr_partial` / `asr_final`
Schema:
```json
{
  "type": "asr_partial" | "asr_final",
  "t0": 123.4,
  "t1": 126.2,
  "text": "...",
  "confidence": 0.95,
  "source": "mic" | "system",
  "speaker": "Speaker 1",
  "stable": false
}
```

### Analysis: `entities_update`
Schema:
```json
{
  "type": "entities_update",
  "people": [{"name":"...","type":"person","count":5,"last_seen":123.4,"confidence":0.8}],
  "orgs": [{"name":"...","type":"org","count":1,"last_seen":123.4,"confidence":0.8}],
  ...
}
```

### Status: `status`
Schema:
```json
{
  "type": "status",
  "state": "streaming|reconnecting|error",
  "message": "..."
}
```

Examples:
```json
{"type":"status","state":"streaming","message":"Connected"}
```

```json
{"type":"status","state":"reconnecting","message":"Retrying in 2s"}
```

### Finalization: `final_summary`
Schema:
```json
{
  "type": "final_summary",
  "markdown": "...",
  "json": {}
}
```

Example:
```json
{"type":"final_summary","markdown":"# Summary\\n...","json":{"session_id":"2E3B2BC2-0F6D-46E0-8B7A-5D80A8B8BE68","actions":[],"decisions":[],"risks":[],"entities":{}}}
```

## Error handling expectations
- If the server cannot process audio, it sends a `status` with `state:"error"` and a human-readable `message`.
- If the client disconnects unexpectedly, the server may terminate analysis for that session unless it supports resumption (not required in v0.1).
