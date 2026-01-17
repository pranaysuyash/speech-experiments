# Run Session Sentinel Contract

## Format

Exactly one line, emitted at end of successful run:

```
RUN_SESSION_RESULT={"run_id":"...","run_dir":"...","console_url":"..."}
```

## Requirements

- **Prefix**: Exactly `RUN_SESSION_RESULT=`
- **JSON**: Compact (no whitespace, no newlines)
- **Fields**: `run_id`, `run_dir`, `console_url` (all strings)
- **Emission**: End of run, before success message
- **Flush**: `flush=True` to ensure immediate output

## Parsing

```python
import re
import json

SENTINEL_RE = re.compile(r"^RUN_SESSION_RESULT=(\{.*\})\s*$", re.MULTILINE)

def parse_sentinel(stdout: str) -> dict | None:
    match = SENTINEL_RE.search(stdout)
    return json.loads(match.group(1)) if match else None
```

## Implementation

```python
print("RUN_SESSION_RESULT=" + json.dumps(payload, separators=(",", ":"), ensure_ascii=False), flush=True)
```

## Never Change

This format is a permanent contract. Wrappers depend on it.
