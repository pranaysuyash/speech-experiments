# Resource Limits

This system operates with shared hardware constraints. These limits ensure reliability for all builders in the public lab.

## 1. File Size
- **Limit**: 200 MB per file.
- **What happens if exceeded?**
    - The API will reject the upload immediately (`413 Payload Too Large`).
    - The run will not be created.
- **Why?**
    - Ingesting and processing large audio files blocks shared workers and consumes excessive RAM, potentially crashing the lab for everyone.

## 2. Concurrency
- **Limit**: 3 Concurrent Runs Global.
- **What happens if exceeded?**
    - The API returns `409 Conflict` (Runner Busy).
    - The request is rejected. You must wait and retry.
- **Why?**
    - We have limited GPU/CPU capacity. Queuing runs indefinitely causes "stalled" UX. Fast rejection is better than silent waiting.

## 3. Storage & Retention
- **Limit**: Disk Capacity (Volatile).
- **What happens if exceeded?**
    - Maintainers will run a cleanup script deleting the oldest runs.
    - Your run may disappear without warning.
- **Why?**
    - This is a research lab, not a storage locker. We prioritize new experiments over archival data.

## 4. Execution Time
- **Limit**: None (Technically).
- **What happens if a run hangs?**
    - It stays "Running" until a maintainer manually kills it or the server restarts.
- **Why?**
    - We are still characterizing failure modes. We prefer to debug hung processes rather than auto-kill them for now.
