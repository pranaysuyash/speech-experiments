# Hardware Recommendations & Performance Guide

**Provenance**: Ported from `EchoPanel/docs/HARDWARE_AND_PERFORMANCE.md` on 2026-02-05. Useful for local/macOS constraints and model-size guidance.

EchoPanel v0.2 uses on-device AI for privacy and low latency. Performance depends on your Mac's hardware.

## Recommended Specifications

| Feature | Minimum | Recommended | Ideal |
|---------|---------|-------------|-------|
| **Device** | MacBook Air (M1) | MacBook Pro (M1 Pro/M2/M3) | MacBook Pro (M3 Max) |
| **RAM** | 8 GB | 16 GB+ | 32 GB+ |
| **Storage** | 10 GB free | 20 GB free | 50 GB free |
| **OS** | macOS 13 Ventura | macOS 14 Sonoma | macOS 14 Sonoma |

---

## Model Selection Guide

You can choose the Whisper model size in **Settings**.

### 1. Base / Small (Fastest)
- **Best for:** Older Macs, M1 Air, or when battery life is priority.
- **RAM Usage:** ~1 GB
- **Accuracy:** Good for clear English speech. Lower for accents or background noise.
- **Latency:** Very low (< 1s).

### 2. Medium (Balanced)
- **Best for:** M1 Pro, M2/M3 Air.
- **RAM Usage:** ~2-3 GB
- **Accuracy:** High accuracy, handles accents well.
- **Latency:** Low (~1-2s).
- **Default setting.**

### 3. Large-v3-Turbo (Best Accuracy)
- **Best for:** M2/M3 Pro/Max with 32GB+ RAM.
- **RAM Usage:** ~4-5 GB + PyTorch overhead.
- **Accuracy:** State-of-the-art multilingual transcription.
- **Latency:** Moderate (~2-3s).

---

## Performance Optimizations

### Diarization (Speaker Labels)
- Runs at the end of the session.
- **Duration:** Can take 30-60s on M1 Air for a 1-hour meeting.
- **Optimization:** Uses Apple Neural Engine (ANE) / MPS where available.

### Battery Life
- Processing audio consumes power. For long meetings on battery:
  - Use **Base** model.
  - Close other heavy apps.
  - Expect ~15-20% battery usage per hour on M1 Air.
