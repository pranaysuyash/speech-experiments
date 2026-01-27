# LFM2.5-Audio TTS MPS Workaround

**Status**: ✅ RESOLVED  
**Date**: January 9, 2026  
**Component**: TTS / Audio Generation  
**Issue**: `AssertionError: Torch not compiled with CUDA enabled` on Apple Silicon (MPS)

## Problem Description

The `liquid-audio` library (v2.5.0) contains two critical hardcoded CUDA dependencies that prevent it from running on Apple Silicon (MPS):

1. **Processor Loading**: Defaults to `device='cuda'`. (Fixed in previous session)
2. **Audio Detokenizer**: The `LFM2AudioProcessor.audio_detokenizer` property explicitly calls `.cuda()` when initializing the detokenizer.

```python
# liquid_audio/processor.py (Simplified)

@property
def audio_detokenizer(self):
    if self._audio_detokenizer is None:
        # ... logic ...
        detok = LFM2AudioDetokenizer(detok_config).eval().cuda()  # <--- HARDCODED CUDA
        self._audio_detokenizer = detok
    return self._audio_detokenizer
```

This causes `processor.decode(tokens)` (which calls `self.audio_detokenizer`) to fail on MPS devices, even if the model and processor were loaded on MPS.

## The Fix: Dynamic Detokenizer Injection

We implemented a workaround in `harness/registry.py` that manually initializes the detokenizer on the correct device (MPS) and injects it into the processor instance _before_ the property is ever accessed.

### Implementation Details

In `load_lfm2_5_audio`:

1. Load `LFM2AudioProcessor` on CPU (standard fix).
2. Move processor to MPS.
3. **Check if `_audio_detokenizer` is None.**
4. If so, manually:
   - Import `Lfm2Config`, `LFM2AudioDetokenizer` from `liquid_audio.processor`.
   - Load the detokenizer config.
   - Fix layer type names (duplicate logic from library).
   - Instantiate `LFM2AudioDetokenizer`.
   - **Move to `actual_device` (MPS) instead of `.cuda()`.**
   - Load weights using `safetensors`.
   - Assign to `processor._audio_detokenizer`.

By pre-populating `_audio_detokenizer`, the library's property getter simply returns our patched instance, bypassing the faulty code.

### Code Snippet

```python
# harness/registry.py

# ... (processor loaded and moved to actual_device) ...

if processor._audio_detokenizer is None:
    try:
        from liquid_audio.processor import Lfm2Config, LFM2AudioDetokenizer
        from safetensors.torch import load_file

        # Load config logic ...

        # Initialize on CORRECT DEVICE
        detok = LFM2AudioDetokenizer(detok_config).eval().to(actual_device)

        # Load weights logic ...

        # Inject patched instance
        processor._audio_detokenizer = detok
        logger.info(f"✓ Audio detokenizer initialized on {actual_device}")

    except Exception as e:
        logger.warning(f"Failed to manually initialize detokenizer: {e}")
```

## Verification

The fix was verified by running `scripts/run_tts.py` on an Apple Silicon M3 (MPS).

- **Before**: `AssertionError: Torch not compiled with CUDA enabled`
- **After**: Successful audio generation.
  - Latency: ~10s for short prompts.
  - RTF: ~2.1x (Real-time factor).
  - Output: WAV files generated in `runs/lfm2_5_audio/tts/`.
