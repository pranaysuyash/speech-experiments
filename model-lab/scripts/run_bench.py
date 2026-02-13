#!/usr/bin/env python3
"""
Benchmark entry point with MPS threading fix.

CRITICAL: This script must set multiprocessing start method BEFORE
importing torch to avoid "mutex lock failed: Invalid argument" on macOS.
"""
import multiprocessing
import sys
from pathlib import Path

# MUST be set before ANY torch import
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now safe to import
import argparse
import json

from bench.runner import (
    run_batch_asr_bench,
    run_batch_asr_sweep,
    run_classify_bench,
    run_embed_bench,
    run_enhance_bench,
    run_separate_bench,
    run_tts_bench,
    save_result,
    format_result_table,
)


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks with MPS support")
    parser.add_argument("surface", choices=["asr", "asr-sweep", "enhance", "classify", "embed", "separate", "tts"])
    parser.add_argument("--model", "-m", required=False, help="Model ID")
    parser.add_argument("--audio", "-a", required=False, help="Audio file path")
    parser.add_argument("--text", "-t", required=False, help="Text for TTS")
    parser.add_argument("--reference", "-r", required=False, help="Reference text for WER")
    parser.add_argument("--device", "-d", default="mps", help="Device (cpu/mps/cuda)")
    parser.add_argument("--clean", "-c", required=False, help="Clean audio for SI-SNR (enhance)")
    parser.add_argument("--output", "-o", action="store_true", help="Save to results/")
    parser.add_argument("--chunk-len", type=int, default=None, help="Chunk length in seconds (ASR)")
    parser.add_argument("--stride-len", type=int, nargs=2, default=None, help="Stride length (left, right) in seconds (ASR)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (ASR)")
    
    args = parser.parse_args()
    
    if args.surface == "asr":
        if not args.model or not args.audio:
            parser.error("asr requires --model and --audio")
        result = run_batch_asr_bench(
            args.model, 
            args.audio, 
            reference=args.reference, 
            device=args.device,
            chunk_length_s=args.chunk_len,
            stride_length_s=tuple(args.stride_len) if args.stride_len else None,
            batch_size=args.batch_size,
        )
        
    elif args.surface == "asr-sweep":
        if not args.audio:
            parser.error("asr-sweep requires --audio")
        results = run_batch_asr_sweep(args.audio, reference=args.reference, device=args.device)
        if args.output:
            for r in results:
                save_result(r)
        print(format_result_table(results))
        return
        
    elif args.surface == "enhance":
        if not args.model or not args.audio:
            parser.error("enhance requires --model and --audio")
        result = run_enhance_bench(args.model, args.audio, clean_path=args.clean, device=args.device)
        
    elif args.surface == "classify":
        if not args.model or not args.audio:
            parser.error("classify requires --model and --audio")
        result = run_classify_bench(args.model, args.audio, device=args.device)
        
    elif args.surface == "embed":
        if not args.model or not args.audio:
            parser.error("embed requires --model and --audio")
        result = run_embed_bench(args.model, args.audio, device=args.device)
        
    elif args.surface == "separate":
        if not args.model or not args.audio:
            parser.error("separate requires --model and --audio")
        result = run_separate_bench(args.model, args.audio, device=args.device)
        
    elif args.surface == "tts":
        if not args.model or not args.text:
            parser.error("tts requires --model and --text")
        result = run_tts_bench(args.model, args.text, device=args.device)
    
    if args.output:
        save_result(result)
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
