#!/usr/bin/env python3
"""
Speech Quality Gate - Single-command regression test for STT/TTS
Implements audio integrity checks + ASR quality gates vs baselines.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile

# ----------------------------
# Text normalization + edit distance
# ----------------------------


def normalize_for_wer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str) -> list[str]:
    t = normalize_for_wer(text)
    return t.split() if t else []


def levenshtein(a: list[str], b: list[str]) -> int:
    # classic DP, O(len(a)*len(b)), good enough for small tests
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,  # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,  # substitution
            )
            prev = cur
    return dp[m]


def wer(ref: str, hyp: str) -> float:
    r = tokenize_words(ref)
    h = tokenize_words(hyp)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    dist = levenshtein(r, h)
    return dist / len(r)


def cer(ref: str, hyp: str) -> float:
    r = normalize_for_wer(ref).replace(" ", "")
    h = normalize_for_wer(hyp).replace(" ", "")
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    dist = levenshtein(list(r), list(h))
    return dist / len(r)


def glossary_hits(text: str, terms: list[str]) -> dict[str, bool]:
    n = normalize_for_wer(text)
    hits = {}
    for t in terms:
        # keep hyphenated terms: normalize term similarly, but allow either hyphen or space in hyp
        tn = normalize_for_wer(t)
        # e.g. "self-supervised" -> "self supervised" after normalize; treat as phrase match
        hits[t] = tn in n
    return hits


def number_hits(text: str, anchors: list[str]) -> dict[str, bool]:
    raw = text
    hits = {}
    for a in anchors:
        hits[a] = a in raw
    return hits


# ----------------------------
# ffmpeg helpers for loudness + normalization
# ----------------------------


def require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH")


def ffmpeg_loudness_json(wav_path: str) -> dict:
    """
    Uses ffmpeg loudnorm filter to measure LUFS and true peak.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        wav_path,
        "-af",
        "loudnorm=I=-20:TP=-1:LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    stderr = p.stderr

    # loudnorm JSON appears in stderr; extract last {...}
    m = re.findall(r"\{[\s\S]*?\}", stderr)
    if not m:
        raise RuntimeError(f"Could not parse loudnorm JSON from ffmpeg output for {wav_path}")
    j = json.loads(m[-1])
    return j


def ffmpeg_normalize_to_target(
    wav_path: str, out_path: str, target_lufs: float, tp_limit: float
) -> dict:
    """
    Two-pass loudnorm: pass 1 to measure, pass 2 to normalize.
    Returns measurement JSON.
    """
    meas = ffmpeg_loudness_json(wav_path)
    # Pass 2 uses measured values for deterministic normalization
    af = (
        f"loudnorm=I={target_lufs}:TP={tp_limit}:LRA=11:"
        f"measured_I={meas['input_i']}:measured_TP={meas['input_tp']}:"
        f"measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}:"
        f"offset={meas['target_offset']}:linear=true:print_format=summary"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        wav_path,
        "-af",
        af,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        out_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg normalization failed for {wav_path}: {p.stderr[:400]}")
    return meas


# ----------------------------
# Simple audio integrity signals
# ----------------------------


@dataclass
class IntegrityResult:
    sr: int
    duration_s: float
    speech_ratio: float
    snr_db: float
    input_lufs: float
    input_tp: float
    normalized_used: bool
    normalized_path: str


def read_wav_mono_float(wav_path: str) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(wav_path)
    if x.ndim > 1:
        x = x[:, 0]
    # int16 -> float32
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    return sr, x


def compute_wav_fingerprint(wav_path: str) -> dict:
    """Compute SHA256, duration, and sample rate for fixture stability."""
    with open(wav_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    sr, x = read_wav_mono_float(wav_path)
    duration_sec = len(x) / sr if sr > 0 else 0.0

    return {"sha256": sha256, "duration_sec": round(float(duration_sec), 2), "sample_rate": int(sr)}


def frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        return np.array([], dtype=np.float32)
    n = 1 + (len(x) - frame) // hop
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        w = x[s : s + frame]
        rms[i] = float(np.sqrt(np.mean(w * w) + 1e-12))
    return rms


def estimate_snr_db(x: np.ndarray, sr: int) -> float:
    # crude but stable: compare top 10% frame RMS vs bottom 10% frame RMS
    frame = int(0.03 * sr)
    hop = int(0.01 * sr)
    rms = frame_rms(x, frame, hop)
    if rms.size < 10:
        return 0.0
    lo = np.percentile(rms, 10)
    hi = np.percentile(rms, 90)
    snr = 20.0 * math.log10((hi + 1e-9) / (lo + 1e-9))
    return float(snr)


def estimate_speech_ratio(x: np.ndarray, sr: int) -> float:
    # energy-based VAD proxy: frames above percentile threshold
    frame = int(0.03 * sr)
    hop = int(0.01 * sr)
    rms = frame_rms(x, frame, hop)
    if rms.size == 0:
        return 0.0
    thr = np.percentile(rms, 60)  # moderate gate
    voiced = (rms >= thr).sum()
    return float(voiced / rms.size)


def integrity_gate(wav_path: str, policy: dict, reports_dir: str) -> IntegrityResult:
    require_ffmpeg()

    meas = ffmpeg_loudness_json(wav_path)
    input_lufs = float(meas["input_i"])
    input_tp = float(meas["input_tp"])

    sr, x = read_wav_mono_float(wav_path)
    duration_s = len(x) / sr if sr > 0 else 0.0
    snr_db = estimate_snr_db(x, sr)
    speech_ratio = estimate_speech_ratio(x, sr)

    lufs_lo, lufs_hi = policy["lufs_range_for_normalization"]
    target_lufs = policy["target_lufs"]
    tp_limit = policy["true_peak_limit_dbTP"]

    normalized_used = False
    normalized_path = wav_path

    if input_lufs < lufs_lo or input_lufs > lufs_hi:
        normalized_used = True
        base = os.path.basename(wav_path).replace(".wav", "")
        normalized_path = os.path.join(reports_dir, f"{base}.norm.wav")
        ffmpeg_normalize_to_target(wav_path, normalized_path, target_lufs, tp_limit)

    return IntegrityResult(
        sr=sr,
        duration_s=float(duration_s),
        speech_ratio=float(speech_ratio),
        snr_db=float(snr_db),
        input_lufs=float(input_lufs),
        input_tp=float(input_tp),
        normalized_used=normalized_used,
        normalized_path=normalized_path,
    )


# ----------------------------
# ASR (optional) via faster-whisper
# ----------------------------


def transcribe_faster_whisper(wav_path: str, asr_cfg: dict) -> str:
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(f"faster-whisper not available: {e}")

    model = WhisperModel(asr_cfg["model"], compute_type=asr_cfg.get("compute_type", "int8"))
    segments, info = model.transcribe(
        wav_path,
        language=asr_cfg.get("language", "en"),
        beam_size=int(asr_cfg.get("beam_size", 5)),
        vad_filter=bool(asr_cfg.get("vad_filter", True)),
    )
    parts = []
    for seg in segments:
        parts.append(seg.text)
    return " ".join(parts).strip()


# ----------------------------
# Gate evaluation
# ----------------------------


@dataclass
class GateOutcome:
    status: str  # PASS/WARN/FAIL
    reasons: list[str]


def merge_status(a: str, b: str) -> str:
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return a if order[a] >= order[b] else b


def main():
    ap = argparse.ArgumentParser(
        description="Speech quality gate with audio integrity and ASR regression checks"
    )
    ap.add_argument(
        "--init-baseline",
        action="store_true",
        help="Populate baselines.json expected metrics from current measured results (no gating).",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these fixture relative paths, e.g. fixtures/tts_voice1_clean.wav",
    )
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    baselines_path = os.path.join(root, "baselines.json")
    fixtures_dir = os.path.join(root, "fixtures")
    reports_dir = os.path.join(root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    with open(baselines_path, encoding="utf-8") as f:
        cfg = json.load(f)

    policy_ai = cfg["policy"]["audio_integrity"]
    policy_asr = cfg["policy"]["asr_quality"]

    glossary = cfg.get("glossary_terms", [])
    anchors = cfg.get("number_anchors", [])
    asr_cfg = cfg.get("asr", {})

    report = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": cfg.get("version", 1),
        "results": {},
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
    }

    run_dir = os.path.join(reports_dir, report["run_id"])
    os.makedirs(run_dir, exist_ok=True)

    overall = GateOutcome(status="PASS", reasons=[])

    for wav_rel, meta in cfg["fixtures"].items():
        if args.only is not None and len(args.only) > 0:
            if wav_rel not in set(args.only):
                continue

        wav_path = os.path.join(root, wav_rel)
        res_key = wav_rel

        if not os.path.exists(wav_path):
            status = "FAIL"
            reasons = [f"Missing fixture: {wav_rel}"]
            report["results"][res_key] = {"status": status, "reasons": reasons}
            overall.status = merge_status(overall.status, status)
            overall.reasons.extend(reasons)
            continue

        expected = meta["expected"]

        # Check for frozen reference text (per-fixture .ref.txt)
        ref_frozen_path = wav_path.replace(".wav", ".ref.txt")
        if not os.path.exists(ref_frozen_path):
            status = "FAIL"
            reasons = [f"Missing frozen reference: {os.path.basename(ref_frozen_path)}"]
            report["results"][res_key] = {"status": status, "reasons": reasons}
            overall.status = merge_status(overall.status, status)
            overall.reasons.extend(reasons)
            continue

        with open(ref_frozen_path, encoding="utf-8") as f:
            ref_text = f.read()

        # Check fixture fingerprint
        current_fp = compute_wav_fingerprint(wav_path)
        stored_fp = meta.get("fingerprint", {})

        if args.init_baseline:
            # Populate fingerprint during init
            meta["fingerprint"] = current_fp
        elif stored_fp.get("sha256") is not None:
            # Verify fingerprint if present
            if current_fp["sha256"] != stored_fp["sha256"]:
                status = "FAIL"
                reasons = [
                    f"Fixture changed: SHA256 mismatch (expected {stored_fp['sha256'][:8]}..., got {current_fp['sha256'][:8]}...)"
                ]
                report["results"][res_key] = {
                    "status": status,
                    "reasons": reasons,
                    "fingerprint": current_fp,
                }
                overall.status = merge_status(overall.status, status)
                overall.reasons.extend(reasons)
                continue

        # Integrity
        integ = integrity_gate(wav_path, policy_ai, reports_dir)

        status = "PASS"
        reasons = []

        # hard fail: clipping risk based on measured true peak
        if integ.input_tp > policy_ai["true_peak_fail_dbTP"]:
            status = "FAIL"
            reasons.append(
                f"True peak too high ({integ.input_tp:.2f} dBTP > {policy_ai['true_peak_fail_dbTP']})"
            )

        # warn: low SNR
        if integ.snr_db < policy_ai["snr_warn_db"]:
            status = merge_status(status, "WARN")
            reasons.append(f"Low SNR estimate ({integ.snr_db:.1f} dB < {policy_ai['snr_warn_db']})")

        # speech coverage expectations
        if meta["kind"] in ("single_speaker", "multi_speaker"):
            if integ.speech_ratio < policy_ai["speech_coverage_min_ratio"]:
                status = "FAIL"
                reasons.append(
                    f"Low speech coverage ({integ.speech_ratio:.2f} < {policy_ai['speech_coverage_min_ratio']})"
                )

        # ASR (optional, but default is required for this gate)
        hyp_text = ""
        asr_error = None
        try:
            hyp_text = transcribe_faster_whisper(integ.normalized_path, asr_cfg)
        except Exception as e:
            asr_error = str(e)

        if asr_error:
            status = "FAIL"
            reasons.append(f"ASR failed: {asr_error}")
            report["results"][res_key] = {
                "status": status,
                "reasons": reasons,
                "integrity": integ.__dict__,
                "asr": {"error": asr_error},
            }
            overall.status = merge_status(overall.status, status)
            overall.reasons.extend(reasons)
            continue

        w = wer(ref_text, hyp_text)
        c = cer(ref_text, hyp_text)

        w_base = float(expected["wer_norm_baseline"])
        c_base = float(expected["cer_norm_baseline"])
        wer_delta_pp = (w - w_base) * 100.0
        cer_delta_pp = (c - c_base) * 100.0

        # Glossary + numbers
        g_hits = glossary_hits(hyp_text, glossary)
        g_count = sum(1 for v in g_hits.values() if v)
        g_base = int(expected["glossary_hits_baseline"])
        g_missing = max(0, g_base - g_count)

        n_hits = number_hits(hyp_text, anchors)
        n_count = sum(1 for v in n_hits.values() if v)
        n_base = int(expected["numbers_hits_baseline"])
        n_missing = max(0, n_base - n_count)

        # Optional baseline init: overwrite expected values with measured values
        if args.init_baseline:
            meta["expected"]["wer_norm_baseline"] = round(float(w), 4)
            meta["expected"]["cer_norm_baseline"] = round(float(c), 4)
            meta["expected"]["glossary_hits_baseline"] = int(g_count)
            meta["expected"]["numbers_hits_baseline"] = int(n_count)
            status = "PASS"
            reasons = ["Baseline initialized from current measurement"]

        # WER delta gating
        if (not args.init_baseline) and wer_delta_pp > policy_asr["wer_warn_delta_pp"]:
            status = "FAIL"
            reasons.append(
                f"WER regression {wer_delta_pp:.2f} pp (> {policy_asr['wer_warn_delta_pp']})"
            )
        elif (not args.init_baseline) and wer_delta_pp > policy_asr["wer_pass_delta_pp"]:
            status = merge_status(status, "WARN")
            reasons.append(
                f"WER regression {wer_delta_pp:.2f} pp (> {policy_asr['wer_pass_delta_pp']})"
            )

        # Glossary gating
        if (not args.init_baseline) and g_missing >= policy_asr["glossary_fail_missing"]:
            status = "FAIL"
            reasons.append(
                f"Glossary missing {g_missing} terms (>= {policy_asr['glossary_fail_missing']})"
            )
        elif (not args.init_baseline) and g_missing >= policy_asr["glossary_warn_missing"]:
            status = merge_status(status, "WARN")
            reasons.append(
                f"Glossary missing {g_missing} terms (>= {policy_asr['glossary_warn_missing']})"
            )

        # Numbers gating
        if (not args.init_baseline) and n_missing >= policy_asr["numbers_fail_missing"]:
            status = "FAIL"
            reasons.append(
                f"Numbers missing {n_missing} anchors (>= {policy_asr['numbers_fail_missing']})"
            )
        elif (not args.init_baseline) and n_missing >= policy_asr["numbers_warn_missing"]:
            status = merge_status(status, "WARN")
            reasons.append(
                f"Numbers missing {n_missing} anchors (>= {policy_asr['numbers_warn_missing']})"
            )

        report["results"][res_key] = {
            "status": status,
            "reasons": reasons,
            "integrity": integ.__dict__,
            "metrics": {
                "wer_norm": w,
                "cer_norm": c,
                "wer_delta_pp": wer_delta_pp,
                "cer_delta_pp": cer_delta_pp,
                "glossary_hits": g_count,
                "glossary_baseline": g_base,
                "numbers_hits": n_count,
                "numbers_baseline": n_base,
            },
            "fingerprint": current_fp,
            "hypothesis_preview": hyp_text[:500],
        }

        # Save per-fixture transcript
        fixture_name = os.path.basename(wav_rel).replace(".wav", "")
        hyp_path = os.path.join(run_dir, f"{fixture_name}.hyp.txt")
        norm_hyp_path = os.path.join(run_dir, f"{fixture_name}.norm.hyp.txt")

        with open(hyp_path, "w", encoding="utf-8") as f:
            f.write(hyp_text)
        with open(norm_hyp_path, "w", encoding="utf-8") as f:
            f.write(normalize_for_wer(hyp_text))

        overall.status = merge_status(overall.status, status)
        overall.reasons.extend([f"{wav_rel}: {r}" for r in reasons])

    # Persist updated baselines if init mode was used
    if args.init_baseline:
        with open(baselines_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"\n✅ Baselines updated in {baselines_path}")

    out_path = os.path.join(reports_dir, "latest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Symlink to latest run directory
    latest_run_link = os.path.join(reports_dir, "latest")
    if os.path.islink(latest_run_link):
        os.unlink(latest_run_link)
    os.symlink(report["run_id"], latest_run_link)

    # Console summary (short)
    print(f"\nSPEECH GATE: {overall.status}")
    if overall.reasons:
        top = overall.reasons[:8]
        for r in top:
            print(f"- {r}")
        if len(overall.reasons) > len(top):
            print(f"- ... ({len(overall.reasons) - len(top)} more)")
    print(f"\nReport: {out_path}")

    if overall.status == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
