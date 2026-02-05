#!/usr/bin/env python3
"""
Upgrade model-lab/data/model_catalog.csv to the expanded_plus schema.

Goals:
- Use the richer schema from the Downloads import:
  data/ports/audio_2026-02-05/audio_model_audit_expanded_plus.csv
- Preserve any locally-added catalog rows (by model_id) from the current catalog.
- Populate evidence_links using:
  - docs_url (always, when present)
  - chat-provided bibliography subset (data/from_chat/audio_model_audit_citations_deduped_2026-02-05.csv)

This script intentionally uses only the Python stdlib (no pandas).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CitationSet:
    gladia_asr_2024: str
    gladia_tts_2026: str
    assemblyai_deepspeech: str
    medium_wav2vec_compress: str
    resemble: str
    openvoice: str
    picovoice_noise: str
    techcrunch_audiocraft: str
    mbw_stability: str
    stability_stable_audio_open: str
    medium_panns_review: str
    pmc_deepfake_survey: str
    hf_deepfake_model: str


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        header = list(reader.fieldnames)
        rows: list[dict[str, str]] = []
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
        return header, rows


def _write_csv_rows(path: Path, header: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_url(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return ""


def _load_citations(path: Path) -> CitationSet:
    """
    Load the deduped citations CSV produced from the chat-provided bibliography.
    This is a small, curated list used to populate evidence_links heuristically.
    """
    _, rows = _read_csv_rows(path)
    urls = {r.get("url", "").strip() for r in rows if r.get("url")}

    def must(url: str) -> str:
        if url not in urls:
            raise ValueError(f"Expected citation URL missing from {path}: {url}")
        return url

    return CitationSet(
        gladia_asr_2024=must(
            "https://www.gladia.io/blog/a-review-of-the-best-asr-engines-and-the-models-powering-them-in-2024"
        ),
        gladia_tts_2026=must(
            "https://www.gladia.io/blog/best-tts-apis-for-developers-in-2026-top-7-text-to-speech-services"
        ),
        assemblyai_deepspeech=must(
            "https://www.assemblyai.com/blog/deepspeech-for-dummies-a-tutorial-and-overview-part-1"
        ),
        medium_wav2vec_compress=must(
            "https://medium.com/georgian-impact-blog/compressing-wav2vec-2-0-f41166e82dc2"
        ),
        resemble=must("https://www.resemble.ai/"),
        openvoice=must("https://github.com/myshell-ai/OpenVoice"),
        picovoice_noise=must("https://picovoice.ai/blog/top-noise-suppression-software-free-paid/"),
        techcrunch_audiocraft=must(
            "https://techcrunch.com/2023/08/02/meta-open-sources-models-for-generating-sounds-and-music/"
        ),
        mbw_stability=must(
            "https://www.musicbusinessworldwide.com/stability-ai-launches-ai-model-for-brands-to-create-custom-sounds/"
        ),
        stability_stable_audio_open=must("https://stability.ai/news/introducing-stable-audio-open"),
        medium_panns_review=must(
            "https://sh-tsang.medium.com/brief-review-panns-large-scale-pretrained-audio-neural-networks-for-audio-pattern-recognition-fad0e7c0c117"
        ),
        pmc_deepfake_survey=must("https://pmc.ncbi.nlm.nih.gov/articles/PMC11991371/"),
        hf_deepfake_model=must("https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2"),
    )


def _add_evidence_links(row: dict[str, str], citations: CitationSet) -> dict[str, str]:
    """
    Heuristic wiring:
    - Always include docs_url if present.
    - Add bibliography links when row appears related.
    """
    model_id = (row.get("model_id") or "").lower()
    family = (row.get("family") or "").lower()
    category = (row.get("category") or "").lower()
    tasks = (row.get("tasks_supported") or "").lower()
    docs_url = _safe_url(row.get("docs_url", ""))

    evidence: list[str] = []
    if docs_url:
        evidence.append(docs_url)

    # ASR / general reviews
    if category == "asr" or "automatic speech recognition" in tasks or "asr" in tasks:
        evidence.append(citations.gladia_asr_2024)

    # DeepSpeech / Coqui STT
    if "deepspeech" in model_id or "deepspeech" in family:
        evidence.append(citations.assemblyai_deepspeech)

    # wav2vec2 family
    if "wav2vec" in model_id or "wav2vec" in family:
        evidence.append(citations.medium_wav2vec_compress)
        evidence.append(citations.gladia_asr_2024)

    # TTS vendor roundups
    if category == "tts" or "text-to-speech" in tasks or "tts" in tasks:
        evidence.append(citations.gladia_tts_2026)

    # Resemble
    if "resemble" in model_id or "resemble" in family:
        evidence.append(citations.resemble)

    # OpenVoice
    if "openvoice" in model_id or "openvoice" in family:
        evidence.append(citations.openvoice)

    # Noise suppression / enhancement
    if "noise" in category or "suppression" in category or "denoise" in tasks:
        evidence.append(citations.picovoice_noise)

    # AudioCraft / MusicGen / AudioGen
    if "audiocraft" in model_id or "musicgen" in model_id or "audiogen" in model_id:
        evidence.append(citations.techcrunch_audiocraft)

    # Stable Audio
    if "stable-audio" in model_id or "stable audio" in family:
        evidence.append(citations.stability_stable_audio_open)
        evidence.append(citations.mbw_stability)

    # PANNs
    if "panns" in model_id or "panns" in family:
        evidence.append(citations.medium_panns_review)

    # Deepfake detection / forensics
    if "deepfake" in model_id or "deepfake" in family or "spoof" in tasks or "deepfake" in tasks:
        evidence.append(citations.pmc_deepfake_survey)
        evidence.append(citations.hf_deepfake_model)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for url in evidence:
        url = url.strip()
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(url)

    row["evidence_links"] = " | ".join(deduped)
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "ports"
            / "audio_2026-02-05"
            / "audio_model_audit_expanded_plus.csv"
        ),
        help="Path to expanded_plus CSV (default: imported Downloads copy)",
    )
    p.add_argument(
        "--current",
        default=str(Path(__file__).resolve().parents[1] / "data" / "model_catalog.csv"),
        help="Path to current model_catalog.csv",
    )
    p.add_argument(
        "--citations",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "from_chat"
            / "audio_model_audit_citations_deduped_2026-02-05.csv"
        ),
        help="Path to deduped citations CSV",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output path (default: overwrite --current)",
    )
    args = p.parse_args()

    base_path = Path(args.base).expanduser().resolve()
    current_path = Path(args.current).expanduser().resolve()
    citations_path = Path(args.citations).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else current_path

    if not base_path.exists():
        raise SystemExit(f"Base expanded_plus CSV not found: {base_path}")
    if not current_path.exists():
        raise SystemExit(f"Current model_catalog not found: {current_path}")
    if not citations_path.exists():
        raise SystemExit(f"Citations CSV not found: {citations_path}")

    base_header, base_rows = _read_csv_rows(base_path)
    cur_header, cur_rows = _read_csv_rows(current_path)

    base_by_id = {r.get("model_id", ""): r for r in base_rows if r.get("model_id")}
    cur_by_id = {r.get("model_id", ""): r for r in cur_rows if r.get("model_id")}

    # Start with base rows (expanded_plus).
    merged: dict[str, dict[str, str]] = {k: dict(v) for k, v in base_by_id.items()}

    # Preserve any current rows that are not in base (schema expand with blanks).
    for model_id, r in cur_by_id.items():
        if model_id in merged:
            continue
        expanded = {k: "" for k in base_header}
        for k in base_header:
            if k in r:
                expanded[k] = r.get(k, "")
        # Ensure required defaults.
        expanded["model_id"] = model_id
        expanded.setdefault("last_reviewed_utc", _now_utc_iso())
        merged[model_id] = expanded

    citations = _load_citations(citations_path)
    now = _now_utc_iso()
    for r in merged.values():
        # Update review timestamp for deterministic refresh.
        r["last_reviewed_utc"] = now
        _add_evidence_links(r, citations)

    # Stable ordering: keep base order, then append extras sorted by model_id.
    ordered_ids: list[str] = []
    for r in base_rows:
        mid = r.get("model_id", "")
        if mid and mid in merged:
            ordered_ids.append(mid)
    extras = sorted([mid for mid in merged.keys() if mid not in ordered_ids])
    ordered_ids.extend(extras)

    out_rows = [merged[mid] for mid in ordered_ids]
    _write_csv_rows(out_path, base_header, out_rows)
    print(f"Wrote upgraded catalog: {out_path} (rows={len(out_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

