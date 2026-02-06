from __future__ import annotations

from pathlib import Path

import pytest

from harness.media_ingest import IngestConfig, ingest_media


@pytest.mark.parametrize("normalize_a,normalize_b", [(False, True)])
def test_ingest_hash_changes_when_config_changes(
    tmp_path: Path, test_wav_path: Path, normalize_a: bool, normalize_b: bool
) -> None:
    # Use different output dirs to prevent file collision, though artifacts are namespaced
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    a = ingest_media(test_wav_path, dir_a, IngestConfig(normalize=normalize_a, trim_silence=False))
    b = ingest_media(test_wav_path, dir_b, IngestConfig(normalize=normalize_b, trim_silence=False))

    # Fingerprint should differ because config differs
    assert a["audio_fingerprint"] != b["audio_fingerprint"]

    # Preprocess hash should definitely differ
    assert a["preprocess_hash"] != b["preprocess_hash"]

    # Content hash MIGHT be same if normalization didn't change bits (e.g. silence), or different
    # But fingerprint includes config hash, so it guarantees difference.
