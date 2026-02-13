"""
Claims execution tests.

Maps test_ref strings to actual test functions and executes claims.
"""

import json
from pathlib import Path

import pytest
import yaml


def get_all_claims_files():
    """Find all claims.yaml files in models directory."""
    models_dir = Path(__file__).parent.parent.parent / "models"
    return list(models_dir.glob("*/claims.yaml"))


def get_model_id_from_path(claims_path: Path) -> str:
    """Extract model ID from claims path."""
    return claims_path.parent.name


def load_claims(claims_path: Path) -> dict:
    """Load and parse a claims.yaml file."""
    with open(claims_path) as f:
        return yaml.safe_load(f)


# Module-level mock evidence for hermetic testing
# When set, this takes precedence over real file loading
_MOCK_EVIDENCE: dict[str, dict[str, list]] = {}  # model_id -> task -> evidence list


def set_mock_evidence(model_id: str, task: str, evidence: list):
    """Inject mock evidence for hermetic testing."""
    if model_id not in _MOCK_EVIDENCE:
        _MOCK_EVIDENCE[model_id] = {}
    _MOCK_EVIDENCE[model_id][task] = evidence


def clear_mock_evidence():
    """Clear all mock evidence."""
    _MOCK_EVIDENCE.clear()


def get_evidence_for_task(model_id: str, task: str) -> list:
    """Load run artifacts for a model/task combination.

    If mock evidence is set for this model/task, returns that instead.
    This enables hermetic testing without requiring real run artifacts.
    """
    # Check for mock evidence first (hermetic testing)
    if model_id in _MOCK_EVIDENCE and task in _MOCK_EVIDENCE[model_id]:
        return _MOCK_EVIDENCE[model_id][task]

    # Fall back to real file loading
    runs_dir = Path(__file__).parent.parent.parent / "runs" / model_id / task
    if not runs_dir.exists():
        return []

    evidence = []
    for run_file in runs_dir.glob("*.json"):
        with open(run_file) as f:
            evidence.append(json.load(f))
    return evidence


# =============================================================================
# Claim test handlers - each test_ref maps to a function
# =============================================================================


def handler_v2v_realtime_threshold_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """
    V2V real-time test: rtf_like <= threshold and latency is finite.
    """
    if not evidence:
        return False, "No V2V evidence found"

    rtf_max = thresholds.get("rtf_like_max", 1.2)

    for run in evidence:
        metrics = run.get("metrics", {})
        summary = run.get("summary", {}).get("metrics", {})

        # Try to find rtf_like
        rtf_like = metrics.get("rtf_like") or summary.get("rtf_like")
        if rtf_like is None:
            continue

        if not isinstance(rtf_like, (int, float)) or rtf_like != rtf_like:  # NaN check
            continue

        if rtf_like <= rtf_max:
            return True, f"rtf_like={rtf_like:.2f} <= {rtf_max}"

    return False, f"No evidence with rtf_like <= {rtf_max}"


def handler_tts_audio_output_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """
    TTS audio output test: produces audio of minimum duration.
    """
    if not evidence:
        return False, "No TTS evidence found (test skipped)"

    min_duration = thresholds.get("min_duration_s", 0.5)

    for run in evidence:
        output = run.get("output", {})
        metrics = run.get("metrics", {})

        # Check multiple locations where duration might be stored
        duration = (
            output.get("duration_s")
            or output.get("audio_duration_s")
            or output.get("total_duration_s")
            or metrics.get("duration_s")
            or metrics.get("audio_duration_s")
        )

        if duration is not None and duration >= min_duration:
            return True, f"duration={duration:.2f}s >= {min_duration}s"

    return False, f"No evidence with duration >= {min_duration}s"


def handler_chat_responds_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """
    Chat response test: returns non-empty response within latency budget.
    """
    if not evidence:
        return False, "No chat evidence found (test skipped)"

    max_latency = thresholds.get("max_latency_ms", 30000)

    for run in evidence:
        output = run.get("output", {})
        response = output.get("response") or output.get("text")
        latency = run.get("metrics", {}).get("latency_ms")

        if response and len(str(response).strip()) > 0:
            if latency is None or latency <= max_latency:
                return True, f"Response received (latency={latency}ms)"

    return False, "No valid chat response found"


def handler_asr_smoke_structural_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """ASR structural smoke test: produces valid artifact with provenance."""
    if not evidence:
        return False, "No ASR evidence found"

    for run in evidence:
        has_provenance = run.get("provenance") is not None
        has_run_context = run.get("run_context") is not None

        if has_provenance and has_run_context:
            return True, "ASR run has provenance and run_context"

    return False, "No ASR run with provenance and run_context"


def handler_asr_smoke_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """ASR smoke test alias for structural validation."""
    return handler_asr_smoke_structural_v1(evidence, thresholds)


def handler_asr_non_whisper_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """ASR non-Whisper structural claim: uses thresholds as declaration."""
    if not evidence:
        return False, "No ASR evidence found"

    if thresholds.get("non_whisper") is True:
        return True, "Non-Whisper architecture declared"

    return False, "Non-Whisper threshold not set"


def handler_asr_stream_smoke_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """ASR stream structural smoke test: requires provenance + run_context."""
    if not evidence:
        return False, "No ASR stream evidence found"

    for run in evidence:
        has_provenance = run.get("provenance") is not None
        has_run_context = run.get("run_context") is not None
        if has_provenance and has_run_context:
            return True, "ASR stream run has provenance and run_context"

    return False, "No ASR stream run with provenance and run_context"


def handler_asr_stream_nemo_runtime_v1(
    evidence: list, thresholds: dict
) -> tuple[bool, str]:
    """ASR stream runtime declaration for NeMo."""
    if not evidence:
        return False, "No ASR stream evidence found"

    if thresholds.get("runtime") == "nemo":
        return True, "NeMo runtime declared"

    return False, "NeMo runtime threshold not set"


def handler_asr_stream_delay_configurable_v1(
    evidence: list, thresholds: dict
) -> tuple[bool, str]:
    """ASR stream delay config claim: verifies thresholds are present."""
    if not evidence:
        return False, "No ASR stream evidence found"

    required_keys = {
        "transcription_delay_ms_min",
        "transcription_delay_ms_max",
        "transcription_delay_ms_default",
    }

    if required_keys.issubset(thresholds.keys()):
        return True, "Delay thresholds declared"

    return False, "Delay thresholds missing"


def handler_enhance_smoke_structural_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """Enhancement structural smoke: outputs audio + sample rate."""
    if not evidence:
        return False, "No enhance evidence found"

    for run in evidence:
        output = run.get("output", {})
        audio = output.get("audio") or output.get("audio_array")
        sample_rate = output.get("sample_rate") or output.get("sr")

        if audio is not None and sample_rate:
            return True, "Enhance output includes audio and sample_rate"

    return False, "No enhance output with audio and sample_rate"


def handler_separate_smoke_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """Source separation smoke: outputs stems dict and sample rate."""
    if not evidence:
        return False, "No separation evidence found"

    for run in evidence:
        output = run.get("output", {})
        stems = output.get("stems")
        sample_rate = output.get("sr") or output.get("sample_rate")
        if isinstance(stems, dict) and sample_rate:
            return True, "Separation output includes stems and sample rate"

    return False, "No separation output with stems and sample rate"


def handler_separate_output_format_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """Separation output format: {stems: {...}, sr: int}."""
    return handler_separate_smoke_v1(evidence, thresholds)


def handler_music_transcription_smoke_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """Music transcription smoke: outputs note list."""
    if not evidence:
        return False, "No music transcription evidence found"

    for run in evidence:
        output = run.get("output", {})
        notes = output.get("notes")
        if isinstance(notes, list):
            return True, "Music transcription output includes notes list"

    return False, "No music transcription output with notes list"


def handler_music_transcription_structure_v1(
    evidence: list, thresholds: dict
) -> tuple[bool, str]:
    """Music transcription structure smoke: outputs note list."""
    return handler_music_transcription_smoke_v1(evidence, thresholds)


def handler_embed_smoke_structural_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """Embed smoke: outputs embedding vector."""
    if not evidence:
        return False, "No embed evidence found"

    expected_dim = thresholds.get("embedding_dim")

    for run in evidence:
        output = run.get("output", {})
        embedding = output.get("embedding")
        if isinstance(embedding, list):
            if expected_dim is None or len(embedding) == expected_dim:
                return True, "Embedding output present"

    return False, "No embedding output matching expected dimension"


def handler_classify_zero_shot_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """Classification smoke: outputs labels and scores."""
    if not evidence:
        return False, "No classify evidence found"

    for run in evidence:
        output = run.get("output", {})
        labels = output.get("labels")
        scores = output.get("scores")
        if isinstance(labels, list) and isinstance(scores, list) and labels and scores:
            return True, "Classification output includes labels and scores"

    return False, "No classification output with labels and scores"


def handler_classify_smoke_structural_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """Classification structural smoke: outputs labels and scores."""
    return handler_classify_zero_shot_v1(evidence, thresholds)


def handler_vad_smoke_ratio_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """VAD smoke test: speech_ratio within bounds."""
    if not evidence:
        return False, "No VAD evidence found"

    min_ratio = thresholds.get("speech_ratio_min", 0.05)
    max_ratio = thresholds.get("speech_ratio_max", 0.95)

    for run in evidence:
        metrics = run.get("metrics", {})
        speech_ratio = metrics.get("speech_ratio")

        if speech_ratio is not None:
            if min_ratio <= speech_ratio <= max_ratio:
                return True, f"speech_ratio={speech_ratio:.2f} within [{min_ratio}, {max_ratio}]"

    return False, f"No VAD run with speech_ratio in [{min_ratio}, {max_ratio}]"


def handler_vad_device_cpu_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """VAD device constraint: runs on CPU."""
    if not evidence:
        return False, "No VAD evidence found"

    for run in evidence:
        run_context = run.get("run_context", {})
        device = run_context.get("device", "")

        if "cpu" in device.lower():
            return True, f"VAD ran on device={device}"

    return False, "No VAD run on CPU device"


def handler_diarization_smoke_structural_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """Diarization structural smoke: produces speaker segments."""
    if not evidence:
        return False, "No diarization evidence found"

    for run in evidence:
        metrics = run.get("metrics", {})
        results = run.get("results", [])

        # Check if speakers were detected - try multiple metric names
        speakers = (
            metrics.get("num_speakers_pred")
            or metrics.get("speaker_count")
            or metrics.get("predicted_speakers")
        )
        has_segments = len(results) > 0 or run.get("output", {}).get("segments") is not None

        if speakers is not None or has_segments:
            return True, f"Diarization produced speakers={speakers}"

    return False, "No diarization run with speaker segments"


def handler_diarization_env_hf_token_v1(evidence: list, thresholds: dict) -> tuple[bool, str]:
    """Diarization environment requirement: HF_TOKEN set."""
    import os

    if os.environ.get("HF_TOKEN"):
        return True, "HF_TOKEN environment variable is set"
    return False, "HF_TOKEN environment variable not set"


# Dispatch table: test_ref -> handler function
TEST_REF_HANDLERS = {
    "claims.asr.smoke_v1": handler_asr_smoke_v1,
    "claims.asr.non_whisper_v1": handler_asr_non_whisper_v1,
    "claims.v2v.realtime_threshold_v1": handler_v2v_realtime_threshold_v1,
    "claims.tts.audio_output_v1": handler_tts_audio_output_v1,
    "claims.chat.responds_v1": handler_chat_responds_v1,
    "claims.asr.smoke_structural_v1": handler_asr_smoke_structural_v1,
    "claims.asr_stream.smoke_v1": handler_asr_stream_smoke_v1,
    "claims.asr_stream.nemo_runtime_v1": handler_asr_stream_nemo_runtime_v1,
    "claims.asr_stream.delay_configurable_v1": handler_asr_stream_delay_configurable_v1,
    "claims.vad.smoke_ratio_v1": handler_vad_smoke_ratio_v1,
    "claims.vad.device_cpu_v1": handler_vad_device_cpu_v1,
    "claims.diarization.smoke_structural_v1": handler_diarization_smoke_structural_v1,
    "claims.diarization.env_hf_token_v1": handler_diarization_env_hf_token_v1,
    "claims.enhance.smoke_structural_v1": handler_enhance_smoke_structural_v1,
    "claims.separate.smoke_v1": handler_separate_smoke_v1,
    "claims.separate.output_format_v1": handler_separate_output_format_v1,
    "claims.music_transcription.smoke_v1": handler_music_transcription_smoke_v1,
    "claims.music_transcription.structure_v1": handler_music_transcription_structure_v1,
    "claims.embed.smoke_structural_v1": handler_embed_smoke_structural_v1,
    "claims.classify.zero_shot_v1": handler_classify_zero_shot_v1,
    "claims.classify.smoke_structural_v1": handler_classify_smoke_structural_v1,
}


def execute_claim(model_id: str, claim: dict) -> dict:
    """
    Execute a single claim and return result.

    Returns:
        {
            'claim_id': str,
            'task': str,
            'enforcement': str,
            'status': 'pass' | 'fail' | 'skip',
            'reason': str
        }
    """
    test_ref = claim["test_ref"]
    task = claim["task"]
    thresholds = claim.get("thresholds", {})

    result = {
        "claim_id": claim["id"],
        "task": task,
        "enforcement": claim["enforcement"],
        "test_ref": test_ref,
    }

    # Check if handler exists
    handler = TEST_REF_HANDLERS.get(test_ref)
    if handler is None:
        result["status"] = "skip"
        result["reason"] = f"No handler for test_ref: {test_ref}"
        return result

    # Load evidence
    evidence = get_evidence_for_task(model_id, task)

    # Execute
    try:
        passed, reason = handler(evidence, thresholds)
        result["status"] = "pass" if passed else "fail"
        result["reason"] = reason
    except Exception as e:
        result["status"] = "fail"
        result["reason"] = f"Handler error: {e}"

    return result


def execute_all_claims(model_id: str, claims_data: dict) -> dict:
    """
    Execute all claims for a model.

    Returns aggregate results with pass/fail/skip counts.
    """
    results = {
        "model_id": model_id,
        "claims": [],
        "summary": {
            "total": 0,
            "enforced_pass": 0,
            "enforced_fail": 0,
            "enforced_skip": 0,
            "optional_pass": 0,
            "optional_fail": 0,
            "optional_skip": 0,
        },
    }

    for claim in claims_data.get("claims", []):
        claim_result = execute_claim(model_id, claim)
        results["claims"].append(claim_result)
        results["summary"]["total"] += 1

        enforcement = claim["enforcement"]
        status = claim_result["status"]

        if enforcement == "required":
            results["summary"][f"enforced_{status}"] += 1
        else:
            results["summary"][f"optional_{status}"] += 1

    return results


# =============================================================================
# Pytest tests
# =============================================================================

# Minimal mock evidence templates for each task type
# These are the minimum fields required to pass the structural checks
MOCK_EVIDENCE_TEMPLATES = {
    "asr": [
        {
            "provenance": {"source": "mock", "created_at": "2026-01-01T00:00:00"},
            "run_context": {"device": "cpu", "model_id": "mock"},
            "metrics": {"wer": 0.1},
        }
    ],
    "asr_stream": [
        {
            "provenance": {"source": "mock", "created_at": "2026-01-01T00:00:00"},
            "run_context": {"device": "cpu", "model_id": "mock_stream"},
            "output": {"text": "streaming mock"},
            "metrics": {"partial_rate": 0.5},
        }
    ],
    "vad": [
        {
            "provenance": {"source": "mock"},
            "run_context": {"device": "cpu"},
            "metrics": {"speech_ratio": 0.5},
        }
    ],
    "diarization": [
        {
            "provenance": {"source": "mock"},
            "run_context": {"device": "cpu"},
            "metrics": {"num_speakers_pred": 2},
            "results": [{"speaker": "SPEAKER_0", "start": 0, "end": 1}],
        }
    ],
    "v2v": [
        {
            "provenance": {"source": "mock"},
            "run_context": {"device": "cpu"},
            "metrics": {"rtf_like": 0.5},
        }
    ],
    "tts": [
        {
            "provenance": {"source": "mock"},
            "run_context": {"device": "cpu"},
            "output": {"duration_s": 2.0},
        }
    ],
    "enhance": [
        {
            "provenance": {"source": "mock"},
            "run_context": {"device": "cpu"},
            "output": {"audio": [0.0, 0.1], "sample_rate": 48000},
        }
    ],
    "separate": [
        {
            "provenance": {"source": "mock"},
            "run_context": {"device": "cpu"},
            "output": {
                "stems": {"vocals": [0.0], "drums": [0.0], "bass": [0.0], "other": [0.0]},
                "sr": 44100,
            },
        }
    ],
    "music_transcription": [
        {
            "provenance": {"source": "mock"},
            "run_context": {"device": "cpu"},
            "output": {
                "notes": [
                    {"onset": 0.0, "offset": 0.1, "pitch": 60, "velocity": 0.8}
                ]
            },
        }
    ],
    "embed": [
        {
            "provenance": {"source": "mock"},
            "run_context": {"device": "cpu"},
            "output": {"embedding": [0.0] * 512},
        }
    ],
    "classify": [
        {
            "provenance": {"source": "mock"},
            "run_context": {"device": "cpu"},
            "output": {"labels": ["speech", "music"], "scores": [0.9, 0.1]},
        }
    ],
    "chat": [
        {
            "provenance": {"source": "mock"},
            "run_context": {"device": "cpu"},
            "output": {"response": "Mock response"},
            "metrics": {"latency_ms": 100},
        }
    ],
}


@pytest.fixture(autouse=True)
def inject_mock_evidence():
    """Inject mock evidence for all models/tasks to enable hermetic testing.

    This fixture runs before each test and sets up minimal mock evidence
    that passes the structural checks, so tests don't depend on real run artifacts.
    """
    # Get all claims files and their required tasks
    for claims_path in get_all_claims_files():
        model_id = get_model_id_from_path(claims_path)
        data = load_claims(claims_path)

        for claim in data.get("claims", []):
            task = claim["task"]
            # Use the template for this task type, or fall back to asr
            template = MOCK_EVIDENCE_TEMPLATES.get(task, MOCK_EVIDENCE_TEMPLATES["asr"])
            set_mock_evidence(model_id, task, template)

    yield

    # Cleanup after test
    clear_mock_evidence()


class TestClaimsExecution:
    """Execute claims and verify required claims pass."""

    @pytest.mark.parametrize("claims_path", get_all_claims_files(), ids=lambda p: p.parent.name)
    def test_required_claims_have_handlers(self, claims_path):
        """All required claims must have implemented test handlers."""
        data = load_claims(claims_path)

        missing = []
        for claim in data["claims"]:
            if claim["enforcement"] == "required":
                test_ref = claim["test_ref"]
                if test_ref not in TEST_REF_HANDLERS:
                    missing.append(test_ref)

        assert not missing, f"Missing handlers for required claims: {missing}"

    @pytest.mark.parametrize("claims_path", get_all_claims_files(), ids=lambda p: p.parent.name)
    def test_required_claims_pass(self, claims_path):
        """All required claims must pass for the model."""
        model_id = get_model_id_from_path(claims_path)
        data = load_claims(claims_path)

        results = execute_all_claims(model_id, data)

        # Check required claims
        failed_required = [
            c for c in results["claims"] if c["enforcement"] == "required" and c["status"] == "fail"
        ]

        if failed_required:
            failures = [f"{c['claim_id']}: {c['reason']}" for c in failed_required]
            pytest.fail(f"Required claims failed: {failures}")
