"""
Test API Compatibility - Ensure all v0.4.2 API contracts are preserved.

Tests endpoint paths, request/response schemas, auth, status codes,
alias fields, idempotent behavior.
"""
import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestAuthCompatibility:
    """Auth must accept both Authorization: Bearer and X-Api-Token headers."""

    def test_extract_token_bearer(self):
        from app.api.auth import _extract_token
        req = MagicMock()
        req.headers = {"authorization": "Bearer test123"}
        assert _extract_token(req) == "test123"

    def test_extract_token_x_api_token(self):
        from app.api.auth import _extract_token
        req = MagicMock()
        req.headers = {"x-api-token": "test456", "authorization": ""}
        assert _extract_token(req) == "test456"

    def test_extract_token_bearer_priority(self):
        from app.api.auth import _extract_token
        req = MagicMock()
        req.headers = {"authorization": "Bearer bearer_tok", "x-api-token": "api_tok"}
        assert _extract_token(req) == "bearer_tok"

    def test_extract_token_none_when_missing(self):
        from app.api.auth import _extract_token
        req = MagicMock()
        req.headers = {}
        assert _extract_token(req) is None


class TestSessionCreation:
    """Session creation must preserve all alias behaviors."""

    def test_create_session_returns_id_and_url(self, tmp_sessions_dir):
        from app.storage.session_store import create_session
        result = create_session({})
        assert "session_id" in result
        assert "upload_url" in result
        assert len(result["session_id"]) > 8
        assert "/chunks" in result["upload_url"]

    def test_create_session_sample_rate_alias(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, get_session_meta
        result = create_session({"sample_rate": 48000})
        meta = get_session_meta(result["session_id"])
        assert meta["sample_rate_hz"] == 48000

    def test_create_session_device_import_mode(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, get_session_meta
        result = create_session({"source_type": "device_import"})
        meta = get_session_meta(result["session_id"])
        assert meta["mode"] == "file"

    def test_create_session_diarization_hint(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, get_session_meta
        result = create_session({"diarization": True})
        meta = get_session_meta(result["session_id"])
        assert meta["diarization_hint"] is True

    def test_create_session_initial_state(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, get_session_meta
        result = create_session({})
        meta = get_session_meta(result["session_id"])
        assert meta["state"] == "created"


class TestStatusMapping:
    """Internal statuses must map to correct public states."""

    def test_status_mapping(self):
        from app.api.api_v2 import _map_status_to_public
        assert _map_status_to_public("created") == "queued"
        assert _map_status_to_public("uploaded") == "queued"
        assert _map_status_to_public("pending") == "queued"
        assert _map_status_to_public("processing") == "running"
        assert _map_status_to_public("running") == "running"
        assert _map_status_to_public("done") == "done"
        assert _map_status_to_public("error") == "error"
        assert _map_status_to_public("cancelled") == "error"


class TestTranscriptFieldPreservation:
    """Transcript response must preserve all backward-compatible fields."""

    def test_segment_normalization_v2_format(self):
        from app.api.api_v2 import _normalize_segments
        segs = [{"start_ms": 400, "end_ms": 1145, "speaker": "S0", "text": "hello"}]
        result = _normalize_segments(segs)
        assert len(result) == 1
        assert result[0]["start_ms"] == 400
        assert result[0]["end_ms"] == 1145

    def test_segment_normalization_legacy_float_format(self):
        from app.api.api_v2 import _normalize_segments
        segs = [{"start": 0.4, "end": 1.145, "text": "hello"}]
        result = _normalize_segments(segs)
        assert len(result) == 1
        assert result[0]["start_ms"] == 400
        assert result[0]["end_ms"] == 1145

    def test_segment_normalization_start_s_format(self):
        from app.api.api_v2 import _normalize_segments
        segs = [{"start_s": 0.0, "end_s": 1.5, "text": "hello"}]
        result = _normalize_segments(segs)
        assert result[0]["start_ms"] == 0
        assert result[0]["end_ms"] == 1500

    def test_empty_text_segments_filtered(self):
        from app.api.api_v2 import _normalize_segments
        segs = [
            {"start_ms": 0, "end_ms": 1000, "text": "hello"},
            {"start_ms": 1000, "end_ms": 2000, "text": "   "},
        ]
        result = _normalize_segments(segs)
        assert len(result) == 1

    def test_segments_sorted_by_start(self):
        from app.api.api_v2 import _normalize_segments
        segs = [
            {"start_ms": 2000, "end_ms": 3000, "text": "second"},
            {"start_ms": 0, "end_ms": 1000, "text": "first"},
        ]
        result = _normalize_segments(segs)
        assert result[0]["text"] == "first"
        assert result[1]["text"] == "second"

    def test_corruption_flags_preserved(self):
        from app.api.api_v2 import _normalize_segments
        segs = [{"start_ms": 0, "end_ms": 1000, "text": "hi", "corruption_flags": ["low_confidence"]}]
        result = _normalize_segments(segs)
        assert result[0]["corruption_flags"] == ["low_confidence"]


class TestSpeakerTimestampedSynthesis:
    """Speaker timestamped text synthesis."""

    def test_synthesized_from_segments(self):
        from app.api.api_v2 import _synthesize_speaker_timestamped
        segs = [
            {"start_ms": 0, "end_ms": 1500, "speaker": "SPEAKER_00", "text": "Hello"},
            {"start_ms": 1500, "end_ms": 3000, "speaker": "SPEAKER_01", "text": "World"},
        ]
        result = _synthesize_speaker_timestamped(segs)
        assert result is not None
        assert "SPEAKER_01" in result
        assert "World" in result
        assert "1.5s" in result

    def test_returns_none_when_empty(self):
        from app.api.api_v2 import _synthesize_speaker_timestamped
        assert _synthesize_speaker_timestamped([]) is None

    def test_returns_none_when_all_empty_text(self):
        from app.api.api_v2 import _synthesize_speaker_timestamped
        segs = [{"start_ms": 0, "end_ms": 1000, "speaker": "S0", "text": "  "}]
        assert _synthesize_speaker_timestamped(segs) is None


class TestModelSizeValidation:
    """Model size validation preserves old behavior."""

    def test_valid_sizes(self):
        from app.models.registry import VALID_MODEL_SIZES
        for size in ("tiny", "base", "small", "medium", "large-v2", "large-v3"):
            assert size in VALID_MODEL_SIZES

    def test_resolve_legacy_size(self):
        from app.models.registry import resolve_model_id
        assert resolve_model_id("small") == "faster-whisper:small"
        assert resolve_model_id("large-v3") == "faster-whisper:large-v3"

    def test_resolve_full_id(self):
        from app.models.registry import resolve_model_id
        assert resolve_model_id("faster-whisper:medium") == "faster-whisper:medium"
        assert resolve_model_id("nemo-asr:parakeet-tdt-0.6b-v3") == "nemo-asr:parakeet-tdt-0.6b-v3"
