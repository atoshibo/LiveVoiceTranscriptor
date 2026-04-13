"""
Test API Compatibility - Ensure all v0.4.2 API contracts are preserved.

Tests endpoint paths, request/response schemas, auth, status codes,
alias fields, idempotent behavior.
"""
import os
import sys
import json
import pytest
from datetime import datetime, timedelta, timezone
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


class TestSessionListSummaries:
    def test_list_sessions_sorted_by_activity_time(self, tmp_sessions_dir, monkeypatch):
        from app.storage import session_store

        ids = iter(["z-session", "a-session"])
        monkeypatch.setattr(session_store.uuid, "uuid4", lambda: next(ids))

        older = session_store.create_session({})
        newer = session_store.create_session({})

        now = datetime.now(timezone.utc)
        session_store.update_session_meta(older["session_id"], {
            "created_at": (now - timedelta(minutes=10)).isoformat(),
        })
        session_store.update_session_meta(newer["session_id"], {
            "created_at": (now - timedelta(minutes=5)).isoformat(),
        })

        sessions = session_store.list_sessions(limit=10)
        assert [item["session_id"] for item in sessions[:2]] == ["a-session", "z-session"]

    def test_cleanup_removes_stale_empty_draft_sessions(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, cleanup_abandoned_draft_sessions, session_exists, update_session_meta

        sid = create_session({})["session_id"]
        update_session_meta(sid, {
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        })

        result = cleanup_abandoned_draft_sessions(max_age_minutes=120)

        assert result["deleted_count"] == 1
        assert sid in result["deleted_session_ids"]
        assert session_exists(sid) is False

    def test_cleanup_preserves_sessions_that_have_received_chunks(self, tmp_sessions_dir, make_wav_bytes):
        from app.storage.session_store import (
            cleanup_abandoned_draft_sessions,
            create_session,
            register_chunk,
            session_dir,
            session_exists,
            update_session_meta,
        )

        sid = create_session({})["session_id"]
        sd = session_dir(sid)
        (sd / "chunks" / "chunk_0000.wav").write_bytes(make_wav_bytes(duration_s=0.1))
        register_chunk(sid, 0, {
            "chunk_index": 0,
            "chunk_started_ms": 0,
            "chunk_duration_ms": 100,
        })
        update_session_meta(sid, {
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        })

        result = cleanup_abandoned_draft_sessions(max_age_minutes=120)

        assert sid not in result["deleted_session_ids"]
        assert session_exists(sid) is True

    def test_list_sessions_auto_cleans_stale_empty_drafts(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, list_sessions, session_exists, update_session_meta

        stale_sid = create_session({})["session_id"]
        fresh_sid = create_session({})["session_id"]
        update_session_meta(stale_sid, {
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat(),
        })
        update_session_meta(fresh_sid, {
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        sessions = list_sessions(limit=10)
        session_ids = [item["session_id"] for item in sessions]

        assert stale_sid not in session_ids
        assert fresh_sid in session_ids
        assert session_exists(stale_sid) is False

    def test_grouped_sessions_use_worker_state_for_completed_runs(self, tmp_sessions_dir):
        from app.storage.session_store import (
            create_session,
            list_sessions_grouped,
            update_session_meta,
            update_status,
        )

        sid = create_session({"source_type": "device_import"})["session_id"]
        update_session_meta(sid, {"state": "finalized"})
        update_status(sid, "done")

        grouped = list_sessions_grouped()
        summary = next(
            sess
            for group in grouped["groups"]
            for sess in group["sessions"]
            if sess["session_id"] == sid
        )

        assert summary["state"] == "done"
        assert summary["backend_outcome"] == "completed"

    def test_grouped_sessions_surface_finalized_pending_as_queued(self, tmp_sessions_dir):
        from app.storage.session_store import (
            create_session,
            list_sessions_grouped,
            update_session_meta,
            update_status,
        )

        sid = create_session({"source_type": "device_import"})["session_id"]
        update_session_meta(sid, {"state": "finalized"})
        update_status(sid, "uploaded")

        grouped = list_sessions_grouped()
        summary = next(
            sess
            for group in grouped["groups"]
            for sess in group["sessions"]
            if sess["session_id"] == sid
        )

        assert summary["state"] == "queued"


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


class TestJobEnqueueContract:
    def test_enqueue_job_uses_latest_finalize_contract(self, monkeypatch):
        from app.api import api_v2

        pushed = []

        class DummyRedis:
            def rpush(self, queue, payload):
                pushed.append((queue, payload))

        monkeypatch.setattr(api_v2, "_get_redis", lambda: DummyRedis())
        monkeypatch.setattr(api_v2, "update_status", lambda *args, **kwargs: None)

        meta = {
            "language": "auto",
            "model_size": "small",
            "allowed_languages": ["fr"],
            "forced_language": None,
            "transcription_mode": "verbatim_multilingual",
            "diarization_policy": "forced",
            "speaker_count": 2,
        }
        body = {
            "model_size": "small",
            "allowed_languages": ["fr"],
            "speaker_count": 2,
        }

        api_v2._enqueue_job("session-1", meta, body)

        assert len(pushed) == 1
        payload = json.loads(pushed[0][1])
        assert payload["job_type"] == "v2"
        assert payload["model_size"] == "small"
        assert payload["allowed_languages"] == ["fr"]
        assert payload["diarization_policy"] == "forced"
        assert payload["run_diarization"] is True
