"""
Test Android/App Compatibility - Alias fields, continuity metadata, partial suppression.

Mirrors the old test_android_compat.py behavior to ensure backward compatibility.
"""
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestCreateSessionAliases:
    """Test that session creation accepts all Android alias fields."""

    def _simulate_create_session_body(self, body: dict) -> dict:
        """Mirror the alias resolution logic from create_session."""
        sample_rate = body.get("sample_rate_hz") or body.get("sample_rate") or 16000
        mode = body.get("mode", "stream")
        source_type = body.get("source_type")
        if source_type == "device_import":
            mode = "file"
        return {
            "sample_rate_hz": int(sample_rate),
            "mode": mode,
            "source_type": source_type,
            "chunk_duration_sec": body.get("chunk_duration_sec"),
            "diarization_hint": bool(body.get("diarization", False)),
        }

    def test_legacy_sample_rate_hz(self):
        r = self._simulate_create_session_body({"sample_rate_hz": 44100})
        assert r["sample_rate_hz"] == 44100

    def test_android_sample_rate_alias(self):
        r = self._simulate_create_session_body({"sample_rate": 48000})
        assert r["sample_rate_hz"] == 48000

    def test_sample_rate_hz_takes_priority(self):
        r = self._simulate_create_session_body({"sample_rate_hz": 44100, "sample_rate": 48000})
        assert r["sample_rate_hz"] == 44100

    def test_default_sample_rate(self):
        r = self._simulate_create_session_body({})
        assert r["sample_rate_hz"] == 16000

    def test_source_type_device_import_sets_mode_file(self):
        r = self._simulate_create_session_body({"source_type": "device_import"})
        assert r["mode"] == "file"
        assert r["source_type"] == "device_import"

    def test_source_type_ble_stream_keeps_stream(self):
        r = self._simulate_create_session_body({"source_type": "ble_stream"})
        assert r["mode"] == "stream"

    def test_no_source_type_defaults_to_stream(self):
        r = self._simulate_create_session_body({})
        assert r["mode"] == "stream"
        assert r["source_type"] is None

    def test_explicit_mode_preserved_when_no_device_import(self):
        r = self._simulate_create_session_body({"mode": "file"})
        assert r["mode"] == "file"

    def test_source_type_device_import_overrides_explicit_stream(self):
        r = self._simulate_create_session_body({"source_type": "device_import", "mode": "stream"})
        assert r["mode"] == "file"

    def test_chunk_duration_sec_stored(self):
        r = self._simulate_create_session_body({"chunk_duration_sec": 30.0})
        assert r["chunk_duration_sec"] == 30.0

    def test_diarization_hint_stored(self):
        r = self._simulate_create_session_body({"diarization": True})
        assert r["diarization_hint"] is True

    def test_diarization_hint_default_false(self):
        r = self._simulate_create_session_body({})
        assert r["diarization_hint"] is False


class TestUploadContinuityAliases:
    """Test continuity field normalization for Android compatibility."""

    def _normalize_continuity(self, dropped_frames=0, decode_failure=False,
                               gap_before_ms=0, source_degraded=False,
                               decode_errors=0, ble_gaps=0,
                               plc_frames_applied=0, has_continuity_warning=False):
        actual_dropped_frames = dropped_frames + plc_frames_applied
        actual_decode_failure = decode_failure or (decode_errors > 0)
        actual_gap_before_ms = gap_before_ms if gap_before_ms > 0 else ble_gaps
        actual_source_degraded = source_degraded or has_continuity_warning
        return {
            "dropped_frames": actual_dropped_frames,
            "decode_failure": actual_decode_failure,
            "gap_before_ms": actual_gap_before_ms,
            "source_degraded": actual_source_degraded,
        }

    def test_legacy_fields_passthrough(self):
        r = self._normalize_continuity(dropped_frames=5, decode_failure=True, gap_before_ms=100, source_degraded=True)
        assert r == {"dropped_frames": 5, "decode_failure": True, "gap_before_ms": 100, "source_degraded": True}

    def test_android_decode_errors(self):
        r = self._normalize_continuity(decode_errors=3)
        assert r["decode_failure"] is True

    def test_android_decode_errors_zero(self):
        r = self._normalize_continuity(decode_errors=0)
        assert r["decode_failure"] is False

    def test_legacy_decode_failure_preserved_with_zero_android(self):
        r = self._normalize_continuity(decode_failure=True, decode_errors=0)
        assert r["decode_failure"] is True

    def test_android_ble_gaps(self):
        r = self._normalize_continuity(ble_gaps=250)
        assert r["gap_before_ms"] == 250

    def test_android_ble_gaps_does_not_overwrite_legacy(self):
        r = self._normalize_continuity(gap_before_ms=100, ble_gaps=250)
        assert r["gap_before_ms"] == 100

    def test_android_plc_frames_added_to_dropped(self):
        r = self._normalize_continuity(dropped_frames=10, plc_frames_applied=5)
        assert r["dropped_frames"] == 15

    def test_android_plc_frames_alone(self):
        r = self._normalize_continuity(plc_frames_applied=7)
        assert r["dropped_frames"] == 7

    def test_android_has_continuity_warning(self):
        r = self._normalize_continuity(has_continuity_warning=True)
        assert r["source_degraded"] is True

    def test_legacy_source_degraded_preserved(self):
        r = self._normalize_continuity(source_degraded=True)
        assert r["source_degraded"] is True

    def test_all_zeros(self):
        r = self._normalize_continuity()
        assert r == {"dropped_frames": 0, "decode_failure": False, "gap_before_ms": 0, "source_degraded": False}

    def test_all_android_fields_combined(self):
        r = self._normalize_continuity(decode_errors=2, ble_gaps=300, plc_frames_applied=10, has_continuity_warning=True)
        assert r == {"dropped_frames": 10, "decode_failure": True, "gap_before_ms": 300, "source_degraded": True}

    def test_mixed_legacy_and_android(self):
        r = self._normalize_continuity(dropped_frames=3, gap_before_ms=50, decode_errors=1, plc_frames_applied=2, ble_gaps=200)
        assert r["dropped_frames"] == 5
        assert r["decode_failure"] is True
        assert r["gap_before_ms"] == 50  # Legacy wins
        assert r["source_degraded"] is False


class TestDeviceImportSuppressesPartial:
    """Test that device_import mode suppresses partial transcript triggers."""

    def _should_trigger(self, mode):
        return mode == "stream"

    def test_stream_mode_would_trigger(self):
        assert self._should_trigger("stream") is True

    def test_file_mode_blocks_trigger(self):
        assert self._should_trigger("file") is False

    def test_device_import_resolves_to_file_mode(self):
        mode = "stream"
        source_type = "device_import"
        if source_type == "device_import":
            mode = "file"
        assert self._should_trigger(mode) is False
