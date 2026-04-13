"""
Test Canonical Pipeline - Pipeline geometry, stages, model replacement, assembly.

Ensures the frozen 30s/15s/15s geometry, Parakeet replacement of Turbo,
canonical assembly, and provenance are correct.
"""
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestCanonicalStages:
    """Verify the frozen V1 stage list."""

    def test_exact_list(self):
        from app.pipeline.run import CANONICAL_V1_STAGES
        expected = [
            "normalize_audio",
            "acoustic_triage",
            "decode_lattice",
            "first_pass_medium",
            "candidate_asr_large_v3",
            "candidate_asr_parakeet",
            "stripe_grouping",
            "reconciliation",
            "canonical_assembly",
            "selective_enrichment",
            "derived_outputs",
        ]
        assert CANONICAL_V1_STAGES == expected

    def test_count(self):
        from app.pipeline.run import CANONICAL_V1_STAGES
        assert len(CANONICAL_V1_STAGES) == 11

    def test_no_turbo_in_stages(self):
        """Turbo must NOT be in the canonical stage list."""
        from app.pipeline.run import CANONICAL_V1_STAGES
        assert "candidate_asr_turbo_hf" not in CANONICAL_V1_STAGES

    def test_parakeet_replaces_turbo(self):
        """Parakeet must be in the stage list where turbo was."""
        from app.pipeline.run import CANONICAL_V1_STAGES
        assert "candidate_asr_parakeet" in CANONICAL_V1_STAGES

    def test_turbo_in_legacy_stages(self):
        """Turbo stage name kept in legacy list for compat detection."""
        from app.pipeline.run import LEGACY_STAGES
        assert "candidate_asr_turbo_hf" in LEGACY_STAGES


class TestPipelineGeometry:
    """Verify the frozen 30s/15s/15s geometry."""

    def test_decode_windows_basic(self):
        from app.pipeline.decode_lattice import build_decode_windows
        windows = build_decode_windows(60000)
        # Should produce: W0=[0,30], W1=[15,45], W2=[30,60], W3=[45,60]
        assert len(windows) >= 3
        assert windows[0]["start_ms"] == 0
        assert windows[0]["end_ms"] == 30000
        assert windows[0]["window_type"] == "full"
        assert windows[1]["start_ms"] == 15000
        assert windows[1]["end_ms"] == 45000
        assert windows[1]["window_type"] == "bridge"

    def test_15s_stride(self):
        from app.pipeline.decode_lattice import build_decode_windows
        windows = build_decode_windows(90000)
        # Check stride
        for i in range(1, len(windows)):
            stride = windows[i]["start_ms"] - windows[i-1]["start_ms"]
            assert stride == 15000

    def test_bridge_windows_span_chunks(self):
        from app.pipeline.decode_lattice import build_decode_windows
        windows = build_decode_windows(60000)
        bridges = [w for w in windows if w["window_type"] == "bridge"]
        assert len(bridges) >= 1
        # Bridge should span two source chunks
        for b in bridges:
            if len(b["source_chunks"]) > 1:
                assert len(b["source_chunks"]) >= 2

    def test_every_interior_stripe_has_two_windows(self):
        """Every interior 15s stripe must be observable from 2 windows."""
        from app.pipeline.decode_lattice import build_decode_windows
        from app.pipeline.stripe_grouping import build_stripes
        windows = build_decode_windows(90000)
        stripes = build_stripes(90000)

        # Interior stripes (not first or last)
        for stripe in stripes[1:-1]:
            supporting = []
            for w in windows:
                overlap_start = max(stripe["start_ms"], w["start_ms"])
                overlap_end = min(stripe["end_ms"], w["end_ms"])
                if overlap_end > overlap_start:
                    supporting.append(w)
            assert len(supporting) >= 2, f"Stripe {stripe['stripe_id']} has only {len(supporting)} supporting windows"

    def test_bridge_window_required_when_speech_crosses_chunk_boundary(self):
        from app.pipeline.decode_lattice import build_decode_windows

        windows = build_decode_windows(60000, speech_islands=[
            {"start_ms": 29000, "end_ms": 31000},
        ])
        bridge = next(w for w in windows if w["window_type"] == "bridge" and w["start_ms"] == 15000)

        assert bridge["bridge_required"] is True
        assert bridge["scheduled"] is True

    def test_bridge_window_skipped_when_boundary_has_no_crossing_speech(self):
        from app.pipeline.decode_lattice import build_decode_windows

        windows = build_decode_windows(60000, speech_islands=[
            {"start_ms": 0, "end_ms": 1000},
        ])
        bridge = next(w for w in windows if w["window_type"] == "bridge" and w["start_ms"] == 15000)

        assert bridge["bridge_required"] is False
        assert bridge["scheduled"] is False

    def test_decode_windows_stay_conservative_when_triage_is_unavailable(self):
        from app.pipeline.decode_lattice import build_decode_windows

        windows = build_decode_windows(60000, speech_islands=None)

        assert all(window["scheduled"] for window in windows)


class TestStripeGrouping:
    """Test 15-second commit stripes."""

    def test_stripe_size(self):
        from app.pipeline.stripe_grouping import build_stripes
        stripes = build_stripes(60000)
        for s in stripes[:-1]:  # Last may be shorter
            assert s["duration_ms"] == 15000

    def test_stripe_coverage(self):
        from app.pipeline.stripe_grouping import build_stripes
        stripes = build_stripes(60000)
        assert stripes[0]["start_ms"] == 0
        assert stripes[-1]["end_ms"] == 60000

    def test_no_stripe_gaps(self):
        from app.pipeline.stripe_grouping import build_stripes
        stripes = build_stripes(60000)
        for i in range(1, len(stripes)):
            assert stripes[i]["start_ms"] == stripes[i-1]["end_ms"]

    def test_extract_stripe_text_uses_explicit_seconds_unit(self):
        from app.pipeline.stripe_grouping import _extract_stripe_text

        candidate = {
            "window_start_ms": 600000,
            "window_end_ms": 630000,
            "raw_text": "privet mir",
            "segments": [
                {"start": 0.0, "end": 28.5, "text": "privet mir"},
            ],
            "decode_metadata": {"segment_timestamp_unit": "seconds"},
        }

        text = _extract_stripe_text(candidate, 615000, 630000)
        assert text == "privet mir"

    def test_extract_stripe_text_infers_seconds_from_window_duration(self):
        from app.pipeline.stripe_grouping import _extract_stripe_text

        candidate = {
            "window_start_ms": 600000,
            "window_end_ms": 630000,
            "raw_text": "dobryy den",
            "segments": [
                {"start": 0.0, "end": 28.5, "text": "dobryy den"},
            ],
        }

        text = _extract_stripe_text(candidate, 615000, 630000)
        assert text == "dobryy den"

    def test_extract_stripe_text_preserves_millisecond_segments(self):
        from app.pipeline.stripe_grouping import _extract_stripe_text

        candidate = {
            "window_start_ms": 600000,
            "window_end_ms": 630000,
            "raw_text": "millisecond text",
            "segments": [
                {"start": 1200, "end": 1800, "text": "millisecond text"},
            ],
        }

        text = _extract_stripe_text(candidate, 601000, 602000)
        assert text == "millisecond text"

    def test_run_stripe_grouping_includes_first_pass_candidates(self, tmp_sessions_dir):
        from app.core.atomic_io import atomic_write_json
        from app.pipeline.stripe_grouping import run_stripe_grouping
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        window = {
            "window_id": "W000000",
            "start_ms": 0,
            "end_ms": 30000,
            "window_type": "full",
            "scheduled": True,
        }
        shared_segments = [{"start": 0.0, "end": 15.0, "text": "privet"}]

        atomic_write_json(str(sd / "candidates" / "cand_medium.json"), {
            "candidate_id": "cand_medium",
            "session_id": sid,
            "model_id": "faster-whisper:medium",
            "window_id": "W000000",
            "window_start_ms": 0,
            "window_end_ms": 30000,
            "window_type": "full",
            "raw_text": "privet",
            "segments": shared_segments,
            "language_evidence": {"detected_language": "ru"},
            "confidence_features": {"success": True, "degraded": False},
            "decode_metadata": {"segment_timestamp_unit": "seconds"},
        })
        atomic_write_json(str(sd / "candidates" / "cand_large.json"), {
            "candidate_id": "cand_large",
            "session_id": sid,
            "model_id": "faster-whisper:large-v3",
            "window_id": "W000000",
            "window_start_ms": 0,
            "window_end_ms": 30000,
            "window_type": "full",
            "raw_text": "privet",
            "segments": shared_segments,
            "language_evidence": {"detected_language": "ru"},
            "confidence_features": {"success": True, "degraded": False},
            "decode_metadata": {"segment_timestamp_unit": "seconds"},
        })

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        result = run_stripe_grouping(sid, [window], 30000, StageStub())
        first_stripe = result["stripes"][0]

        assert first_stripe["model_ids"] == ["faster-whisper:large-v3", "faster-whisper:medium"]
        assert {item["model_id"] for item in first_stripe["evidence"]} == {
            "faster-whisper:large-v3",
            "faster-whisper:medium",
        }


class TestModelReplacement:
    """Verify Turbo is replaced by Parakeet everywhere."""

    def test_default_candidate_b_is_parakeet(self):
        from app.models.registry import DEFAULT_CANDIDATE_B
        assert "parakeet" in DEFAULT_CANDIDATE_B.lower()

    def test_turbo_not_in_default_pipeline(self):
        from app.models.registry import DEFAULT_FIRST_PASS, DEFAULT_CANDIDATE_A, DEFAULT_CANDIDATE_B
        for model in [DEFAULT_FIRST_PASS, DEFAULT_CANDIDATE_A, DEFAULT_CANDIDATE_B]:
            assert "turbo" not in model.lower()

    def test_parakeet_in_registry(self):
        from app.models.registry import get_registry
        reg = get_registry(refresh=True)
        assert "nemo-asr:parakeet-tdt-0.6b-v3" in reg

    def test_turbo_marked_deprecated(self):
        from app.models.registry import get_registry
        reg = get_registry(refresh=True)
        turbo = reg.get("transformers-whisper:whisper-large-v3-turbo-hf")
        if turbo:
            assert "DEPRECATED" in turbo.display_name or "DEPRECATED" in (turbo.notes or "")

    def test_parakeet_honest_language_coverage(self):
        """Parakeet must honestly report English-only."""
        from app.models.registry import get_registry
        reg = get_registry(refresh=True)
        parakeet = reg["nemo-asr:parakeet-tdt-0.6b-v3"]
        assert parakeet.languages == ["en"]
        assert parakeet.capabilities.multilingual is False

    def test_canary_not_in_default_pipeline(self):
        from app.models.registry import DEFAULT_FIRST_PASS, DEFAULT_CANDIDATE_A, DEFAULT_CANDIDATE_B
        for model in [DEFAULT_FIRST_PASS, DEFAULT_CANDIDATE_A, DEFAULT_CANDIDATE_B]:
            assert "canary" not in model.lower()

    def test_models_endpoint_reflects_parakeet(self):
        from app.models.registry import list_all_models
        models = list_all_models()
        model_ids = [m.model_id for m in models]
        assert "nemo-asr:parakeet-tdt-0.6b-v3" in model_ids

    def test_default_first_pass_is_medium(self):
        from app.models.registry import DEFAULT_FIRST_PASS

        assert DEFAULT_FIRST_PASS == "faster-whisper:medium"


class TestCanonicalAssembly:
    """Test canonical segment assembly from reconciliation."""

    def test_dedup_join(self):
        from app.pipeline.canonical_assembly import _dedup_join
        # Overlapping text
        result = _dedup_join("hello world is great", "is great today")
        assert "is great" in result
        # Should not duplicate "is great"
        assert result.count("is great") == 1

    def test_dedup_join_no_overlap(self):
        from app.pipeline.canonical_assembly import _dedup_join
        result = _dedup_join("hello world", "foo bar")
        assert result == "hello world foo bar"

    def test_merge_into_segments(self):
        from app.pipeline.canonical_assembly import merge_into_segments
        records = [
            {"stripe_id": "S0", "start_ms": 0, "end_ms": 15000,
             "chosen_text": "hello world", "chosen_source": "faster-whisper:large-v3",
             "confidence": 0.8, "stabilization_state": "stabilized", "support_window_count": 2},
            {"stripe_id": "S1", "start_ms": 15000, "end_ms": 30000,
             "chosen_text": "foo bar", "chosen_source": "faster-whisper:large-v3",
             "confidence": 0.9, "stabilization_state": "stabilized", "support_window_count": 2},
        ]
        segments = merge_into_segments(records)
        assert len(segments) >= 1
        # All segments have required fields
        for seg in segments:
            assert "segment_id" in seg
            assert "start_ms" in seg
            assert "end_ms" in seg
            assert "text" in seg
            assert "support_models" in seg
            assert "stabilization_state" in seg

    def test_provisional_vs_stabilized(self):
        from app.pipeline.canonical_assembly import stabilize_stripes
        records = [{"stripe_id": "S1", "start_ms": 15000, "end_ms": 30000,
                     "chosen_text": "hello", "confidence": 0.8}]
        packets = [
            {"stripe_id": "S0", "support_window_count": 2},
            {"stripe_id": "S1", "support_window_count": 1},
            {"stripe_id": "S2", "support_window_count": 2},
        ]
        result = stabilize_stripes(records, packets)
        assert result[0]["stabilization_state"] == "provisional"

        packets[1]["support_window_count"] = 2
        result = stabilize_stripes(records, packets)
        assert result[0]["stabilization_state"] == "stabilized"

    def test_boundary_stripes_finalize_with_single_support(self):
        from app.pipeline.canonical_assembly import stabilize_stripes

        records = [
            {"stripe_id": "S0", "start_ms": 0, "end_ms": 15000, "chosen_text": "alpha", "confidence": 0.8},
            {"stripe_id": "S2", "start_ms": 30000, "end_ms": 45000, "chosen_text": "omega", "confidence": 0.8},
        ]
        packets = [
            {"stripe_id": "S0", "support_window_count": 1},
            {"stripe_id": "S1", "support_window_count": 2},
            {"stripe_id": "S2", "support_window_count": 1},
        ]

        result = stabilize_stripes(records, packets)
        assert [row["stabilization_state"] for row in result] == ["stabilized", "stabilized"]

    def test_final_transcript_keeps_boundary_stripes(self, tmp_sessions_dir):
        from app.pipeline.canonical_assembly import (
            build_transcript_surfaces,
            merge_into_segments,
            stabilize_stripes,
        )
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        assert sd.is_dir()

        records = [
            {
                "stripe_id": "S0",
                "start_ms": 0,
                "end_ms": 15000,
                "chosen_text": "privet",
                "chosen_source": "faster-whisper:medium",
                "confidence": 0.8,
                "language": "ru",
            },
            {
                "stripe_id": "S1",
                "start_ms": 15000,
                "end_ms": 30000,
                "chosen_text": "mir",
                "chosen_source": "faster-whisper:large-v3",
                "confidence": 0.9,
                "language": "ru",
            },
        ]
        packets = [
            {
                "stripe_id": "S0",
                "support_window_count": 1,
                "support_windows": ["W000000"],
                "support_models": ["faster-whisper:medium"],
            },
            {
                "stripe_id": "S1",
                "support_window_count": 1,
                "support_windows": ["W000001"],
                "support_models": ["faster-whisper:large-v3"],
            },
        ]

        stabilized = stabilize_stripes(records, packets)
        segments = merge_into_segments(stabilized)
        surfaces = build_transcript_surfaces(segments, sid)

        assert surfaces["text"] == "privet mir"
        assert surfaces["final_transcript"]["segment_count"] == 2
        assert surfaces["stabilized_partial"]["text"] == "privet mir"


class TestPipelineRun:
    """Test pipeline run tracking."""

    def test_create_run(self, tmp_sessions_dir):
        from app.pipeline.run import create_canonical_run, CANONICAL_V1_STAGES
        sd = tmp_sessions_dir / "test-session"
        sd.mkdir()
        run = create_canonical_run(str(sd), "test-session", {"test": True})
        assert run.run_id
        assert run._stage_names == CANONICAL_V1_STAGES

    def test_stage_lifecycle(self, tmp_sessions_dir):
        from app.pipeline.run import create_canonical_run, STAGE_DONE
        sd = tmp_sessions_dir / "test-session2"
        sd.mkdir()
        run = create_canonical_run(str(sd), "test-session2", {})
        stage = run.start_stage("normalize_audio")
        assert stage.status == "running"
        stage.commit(["audio.wav"])
        assert stage.status == STAGE_DONE
        assert run.is_stage_done("normalize_audio")

    def test_skip_stage(self, tmp_sessions_dir):
        from app.pipeline.run import create_canonical_run
        sd = tmp_sessions_dir / "test-session3"
        sd.mkdir()
        run = create_canonical_run(str(sd), "test-session3", {})
        run.skip_stage("selective_enrichment", "not_justified")
        assert run.is_stage_done("selective_enrichment")

    def test_fallback_stage(self, tmp_sessions_dir):
        from app.pipeline.run import create_canonical_run, STAGE_DONE
        sd = tmp_sessions_dir / "test-session4"
        sd.mkdir()
        run = create_canonical_run(str(sd), "test-session4", {})
        stage = run.start_stage("acoustic_triage")
        stage.commit_with_fallback([], "triage_unavailable")
        assert stage.status == STAGE_DONE
        assert run.is_stage_done("acoustic_triage")
        assert stage.error.startswith("fallback:")

    def test_unknown_stage_raises(self, tmp_sessions_dir):
        from app.pipeline.run import create_canonical_run
        sd = tmp_sessions_dir / "test-session5"
        sd.mkdir()
        run = create_canonical_run(str(sd), "test-session5", {})
        with pytest.raises(ValueError, match="Unknown stage"):
            run.get_stage("diarization")  # Old legacy name


class TestFailureHandling:
    """Test graceful degradation on failures."""

    def test_model_priority_tracks_default_first_pass(self):
        from app.models.registry import DEFAULT_FIRST_PASS, resolve_model_id
        from app.pipeline.reconciliation import MODEL_PRIORITY

        assert resolve_model_id(DEFAULT_FIRST_PASS) in MODEL_PRIORITY

    def test_reconciliation_deterministic_fallback(self):
        from app.pipeline.reconciliation import _select_fallback
        evidence = [
            {"model_id": "faster-whisper:large-v3", "text": "hello world", "trust_score": 0.9},
            {"model_id": "nemo-asr:parakeet-tdt-0.6b-v3", "text": "hello world!", "trust_score": 0.7},
        ]
        result = _select_fallback(evidence)
        assert result["method"] == "fallback"
        assert result["chosen_text"] in ("hello world", "hello world!")
        assert result["confidence"] >= 0.0

    def test_fallback_with_empty_evidence(self):
        from app.pipeline.reconciliation import _select_fallback
        result = _select_fallback([])
        assert result["chosen_text"] == ""
        assert result["confidence"] == 0.0

    def test_fallback_with_all_empty_text(self):
        from app.pipeline.reconciliation import _select_fallback
        evidence = [{"model_id": "m1", "text": "", "trust_score": 0.5}]
        result = _select_fallback(evidence)
        assert result["chosen_text"] == ""

    def test_fallback_uses_medium_when_large_is_empty(self):
        from app.pipeline.reconciliation import _select_fallback

        evidence = [
            {"model_id": "faster-whisper:large-v3", "text": "", "trust_score": 0.95},
            {"model_id": "faster-whisper:medium", "text": "privet mir", "trust_score": 0.8},
        ]
        result = _select_fallback(evidence)
        assert result["chosen_source"] == "faster-whisper:medium"
        assert result["chosen_text"] == "privet mir"

    def test_llm_response_parsing(self):
        from app.pipeline.reconciliation import _parse_llm_response
        # Valid JSON
        result = _parse_llm_response('{"text": "hello", "source_model": "m1", "confidence": 0.9}')
        assert result["text"] == "hello"

        # With markdown fences
        result = _parse_llm_response('```json\n{"text": "hello", "confidence": 0.8}\n```')
        assert result["text"] == "hello"

        # Invalid
        result = _parse_llm_response("not json at all")
        assert result is None

        # Empty
        result = _parse_llm_response("")
        assert result is None


class TestDerivedOutputs:
    """Test derived output generation."""

    def test_srt_generation(self):
        from app.pipeline.derived_outputs import generate_srt
        segments = [
            {"start_ms": 0, "end_ms": 5000, "speaker": "S0", "text": "Hello"},
            {"start_ms": 5000, "end_ms": 10000, "speaker": "S1", "text": "World"},
        ]
        srt = generate_srt(segments)
        assert "00:00:00,000" in srt
        assert "00:00:05,000" in srt
        assert "[S0] Hello" in srt

    def test_vtt_generation(self):
        from app.pipeline.derived_outputs import generate_vtt
        segments = [{"start_ms": 0, "end_ms": 5000, "text": "Hello"}]
        vtt = generate_vtt(segments)
        assert vtt.startswith("WEBVTT")
        assert "00:00:00.000" in vtt

    def test_quality_report(self):
        from app.pipeline.derived_outputs import generate_quality_report
        segments = [{"start_ms": 0, "end_ms": 5000, "text": "Hello", "confidence": 0.9}]
        report = generate_quality_report(segments, "Hello", 10000)
        assert report["analysis_version"] == "1.1"
        assert report["segment_count"] == 1
        assert "issues" in report
        assert "reading_text" in report

    def test_retrieval_index_grounded_on_segments(self, tmp_sessions_dir):
        import json
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.derived_outputs import build_derived_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        segments = [{
            "segment_id": "seg_000001",
            "start_ms": 1000,
            "end_ms": 4000,
            "speaker": None,
            "text": "Bonjour tout le monde",
            "language": "fr",
            "confidence": 0.9,
            "support_windows": ["W000001", "W000002"],
            "support_models": ["faster-whisper:large-v3"],
            "stabilization_state": "stabilized",
        }]

        build_derived_dir(sid, segments, "Bonjour tout le monde", 5000)
        retrieval_path = sd / "derived" / "retrieval_index.json"
        assert retrieval_path.is_file()

        payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
        assert payload["entry_count"] == 1
        assert payload["entries"][0]["grounding"]["segment_id"] == "seg_000001"
        assert payload["entries"][0]["grounding"]["canonical_path"] == "canonical/canonical_segments.json"


class TestSelectiveEnrichment:
    def test_honors_off_policy(self, tmp_sessions_dir, monkeypatch):
        from app.storage.session_store import create_session, update_session_meta
        from app.pipeline.selective_enrichment import run_selective_enrichment

        sid = create_session({"mode": "stream"})["session_id"]
        update_session_meta(sid, {"diarization_policy": "off", "run_diarization": True})

        calls = []

        def fake_run_diarization(*args, **kwargs):
            calls.append((args, kwargs))
            return [{"speaker": "SPEAKER_01", "start_s": 0.0, "end_s": 1.0}]

        monkeypatch.setattr("app.pipeline.selective_enrichment.run_diarization", fake_run_diarization)

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        stage = StageStub()
        segments = [{
            "segment_id": "seg_000000",
            "start_ms": 0,
            "end_ms": 1000,
            "speaker": None,
            "text": "hello",
            "language": "en",
            "confidence": 0.9,
            "support_windows": ["W000000"],
            "support_models": ["faster-whisper:large-v3"],
            "stabilization_state": "stabilized",
            "stripes": ["S0"],
            "assembly_decisions": [],
        }]

        result = run_selective_enrichment(sid, segments, "unused.wav", stage)
        assert calls == []
        assert result["diarization_policy"] == "off"


class TestIngestAndTimeline:
    """Test ingest and normalization."""

    def test_merge_chunks(self, make_wav_file, tmp_path):
        from app.pipeline.ingest import merge_chunks
        c1 = make_wav_file("c1.wav", duration_s=1.0)
        c2 = make_wav_file("c2.wav", duration_s=1.0)
        output = str(tmp_path / "merged.wav")
        result = merge_chunks([c1, c2], output)
        assert result["success"] is True
        assert result["chunk_count"] == 2
        assert os.path.isfile(output)

    def test_merge_empty(self, tmp_path):
        from app.pipeline.ingest import merge_chunks
        output = str(tmp_path / "merged.wav")
        result = merge_chunks([], output)
        assert result["success"] is False

    def test_normalize_audio(self, make_wav_file, tmp_path):
        from app.pipeline.ingest import normalize_audio_file
        src = make_wav_file("src.wav", duration_s=0.5)
        dst = str(tmp_path / "norm.wav")
        result = normalize_audio_file(src, dst)
        assert result["success"] is True
        assert os.path.isfile(dst)

    def test_render_session_timeline_preserves_gap(self, tmp_sessions_dir, make_wav_bytes):
        import wave
        from app.storage.session_store import create_session, register_chunk, session_dir
        from app.pipeline.ingest import build_session_timeline, render_session_timeline_audio

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        (sd / "chunks" / "chunk_0000.wav").write_bytes(make_wav_bytes(duration_s=1.0))
        register_chunk(sid, 0, {
            "chunk_index": 0,
            "chunk_started_ms": 0,
            "chunk_duration_ms": 1000,
        })

        (sd / "chunks" / "chunk_0001.wav").write_bytes(make_wav_bytes(duration_s=1.0))
        register_chunk(sid, 1, {
            "chunk_index": 1,
            "chunk_started_ms": 2000,
            "chunk_duration_ms": 1000,
        })

        timeline = build_session_timeline(sid)
        assert timeline["integrity"]["has_gaps"] is True
        assert timeline["integrity"]["total_gap_ms"] == 1000
        assert timeline["total_duration_ms"] == 3000

        output = str(sd / "normalized" / "audio.wav")
        result = render_session_timeline_audio(sid, output, timeline=timeline)
        assert result["success"] is True
        assert result["normalized_duration_ms"] == 3000

        with wave.open(output, "rb") as wf:
            duration_ms = int(round(wf.getnframes() / wf.getframerate() * 1000))
        assert duration_ms == 3000


class TestAcousticTriage:
    """Test acoustic triage."""

    def test_speech_island_merge(self):
        from app.pipeline.acoustic_triage import build_speech_islands
        regions = [
            {"start_ms": 0, "end_ms": 1000, "tag": "speech"},
            {"start_ms": 1000, "end_ms": 2000, "tag": "speech"},
            {"start_ms": 5000, "end_ms": 6000, "tag": "speech"},
        ]
        islands = build_speech_islands(regions, merge_gap_ms=2000)
        # First two should merge (gap=0 < 2000ms)
        assert len(islands) >= 1
        assert islands[0]["start_ms"] == 0

    def test_non_speech_excluded(self):
        from app.pipeline.acoustic_triage import build_speech_islands
        regions = [
            {"start_ms": 0, "end_ms": 1000, "tag": "non_speech"},
            {"start_ms": 1000, "end_ms": 2000, "tag": "noise"},
        ]
        islands = build_speech_islands(regions)
        assert len(islands) == 0


class TestEndToEndIntegration:
    """Integration test: chunk upload -> finalize -> processing -> retrieval."""

    def test_session_lifecycle(self, tmp_sessions_dir, make_wav_bytes):
        from app.storage.session_store import (
            create_session, get_session_meta, register_chunk,
            update_session_meta, session_dir, get_status,
        )

        # Create session
        result = create_session({"mode": "stream"})
        sid = result["session_id"]
        assert sid

        # Upload chunks
        sd = session_dir(sid)
        for i in range(2):
            wav_data = make_wav_bytes(duration_s=1.0)
            chunk_path = sd / "chunks" / f"chunk_{i:04d}.wav"
            chunk_path.write_bytes(wav_data)
            count = register_chunk(sid, i, {
                "chunk_index": i,
                "chunk_started_ms": i * 30000,
                "chunk_duration_ms": 30000,
            })

        meta = get_session_meta(sid)
        assert meta["state"] == "receiving"
        assert len(meta["chunks"]) == 2

        # Finalize
        update_session_meta(sid, {"state": "finalized"})
        meta = get_session_meta(sid)
        assert meta["state"] == "finalized"

    def test_bridge_window_across_boundary(self):
        """Bridge windows must be created at chunk boundaries."""
        from app.pipeline.decode_lattice import build_decode_windows
        # 2 chunks x 30s = 60s total
        windows = build_decode_windows(60000)
        bridge_at_30s = [w for w in windows
                         if w["start_ms"] < 30000 < w["end_ms"]
                         and w["window_type"] == "bridge"]
        assert len(bridge_at_30s) >= 1, "Must have bridge window spanning the 30s chunk boundary"
