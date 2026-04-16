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
    """Verify the current canonical stage list."""

    def test_exact_list(self):
        from app.pipeline.run import CANONICAL_V1_STAGES
        expected = [
            "normalize_audio",
            "acoustic_triage",
            "decode_lattice",
            "first_pass_medium",
            "candidate_asr_large_v3",
            "candidate_asr_secondary",
            "stripe_grouping",
            "reconciliation",
            "canonical_assembly",
            "selective_enrichment",
            "semantic_marking",
            "memory_graph_update",
            "derived_outputs",
            "nosql_projection",
            "thread_linking",
        ]
        assert CANONICAL_V1_STAGES == expected

    def test_count(self):
        from app.pipeline.run import CANONICAL_V1_STAGES
        assert len(CANONICAL_V1_STAGES) == 15

    def test_memory_graph_update_follows_semantic_marking(self):
        """Memory graph update must sit between semantic marking and derived outputs."""
        from app.pipeline.run import CANONICAL_V1_STAGES
        idx = CANONICAL_V1_STAGES.index("memory_graph_update")
        assert CANONICAL_V1_STAGES[idx - 1] == "semantic_marking"
        assert CANONICAL_V1_STAGES[idx + 1] == "derived_outputs"

    def test_nosql_projection_follows_derived_outputs(self):
        """NoSQL projection must follow derived_outputs."""
        from app.pipeline.run import CANONICAL_V1_STAGES
        idx = CANONICAL_V1_STAGES.index("nosql_projection")
        assert CANONICAL_V1_STAGES[idx - 1] == "derived_outputs"
        assert CANONICAL_V1_STAGES[idx + 1] == "thread_linking"

    def test_thread_linking_is_last_stage(self):
        """Thread linking must be the final stage."""
        from app.pipeline.run import CANONICAL_V1_STAGES
        assert CANONICAL_V1_STAGES[-1] == "thread_linking"

    def test_no_turbo_in_stages(self):
        """Turbo must NOT be in the canonical stage list."""
        from app.pipeline.run import CANONICAL_V1_STAGES
        assert "candidate_asr_turbo_hf" not in CANONICAL_V1_STAGES

    def test_secondary_stage_replaces_model_branded_name(self):
        """The pipeline stage must describe the role, not the model brand."""
        from app.pipeline.run import CANONICAL_V1_STAGES
        assert "candidate_asr_secondary" in CANONICAL_V1_STAGES

    def test_stage_alias_maps_legacy_parakeet_name(self):
        from app.pipeline.run import normalize_stage_name

        assert normalize_stage_name("candidate_asr_parakeet") == "candidate_asr_secondary"

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

    def test_live_decode_windows_exclude_trailing_partial_window(self):
        from app.pipeline.decode_lattice import build_decode_windows

        windows = build_decode_windows(150000, allow_trailing_partial_window=False)

        assert all((w["end_ms"] - w["start_ms"]) == 30000 for w in windows)
        assert windows[-1]["start_ms"] == 120000
        assert windows[-1]["end_ms"] == 150000


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

    def test_dedup_join_drops_contained_duplicate(self):
        from app.pipeline.canonical_assembly import _dedup_join

        text_a = "the weather today is cloudy with light rain expected later this afternoon"
        text_b = "weather today is cloudy with light rain expected later this afternoon"

        result = _dedup_join(text_a, text_b)
        assert result == text_a

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

    def test_live_boundary_keeps_trailing_single_support_provisional(self):
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

        result = stabilize_stripes(records, packets, finalize_last_boundary=False)
        assert [row["stabilization_state"] for row in result] == ["stabilized", "provisional"]

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

    def test_live_transcript_surfaces_do_not_emit_final_surface(self, tmp_sessions_dir):
        from app.pipeline.canonical_assembly import build_transcript_surfaces
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        segments = [
            {
                "segment_id": "seg_000000",
                "start_ms": 0,
                "end_ms": 15000,
                "speaker": None,
                "text": "alpha",
                "language": "en",
                "confidence": 0.8,
                "support_windows": ["W000000"],
                "support_models": ["faster-whisper:medium"],
                "stabilization_state": "stabilized",
                "corruption_flags": [],
                "stripes": ["S0"],
                "assembly_decisions": [],
            },
            {
                "segment_id": "seg_000001",
                "start_ms": 15000,
                "end_ms": 30000,
                "speaker": None,
                "text": "omega",
                "language": "en",
                "confidence": 0.8,
                "support_windows": ["W000001"],
                "support_models": ["faster-whisper:medium"],
                "stabilization_state": "provisional",
                "corruption_flags": ["single_window_support"],
                "stripes": ["S1"],
                "assembly_decisions": [],
            },
        ]

        surfaces = build_transcript_surfaces(segments, sid, emit_final_surface=False)
        assert surfaces["final_transcript"] is None
        assert not (sd / "canonical" / "final_transcript.json").exists()
        assert surfaces["stabilized_partial"]["text"] == "alpha"


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

    def test_candidate_b_defaults_to_parakeet_for_auto_sessions(self, monkeypatch):
        from types import SimpleNamespace
        from app.models.registry import DEFAULT_CANDIDATE_B
        from app.workers import worker

        monkeypatch.setattr(
            worker,
            "get_registry",
            lambda refresh=True: {
                DEFAULT_CANDIDATE_B: SimpleNamespace(is_usable=True),
                "nemo-asr:canary-1b-v2": SimpleNamespace(is_usable=True),
            },
        )
        monkeypatch.setattr(
            worker,
            "_first_pass_language_evidence",
            lambda session_id: {"success_count": 0, "language_counts": {}, "dominant_language": None},
        )

        model_id, reason = worker._select_candidate_b_model(
            {
                "allowed_languages": [],
                "forced_language": None,
                "transcription_mode": "verbatim_multilingual",
            },
            session_id="sess-auto",
        )

        assert model_id == DEFAULT_CANDIDATE_B
        assert reason == "auto_session_defaulted_to_parakeet_pending_language_evidence"

    def test_candidate_b_uses_canary_when_first_pass_detects_non_english(self, monkeypatch):
        from types import SimpleNamespace
        from app.models.registry import DEFAULT_CANDIDATE_B
        from app.workers import worker

        monkeypatch.setattr(
            worker,
            "get_registry",
            lambda refresh=True: {
                DEFAULT_CANDIDATE_B: SimpleNamespace(is_usable=True),
                "nemo-asr:canary-1b-v2": SimpleNamespace(is_usable=True),
            },
        )
        monkeypatch.setattr(
            worker,
            "_first_pass_language_evidence",
            lambda session_id: {"success_count": 2, "language_counts": {"fr": 2}, "dominant_language": "fr"},
        )

        model_id, reason = worker._select_candidate_b_model(
            {
                "allowed_languages": [],
                "forced_language": None,
                "transcription_mode": "verbatim_multilingual",
            },
            session_id="sess-fr",
        )

        assert model_id == "nemo-asr:canary-1b-v2"
        assert reason == "first_pass_detected_fr"

    def test_parakeet_routing_forces_english_execution_context(self):
        from app.models.registry import DEFAULT_CANDIDATE_B
        from app.workers.worker import _candidate_b_execution_language_ctx

        effective = _candidate_b_execution_language_ctx(
            DEFAULT_CANDIDATE_B,
            "first_pass_detected_english",
            {
                "allowed_languages": [],
                "forced_language": None,
                "requested_language": None,
                "transcription_mode": "verbatim_multilingual",
            },
        )

        assert effective["language"] == "en"
        assert effective["allowed_languages"] == ["en"]
        assert effective["forced_language"] == "en"

    def test_faster_whisper_retries_on_cpu_after_cuda_runtime_failure(self, monkeypatch):
        import sys
        from types import SimpleNamespace
        from app.pipeline import asr_executor

        class FakeSegment:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        class FakeInfo:
            language = "en"
            language_probability = 0.99

        class FakeWhisperModel:
            def __init__(self, model_size, device, compute_type, device_index=0):
                self._model_size = model_size
                self._device = device

            def transcribe(self, audio_path, **kwargs):
                if self._device == "cuda":
                    raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
                return iter([FakeSegment(0.0, 1.0, "hello world")]), FakeInfo()

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: None,
                synchronize=lambda: None,
                ipc_collect=lambda: None,
                reset_peak_memory_stats=lambda: None,
                device_count=lambda: 1,
                get_device_name=lambda idx=0: "FakeGPU",
            ),
            version=SimpleNamespace(cuda="12.8"),
            backends=SimpleNamespace(cuda=SimpleNamespace(is_built=lambda: True)),
        )

        asr_executor.unload_all_models()
        asr_executor._faster_whisper_runtime_preferences.clear()
        monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
        monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 1))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        # Ensure strict_cuda is off so CPU fallback is allowed
        monkeypatch.setenv("WHISPER_STRICT_CUDA", "0")
        monkeypatch.setenv("STRICT_CUDA", "false")
        from app.core.config import reset_config
        reset_config()

        result = asr_executor._transcribe_faster_whisper("dummy.wav", "medium", "en")

        assert result["success"] is True
        assert result["text"] == "hello world"
        # CPU fallback succeeded — the function tried CUDA, failed, fell back to CPU
        reset_config()  # restore

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


class TestLexicalTruthContract:
    def test_reconcile_stripe_emits_richer_lexical_contract(self):
        from app.pipeline.reconciliation import reconcile_stripe

        stripe_packet = {
            "stripe_id": "S0001",
            "start_ms": 15000,
            "end_ms": 30000,
            "support_window_count": 2,
            "support_windows": ["W000001", "W000002"],
            "support_models": ["faster-whisper:large-v3"],
            "evidence": [
                {
                    "candidate_id": "cand_alpha",
                    "model_id": "faster-whisper:large-v3",
                    "window_id": "W000001",
                    "text": "bonjour le monde",
                    "trust_score": 0.9,
                    "language_evidence": {"detected_language": "fr"},
                }
            ],
        }

        record = reconcile_stripe(stripe_packet, llm=None)

        assert record["final_text"] == "bonjour le monde"
        assert record["chosen_text"] == "bonjour le monde"
        assert record["assembly_mode"] == "deterministic_fallback"
        assert record["used_candidate_ids"] == ["cand_alpha"]
        assert record["unsupported_tokens"] == []
        assert record["token_support_ratio"] == 1.0
        assert record["source_language"] == "fr"
        assert record["output_language"] == "fr"
        assert record["validation_status"].startswith("accepted")

    def test_run_reconciliation_writes_v2_side_artifacts(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.reconciliation import run_reconciliation

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        stripe_packets = [{
            "stripe_id": "S0000",
            "start_ms": 0,
            "end_ms": 15000,
            "support_window_count": 1,
            "support_windows": ["W000000"],
            "support_models": ["faster-whisper:medium"],
            "evidence": [{
                "candidate_id": "cand_medium",
                "model_id": "faster-whisper:medium",
                "window_id": "W000000",
                "text": "hello world",
                "trust_score": 0.8,
                "language_evidence": {"detected_language": "en"},
            }],
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        stage = StageStub()
        result = run_reconciliation(sid, stripe_packets, stage)

        assert (sd / "reconciliation" / "reconciliation_result.json").is_file()
        assert (sd / "reconciliation" / "lexical_synthesis_result.json").is_file()
        assert (sd / "reconciliation" / "validation_audit.json").is_file()
        assert "lexical_synthesis_result.json" in stage.artifacts
        assert result["records"][0]["final_text"] == "hello world"


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
        assert report["analysis_version"] == "1.2"
        assert report["segment_count"] == 1
        assert "issues" in report
        assert "reading_text" in report

    def test_retrieval_index_grounded_on_segments(self, tmp_sessions_dir):
        import json
        from app.core.atomic_io import atomic_write_json
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

        atomic_write_json(str(sd / "enrichment" / "segment_markers.json"), {
            "session_id": sid,
            "marker_count": 1,
            "markers": [{
                "segment_id": "seg_000001",
                "entity_mentions": [{"surface_form": "Bonjour", "entity_id": None, "mention_type": "capitalized_phrase", "confidence": 0.5}],
                "topic_tags": ["meeting"],
                "relation_tags": [],
                "project_tags": ["Orion"],
                "emotion_tags": [],
                "retrieval_terms": ["bonjour", "orion", "meeting"],
                "ambiguity_flags": [],
                "marker_confidence": 0.6,
                "marker_source": "heuristic_segment_text",
            }],
        })

        build_derived_dir(sid, segments, "Bonjour tout le monde", 5000)
        retrieval_path = sd / "derived" / "retrieval_index.json"
        retrieval_v2_path = sd / "derived" / "retrieval_index_v2.json"
        retrieval_v3_path = sd / "derived" / "retrieval_index_v3.json"
        assert retrieval_path.is_file()
        assert retrieval_v2_path.is_file()
        assert retrieval_v3_path.is_file()

        payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
        assert payload["entry_count"] == 1
        assert payload["entries"][0]["grounding"]["segment_id"] == "seg_000001"
        assert payload["entries"][0]["grounding"]["canonical_path"] == "canonical/canonical_segments.json"
        assert payload["entries"][0]["grounding"]["markers_path"] == "enrichment/segment_markers.json"
        assert payload["entries"][0]["project_tags"] == ["Orion"]

        payload_v2 = json.loads(retrieval_v2_path.read_text(encoding="utf-8"))
        assert payload_v2["version"] == 2

    def test_retrieval_v3_grounded_on_context_spans(self, tmp_sessions_dir):
        import json
        from app.core.atomic_io import atomic_write_json
        from app.pipeline.derived_outputs import build_derived_dir
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        segments = [
            {
                "segment_id": "seg_000001",
                "start_ms": 0,
                "end_ms": 4000,
                "speaker": "SPEAKER_00",
                "text": "Need to wire money to Omar",
                "language": "en",
                "confidence": 0.9,
                "support_windows": ["W0"],
                "support_models": ["faster-whisper:large-v3"],
                "stabilization_state": "stabilized",
            },
            {
                "segment_id": "seg_000002",
                "start_ms": 4500,
                "end_ms": 8500,
                "speaker": "SPEAKER_00",
                "text": "Bank transfer tomorrow morning",
                "language": "en",
                "confidence": 0.9,
                "support_windows": ["W1"],
                "support_models": ["faster-whisper:large-v3"],
                "stabilization_state": "stabilized",
            },
        ]

        atomic_write_json(str(sd / "enrichment" / "segment_markers.json"), {
            "session_id": sid,
            "marker_count": 2,
            "markers": [
                {
                    "segment_id": "seg_000001",
                    "entity_mentions": [{"surface_form": "Omar", "entity_id": "person_omar", "mention_type": "curated_pack"}],
                    "topic_tags": ["relationship"],
                    "topic_candidates": ["money_help"],
                    "relation_tags": ["partner_support"],
                    "project_tags": [],
                    "emotion_tags": [],
                    "retrieval_terms": ["omar", "wire money"],
                    "ambiguity_flags": [],
                    "marker_confidence": 0.8,
                },
                {
                    "segment_id": "seg_000002",
                    "entity_mentions": [],
                    "topic_tags": ["finance"],
                    "topic_candidates": ["money_help"],
                    "relation_tags": [],
                    "project_tags": [],
                    "emotion_tags": [],
                    "retrieval_terms": ["bank transfer"],
                    "ambiguity_flags": [],
                    "marker_confidence": 0.8,
                },
            ],
        })
        atomic_write_json(str(sd / "enrichment" / "context_spans.json"), {
            "session_id": sid,
            "span_count": 1,
            "spans": [{
                "context_id": "ctx_000001",
                "session_id": sid,
                "start_ms": 0,
                "end_ms": 8500,
                "segment_ids": ["seg_000001", "seg_000002"],
                "segment_count": 2,
                "speaker_ids": ["SPEAKER_00"],
                "language_profile": {"primary": "en", "ratio": 1.0, "languages": {"en": 2}},
                "topic_tags": ["relationship", "finance"],
                "topic_candidates": ["money_help"],
                "entity_ids": ["person_omar"],
                "alias_hits": [{"entity_id": "person_omar", "surface_form": "Omar"}],
                "confidence": 0.82,
            }],
        })

        build_derived_dir(sid, segments, "Need to wire money to Omar Bank transfer tomorrow morning", 9000)
        retrieval_v3 = json.loads((sd / "derived" / "retrieval_index_v3.json").read_text(encoding="utf-8"))
        assert retrieval_v3["version"] == 3
        assert retrieval_v3["entry_count"] == 1
        entry = retrieval_v3["entries"][0]
        assert entry["context_id"] == "ctx_000001"
        assert entry["segment_ids"] == ["seg_000001", "seg_000002"]
        assert "money_help" in entry["topic_candidates"]
        assert "person_omar" in entry["entity_ids"]


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


class TestSemanticMarking:
    def test_semantic_marking_writes_grounded_segment_markers(self, tmp_sessions_dir):
        import json
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.semantic_marking import run_semantic_marking

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        segments = [{
            "segment_id": "seg_000010",
            "start_ms": 0,
            "end_ms": 5000,
            "speaker": "SPEAKER_00",
            "text": "Project Orion meeting with Alice about the release branch",
            "language": "en",
            "confidence": 0.91,
            "support_windows": ["W000000"],
            "support_models": ["faster-whisper:large-v3"],
            "stabilization_state": "stabilized",
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        stage = StageStub()
        result = run_semantic_marking(sid, segments, stage)

        marker_path = sd / "enrichment" / "segment_markers.json"
        assert marker_path.is_file()
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        marker = payload["markers"][0]

        assert result["marker_count"] == 1
        assert "segment_markers.json" in stage.artifacts
        assert marker["segment_id"] == "seg_000010"
        assert marker["project_tags"] == ["Orion"]
        assert "work" in marker["topic_tags"]
        assert marker["retrieval_terms"]


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


# ============================================================
# Misconception repair tests — canonical spec V2
# ============================================================

class TestWitnessDiagnostics:
    """Canonical spec 9.6: every candidate must carry local witness diagnostics."""

    def test_language_mismatch_flag(self):
        from app.pipeline.witness_diagnostics import compute_candidate_flags
        result = compute_candidate_flags(
            raw_text="bonjour tout le monde",
            detected_language="fr",
            requested_language="ru",
            transcription_mode="verbatim_multilingual",
            success=True,
            degraded=False,
            duration_s=15.0,
        )
        assert "language_mismatch" in result["candidate_flags"]

    def test_possible_translation_cyrillic_to_latin(self):
        """If a Russian session returned Latin text, flag possible translation drift."""
        from app.pipeline.witness_diagnostics import compute_candidate_flags
        result = compute_candidate_flags(
            raw_text="hello world",
            detected_language="ru",
            requested_language="ru",
            transcription_mode="verbatim_multilingual",
            success=True,
            degraded=False,
            duration_s=15.0,
        )
        assert "possible_translation" in result["candidate_flags"]
        assert "script_mismatch" in result["candidate_flags"]

    def test_edge_truncation_detected(self):
        from app.pipeline.witness_diagnostics import compute_candidate_flags
        result = compute_candidate_flags(
            raw_text="ld today was great and the weather was ni",
            detected_language="en",
            requested_language="en",
            transcription_mode="verbatim_multilingual",
            success=True,
            degraded=False,
            duration_s=15.0,
        )
        assert "edge_truncation_suspected" in result["candidate_flags"]

    def test_empty_candidate_flag(self):
        from app.pipeline.witness_diagnostics import compute_candidate_flags
        result = compute_candidate_flags(
            raw_text="",
            detected_language=None,
            requested_language="en",
            transcription_mode="verbatim_multilingual",
            success=False,
            degraded=True,
            duration_s=15.0,
        )
        assert "empty_candidate" in result["candidate_flags"]
        assert "provider_degraded" in result["candidate_flags"]

    def test_repetition_anomaly(self):
        from app.pipeline.witness_diagnostics import compute_candidate_flags
        result = compute_candidate_flags(
            raw_text="thank you thank you thank you thank you thank you",
            detected_language="en",
            requested_language="en",
            transcription_mode="verbatim_multilingual",
            success=True,
            degraded=False,
            duration_s=15.0,
        )
        assert "repetition_anomaly" in result["candidate_flags"]

    def test_clean_candidate_has_no_flags(self):
        from app.pipeline.witness_diagnostics import compute_candidate_flags
        result = compute_candidate_flags(
            raw_text="Привет мир как у тебя дела сегодня",
            detected_language="ru",
            requested_language="ru",
            transcription_mode="verbatim_multilingual",
            success=True,
            degraded=False,
            duration_s=15.0,
        )
        assert result["candidate_flags"] == []

    def test_persist_candidate_emits_flags(self, tmp_sessions_dir):
        from app.pipeline.asr_executor import persist_candidate
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        window = {"window_id": "W000000", "start_ms": 0, "end_ms": 30000, "window_type": "full"}
        result = {
            "text": "hello world",
            "segments": [],
            "detection_info": {
                "detected_language": "en",
                "requested_language": "ru",
            },
            "success": True,
            "degraded": False,
            "transcription_mode": "verbatim_multilingual",
            "segment_timestamp_unit": "seconds",
        }

        candidate = persist_candidate(sid, window, "faster-whisper:medium", result)
        assert "candidate_flags" in candidate
        assert "language_mismatch" in candidate["candidate_flags"]
        assert candidate["decode_metadata"]["requested_language"] == "ru"

    def test_persist_candidate_is_deterministic_per_window_and_model(self, tmp_sessions_dir):
        from app.pipeline.asr_executor import persist_candidate
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        window = {"window_id": "W000123", "start_ms": 0, "end_ms": 30000, "window_type": "full"}
        result = {
            "text": "first text",
            "segments": [],
            "detection_info": {"detected_language": "en", "requested_language": "en"},
            "success": True,
            "degraded": False,
            "transcription_mode": "verbatim_multilingual",
            "segment_timestamp_unit": "seconds",
        }

        first = persist_candidate(sid, window, "faster-whisper:medium", result)
        result["text"] = "second text"
        second = persist_candidate(sid, window, "faster-whisper:medium", result)

        candidate_files = list((sd / "candidates").glob("cand_*.json"))
        assert len(candidate_files) == 1
        assert first["candidate_id"] == second["candidate_id"]
        persisted = candidate_files[0].read_text(encoding="utf-8")
        assert "second text" in persisted


class TestReconciliationAuditRates:
    """Misconception 5: run must expose LLM usage / fallback rates explicitly."""

    def test_validation_audit_reports_usage_rates(self, tmp_sessions_dir):
        import json
        from app.pipeline.reconciliation import run_reconciliation
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        stripe_packets = [
            {
                "stripe_id": f"S{idx:04d}",
                "start_ms": idx * 15000,
                "end_ms": (idx + 1) * 15000,
                "support_window_count": 2 if idx > 0 else 1,
                "support_windows": [f"W{idx:06d}"],
                "support_models": ["faster-whisper:medium"],
                "evidence": [{
                    "candidate_id": f"cand_{idx}",
                    "model_id": "faster-whisper:medium",
                    "window_id": f"W{idx:06d}",
                    "text": f"sample text {idx}",
                    "trust_score": 0.7,
                    "candidate_flags": ["single_window_support"] if idx == 0 else [],
                    "language_evidence": {"detected_language": "en"},
                }],
            }
            for idx in range(3)
        ]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_reconciliation(sid, stripe_packets, StageStub())
        audit = json.loads((sd / "reconciliation" / "validation_audit.json").read_text(encoding="utf-8"))

        assert "usage_rates" in audit
        rates = audit["usage_rates"]
        assert rates["fallback_resolved_rate"] == 1.0  # no LLM present
        assert rates["llm_resolved_rate"] == 0.0
        assert "avg_support_window_count" in rates
        assert "witness_flag_counts" in audit

    def test_adversarial_stripe_disagreement_is_surfaced(self, tmp_sessions_dir):
        """Canonical reconciliation must audit disagreement, not hide it."""
        import json
        from app.pipeline.reconciliation import run_reconciliation
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        stripe_packets = [{
            "stripe_id": "S0007",
            "start_ms": 105000,
            "end_ms": 120000,
            "support_window_count": 2,
            "support_windows": ["W000008", "W000009"],
            "support_models": ["faster-whisper:large-v3", "faster-whisper:medium"],
            "evidence": [
                {
                    "candidate_id": "cand_a",
                    "model_id": "faster-whisper:large-v3",
                    "window_id": "W000008",
                    "text": "я опять думал про наркомана из Бангкока",
                    "trust_score": 0.9,
                    "language_evidence": {"detected_language": "ru"},
                    "candidate_flags": [],
                },
                {
                    "candidate_id": "cand_b",
                    "model_id": "faster-whisper:medium",
                    "window_id": "W000009",
                    "text": "I was thinking about Matthew from Bangkok",
                    "trust_score": 0.6,
                    "language_evidence": {"detected_language": "en"},
                    "candidate_flags": ["possible_translation", "language_mismatch"],
                },
            ],
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_reconciliation(sid, stripe_packets, StageStub())
        audit = json.loads((sd / "reconciliation" / "validation_audit.json").read_text(encoding="utf-8"))

        assert audit["disagreement_stripe_count"] == 1
        assert audit["witness_flag_counts"].get("possible_translation", 0) >= 1
        assert audit["witness_flag_counts"].get("language_mismatch", 0) >= 1


class TestCanonicalOverMergeGuard:
    """Misconception 7: canonical segments must not collapse into multi-minute blobs."""

    def test_segments_are_capped_at_max_segment_ms(self):
        from app.pipeline.canonical_assembly import merge_into_segments, MAX_SEGMENT_MS

        # 10 contiguous 15s stripes with identical source+language.  Without the
        # cap they would collapse into a single 150s segment.
        records = [
            {
                "stripe_id": f"S{i:04d}",
                "start_ms": i * 15000,
                "end_ms": (i + 1) * 15000,
                "chosen_text": f"word_{i}",
                "chosen_source": "faster-whisper:large-v3",
                "language": "en",
                "output_language": "en",
                "confidence": 0.8,
                "stabilization_state": "stabilized",
                "support_window_count": 2,
            }
            for i in range(10)
        ]
        segments = merge_into_segments(records)
        assert len(segments) >= 2, "Over-merge guard must split long runs"
        for seg in segments:
            assert (seg["end_ms"] - seg["start_ms"]) <= MAX_SEGMENT_MS

    def test_corruption_flags_propagate_from_stripes(self):
        from app.pipeline.canonical_assembly import merge_into_segments

        records = [{
            "stripe_id": "S0000",
            "start_ms": 0,
            "end_ms": 15000,
            "chosen_text": "weak text",
            "chosen_source": "faster-whisper:large-v3",
            "language": "en",
            "output_language": "en",
            "confidence": 0.4,
            "stabilization_state": "stabilized",
            "support_window_count": 1,
            "uncertainty_flags": ["low_confidence", "single_window_support"],
            "unsupported_tokens": ["weak"],
            "validation_status": "accepted_with_warnings",
            "llm_validation_rejected": True,
        }]
        segments = merge_into_segments(records)
        assert segments, "expected at least one segment"
        flags = set(segments[0]["corruption_flags"])
        assert "low_confidence" in flags
        assert "unsupported_tokens_present" in flags
        assert "validator_warning" in flags
        assert "llm_selection_rejected" in flags


class TestSemanticSpansArtifact:
    """Misconception 8: retrieval needs finer-grained semantic access paths."""

    def test_semantic_spans_artifact_emitted(self, tmp_sessions_dir):
        import json
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.semantic_marking import run_semantic_marking

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        segments = [
            {
                "segment_id": f"seg_{i:06d}",
                "start_ms": i * 30000,
                "end_ms": (i + 1) * 30000,
                "speaker": "SPEAKER_00",
                "text": "Project Orion stand-up meeting with Alice about release branch",
                "language": "en",
                "confidence": 0.9,
                "support_windows": [f"W{i:06d}"],
                "support_models": ["faster-whisper:large-v3"],
                "stabilization_state": "stabilized",
            }
            for i in range(3)
        ]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        stage = StageStub()
        result = run_semantic_marking(sid, segments, stage)

        assert (sd / "enrichment" / "semantic_spans.json").is_file()
        assert (sd / "enrichment" / "marker_audit.json").is_file()
        assert "semantic_spans.json" in stage.artifacts
        assert "marker_audit.json" in stage.artifacts
        spans = json.loads((sd / "enrichment" / "semantic_spans.json").read_text(encoding="utf-8"))
        assert spans["span_count"] >= 1
        assert spans["spans"][0]["segment_count"] >= 1
        assert result["semantic_span_count"] == spans["span_count"]


class TestMemoryGraphArtifacts:
    """Misconceptions 1, 8, 9: the V2 direction requires a real memory layer."""

    def test_memory_artifacts_created(self, tmp_sessions_dir):
        import json
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.memory_graph import run_memory_graph_update

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        markers = [{
            "segment_id": "seg_000001",
            "entity_mentions": [
                {"surface_form": "Mattiew", "mention_type": "capitalized_phrase", "confidence": 0.6, "entity_id": None},
                {"surface_form": "Dynatrace", "mention_type": "capitalized_phrase", "confidence": 0.6, "entity_id": None},
            ],
            "topic_tags": ["relationship"],
            "project_tags": ["WebLogic"],
            "relation_tags": [],
            "emotion_tags": [],
            "retrieval_terms": ["mattiew", "dynatrace"],
            "ambiguity_flags": [],
            "marker_confidence": 0.6,
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        stage = StageStub()
        result = run_memory_graph_update(sid, markers, stage)

        memory_dir = sd / "memory"
        assert (memory_dir / "entity_registry.json").is_file()
        assert (memory_dir / "alias_graph.json").is_file()
        assert (memory_dir / "graph_updates.json").is_file()
        assert (memory_dir / "context_pack_summary.json").is_file()
        assert (memory_dir / "context_packs" / "session_auto_context.json").is_file()

        registry = json.loads((memory_dir / "entity_registry.json").read_text(encoding="utf-8"))
        assert registry["entity_count"] >= 2
        assert any(e["display_name"] == "Mattiew" for e in registry["entities"])
        assert result["entity_count"] >= 2

    def test_curated_pack_merges_and_overrides_inferred(self, tmp_sessions_dir):
        import json
        from app.core.atomic_io import atomic_write_json
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.memory_graph import run_memory_graph_update

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        curated_dir = sd / "memory" / "curated_packs"
        curated_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(curated_dir / "relationships.json"), {
            "pack_id": "relationships",
            "version": 3,
            "origin": "curated_human",
            "entities": [{
                "entity_id": "person_omar",
                "entity_type": "person",
                "display_name": "Omar",
                "aliases": ["Omar", "мой муж", "my husband"],
                "roles": ["spouse"],
                "confidence": 0.95,
                "status": "active",
            }],
        })

        markers = [{
            "segment_id": "seg_000001",
            "entity_mentions": [
                {"surface_form": "мой муж", "mention_type": "capitalized_phrase", "confidence": 0.4, "entity_id": None},
            ],
            "topic_tags": ["relationship"],
            "retrieval_terms": ["мой муж"],
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_memory_graph_update(sid, markers, StageStub())
        registry = json.loads((sd / "memory" / "entity_registry.json").read_text(encoding="utf-8"))
        person_omar = next(
            (e for e in registry["entities"] if e["entity_id"] == "person_omar"),
            None,
        )
        assert person_omar is not None
        assert "мой муж" in person_omar["aliases"]
        assert person_omar["origin"] == "curated_pack"
        assert person_omar["confidence"] >= 0.9  # curated always dominates inferred

    def test_alias_graph_marks_ambiguous_aliases(self, tmp_sessions_dir):
        import json
        from app.core.atomic_io import atomic_write_json
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.memory_graph import run_memory_graph_update

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        curated_dir = sd / "memory" / "curated_packs"
        curated_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(curated_dir / "people.json"), {
            "pack_id": "people",
            "entities": [
                {"entity_id": "person_a", "display_name": "Alex", "aliases": ["Alex"]},
                {"entity_id": "person_b", "display_name": "Alex", "aliases": ["Alex"]},
            ],
        })

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_memory_graph_update(sid, [], StageStub())
        alias_graph = json.loads((sd / "memory" / "alias_graph.json").read_text(encoding="utf-8"))
        alex_entry = next(entry for entry in alias_graph["aliases"] if entry["alias"] == "alex")
        assert alex_entry["ambiguous"] is True
        assert set(alex_entry["entity_ids"]) == {"person_a", "person_b"}


class TestStripeEvidencePropagatesWitnessFlags:
    """Misconception 6: per-stripe witness diagnostics must actually reach stripe packets."""

    def test_evidence_entries_carry_candidate_flags(self, tmp_sessions_dir):
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
        atomic_write_json(str(sd / "candidates" / "cand_medium.json"), {
            "candidate_id": "cand_medium",
            "session_id": sid,
            "model_id": "faster-whisper:medium",
            "window_id": "W000000",
            "window_start_ms": 0,
            "window_end_ms": 30000,
            "window_type": "full",
            "raw_text": "hello world",
            "segments": [{"start": 0.0, "end": 3.0, "text": "hello world"}],
            "language_evidence": {"detected_language": "en", "requested_language": "ru"},
            "confidence_features": {"success": True, "degraded": False},
            "candidate_flags": ["language_mismatch", "possible_translation"],
            "witness_audit": {"script": "latin", "expected_script": "cyrillic"},
            "decode_metadata": {"segment_timestamp_unit": "seconds"},
        })

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        result = run_stripe_grouping(sid, [window], 30000, StageStub())
        evidence = result["stripes"][0]["evidence"][0]
        assert "language_mismatch" in evidence["candidate_flags"]
        assert evidence["witness_audit"]["script"] == "latin"


class TestStreamingContract:
    """Misconception 10: provisional vs stabilized vs final must remain distinct."""

    def test_three_surfaces_are_published_separately(self, tmp_sessions_dir):
        import json
        from app.pipeline.canonical_assembly import (
            build_transcript_surfaces,
            merge_into_segments,
            stabilize_stripes,
        )
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        records = [
            {
                "stripe_id": "S0",
                "start_ms": 0,
                "end_ms": 15000,
                "chosen_text": "hello",
                "chosen_source": "faster-whisper:large-v3",
                "confidence": 0.9,
                "language": "en",
                "output_language": "en",
            },
            # Interior stripe supported by a single window — should remain
            # provisional and must NOT appear in final_transcript.
            {
                "stripe_id": "S1",
                "start_ms": 15000,
                "end_ms": 30000,
                "chosen_text": "there",
                "chosen_source": "faster-whisper:medium",
                "confidence": 0.5,
                "language": "en",
                "output_language": "en",
            },
            {
                "stripe_id": "S2",
                "start_ms": 30000,
                "end_ms": 45000,
                "chosen_text": "world",
                "chosen_source": "faster-whisper:large-v3",
                "confidence": 0.88,
                "language": "en",
                "output_language": "en",
            },
        ]
        packets = [
            {"stripe_id": "S0", "support_window_count": 2},
            {"stripe_id": "S1", "support_window_count": 1},
            {"stripe_id": "S2", "support_window_count": 2},
        ]
        stabilized = stabilize_stripes(records, packets)
        segments = merge_into_segments(stabilized)
        build_transcript_surfaces(segments, sid, stripe_decisions=stabilized)

        canonical_dir = sd / "canonical"
        provisional = json.loads((canonical_dir / "provisional_partial.json").read_text(encoding="utf-8"))
        stabilized_surface = json.loads((canonical_dir / "stabilized_partial.json").read_text(encoding="utf-8"))
        final = json.loads((canonical_dir / "final_transcript.json").read_text(encoding="utf-8"))

        assert provisional["semantic_layer"] == "provisional_partial"
        assert stabilized_surface["semantic_layer"] == "stabilized_partial"
        assert final["semantic_layer"] == "final_transcript"

        provisional_texts = {seg["text"] for seg in provisional["segments"]}
        final_texts = {seg["text"] for seg in final["segments"]}
        # Provisional is a superset: contains interior single-support stripe,
        # final must only contain stabilized content.
        assert "there" in " ".join(provisional_texts)
        assert "there" not in " ".join(final_texts)


class TestRetrievalV2Grounding:
    """Misconception 8: retrieval_index_v2 must carry markers, not just literal text."""

    def test_retrieval_entries_inherit_semantic_markers(self, tmp_sessions_dir):
        import json
        from app.core.atomic_io import atomic_write_json
        from app.pipeline.derived_outputs import build_derived_dir
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        atomic_write_json(str(sd / "enrichment" / "segment_markers.json"), {
            "session_id": sid,
            "marker_count": 1,
            "markers": [{
                "segment_id": "seg_000001",
                "entity_mentions": [
                    {"surface_form": "Mattiew", "entity_id": "person_mattiew",
                     "mention_type": "capitalized_phrase", "confidence": 0.8},
                ],
                "topic_tags": ["relationship"],
                "relation_tags": ["past_relationship"],
                "project_tags": [],
                "emotion_tags": ["anxiety"],
                "retrieval_terms": ["Mattiew", "наркоман из Бангкока"],
                "ambiguity_flags": [],
                "marker_confidence": 0.8,
            }],
        })

        segments = [{
            "segment_id": "seg_000001",
            "start_ms": 0,
            "end_ms": 5000,
            "text": "я опять думал про наркомана из Бангкока",
            "language": "ru",
            "confidence": 0.9,
            "support_windows": ["W0"],
            "support_models": ["faster-whisper:large-v3"],
            "stabilization_state": "stabilized",
        }]

        build_derived_dir(sid, segments, segments[0]["text"], 5000)
        retrieval_v2 = json.loads((sd / "derived" / "retrieval_index_v2.json").read_text(encoding="utf-8"))
        entry = retrieval_v2["entries"][0]
        assert retrieval_v2["version"] == 2
        assert "person_mattiew" in entry["entity_ids"]
        assert "Mattiew" in entry["aliases"] or "наркоман из Бангкока" in entry["aliases"]
        assert "past_relationship" in entry["relation_tags"]


class TestSelectiveDiarizationPolicies:
    """Misconception 9: diarization must be selective, not universal."""

    def test_auto_policy_skips_when_no_evidence(self):
        from app.pipeline.selective_enrichment import should_run_diarization
        assert should_run_diarization({}, [], policy="auto") is False

    def test_auto_policy_runs_when_speaker_count_hint(self):
        from app.pipeline.selective_enrichment import should_run_diarization
        assert should_run_diarization({"speaker_count": 2}, [], policy="auto") is True

    def test_forced_policy_overrides(self):
        from app.pipeline.selective_enrichment import should_run_diarization
        assert should_run_diarization({}, [], policy="forced") is True

    def test_off_policy_overrides_hint(self):
        from app.pipeline.selective_enrichment import should_run_diarization
        assert should_run_diarization({"speaker_count": 3}, [], policy="off") is False


class TestAcousticTriageNonSpeech:
    """Misconception 3: triage must separate non-speech regions."""

    def test_mixed_and_noise_regions_excluded(self):
        from app.pipeline.acoustic_triage import build_speech_islands
        regions = [
            {"start_ms": 0, "end_ms": 30000, "tag": "non_speech"},
            {"start_ms": 30000, "end_ms": 60000, "tag": "music_media"},
            {"start_ms": 60000, "end_ms": 65000, "tag": "speech"},
            {"start_ms": 65000, "end_ms": 80000, "tag": "noise"},
        ]
        islands = build_speech_islands(regions)
        # Only a single short speech island should remain.
        assert len(islands) == 1
        assert islands[0]["start_ms"] == 60000
        assert islands[0]["end_ms"] == 65000

    def test_decode_windows_skip_pure_music_regions(self):
        from app.pipeline.decode_lattice import build_decode_windows

        windows = build_decode_windows(
            120000,
            speech_islands=[{"start_ms": 0, "end_ms": 5000}],
        )
        # Any window that does not intersect speech should be marked
        # not-scheduled, preserving the misconception-3 invariant that
        # triage is conservative about compute.
        later_windows = [w for w in windows if w["start_ms"] >= 30000]
        assert later_windows, "expected follow-up windows"
        assert all(w["scheduled"] is False for w in later_windows)


class TestMultilingualCodeSwitchFixtures:
    """Misconception 2: language attribution must remain local and time-varying."""

    def test_local_language_varies_per_candidate(self):
        """Per-window language evidence must stay local, never overwritten session-wide."""
        from app.pipeline.witness_diagnostics import compute_candidate_flags

        # Three adjacent stripes: RU, FR, EN.  Each one is flagged according
        # to its OWN detected language — there is no session-wide override.
        for expected_lang, text in [
            ("ru", "Привет мир"),
            ("fr", "Bonjour tout le monde"),
            ("en", "Hello world"),
        ]:
            result = compute_candidate_flags(
                raw_text=text,
                detected_language=expected_lang,
                requested_language=expected_lang,
                transcription_mode="verbatim_multilingual",
                success=True,
                degraded=False,
                duration_s=15.0,
            )
            assert "language_mismatch" not in result["candidate_flags"]
            assert "possible_translation" not in result["candidate_flags"]

    def test_code_switch_stripe_packets_accept_multiple_languages(self, tmp_sessions_dir):
        """An allowed_languages=[ru,fr,en] session must not filter out code-switches."""
        from app.pipeline.stripe_grouping import group_evidence_by_stripe

        stripes = [
            {"stripe_id": "S0", "start_ms": 0, "end_ms": 15000},
            {"stripe_id": "S1", "start_ms": 15000, "end_ms": 30000},
        ]
        windows = [
            {"window_id": "W0", "start_ms": 0, "end_ms": 30000, "window_type": "full"},
        ]
        candidates = [
            {
                "candidate_id": "cand_ru",
                "model_id": "faster-whisper:large-v3",
                "window_id": "W0",
                "window_start_ms": 0,
                "window_end_ms": 30000,
                "window_type": "full",
                "raw_text": "Привет",
                "segments": [{"start": 0.0, "end": 5.0, "text": "Привет"}],
                "language_evidence": {"detected_language": "ru"},
                "decode_metadata": {"segment_timestamp_unit": "seconds"},
                "candidate_flags": [],
            },
            {
                "candidate_id": "cand_fr",
                "model_id": "faster-whisper:large-v3",
                "window_id": "W0",
                "window_start_ms": 0,
                "window_end_ms": 30000,
                "window_type": "full",
                "raw_text": "Bonjour",
                "segments": [{"start": 20.0, "end": 25.0, "text": "Bonjour"}],
                "language_evidence": {"detected_language": "fr"},
                "decode_metadata": {"segment_timestamp_unit": "seconds"},
                "candidate_flags": [],
            },
        ]

        packets = group_evidence_by_stripe(
            stripes,
            windows,
            candidates,
            allowed_languages=["ru", "fr", "en"],
            forced_language=None,
            transcription_mode="verbatim_multilingual",
        )
        langs = {
            item.get("language_evidence", {}).get("detected_language")
            for packet in packets
            for item in packet.get("evidence") or []
        }
        assert {"ru", "fr"} <= langs, "code-switched candidates must both survive grouping"


class TestMemoryOrientedRetrieval:
    """Misconception 13: real test must retrieve by alias / entity / relation."""

    def test_indirect_alias_resolves_to_entity(self, tmp_sessions_dir):
        from app.core.atomic_io import atomic_write_json
        from app.pipeline.memory_graph import run_memory_graph_update
        from app.storage.session_store import create_session, session_dir
        import json

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        curated_dir = sd / "memory" / "curated_packs"
        curated_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(curated_dir / "relationships.json"), {
            "pack_id": "relationships",
            "entities": [{
                "entity_id": "person_mattiew",
                "display_name": "Mattiew",
                "aliases": ["Mattiew", "наркоман из Бангкока", "the addict from Bangkok"],
                "roles": ["past_partner"],
                "confidence": 0.9,
            }],
        })

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_memory_graph_update(sid, [], StageStub())
        alias_graph = json.loads((sd / "memory" / "alias_graph.json").read_text(encoding="utf-8"))
        matches = {
            entry["alias"]: entry["entity_ids"]
            for entry in alias_graph["aliases"]
        }
        assert "наркоман из бангкока" in matches
        assert matches["наркоман из бангкока"] == ["person_mattiew"]


# ============================================================
# Repair coverage for the 2026-04-15 server audit
# ============================================================


class TestInterruptedRunResumability:
    """Pointer must be written at initialize so interrupted runs resume."""

    def test_pointer_written_on_initialize(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.run import create_canonical_run, get_canonical_run_id

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        run = create_canonical_run(str(sd), sid, {"session_id": sid})
        # Don't call complete() -- the pointer must already be there.
        assert get_canonical_run_id(str(sd)) == run.run_id

    def test_orphaned_run_discoverable_via_fallback(self, tmp_sessions_dir):
        """If the pointer is missing, get_canonical_run_id falls back to the
        most recent run_meta.json."""
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.run import create_canonical_run, get_canonical_run_id

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        run = create_canonical_run(str(sd), sid, {"session_id": sid})

        # Simulate an older pipeline that never wrote the pointer.
        (sd / "pipeline" / "canonical_run_id.txt").unlink(missing_ok=True)

        assert get_canonical_run_id(str(sd)) == run.run_id


class TestMediaJunkSuppression:
    """Reconciliation must not canonicalize obvious subtitle/media pollution."""

    def test_detector_catches_known_patterns(self):
        from app.pipeline.reconciliation import looks_like_media_junk
        assert looks_like_media_junk("Субтитры сделал DimaTorzok")
        assert looks_like_media_junk("Thanks for watching!")
        assert looks_like_media_junk("Sous-titres réalisés par la communauté")
        assert not looks_like_media_junk("Bonjour, je suis chez BNP Paribas")
        assert not looks_like_media_junk("")

    def test_fallback_suppresses_all_junk_candidates(self):
        from app.pipeline.reconciliation import _select_fallback
        evidence = [
            {"text": "Субтитры сделал DimaTorzok", "model_id": "large-v3",
             "trust_score": 1.0, "candidate_flags": ["media_pollution_suspected"]},
            {"text": "Продолжение следует", "model_id": "medium",
             "trust_score": 0.9, "candidate_flags": ["media_pollution_suspected"]},
        ]
        result = _select_fallback(evidence)
        assert result["chosen_text"] == ""
        assert result["fallback_reason"] == "all_candidates_media_junk"
        assert result["chosen_source"] == "none"

    def test_fallback_picks_clean_over_junk_even_when_junk_has_higher_priority(self):
        from app.pipeline.reconciliation import _select_fallback
        evidence = [
            {"text": "Субтитры сделал DimaTorzok", "model_id": "large-v3",
             "trust_score": 1.0, "candidate_flags": ["media_pollution_suspected"]},
            {"text": "Настоящий текст речи", "model_id": "medium",
             "trust_score": 0.8, "candidate_flags": []},
        ]
        result = _select_fallback(evidence)
        assert result["chosen_text"] == "Настоящий текст речи"
        assert result["suppressed_count"] == 1

    def test_corruption_flags_penalize_fallback_scoring(self):
        from app.pipeline.reconciliation import _select_fallback
        # Two candidates; the "priority 3" one carries a language_mismatch
        # flag strong enough to let the cleaner lower-priority one win.
        evidence = [
            {"text": "tainted", "model_id": "large-v3", "trust_score": 1.0,
             "candidate_flags": ["language_mismatch", "script_mismatch"]},
            {"text": "clean", "model_id": "medium", "trust_score": 1.0,
             "candidate_flags": []},
        ]
        result = _select_fallback(evidence)
        assert result["chosen_text"] == "clean"


class TestAuditedSynthesisValidator:
    """LLM validator must accept bounded synthesis, not only exact candidates."""

    def test_exact_candidate_match_accepted(self):
        from app.pipeline.reconciliation import _validate_llm_selection
        parsed = {"text": "hello world", "source_model": "large-v3", "confidence": 0.9}
        cands = [{"model_id": "large-v3", "text": "hello world"}]
        ok, reason = _validate_llm_selection(parsed, cands)
        assert ok
        assert reason == "exact_candidate_match"

    def test_bounded_synthesis_accepted(self):
        """Words recombined from evidence vocabulary must be accepted."""
        from app.pipeline.reconciliation import _validate_llm_selection
        parsed = {"text": "hello amazing world", "source_model": "llm", "confidence": 0.85}
        cands = [
            {"model_id": "large-v3", "text": "hello world"},
            {"model_id": "medium", "text": "amazing world"},
        ]
        ok, reason = _validate_llm_selection(parsed, cands)
        assert ok
        assert reason.startswith("bounded_synthesis:")

    def test_hallucinated_tokens_rejected(self):
        from app.pipeline.reconciliation import _validate_llm_selection
        parsed = {"text": "quantum fluctuations entropy spiral", "source_model": "llm", "confidence": 0.9}
        cands = [{"model_id": "large-v3", "text": "hello world"}]
        ok, reason = _validate_llm_selection(parsed, cands)
        assert not ok
        assert reason.startswith("unsupported_synthesis:")

    def test_media_junk_llm_output_rejected(self):
        from app.pipeline.reconciliation import _validate_llm_selection
        parsed = {"text": "Субтитры сделал DimaTorzok", "source_model": "llm", "confidence": 0.99}
        cands = [{"model_id": "large-v3", "text": "Субтитры сделал DimaTorzok"}]
        ok, reason = _validate_llm_selection(parsed, cands)
        assert not ok
        assert reason == "llm_emitted_media_junk"


class TestQualityReportDetectsJunk:
    def test_media_pollution_raises_pipeline_health(self):
        from app.pipeline.derived_outputs import generate_quality_report
        segments = [
            {"start_ms": 0, "end_ms": 5000, "text": "Субтитры сделал DimaTorzok",
             "confidence": 0.9, "corruption_flags": []},
            {"start_ms": 5000, "end_ms": 10000, "text": "Hello",
             "confidence": 0.9, "corruption_flags": []},
        ]
        report = generate_quality_report(segments, "Hello", 10000)
        assert report["pipeline_health"]["media_pollution_count"] == 1
        types = {issue["type"] for issue in report["issues"]}
        assert "media_pollution_detected" in types

    def test_corruption_flags_surface_as_issues(self):
        from app.pipeline.derived_outputs import generate_quality_report
        segments = [
            {"start_ms": 0, "end_ms": 5000, "text": "maybe ok",
             "confidence": 0.9,
             "corruption_flags": ["unsupported_tokens_present", "single_window_support"]},
        ]
        report = generate_quality_report(segments, "maybe ok", 5000)
        assert report["pipeline_health"]["corruption_flagged_count"] == 1


class TestCleanTranscriptRealCleanup:
    def test_media_junk_segments_dropped_from_clean_text(self):
        from app.pipeline.derived_outputs import generate_clean_transcript
        segments = [
            {"segment_id": "seg_000000", "start_ms": 0, "end_ms": 5000,
             "text": "Hello world", "speaker": None, "corruption_flags": []},
            {"segment_id": "seg_000001", "start_ms": 5000, "end_ms": 10000,
             "text": "Субтитры сделал DimaTorzok", "speaker": None,
             "corruption_flags": []},
            {"segment_id": "seg_000002", "start_ms": 10000, "end_ms": 15000,
             "text": "Goodbye world", "speaker": None, "corruption_flags": []},
        ]
        clean = generate_clean_transcript(segments, "Hello world Субтитры сделал DimaTorzok Goodbye world")
        assert "DimaTorzok" not in clean["clean_text"]
        assert "Hello world" in clean["clean_text"]
        assert "Goodbye world" in clean["clean_text"]
        assert clean["dropped_count"] == 1
        assert clean["cleanup_version"] == "2.0"

    def test_whitespace_and_punctuation_normalized(self):
        from app.pipeline.derived_outputs import generate_clean_transcript
        segments = [
            {"segment_id": "seg_000000", "start_ms": 0, "end_ms": 5000,
             "text": "Hello   world  !!!!", "speaker": None, "corruption_flags": []},
        ]
        clean = generate_clean_transcript(segments, "Hello   world  !!!!")
        assert clean["clean_text"] == "Hello world!"


class TestFineGrainedDisplaySegmentation:
    def test_subtitles_use_stripe_level_timing(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.derived_outputs import _expand_display_segments

        segments = [
            {
                "segment_id": "seg_000000",
                "start_ms": 0,
                "end_ms": 60000,
                "text": "phrase one phrase two phrase three phrase four",
                "speaker": "SPEAKER_00",
                "language": "en",
                "corruption_flags": [],
                "assembly_decisions": [
                    {"stripe_id": "S0000", "start_ms": 0, "end_ms": 15000, "final_text": "phrase one"},
                    {"stripe_id": "S0001", "start_ms": 15000, "end_ms": 30000, "final_text": "phrase two"},
                    {"stripe_id": "S0002", "start_ms": 30000, "end_ms": 45000, "final_text": "phrase three"},
                    {"stripe_id": "S0003", "start_ms": 45000, "end_ms": 60000, "final_text": "phrase four"},
                ],
            },
        ]
        display = _expand_display_segments(segments)
        assert len(display) == 4
        assert display[0]["text"] == "phrase one"
        assert display[0]["end_ms"] == 15000
        assert display[1]["source_stripe_id"] == "S0001"

    def test_display_skips_media_junk_stripe_within_segment(self):
        from app.pipeline.derived_outputs import _expand_display_segments
        segments = [
            {
                "segment_id": "seg_000000",
                "start_ms": 0,
                "end_ms": 30000,
                "text": "clean phrase Субтитры сделал DimaTorzok",
                "speaker": None,
                "language": "ru",
                "corruption_flags": [],
                "assembly_decisions": [
                    {"stripe_id": "S0", "start_ms": 0, "end_ms": 15000, "final_text": "clean phrase"},
                    {"stripe_id": "S1", "start_ms": 15000, "end_ms": 30000,
                     "final_text": "Субтитры сделал DimaTorzok"},
                ],
            },
        ]
        display = _expand_display_segments(segments)
        assert len(display) == 1
        assert display[0]["text"] == "clean phrase"


class TestRetrievalExcludesJunk:
    def test_junk_segment_not_indexed(self):
        from app.pipeline.derived_outputs import generate_retrieval_index
        segments = [
            {"segment_id": "seg_000000", "start_ms": 0, "end_ms": 5000,
             "text": "Hello world", "language": "en", "corruption_flags": [],
             "stabilization_state": "stabilized",
             "support_windows": ["W0"], "support_models": ["large-v3"]},
            {"segment_id": "seg_000001", "start_ms": 5000, "end_ms": 10000,
             "text": "Субтитры сделал DimaTorzok", "language": "ru",
             "corruption_flags": [], "stabilization_state": "stabilized",
             "support_windows": ["W1"], "support_models": ["large-v3"]},
        ]
        idx = generate_retrieval_index("sid", segments, marker_index={})
        assert idx["entry_count"] == 1
        assert idx["excluded_count"] == 1
        assert idx["entries"][0]["segment_id"] == "seg_000000"


class TestDiarizationDefaults:
    def test_auto_triggers_on_long_audio(self):
        from app.pipeline.selective_enrichment import should_run_diarization
        meta = {"duration_ms": 60 * 60 * 1000}  # 1h
        assert should_run_diarization(meta, [], policy="auto") is True

    def test_auto_triggers_on_voices_filename_hint(self):
        from app.pipeline.selective_enrichment import should_run_diarization
        meta = {"original_filename": "ru_1hours_outside_2voices.wav"}
        assert should_run_diarization(meta, [], policy="auto") is True

    def test_auto_triggers_on_phonecall_hint(self):
        from app.pipeline.selective_enrichment import should_run_diarization
        meta = {"original_filename": "eg_1hours_phonecall_2voices.wav"}
        assert should_run_diarization(meta, [], policy="auto") is True

    def test_auto_skips_short_monologue(self):
        from app.pipeline.selective_enrichment import should_run_diarization
        meta = {"duration_ms": 5000, "original_filename": "quick_note.wav"}
        assert should_run_diarization(meta, [], policy="auto") is False

    def test_off_wins_over_heuristics(self):
        from app.pipeline.selective_enrichment import should_run_diarization
        meta = {"duration_ms": 3600000, "original_filename": "interview_2voices.wav"}
        assert should_run_diarization(meta, [], policy="off") is False

    def test_run_diarization_returns_explicit_no_model_path_status(self, tmp_sessions_dir, monkeypatch):
        from app.pipeline.selective_enrichment import run_diarization
        from app.core.config import reset_config

        # ModelPaths is frozen; force resolve() to miss by pointing MODELS_DIR
        # at an empty directory and clearing any DIARIZATION_MODEL_PATH env
        # override that may be set in the developer's .env.
        empty_models_dir = tmp_sessions_dir / "empty_models"
        empty_models_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("MODELS_DIR", str(empty_models_dir))
        monkeypatch.delenv("DIARIZATION_MODEL_PATH", raising=False)
        reset_config()

        try:
            turns, status = run_diarization("nonexistent.wav", {})
            assert turns is None
            assert status == "no_model_path"
        finally:
            reset_config()


class TestApiExposesCanonicalFirst:
    def test_transcript_reads_canonical_before_current(self, tmp_sessions_dir):
        import json
        from app.storage.session_store import create_session, session_dir, update_status
        from app.core.atomic_io import atomic_write_json, atomic_write_text

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        update_status(sid, "done")

        canonical = sd / "canonical"
        canonical.mkdir(parents=True, exist_ok=True)
        current = sd / "current"
        current.mkdir(parents=True, exist_ok=True)

        # Canonical wins: both surfaces exist but contain different text.
        atomic_write_json(str(canonical / "final_transcript.json"), {
            "text": "CANONICAL TRUTH",
            "segments": [{"segment_id": "seg_000000", "start_ms": 0, "end_ms": 1000,
                          "text": "CANONICAL TRUTH"}],
        })
        atomic_write_json(str(current / "final_transcript.json"), {
            "text": "LEGACY COPY",
            "segments": [{"segment_id": "seg_000000", "start_ms": 0, "end_ms": 1000,
                          "text": "LEGACY COPY"}],
        })
        atomic_write_text(str(current / "transcript.txt"), "LEGACY COPY")

        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        resp = client.get(f"/api/v2/sessions/{sid}/transcript",
                          headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        assert resp.json()["text"] == "CANONICAL TRUTH"

    def test_markers_endpoint_returns_enrichment_payload(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, session_dir
        from app.core.atomic_io import atomic_write_json

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        (sd / "enrichment").mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(sd / "enrichment" / "segment_markers.json"), {
            "markers": [{"segment_id": "seg_000000", "topic_tags": ["meeting"]}],
        })
        atomic_write_json(str(sd / "enrichment" / "semantic_spans.json"), {
            "spans": [{"span_id": "sem_0", "segment_ids": ["seg_000000"]}],
        })
        atomic_write_json(str(sd / "enrichment" / "context_spans.json"), {
            "spans": [{"context_id": "ctx_0", "segment_ids": ["seg_000000"], "topic_candidates": ["meeting"]}],
        })

        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        resp = client.get(f"/api/v2/sessions/{sid}/markers",
                          headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["markers"][0]["topic_tags"] == ["meeting"]
        assert body["semantic_spans"][0]["span_id"] == "sem_0"
        assert body["context_spans"][0]["context_id"] == "ctx_0"

    def test_transcript_exposes_quality_gate_and_context_spans(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, session_dir, update_status
        from app.core.atomic_io import atomic_write_json

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        update_status(sid, "done")

        (sd / "canonical").mkdir(parents=True, exist_ok=True)
        (sd / "enrichment").mkdir(parents=True, exist_ok=True)
        (sd / "derived").mkdir(parents=True, exist_ok=True)

        atomic_write_json(str(sd / "canonical" / "final_transcript.json"), {
            "text": "Need to wire money to Omar",
            "segments": [{"segment_id": "seg_000000", "start_ms": 0, "end_ms": 1000, "text": "Need to wire money to Omar"}],
        })
        atomic_write_json(str(sd / "canonical" / "quality_gate.json"), {
            "session_quality_status": "healthy",
            "semantic_eligible": True,
            "memory_update_eligible": True,
            "reasons": [],
        })
        atomic_write_json(str(sd / "enrichment" / "segment_markers.json"), {
            "markers": [{"segment_id": "seg_000000", "topic_candidates": ["money_help"]}],
        })
        atomic_write_json(str(sd / "enrichment" / "semantic_spans.json"), {
            "spans": [{"span_id": "sem_0"}],
        })
        atomic_write_json(str(sd / "enrichment" / "context_spans.json"), {
            "span_count": 1,
            "spans": [{"context_id": "ctx_0", "segment_ids": ["seg_000000"], "topic_candidates": ["money_help"]}],
        })
        atomic_write_json(str(sd / "derived" / "retrieval_index_v3.json"), {
            "version": 3,
            "entry_count": 1,
            "excluded_count": 0,
            "source": "context_spans+canonical_segments+segment_markers",
            "entries": [{"context_id": "ctx_0"}],
        })

        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        resp = client.get(
            f"/api/v2/sessions/{sid}/transcript",
            headers={"Authorization": "Bearer test-token"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["quality_gate"]["session_quality_status"] == "healthy"
        assert body["context_spans"][0]["context_id"] == "ctx_0"
        assert body["retrieval_summary"]["version"] == 3


class TestDashboardUiBundle:
    def test_root_ui_references_external_dashboard_bundle(self, tmp_sessions_dir):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        resp = client.get("/", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        assert '/ui/dashboard.js' in resp.text
        assert 'Queue Selected Files' in resp.text
        assert 'multiple onchange="onUploadSelectionChanged()"' in resp.text

    def test_dashboard_js_bundle_is_served(self, tmp_sessions_dir):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        resp = client.get("/ui/dashboard.js", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        assert "function startUpload()" in resp.text
        assert "processUploadQueue" in resp.text


class TestSessionStatusDiagnosticsSurface:
    def test_session_status_exposes_diarization_quality_gate_and_context_count(self, tmp_sessions_dir):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.storage.session_store import create_session, session_dir, update_session_meta, update_status
        from app.core.atomic_io import atomic_write_json

        sid = create_session({"mode": "file"})["session_id"]
        sd = session_dir(sid)
        (sd / "canonical").mkdir(parents=True, exist_ok=True)
        (sd / "enrichment").mkdir(parents=True, exist_ok=True)
        update_session_meta(sid, {
            "state": "finalized",
            "original_filename": "meeting.wav",
        })
        update_status(sid, "running")

        atomic_write_json(str(sd / "metadata.json"), {
            "language": "fr",
            "model_size": "auto",
            "diarization_policy": "auto",
        })
        atomic_write_json(str(sd / "canonical" / "quality_gate.json"), {
            "session_quality_status": "healthy",
            "semantic_eligible": True,
            "memory_update_eligible": True,
            "reasons": [],
        })
        atomic_write_json(str(sd / "canonical" / "diarization_status.json"), {
            "policy": "auto",
            "requested": False,
            "status": "skipped",
            "reason": "policy_auto_not_triggered",
            "turn_count": 0,
        })
        atomic_write_json(str(sd / "enrichment" / "context_spans.json"), {
            "span_count": 2,
            "spans": [{"context_id": "ctx_0"}, {"context_id": "ctx_1"}],
        })

        client = TestClient(app)
        resp = client.get(f"/api/v2/sessions/{sid}/status", headers={"Authorization": "Bearer test-token"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["original_filename"] == "meeting.wav"
        assert body["language"] == "fr"
        assert body["diarization"]["status"] == "skipped"
        assert body["quality_gate"]["session_quality_status"] == "healthy"
        assert body["context_span_count"] == 2


class TestSemanticMarkingUsesCuratedPack:
    def test_alias_resolved_to_curated_entity_id(self, tmp_sessions_dir):
        from app.storage.session_store import create_session, session_dir
        from app.core.atomic_io import atomic_write_json
        from app.pipeline.semantic_marking import run_semantic_marking

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        curated_dir = sd / "memory" / "curated_packs"
        curated_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(curated_dir / "pack.json"), {
            "entities": [{
                "entity_id": "person_omar",
                "canonical_name": "Omar",
                "type": "person",
                "confidence": 0.95,
                "aliases": ["omar", "мой муж"],
            }],
            "ontology": {
                "topic_tags": {"observability": ["dynatrace", "weblogic"]},
            },
        })

        segments = [{
            "segment_id": "seg_000000",
            "start_ms": 0,
            "end_ms": 5000,
            "text": "Я говорила с моим мужем про weblogic memory leak",
            "stabilization_state": "stabilized",
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_semantic_marking(sid, segments, StageStub())
        import json
        payload = json.loads((sd / "enrichment" / "segment_markers.json").read_text(encoding="utf-8"))
        marker = payload["markers"][0]
        entity_ids = [m.get("entity_id") for m in marker["entity_mentions"]]
        assert "person_omar" in entity_ids
        assert "observability" in marker["topic_tags"]
        assert marker["marker_source"] == "heuristic+curated_alias_resolution"


class TestTopicCandidatesTaxonomy:
    """Phase 3 - business taxonomy distinct from generic topic_tags."""

    def test_marker_exposes_topic_candidates_field(self, tmp_sessions_dir):
        import json
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.semantic_marking import run_semantic_marking

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        segments = [{
            "segment_id": "seg_000000",
            "start_ms": 0,
            "end_ms": 4000,
            "text": "I need to wire transfer money to my bank account this week",
            "stabilization_state": "stabilized",
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_semantic_marking(sid, segments, StageStub())
        payload = json.loads((sd / "enrichment" / "segment_markers.json").read_text(encoding="utf-8"))
        marker = payload["markers"][0]
        assert "topic_candidates" in marker
        # money_help should be matched by "wire transfer"/"money".
        assert "money_help" in marker["topic_candidates"]

    def test_topic_candidates_are_separate_from_generic_topic_tags(self, tmp_sessions_dir):
        import json
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.semantic_marking import run_semantic_marking

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        segments = [{
            "segment_id": "seg_000000",
            "start_ms": 0,
            "end_ms": 4000,
            "text": "Booking the flight to the airport, hotel reservation included",
            "stabilization_state": "stabilized",
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_semantic_marking(sid, segments, StageStub())
        marker = json.loads(
            (sd / "enrichment" / "segment_markers.json").read_text(encoding="utf-8")
        )["markers"][0]
        # generic heuristic still fires (`travel` in topic_tags) AND
        # business taxonomy adds `travel_logistics`.
        assert "travel" in marker["topic_tags"]
        assert "travel_logistics" in marker["topic_candidates"]
        # The two namespaces must not be flattened into a single key.
        assert "travel_logistics" not in marker["topic_tags"]
        assert "travel" not in marker["topic_candidates"]

    def test_curated_pack_extends_taxonomy(self, tmp_sessions_dir):
        import json
        from app.storage.session_store import create_session, session_dir
        from app.core.atomic_io import atomic_write_json
        from app.pipeline.semantic_marking import run_semantic_marking

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        curated_dir = sd / "memory" / "curated_packs"
        curated_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(curated_dir / "pack.json"), {
            "domain_taxonomy": {
                "money_help": ["airtm", "wise transfer"],
                "custom_topic": ["unique_phrase_xyz"],
            },
        })

        segments = [{
            "segment_id": "seg_000000",
            "start_ms": 0,
            "end_ms": 4000,
            "text": "I will use airtm for the unique_phrase_xyz operation",
            "stabilization_state": "stabilized",
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_semantic_marking(sid, segments, StageStub())
        marker = json.loads(
            (sd / "enrichment" / "segment_markers.json").read_text(encoding="utf-8")
        )["markers"][0]
        assert "money_help" in marker["topic_candidates"]
        assert "custom_topic" in marker["topic_candidates"]


class TestContextSpansBuilder:
    """Phase 2 - continuity-based context spans."""

    @staticmethod
    def _seg(seg_id, start_ms, end_ms, text="hello world there", language="en", speaker="SPEAKER_00"):
        return {
            "segment_id": seg_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "language": language,
            "speaker": speaker,
            "stabilization_state": "stabilized",
        }

    @staticmethod
    def _marker(seg_id, topic_candidates=(), topic_tags=(), entity_ids=()):
        return {
            "segment_id": seg_id,
            "topic_candidates": list(topic_candidates),
            "topic_tags": list(topic_tags),
            "entity_mentions": [
                {"entity_id": eid, "surface_form": eid, "source": "curated_pack"}
                for eid in entity_ids
            ],
        }

    def test_continuous_segments_form_single_span(self):
        from app.pipeline.context_spans import build_context_spans

        segs = [
            self._seg("s0", 0, 3000),
            self._seg("s1", 3500, 6000),
            self._seg("s2", 6500, 9000),
        ]
        markers = [
            self._marker("s0", topic_candidates=["banking"]),
            self._marker("s1", topic_candidates=["banking"]),
            self._marker("s2", topic_candidates=["banking"]),
        ]
        payload = build_context_spans("sess", segs, markers)
        assert payload["span_count"] == 1
        span = payload["spans"][0]
        assert span["segment_ids"] == ["s0", "s1", "s2"]
        assert "banking" in span["topic_candidates"]

    def test_strong_topic_shift_splits_span(self):
        from app.pipeline.context_spans import build_context_spans

        segs = [
            self._seg("s0", 0, 5000, text="loan repayment plan numbers"),
            self._seg("s1", 5500, 9000, text="loan repayment plan amount"),
            # Big gap, no shared entity, totally different topic_candidates.
            self._seg("s2", 80_000, 85_000, text="airport boarding gate latte"),
            self._seg("s3", 86_000, 90_000, text="airport boarding gate flight"),
        ]
        markers = [
            self._marker("s0", topic_candidates=["money_help"]),
            self._marker("s1", topic_candidates=["money_help"]),
            self._marker("s2", topic_candidates=["travel_logistics"]),
            self._marker("s3", topic_candidates=["travel_logistics"]),
        ]
        payload = build_context_spans("sess", segs, markers)
        assert payload["span_count"] == 2
        first, second = payload["spans"]
        assert first["topic_candidates"] == ["money_help"]
        assert second["topic_candidates"] == ["travel_logistics"]

    def test_hard_temporal_break_always_splits(self):
        from app.pipeline.context_spans import build_context_spans

        segs = [
            self._seg("s0", 0, 3000, text="ongoing discussion banking matters"),
            # 5 minutes later, same speaker, same topic -- still splits.
            self._seg("s1", 300_000, 303_000, text="ongoing discussion banking matters"),
        ]
        markers = [
            self._marker("s0", topic_candidates=["banking"], entity_ids=["person_omar"]),
            self._marker("s1", topic_candidates=["banking"], entity_ids=["person_omar"]),
        ]
        payload = build_context_spans("sess", segs, markers)
        assert payload["span_count"] == 2

    def test_speaker_change_alone_does_not_split_in_tight_window(self):
        from app.pipeline.context_spans import build_context_spans

        segs = [
            self._seg("s0", 0, 3000, speaker="SPEAKER_00"),
            self._seg("s1", 3500, 6000, speaker="SPEAKER_01"),
        ]
        markers = [
            self._marker("s0", topic_candidates=["money_help"]),
            self._marker("s1", topic_candidates=["money_help"]),
        ]
        payload = build_context_spans("sess", segs, markers)
        # Within tight window + topic continuity → one span, two speakers.
        assert payload["span_count"] == 1
        assert sorted(payload["spans"][0]["speaker_ids"]) == ["SPEAKER_00", "SPEAKER_01"]

    def test_context_span_payload_carries_required_fields(self):
        from app.pipeline.context_spans import build_context_spans

        segs = [
            self._seg("s0", 0, 4000, language="ru"),
            self._seg("s1", 4500, 9000, language="ru"),
        ]
        markers = [
            self._marker("s0", topic_candidates=["relationship/partner"], entity_ids=["person_omar"]),
            self._marker("s1", topic_candidates=["relationship/partner"], entity_ids=["person_omar"]),
        ]
        payload = build_context_spans("sess-id", segs, markers)
        span = payload["spans"][0]
        for key in (
            "context_id", "session_id", "start_ms", "end_ms",
            "segment_ids", "speaker_ids", "language_profile",
            "topic_candidates", "entity_ids", "alias_hits",
            "continuity_evidence", "confidence", "grounding",
        ):
            assert key in span, f"missing {key}"
        assert span["language_profile"]["primary"] == "ru"
        assert "person_omar" in span["entity_ids"]
        assert any(hit["entity_id"] == "person_omar" for hit in span["alias_hits"])

    def test_run_semantic_marking_writes_context_spans_artifact(self, tmp_sessions_dir):
        import json
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.semantic_marking import run_semantic_marking

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        segments = [
            {
                "segment_id": "seg_000000",
                "start_ms": 0,
                "end_ms": 4000,
                "text": "I should send money to my husband for the bank transfer",
                "language": "en",
                "speaker": "SPEAKER_00",
                "stabilization_state": "stabilized",
            },
            {
                "segment_id": "seg_000001",
                "start_ms": 4500,
                "end_ms": 8500,
                "text": "wire transfer to the bank account today",
                "language": "en",
                "speaker": "SPEAKER_00",
                "stabilization_state": "stabilized",
            },
        ]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        result = run_semantic_marking(sid, segments, StageStub())
        spans_path = sd / "enrichment" / "context_spans.json"
        assert spans_path.is_file()
        payload = json.loads(spans_path.read_text(encoding="utf-8"))
        assert payload["span_count"] >= 1
        assert result["context_span_count"] == payload["span_count"]
        assert "money_help" in payload["spans"][0]["topic_candidates"]

    def test_context_spans_emitted_even_when_quality_gate_suppresses(self, tmp_sessions_dir, monkeypatch):
        import json
        from app.storage.session_store import create_session, session_dir
        from app.pipeline.semantic_marking import run_semantic_marking

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        monkeypatch.setattr(
            "app.pipeline.canonical_assembly.read_quality_gate",
            lambda session_id: {
                "session_quality_status": "degraded",
                "semantic_eligible": False,
                "memory_update_eligible": False,
                "reasons": ["test_forced_suppression"],
            },
        )

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_semantic_marking(sid, [], StageStub())
        spans_path = sd / "enrichment" / "context_spans.json"
        assert spans_path.is_file()
        payload = json.loads(spans_path.read_text(encoding="utf-8"))
        assert payload["span_count"] == 0
        assert payload["gate_status"] == "suppressed_by_quality_gate"
        assert "test_forced_suppression" in payload["gate_reasons"]


class TestQualityGateFallbackPolicy:
    def test_missing_gate_is_strict_after_canonical_artifacts_exist(self, tmp_sessions_dir):
        from app.core.atomic_io import atomic_write_json
        from app.pipeline.canonical_assembly import read_quality_gate
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        (sd / "canonical").mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(sd / "canonical" / "canonical_segments.json"), {
            "segments": [{"segment_id": "seg_000000", "text": "hello"}],
        })

        gate = read_quality_gate(sid)
        assert gate["session_quality_status"] == "missing_after_canonical_stage"
        assert gate["semantic_eligible"] is False
        assert gate["memory_update_eligible"] is False


class TestMemoryGraphPhase3Projection:
    def test_auto_context_pack_tracks_topic_candidates_and_context_summaries(self, tmp_sessions_dir):
        import json
        from app.core.atomic_io import atomic_write_json
        from app.pipeline.memory_graph import run_memory_graph_update
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        (sd / "enrichment").mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(sd / "enrichment" / "context_spans.json"), {
            "span_count": 1,
            "spans": [{
                "context_id": "ctx_000001",
                "start_ms": 0,
                "end_ms": 5000,
                "segment_count": 1,
                "speaker_ids": ["SPEAKER_00"],
                "language_profile": {"primary": "en", "ratio": 1.0, "languages": {"en": 1}},
                "topic_tags": ["relationship"],
                "topic_candidates": ["money_help"],
                "entity_ids": ["person_omar"],
                "alias_hits": [{"entity_id": "person_omar", "surface_form": "Omar"}],
                "confidence": 0.7,
            }],
        })
        markers = [{
            "segment_id": "seg_000001",
            "entity_mentions": [
                {"surface_form": "Omar", "mention_type": "capitalized_phrase", "confidence": 0.8, "entity_id": "person_omar"},
            ],
            "topic_tags": ["relationship"],
            "topic_candidates": ["money_help"],
            "project_tags": [],
            "relation_tags": [],
            "emotion_tags": [],
            "retrieval_terms": ["omar", "wire transfer"],
            "ambiguity_flags": [],
            "marker_confidence": 0.8,
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        run_memory_graph_update(sid, markers, StageStub())
        auto_pack = json.loads((sd / "memory" / "context_packs" / "session_auto_context.json").read_text(encoding="utf-8"))
        assert auto_pack["topic_candidate_frequency"]["money_help"] == 1
        assert auto_pack["context_span_count"] == 1
        assert auto_pack["context_summaries"][0]["context_id"] == "ctx_000001"


class TestPhase4ContextLinks:
    def test_semantic_marking_writes_context_links_with_resolved_and_unresolved_mentions(self, tmp_sessions_dir):
        import json
        from app.core.atomic_io import atomic_write_json
        from app.pipeline.semantic_marking import run_semantic_marking
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        curated_dir = sd / "memory" / "curated_packs"
        curated_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(curated_dir / "relationships.json"), {
            "entities": [{
                "entity_id": "person_omar",
                "canonical_name": "Omar",
                "type": "person",
                "confidence": 0.95,
                "aliases": ["my husband"],
            }],
        })

        segments = [{
            "segment_id": "seg_000000",
            "start_ms": 0,
            "end_ms": 5000,
            "text": "I spoke with my husband about Alice Cooper yesterday",
            "language": "en",
            "speaker": "SPEAKER_00",
            "stabilization_state": "stabilized",
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        result = run_semantic_marking(sid, segments, StageStub())
        links_payload = json.loads((sd / "enrichment" / "context_links.json").read_text(encoding="utf-8"))
        assert result["context_link_count"] == links_payload["link_count"]
        assert links_payload["link_count"] >= 2

        resolved = [
            link for link in links_payload["links"]
            if link["memory_kind"] == "resolved_alias_to_known_entity"
        ]
        unresolved = [
            link for link in links_payload["links"]
            if link["memory_kind"] == "context_local_unresolved_mention"
        ]
        assert any(link["entity_id"] == "person_omar" for link in resolved)
        assert any(link["alias_surface"] == "my husband" for link in resolved)
        assert any("my husband" in (link["mention_text"] or "").lower() for link in resolved)
        assert any((link["mention_text"] or "") == "Alice Cooper" for link in unresolved)
        assert all(link["context_id"] for link in links_payload["links"])
        assert all(link["support_text"] for link in links_payload["links"])


class TestPhase4GraphUpdateProposals:
    def test_memory_graph_writes_contextual_update_proposals(self, tmp_sessions_dir):
        import json
        from app.core.atomic_io import atomic_write_json
        from app.pipeline.memory_graph import run_memory_graph_update
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)
        (sd / "enrichment").mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(sd / "enrichment" / "context_spans.json"), {
            "span_count": 1,
            "spans": [{
                "context_id": "ctx_000001",
                "segment_ids": ["seg_000001"],
                "segment_count": 1,
                "speaker_ids": ["SPEAKER_00"],
                "language_profile": {"primary": "en", "ratio": 1.0, "languages": {"en": 1}},
                "topic_tags": ["relationship"],
                "topic_candidates": ["money_help"],
                "entity_ids": ["person_omar"],
                "alias_hits": [{"entity_id": "person_omar", "surface_form": "my husband"}],
                "confidence": 0.8,
            }],
        })
        atomic_write_json(str(sd / "enrichment" / "context_links.json"), {
            "link_count": 2,
            "links": [
                {
                    "link_id": "ctxlink_000000",
                    "session_id": sid,
                    "context_id": "ctx_000001",
                    "segment_id": "seg_000001",
                    "mention_text": "my husband",
                    "alias_surface": "my husband",
                    "entity_id": "person_omar",
                    "canonical_name": "Omar",
                    "mention_type": "person",
                    "observation_kind": "surface_alias_seen_in_text",
                    "memory_kind": "resolved_alias_to_known_entity",
                    "resolution_status": "resolved_known_entity",
                    "confidence": 0.95,
                    "source": "curated_pack",
                    "support_text": "I spoke with my husband about money",
                },
                {
                    "link_id": "ctxlink_000001",
                    "session_id": sid,
                    "context_id": "ctx_000001",
                    "segment_id": "seg_000001",
                    "mention_text": "Alice Cooper",
                    "alias_surface": "Alice Cooper",
                    "entity_id": None,
                    "canonical_name": None,
                    "mention_type": "capitalized_phrase",
                    "observation_kind": "surface_alias_seen_in_text",
                    "memory_kind": "context_local_unresolved_mention",
                    "resolution_status": "context_local_unresolved",
                    "confidence": 0.58,
                    "source": "heuristic_surface",
                    "support_text": "I spoke with my husband about Alice Cooper yesterday",
                },
            ],
        })

        markers = [{
            "segment_id": "seg_000001",
            "entity_mentions": [
                {
                    "surface_form": "my husband",
                    "entity_id": "person_omar",
                    "mention_type": "person",
                    "confidence": 0.95,
                    "source": "curated_pack",
                },
                {
                    "surface_form": "Alice Cooper",
                    "entity_id": None,
                    "mention_type": "capitalized_phrase",
                    "confidence": 0.58,
                    "source": "heuristic_surface",
                },
            ],
            "topic_tags": ["relationship"],
            "topic_candidates": ["money_help"],
            "project_tags": [],
            "relation_tags": [],
            "emotion_tags": [],
            "retrieval_terms": ["my husband", "Alice Cooper"],
            "ambiguity_flags": [],
            "marker_confidence": 0.8,
        }]

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        result = run_memory_graph_update(sid, markers, StageStub())
        proposals = json.loads((sd / "memory" / "graph_update_proposals.json").read_text(encoding="utf-8"))
        kinds = {proposal["kind"] for proposal in proposals["proposals"]}
        assert result["proposal_count"] == proposals["proposal_count"]
        assert "context_entity_link" in kinds
        assert "alias_resolution_observation" in kinds
        assert "unresolved_context_alias_candidate" in kinds

    def test_graph_update_proposals_suppressed_by_quality_gate(self, tmp_sessions_dir, monkeypatch):
        import json
        from app.pipeline.memory_graph import run_memory_graph_update
        from app.storage.session_store import create_session, session_dir

        sid = create_session({"mode": "stream"})["session_id"]
        sd = session_dir(sid)

        monkeypatch.setattr(
            "app.pipeline.canonical_assembly.read_quality_gate",
            lambda session_id: {
                "session_quality_status": "unhealthy",
                "semantic_eligible": False,
                "memory_update_eligible": False,
                "reasons": ["phase4_test_gate"],
            },
        )

        class StageStub:
            def commit(self, artifacts=None):
                self.artifacts = artifacts or []

        result = run_memory_graph_update(sid, [], StageStub())
        proposals = json.loads((sd / "memory" / "graph_update_proposals.json").read_text(encoding="utf-8"))
        assert result["proposal_count"] == 0
        assert proposals["proposal_count"] == 0
        assert proposals["gate_status"] == "suppressed_by_quality_gate"


# ── Phase 5: NoSQL Projection ──────────────────────────────────────────


class TestNoSQLProjection:
    """Verify the NoSQL projection stage produces correct collection docs."""

    def test_project_segments_basic(self):
        from app.pipeline.nosql_projection import _project_segments

        segments = [
            {
                "segment_id": "seg_000000",
                "start_ms": 0,
                "end_ms": 15000,
                "text": "Hello world",
                "language": "en",
                "confidence": 0.9,
                "speaker": "SPEAKER_00",
                "stabilization_state": "stabilized",
                "segment_quality_status": "good",
                "corruption_flags": [],
                "source_model": "large_v3",
                "support_models": ["large_v3"],
                "support_windows": [0, 1],
            },
            {
                "segment_id": "seg_000001",
                "start_ms": 15000,
                "end_ms": 30000,
                "text": "Testing one two",
                "language": "en",
                "confidence": 0.85,
                "speaker": "SPEAKER_01",
                "stabilization_state": "stabilized",
                "segment_quality_status": "good",
                "corruption_flags": [],
                "source_model": "large_v3",
                "support_models": ["large_v3"],
                "support_windows": [1, 2],
            },
        ]

        seg_docs = _project_segments("test_session", segments)
        assert len(seg_docs) == 2
        assert seg_docs[0]["_collection"] == "segments"
        assert seg_docs[0]["_id"] == "test_session:seg_000000"
        assert seg_docs[0]["text"] == "Hello world"
        assert seg_docs[0]["duration_ms"] == 15000

    def test_project_segments_skips_missing_id(self):
        from app.pipeline.nosql_projection import _project_segments
        segments = [{"text": "no id"}, {"segment_id": "seg_000000", "start_ms": 0, "end_ms": 1000, "text": "ok"}]
        docs = _project_segments("sid", segments)
        assert len(docs) == 1
        assert docs[0]["segment_id"] == "seg_000000"

    def test_project_speaker_turns_from_real_diarization(self):
        """speaker_turns must come from canonical/speaker_turns.json, not synthesized."""
        from app.pipeline.nosql_projection import _project_speaker_turns
        speaker_turns_payload = {
            "session_id": "sid",
            "turns": [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0},
                {"speaker": "SPEAKER_01", "start": 5.0, "end": 12.0},
                {"speaker": "SPEAKER_00", "start": 12.0, "end": 18.5},
            ],
            "turn_count": 3,
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
        }
        diarization_status = {
            "status": "success",
            "reason": "policy_forced",
            "requested": True,
        }
        docs, status = _project_speaker_turns("sid", speaker_turns_payload, diarization_status)
        assert len(docs) == 3
        assert docs[0]["_collection"] == "speaker_turns"
        assert docs[0]["speaker"] == "SPEAKER_00"
        assert docs[0]["start_ms"] == 0
        assert docs[0]["end_ms"] == 5000
        assert docs[1]["speaker"] == "SPEAKER_01"
        assert status["diarization_status"] == "success"

    def test_project_speaker_turns_absent_yields_zero_docs(self):
        """When speaker_turns.json is absent, emit zero docs with explicit status."""
        from app.pipeline.nosql_projection import _project_speaker_turns
        docs, status = _project_speaker_turns("sid", {}, {"status": "skipped", "reason": "policy_off", "requested": False})
        assert len(docs) == 0
        assert status["diarization_status"] == "skipped"
        assert status["diarization_requested"] is False

    def test_project_speaker_turns_empty_turns_list(self):
        """Empty turns list (diarization ran but produced nothing) = zero docs."""
        from app.pipeline.nosql_projection import _project_speaker_turns
        docs, status = _project_speaker_turns(
            "sid",
            {"turns": [], "turn_count": 0},
            {"status": "no_speech", "reason": "auto", "requested": True},
        )
        assert len(docs) == 0
        assert status["diarization_status"] == "no_speech"

    def test_project_context_spans(self):
        from app.pipeline.nosql_projection import _project_context_spans
        spans_payload = {
            "spans": [
                {
                    "context_id": "ctx_abc123",
                    "start_ms": 0,
                    "end_ms": 30000,
                    "segment_ids": ["seg_000000", "seg_000001"],
                    "speaker_ids": ["SPEAKER_00"],
                    "language_profile": {"primary": "en"},
                    "topic_tags": ["work"],
                    "topic_candidates": ["project_test"],
                    "entity_ids": ["person_alice"],
                    "alias_hits": [],
                    "confidence": 0.8,
                    "continuity_evidence": [],
                }
            ]
        }
        docs = _project_context_spans("sid", spans_payload)
        assert len(docs) == 1
        assert docs[0]["_collection"] == "context_spans"
        assert docs[0]["context_id"] == "ctx_abc123"
        assert docs[0]["entity_ids"] == ["person_alice"]

    def test_project_entities(self):
        from app.pipeline.nosql_projection import _project_entities
        registry = {
            "entities": [
                {
                    "entity_id": "person_alice",
                    "entity_type": "person_or_entity",
                    "display_name": "Alice",
                    "aliases": ["Alice", "Al"],
                    "origin": "curated_pack",
                    "status": "active",
                    "confidence": 0.9,
                    "mention_count": 5,
                }
            ]
        }
        docs = _project_entities("sid", registry)
        assert len(docs) == 1
        assert docs[0]["_collection"] == "entities"
        assert docs[0]["display_name"] == "Alice"

    def test_project_aliases(self):
        from app.pipeline.nosql_projection import _project_aliases
        alias_graph = {
            "aliases": [
                {"alias": "alice", "entity_ids": ["person_alice"], "ambiguous": False},
                {"alias": "al", "entity_ids": ["person_alice", "person_alfred"], "ambiguous": True},
            ]
        }
        docs = _project_aliases("sid", alias_graph)
        assert len(docs) == 2
        assert docs[0]["_collection"] == "aliases"
        assert docs[1]["ambiguous"] is True

    def test_project_retrieval_docs_grounding(self):
        """retrieval_doc must point first to context_span, then to segment_ids."""
        from app.pipeline.nosql_projection import _project_retrieval_docs
        spans_payload = {
            "spans": [
                {
                    "context_id": "ctx_abc",
                    "segment_ids": ["seg_000000"],
                    "speaker_ids": ["SPEAKER_00"],
                    "language_profile": {"primary": "en"},
                    "topic_tags": ["work"],
                    "topic_candidates": [],
                    "entity_ids": ["person_alice"],
                    "confidence": 0.8,
                    "start_ms": 0,
                    "end_ms": 15000,
                }
            ]
        }
        segments = [{"segment_id": "seg_000000", "start_ms": 0, "end_ms": 15000, "text": "Hello Alice"}]
        marker_index = {
            "seg_000000": {
                "entity_mentions": [{"entity_id": "person_alice", "surface_form": "Alice"}],
                "retrieval_terms": ["alice", "hello"],
            }
        }
        docs = _project_retrieval_docs("sid", spans_payload, segments, marker_index)
        assert len(docs) == 1
        doc = docs[0]
        assert doc["_collection"] == "retrieval_docs"
        # Rule: grounding points first to context_span
        assert doc["grounding"]["context_span"] == "ctx_abc"
        # Then to canonical segment_ids
        assert doc["grounding"]["segment_ids"] == ["seg_000000"]
        assert "person_alice" in doc["entity_ids"]

    def test_empty_projection(self):
        from app.pipeline.nosql_projection import empty_projection
        payload = empty_projection("sid", "test_gate", ["reason1"])
        assert payload["total_doc_count"] == 0
        assert payload["gate_status"] == "test_gate"
        assert all(len(docs) == 0 for docs in payload["collections"].values())

    def test_no_threads_collection(self):
        """Phase 5 must NOT claim a threads collection -- thread_candidates.json is the real contract."""
        from app.pipeline.nosql_projection import empty_projection, COLLECTION_NAMES
        assert "threads" not in COLLECTION_NAMES
        payload = empty_projection("sid", "test")
        assert "threads" not in payload["collections"]

    def test_all_target_collections_present(self):
        from app.pipeline.nosql_projection import empty_projection, COLLECTION_NAMES
        payload = empty_projection("sid", "test")
        assert set(payload["collections"].keys()) == set(COLLECTION_NAMES)

    def test_session_doc_computes_counts_from_segments(self):
        """segment_count and word_count come from canonical segments, not metadata."""
        from app.pipeline.nosql_projection import _project_session
        meta = {"state": "done", "segment_count": 999, "word_count": 999}  # stale metadata
        gate = {"semantic_eligible": True, "reasons": []}
        segments = [
            {"segment_id": "seg_000000", "text": "Hello world foo"},
            {"segment_id": "seg_000001", "text": "Bar baz"},
        ]
        doc = _project_session("sid", meta, gate, segments)
        assert doc["segment_count"] == 2
        assert doc["word_count"] == 5  # 3 + 2


# ── Phase 6: Thread Linking ─────────────────────────────────────────────


class TestThreadLinking:
    """Verify cross-session thread linking logic."""

    def test_compute_similarity_shared_entities(self):
        from app.pipeline.thread_linking import _compute_similarity, _extract_span_signature
        span_a = {
            "context_id": "ctx_a",
            "session_id": "s1",
            "entity_ids": ["person_alice", "person_bob"],
            "topic_candidates": ["work"],
            "topic_tags": ["meeting"],
            "language_profile": {"primary": "en"},
            "speaker_ids": [],
            "start_ms": 0,
            "end_ms": 30000,
            "confidence": 0.8,
            "retrieval_terms": ["alice", "meeting", "project"],
        }
        span_b = {
            "context_id": "ctx_b",
            "session_id": "s2",
            "entity_ids": ["person_alice"],
            "topic_candidates": ["work"],
            "topic_tags": ["meeting"],
            "language_profile": {"primary": "en"},
            "speaker_ids": [],
            "start_ms": 0,
            "end_ms": 15000,
            "confidence": 0.7,
            "retrieval_terms": ["alice", "meeting", "deploy"],
        }
        sig_a = _extract_span_signature(span_a)
        sig_b = _extract_span_signature(span_b)
        score, evidence = _compute_similarity(sig_a, sig_b)
        assert score > 0.25  # Above MIN_THREAD_SCORE
        assert "shared_entities" in evidence
        assert "person_alice" in evidence["shared_entities"]

    def test_compute_similarity_no_overlap(self):
        from app.pipeline.thread_linking import _compute_similarity, _extract_span_signature
        span_a = {
            "context_id": "ctx_a",
            "session_id": "s1",
            "entity_ids": ["person_alice"],
            "topic_candidates": ["cooking"],
            "topic_tags": ["food"],
            "language_profile": {"primary": "en"},
            "speaker_ids": [],
            "start_ms": 0,
            "end_ms": 30000,
            "confidence": 0.8,
            "retrieval_terms": ["recipe", "ingredients"],
        }
        span_b = {
            "context_id": "ctx_b",
            "session_id": "s2",
            "entity_ids": ["person_charlie"],
            "topic_candidates": ["sports"],
            "topic_tags": ["tennis"],
            "language_profile": {"primary": "fr"},
            "speaker_ids": [],
            "start_ms": 0,
            "end_ms": 15000,
            "confidence": 0.7,
            "retrieval_terms": ["raquette", "match"],
        }
        sig_a = _extract_span_signature(span_a)
        sig_b = _extract_span_signature(span_b)
        score, evidence = _compute_similarity(sig_a, sig_b)
        assert score < 0.25  # Below threshold

    def test_build_thread_candidates_basic(self):
        from app.pipeline.thread_linking import build_thread_candidates
        source = [
            {
                "context_id": "ctx_a",
                "session_id": "s1",
                "entity_ids": ["person_alice"],
                "topic_candidates": ["banking"],
                "topic_tags": ["money"],
                "language_profile": {"primary": "en"},
                "speaker_ids": [],
                "start_ms": 0,
                "end_ms": 30000,
                "confidence": 0.8,
                "retrieval_terms": ["alice", "transfer", "banking"],
            }
        ]
        target = [
            {
                "context_id": "ctx_b",
                "session_id": "s2",
                "entity_ids": ["person_alice"],
                "topic_candidates": ["banking"],
                "topic_tags": ["money"],
                "language_profile": {"primary": "en"},
                "speaker_ids": [],
                "start_ms": 0,
                "end_ms": 15000,
                "confidence": 0.7,
                "retrieval_terms": ["alice", "payment", "banking"],
            }
        ]
        result = build_thread_candidates("s1", source, target)
        assert result["candidate_count"] >= 1
        assert result["candidates"][0]["source_session_id"] == "s1"
        assert result["candidates"][0]["target_session_id"] == "s2"

    def test_build_thread_candidates_empty_sources(self):
        from app.pipeline.thread_linking import build_thread_candidates
        result = build_thread_candidates("s1", [], [{"context_id": "ctx_b"}])
        assert result["candidate_count"] == 0

    def test_build_thread_candidates_empty_targets(self):
        from app.pipeline.thread_linking import build_thread_candidates
        result = build_thread_candidates("s1", [{"context_id": "ctx_a"}], [])
        assert result["candidate_count"] == 0

    def test_language_compatible(self):
        from app.pipeline.thread_linking import _language_compatible
        assert _language_compatible({"primary": "en"}, {"primary": "en"}) is True
        assert _language_compatible({"primary": "en"}, {"primary": "fr"}) is False
        assert _language_compatible({"primary": "en"}, {}) is True  # Unknown is compatible
        assert _language_compatible({}, {}) is True

    def test_group_candidates_into_threads(self):
        from app.pipeline.thread_linking import _group_candidates_into_threads
        candidates = [
            {
                "source_context_id": "ctx_a",
                "source_session_id": "s1",
                "target_context_id": "ctx_b",
                "target_session_id": "s2",
                "similarity_score": 0.7,
                "evidence": {
                    "shared_entities": ["person_alice"],
                    "shared_topics": ["banking"],
                },
            },
            {
                "source_context_id": "ctx_a",
                "source_session_id": "s1",
                "target_context_id": "ctx_c",
                "target_session_id": "s3",
                "similarity_score": 0.6,
                "evidence": {
                    "shared_entities": ["person_alice"],
                    "shared_topics": ["banking"],
                },
            },
        ]
        threads = _group_candidates_into_threads(candidates)
        # Both candidates share ctx_a and entity person_alice -> one thread
        assert len(threads) == 1
        assert threads[0]["session_count"] == 3  # s1, s2, s3
        assert "person_alice" in threads[0]["thread_entity_ids"]

    def test_group_candidates_separate_threads(self):
        from app.pipeline.thread_linking import _group_candidates_into_threads
        candidates = [
            {
                "source_context_id": "ctx_a",
                "source_session_id": "s1",
                "target_context_id": "ctx_b",
                "target_session_id": "s2",
                "similarity_score": 0.7,
                "evidence": {
                    "shared_entities": ["person_alice"],
                    "shared_topics": [],
                },
            },
            {
                "source_context_id": "ctx_x",
                "source_session_id": "s1",
                "target_context_id": "ctx_y",
                "target_session_id": "s4",
                "similarity_score": 0.5,
                "evidence": {
                    "shared_entities": ["person_charlie"],
                    "shared_topics": ["sports"],
                },
            },
        ]
        threads = _group_candidates_into_threads(candidates)
        # Different contexts and different entities -> separate threads
        assert len(threads) == 2

    def test_empty_candidates(self):
        from app.pipeline.thread_linking import empty_candidates
        payload = empty_candidates("sid", "test_gate", ["reason1"])
        assert payload["candidate_count"] == 0
        assert payload["gate_status"] == "test_gate"

    def test_enrich_spans_with_retrieval_terms(self):
        """Source spans must be enriched with retrieval_terms from segment_markers."""
        from app.pipeline.thread_linking import _enrich_spans_with_retrieval_terms
        spans = [
            {
                "context_id": "ctx_a",
                "segment_ids": ["seg_000000", "seg_000001"],
                "topic_tags": ["work"],
            }
        ]
        marker_index = {
            "seg_000000": {"retrieval_terms": ["alice", "deploy"]},
            "seg_000001": {"retrieval_terms": ["deploy", "server"]},
        }
        enriched = _enrich_spans_with_retrieval_terms(spans, marker_index)
        assert enriched[0]["retrieval_terms"] == ["alice", "deploy", "server"]
        # Original should not be mutated
        assert "retrieval_terms" not in spans[0]

    def test_retrieval_terms_for_span(self):
        from app.pipeline.thread_linking import _retrieval_terms_for_span
        span = {"segment_ids": ["seg_000000", "seg_000001"]}
        marker_index = {
            "seg_000000": {"retrieval_terms": ["foo", "bar"]},
            "seg_000001": {"retrieval_terms": ["bar", "baz"]},
        }
        terms = _retrieval_terms_for_span(span, marker_index)
        assert terms == ["foo", "bar", "baz"]  # Deduplicated, order preserved
