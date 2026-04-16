"""
Background Worker - Redis-backed job processor.

Single-threaded to serialize GPU use. Processes:
  - v2_canonical: Full canonical pipeline
  - v2_canonical_live: Incremental canonical lexical preview during recording
  - v2_retranscribe: Re-transcription with different model
  - v2 / legacy: Basic single-model transcription

VRAM discipline: models are loaded/unloaded between stages.
Resumability: canonical pipeline checkpoints per-stage.
"""
import gc
import json
import os
import sys
import time
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_config
from app.core.atomic_io import atomic_write_json, atomic_write_text, safe_read_json
from app.storage.session_store import (
    session_dir, get_session_meta, update_status, update_progress,
    get_chunk_paths, get_status,
)
from app.pipeline.run import (
    PipelineRun, create_canonical_run, get_canonical_run,
    CANONICAL_V1_STAGES, STAGE_DONE, RUN_ERROR,
)
from app.models.registry import (
    get_registry, resolve_model_id, get_model_info,
    DEFAULT_FIRST_PASS,
    DEFAULT_CANDIDATE_A, DEFAULT_CANDIDATE_B,
    VALID_MODEL_SIZES,
)


def _session_language_context(session_id: str, job_data: dict) -> dict:
    meta = get_session_meta(session_id) or {}
    allowed_languages = job_data.get("allowed_languages")
    if not isinstance(allowed_languages, list):
        allowed_languages = meta.get("allowed_languages") or []
    allowed_languages = [str(item).strip() for item in allowed_languages if str(item).strip()]

    forced_language = job_data.get("forced_language")
    if forced_language is None:
        forced_language = meta.get("forced_language")
    if forced_language == "auto":
        forced_language = None

    transcription_mode = (
        job_data.get("transcription_mode")
        or meta.get("transcription_mode")
        or "verbatim_multilingual"
    )

    requested_language = job_data.get("language")
    if requested_language in ("", "auto"):
        requested_language = None

    return {
        "allowed_languages": allowed_languages,
        "forced_language": forced_language,
        "transcription_mode": transcription_mode,
        "requested_language": requested_language,
    }


def _first_pass_language_evidence(session_id: str) -> dict:
    """Summarize successful first-pass language evidence for candidate-B routing."""
    cand_dir = session_dir(session_id) / "candidates"
    counts: Dict[str, int] = {}
    success_count = 0

    if not cand_dir.is_dir():
        return {"success_count": 0, "language_counts": {}, "dominant_language": None}

    for path in cand_dir.glob("cand_*.json"):
        candidate = safe_read_json(str(path)) or {}
        if candidate.get("model_id") != DEFAULT_FIRST_PASS:
            continue
        if not candidate.get("confidence_features", {}).get("success"):
            continue
        success_count += 1
        detected = str(candidate.get("language_evidence", {}).get("detected_language") or "").strip().lower()
        if detected:
            counts[detected] = counts.get(detected, 0) + 1

    dominant_language = None
    if counts:
        dominant_language = max(sorted(counts), key=lambda item: counts[item])

    return {
        "success_count": success_count,
        "language_counts": counts,
        "dominant_language": dominant_language,
    }


def _select_candidate_b_model(language_ctx: dict, session_id: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    reg = get_registry(refresh=True)
    forced = language_ctx.get("forced_language")
    allowed = language_ctx.get("allowed_languages") or []
    transcription_mode = language_ctx.get("transcription_mode", "verbatim_multilingual")
    parakeet = reg.get(DEFAULT_CANDIDATE_B)
    canary_id = "nemo-asr:canary-1b-v2"
    canary = reg.get(canary_id)

    english_only = forced == "en" or allowed == ["en"] or (
        not forced and transcription_mode != "verbatim_multilingual" and allowed == ["en"]
    )

    if english_only:
        if parakeet and parakeet.is_usable:
            return DEFAULT_CANDIDATE_B, "english_restricted_session"
        return None, "parakeet_unavailable_for_english_session"

    explicit_non_english = bool(forced and forced != "en") or any(lang != "en" for lang in allowed)
    if explicit_non_english:
        if canary and canary.is_usable:
            return canary_id, "explicit_non_english_session_routed_to_canary"
        return None, "canary_unavailable_for_explicit_non_english_session"

    first_pass_evidence = _first_pass_language_evidence(session_id) if session_id else {}
    dominant_language = first_pass_evidence.get("dominant_language")

    if dominant_language == "en":
        if parakeet and parakeet.is_usable:
            return DEFAULT_CANDIDATE_B, "first_pass_detected_english"
        if canary and canary.is_usable:
            return canary_id, "first_pass_detected_english_but_parakeet_unavailable"
        return None, "no_candidate_b_available_for_english_session"

    if dominant_language and dominant_language != "en":
        if canary and canary.is_usable:
            return canary_id, f"first_pass_detected_{dominant_language}"
        return None, f"canary_unavailable_for_{dominant_language}"

    if parakeet and parakeet.is_usable:
        return DEFAULT_CANDIDATE_B, "auto_session_defaulted_to_parakeet_pending_language_evidence"
    if canary and canary.is_usable:
        return canary_id, "auto_session_fallback_to_canary"

    return None, "candidate_b_skipped_for_multilingual_session"


def _candidate_b_execution_language_ctx(
    candidate_b_model: Optional[str],
    candidate_b_reason: Optional[str],
    language_ctx: dict,
) -> dict:
    """Translate routing decisions into provider-safe execution language hints."""
    effective = {
        "language": language_ctx.get("forced_language") or language_ctx.get("requested_language"),
        "allowed_languages": list(language_ctx.get("allowed_languages") or []),
        "forced_language": language_ctx.get("forced_language"),
        "transcription_mode": language_ctx.get("transcription_mode", "verbatim_multilingual"),
    }

    if candidate_b_model == DEFAULT_CANDIDATE_B:
        effective["language"] = "en"
        effective["allowed_languages"] = ["en"]
        effective["forced_language"] = "en"

    return effective


def _get_redis():
    """Get Redis connection."""
    try:
        import redis
        cfg = get_config()
        r = redis.Redis(host=cfg.redis.host, port=cfg.redis.port, decode_responses=True)
        r.ping()
        return r
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")
        return None


def _get_gpu_diagnostics() -> dict:
    """Probe GPU health with full CUDA runtime diagnostics."""
    cfg = get_config()
    result = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_reason": "not_checked",
        "config": {
            "whisper_device": cfg.worker.whisper_device,
            "whisper_compute_type": cfg.worker.whisper_compute_type,
            "whisper_compute_type_fallbacks": cfg.worker.whisper_compute_type_fallbacks,
            "strict_cuda": cfg.worker.strict_cuda,
            "cuda_device_index": cfg.worker.cuda_device_index,
        },
        "backend_status": {
            "torch_cuda_available": False,
            "torch_cuda_device_count": 0,
            "ctranslate2_cuda_device_count": 0,
            "faster_whisper_importable": False,
            "nemo_importable": False,
        },
    }

    # --- torch CUDA probe ---
    try:
        import torch
        result["backend_status"]["torch_version"] = torch.__version__
        result["backend_status"]["torch_cuda_built"] = torch.backends.cuda.is_built() if hasattr(torch.backends, 'cuda') else "unknown"
        if torch.cuda.is_available():
            result["gpu_available"] = True
            idx = cfg.worker.cuda_device_index
            result["gpu_name"] = torch.cuda.get_device_name(idx)
            result["gpu_reason"] = "cuda_available"
            props = torch.cuda.get_device_properties(idx)
            result["memory_total_mb"] = round(props.total_memory / 1024 / 1024)
            result["cuda_version"] = torch.version.cuda
            result["backend_status"]["torch_cuda_available"] = True
            result["backend_status"]["torch_cuda_device_count"] = torch.cuda.device_count()
        else:
            result["gpu_reason"] = "torch_cuda_not_available"
    except ImportError:
        result["gpu_reason"] = "torch_not_installed"
    except Exception as e:
        result["gpu_reason"] = f"torch_probe_error:{e}"

    # --- ctranslate2 CUDA probe ---
    try:
        import ctranslate2
        result["backend_status"]["ctranslate2_version"] = getattr(ctranslate2, "__version__", "unknown")
        cuda_devices = int(ctranslate2.get_cuda_device_count())
        result["backend_status"]["ctranslate2_cuda_device_count"] = cuda_devices
        if cuda_devices > 0 and not result["gpu_available"]:
            result["gpu_available"] = True
            result["gpu_reason"] = "ctranslate2_cuda_available"
            result["gpu_name"] = result.get("gpu_name") or "CUDA device via CTranslate2"
    except ImportError:
        result["backend_status"]["ctranslate2_error"] = "not_installed"
    except Exception as e:
        result["backend_status"]["ctranslate2_error"] = str(e)

    # --- faster-whisper import probe ---
    try:
        import faster_whisper
        result["backend_status"]["faster_whisper_importable"] = True
        result["backend_status"]["faster_whisper_version"] = getattr(faster_whisper, "__version__", "unknown")
    except ImportError as e:
        result["backend_status"]["faster_whisper_error"] = str(e)
    except Exception as e:
        result["backend_status"]["faster_whisper_error"] = str(e)

    # --- NeMo import probe ---
    try:
        import nemo.collections.asr  # noqa: F401
        result["backend_status"]["nemo_importable"] = True
    except ImportError:
        result["backend_status"]["nemo_error"] = "not_installed"
    except Exception as e:
        result["backend_status"]["nemo_error"] = str(e)

    # --- Resolve effective runtime ---
    from app.pipeline.asr_executor import _resolve_gpu_runtime
    runtime = _resolve_gpu_runtime(cfg)
    result["effective_device"] = runtime["device"]
    result["effective_compute_type"] = runtime["compute_type"]
    result["device_selection_method"] = runtime["method"]

    return result


def _write_worker_health(gpu_info: dict):
    """Write worker health file for the API to read."""
    cfg = get_config()
    health_path = str(Path(cfg.storage.sessions_dir).parent / "worker_health.json")
    data = {
        **gpu_info,
        "worker_started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        Path(health_path).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(health_path, data)
    except Exception as e:
        logger.warning(f"Failed to write worker health: {e}")


def _unload_all_vram():
    """Release all GPU memory."""
    try:
        from app.pipeline.asr_executor import unload_all_models
        unload_all_models()
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _unload_faster_whisper_only():
    """Unload faster-whisper/ctranslate2 without touching NeMo cache.

    Used between faster-whisper stages and NeMo stages to avoid the
    ctranslate2↔PyTorch CUDA allocator conflict while preserving
    NeMo model cache for reuse across windows.
    """
    try:
        from app.pipeline.asr_executor import unload_faster_whisper
        unload_faster_whisper()
    except Exception:
        pass


def _derive_action_hint(error_msg: str) -> str:
    """Map CUDA errors to human-readable hints."""
    msg = str(error_msg).lower()
    if "cuda" in msg and "out of memory" in msg:
        return "GPU out of memory. Try a smaller model or reduce batch size."
    if "cuda" in msg:
        return "GPU error. Check CUDA installation and driver version."
    if "no such file" in msg or "not found" in msg:
        return "Required file missing. Check model paths and audio files."
    return "Internal error. Check logs for details."


def _live_stage_root(session_id: str) -> Path:
    return session_dir(session_id) / "pipeline" / "live_preview"


class _LiveStage:
    """Minimal stage tracker for canonical live preview runs.

    Live processing must not interfere with the authoritative PipelineRun used
    for finalized sessions, but we still want per-stage diagnostics on disk and
    a writable stage_dir for routing artifacts.
    """

    def __init__(self, session_id: str, name: str):
        self.name = name
        self.session_id = session_id
        self.root = _live_stage_root(session_id)
        self.stage_dir = self.root / "stages" / name
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.status = "running"
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None
        self.error: Optional[str] = None
        self.artifacts: list[str] = []
        self.actual_model: Optional[str] = None
        self.routing_reason: Optional[str] = None
        self._save()

    def commit(self, artifacts=None):
        self.status = "done"
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.artifacts = list(artifacts or [])
        self._save()

    def commit_with_fallback(self, artifacts=None, reason: str = ""):
        self.status = "done"
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.artifacts = list(artifacts or [])
        self.error = f"fallback: {reason}" if reason else "fallback"
        self._save()

    def commit_degraded(self, reason: str, artifacts=None):
        self.status = "degraded"
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.artifacts = list(artifacts or [])
        self.error = reason
        self._save()

    def fail(self, error: str):
        self.status = "error"
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.error = error
        self._save()

    def skip(self, reason: str = ""):
        self.status = "skipped"
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.error = reason
        self._save()

    def _save(self):
        atomic_write_json(
            str(self.stage_dir / "stage_meta.json"),
            {
                "name": self.name,
                "status": self.status,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "artifacts": self.artifacts,
                "actual_model": self.actual_model,
                "routing_reason": self.routing_reason,
                "live_preview": True,
            },
        )


def _write_live_preview_state(
    session_id: str,
    chunk_count: int,
    audio_duration_ms: int,
    surfaces: Dict[str, Any] | None,
) -> None:
    canonical_dir = session_dir(session_id) / "canonical"
    stabilized_segments = (surfaces or {}).get("stabilized_segments") or []
    provisional_segments = [
        seg for seg in ((surfaces or {}).get("segments") or [])
        if seg.get("stabilization_state") != "stabilized"
    ]
    atomic_write_json(
        str(canonical_dir / "live_progress.json"),
        {
            "session_id": session_id,
            "live_preview": True,
            "chunk_count_at_run": chunk_count,
            "audio_duration_ms": audio_duration_ms,
            "stabilized_until_ms": max((seg.get("end_ms", 0) for seg in stabilized_segments), default=0),
            "open_tail_start_ms": min((seg.get("start_ms", 0) for seg in provisional_segments), default=None),
            "stabilized_segment_count": len(stabilized_segments),
            "provisional_segment_count": len(provisional_segments),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _clear_live_lock(session_id: str) -> None:
    lock = session_dir(session_id) / "live_canonical_pending"
    if lock.exists():
        try:
            lock.unlink()
        except OSError:
            pass


def process_canonical_live_job(job_data: dict):
    """Run the lexical canonical pipeline incrementally during an open stream.

    This intentionally stops after canonical assembly.  The finalized session
    still performs the authoritative full pass (including enrichment/memory),
    but live users get true canonical geometry instead of a fake preview path.
    """
    session_id = job_data["session_id"]
    sd = session_dir(session_id)
    logger.info("Starting canonical live pipeline for %s", session_id)

    try:
        status = get_status(session_id) or {}
        meta = get_session_meta(session_id) or {}
        if status.get("status") in ("done", "cancelled"):
            logger.info("Skipping live canonical for %s; status=%s", session_id, status.get("status"))
            return
        if meta.get("state") == "finalized" and (sd / "canonical" / "final_transcript.json").is_file():
            logger.info("Skipping live canonical for %s; finalized output already present", session_id)
            return

        chunk_paths = get_chunk_paths(session_id)
        if len(chunk_paths) < max(1, get_config().worker.partial_every_n_chunks):
            return

        update_status(
            session_id,
            "running",
            live_preview=True,
            live_chunk_count=len(chunk_paths),
            live_generated_at=datetime.now(timezone.utc).isoformat(),
        )
        update_progress(session_id, "canonical_live_initializing", 0)

        language_ctx = _session_language_context(session_id, job_data)
        language = language_ctx["forced_language"] or language_ctx["requested_language"]

        # Stage 1: normalize audio
        update_progress(session_id, "normalize_audio", 5)
        stage = _LiveStage(session_id, "normalize_audio")
        from app.pipeline.ingest import run_ingest_stage
        ingest_result = run_ingest_stage(session_id, stage)
        audio_duration_ms = ingest_result["audio_duration_ms"]
        normalized_audio = str(sd / "normalized" / "audio.wav")

        # Stage 2: acoustic triage
        update_progress(session_id, "acoustic_triage", 18)
        speech_islands = None
        stage = _LiveStage(session_id, "acoustic_triage")
        try:
            from app.pipeline.acoustic_triage import run_acoustic_triage
            run_acoustic_triage(session_id, normalized_audio, audio_duration_ms, stage)
            islands_data = safe_read_json(str(sd / "triage" / "speech_islands.json"))
            speech_islands = (islands_data or {}).get("islands", [])
        except Exception as e:
            stage.commit_with_fallback([], f"triage_error: {e}")

        # Stage 3: decode lattice, but do not create truncated trailing windows
        update_progress(session_id, "decode_lattice", 28)
        stage = _LiveStage(session_id, "decode_lattice")
        from app.pipeline.decode_lattice import run_decode_lattice
        lattice_result = run_decode_lattice(
            session_id,
            normalized_audio,
            audio_duration_ms,
            speech_islands,
            stage,
            allow_trailing_partial_window=False,
        )
        windows = lattice_result.get("windows", [])

        # Stage 4: first-pass ASR
        update_progress(session_id, "first_pass_medium", 36)
        stage = _LiveStage(session_id, "first_pass_medium")
        stage.actual_model = DEFAULT_FIRST_PASS
        stage.routing_reason = "default_first_pass"
        from app.pipeline.asr_executor import run_asr_execution
        summary = run_asr_execution(
            session_id,
            windows,
            [DEFAULT_FIRST_PASS],
            language=language,
            allowed_languages=language_ctx["allowed_languages"],
            forced_language=language_ctx["forced_language"],
            transcription_mode=language_ctx["transcription_mode"],
            progress_callback=lambda p: update_progress(session_id, "first_pass_medium", 36 + int(p * 0.08)),
        )
        atomic_write_json(str(stage.stage_dir / "first_pass_routing.json"), {
            "selected_model": DEFAULT_FIRST_PASS,
            "language_context": language_ctx,
            "asr_summary": summary,
        })
        model_stats = (summary or {}).get("candidates_by_model", {}).get(DEFAULT_FIRST_PASS, {})
        if model_stats.get("success", 0) == 0 and model_stats.get("count", 0) > 0:
            stage.commit_degraded(
                f"all {model_stats['count']} windows failed for {DEFAULT_FIRST_PASS}",
                ["first_pass_routing.json"],
            )
        else:
            stage.commit(["first_pass_routing.json"])

        _unload_all_vram()

        first_pass_evidence = _first_pass_language_evidence(session_id)
        detected_language = first_pass_evidence.get("dominant_language")
        effective_language = language
        if not effective_language and detected_language:
            effective_language = detected_language

        # Stage 5: candidate A
        update_progress(session_id, "candidate_asr_large_v3", 46)
        stage = _LiveStage(session_id, "candidate_asr_large_v3")
        stage.actual_model = DEFAULT_CANDIDATE_A
        stage.routing_reason = "default_candidate_a"
        summary = run_asr_execution(
            session_id,
            windows,
            [DEFAULT_CANDIDATE_A],
            language=effective_language,
            allowed_languages=language_ctx["allowed_languages"],
            forced_language=language_ctx["forced_language"],
            transcription_mode=language_ctx["transcription_mode"],
            progress_callback=lambda p: update_progress(session_id, "candidate_asr_large_v3", 46 + int(p * 0.1)),
        )
        atomic_write_json(str(stage.stage_dir / "candidate_a_routing.json"), {
            "selected_model": DEFAULT_CANDIDATE_A,
            "language_context": language_ctx,
            "asr_summary": summary,
        })
        model_stats = (summary or {}).get("candidates_by_model", {}).get(DEFAULT_CANDIDATE_A, {})
        if model_stats.get("success", 0) == 0 and model_stats.get("count", 0) > 0:
            stage.commit_degraded(
                f"all {model_stats['count']} windows failed for {DEFAULT_CANDIDATE_A}",
                ["candidate_a_routing.json"],
            )
        else:
            stage.commit(["candidate_a_routing.json"])

        _unload_faster_whisper_only()

        # Stage 6: candidate B
        update_progress(session_id, "candidate_asr_secondary", 58)
        stage = _LiveStage(session_id, "candidate_asr_secondary")
        candidate_b_model, candidate_b_reason = _select_candidate_b_model(language_ctx, session_id=session_id)
        candidate_b_language_ctx = _candidate_b_execution_language_ctx(
            candidate_b_model,
            candidate_b_reason,
            language_ctx,
        )
        stage.actual_model = candidate_b_model
        stage.routing_reason = candidate_b_reason
        if candidate_b_model:
            summary = run_asr_execution(
                session_id,
                windows,
                [candidate_b_model],
                language=candidate_b_language_ctx["language"],
                allowed_languages=candidate_b_language_ctx["allowed_languages"],
                forced_language=candidate_b_language_ctx["forced_language"],
                transcription_mode=candidate_b_language_ctx["transcription_mode"],
                progress_callback=lambda p: update_progress(session_id, "candidate_asr_secondary", 58 + int(p * 0.1)),
            )
            atomic_write_json(str(stage.stage_dir / "candidate_b_routing.json"), {
                "selected_model": candidate_b_model,
                "reason": candidate_b_reason,
                "language_context": language_ctx,
                "effective_language_context": candidate_b_language_ctx,
                "first_pass_language_evidence": first_pass_evidence,
                "asr_summary": summary,
            })
            model_stats = (summary or {}).get("candidates_by_model", {}).get(candidate_b_model, {})
            if model_stats.get("success", 0) == 0 and model_stats.get("count", 0) > 0:
                stage.commit_degraded(
                    f"all {model_stats['count']} windows failed for {candidate_b_model}",
                    ["candidate_b_routing.json"],
                )
            else:
                stage.commit(["candidate_b_routing.json"])
        else:
            atomic_write_json(str(stage.stage_dir / "candidate_b_routing.json"), {
                "selected_model": None,
                "reason": candidate_b_reason,
                "language_context": language_ctx,
                "effective_language_context": candidate_b_language_ctx,
                "first_pass_language_evidence": first_pass_evidence,
            })
            stage.commit_with_fallback(["candidate_b_routing.json"], candidate_b_reason)

        _unload_all_vram()

        cand_dir = sd / "candidates"
        successful_candidates = 0
        total_candidates = 0
        if cand_dir.is_dir():
            for f in cand_dir.glob("cand_*.json"):
                c = safe_read_json(str(f))
                if c and any(w.get("window_id") == c.get("window_id") for w in windows):
                    total_candidates += 1
                    if c.get("confidence_features", {}).get("success"):
                        successful_candidates += 1
        if successful_candidates == 0:
            raise RuntimeError(
                f"Live canonical ASR failed — zero successful candidates out of {total_candidates} attempted."
            )

        # Stage 7: stripe grouping
        update_progress(session_id, "stripe_grouping", 70)
        stage = _LiveStage(session_id, "stripe_grouping")
        from app.pipeline.stripe_grouping import run_stripe_grouping
        grouping_result = run_stripe_grouping(session_id, windows, audio_duration_ms, stage)
        stripe_packets = grouping_result.get("stripes", [])

        # Stage 8: reconciliation
        update_progress(session_id, "reconciliation", 79)
        stage = _LiveStage(session_id, "reconciliation")
        from app.pipeline.reconciliation import run_reconciliation
        reconciliation_result = run_reconciliation(session_id, stripe_packets, stage)

        _unload_all_vram()

        # Stage 9: canonical assembly, but never flush trailing single-support stripe as final.
        update_progress(session_id, "canonical_assembly", 87)
        stage = _LiveStage(session_id, "canonical_assembly")
        from app.pipeline.canonical_assembly import run_canonical_assembly, build_transcript_surfaces
        if reconciliation_result.get("records"):
            assembly_result = run_canonical_assembly(
                session_id,
                reconciliation_result,
                stripe_packets,
                stage,
                finalize_last_boundary=False,
                emit_final_surface=False,
            )
        else:
            surfaces = build_transcript_surfaces(
                [],
                session_id,
                stripe_decisions=[],
                emit_final_surface=False,
            )
            stage.commit([
                "transcript.txt",
                "canonical_segments.json",
                "provenance.json",
                "provisional_partial.json",
                "stabilized_partial.json",
                "quality_gate.json",
            ])
            assembly_result = {"surfaces": surfaces, "segment_count": 0}

        surfaces = assembly_result.get("surfaces") or {}
        _write_live_preview_state(session_id, len(chunk_paths), audio_duration_ms, surfaces)
        atomic_write_json(
            str(_live_stage_root(session_id) / "live_meta.json"),
            {
                "session_id": session_id,
                "chunk_count_at_run": len(chunk_paths),
                "audio_duration_ms": audio_duration_ms,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "finalized": False,
                "status": "done",
            },
        )

        from app.storage.session_store import update_session_meta
        update_session_meta(
            session_id,
            {
                "last_live_chunk_count_processed": len(chunk_paths),
                "last_live_processed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        update_status(
            session_id,
            "running",
            live_preview=True,
            live_chunk_count=len(chunk_paths),
            live_stabilized_until_ms=max(
                (seg.get("end_ms", 0) for seg in (surfaces.get("stabilized_segments") or [])),
                default=0,
            ),
        )

        logger.info(
            "Canonical live pipeline complete for %s: chunks=%d stabilized_segments=%d",
            session_id,
            len(chunk_paths),
            len(surfaces.get("stabilized_segments") or []),
        )

    except Exception as e:
        logger.warning("Canonical live pipeline failed for %s: %s", session_id, e)
        atomic_write_json(
            str(sd / "live_error.json"),
            {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        update_status(session_id, "running", live_preview_error=str(e), live_preview=True)
    finally:
        _unload_all_vram()
        _clear_live_lock(session_id)


def process_canonical_pipeline(job_data: dict):
    """Process a full canonical pipeline job aligned to the specification.

    This is the primary job type. It implements the canonical spec:
    30s windows, 15s stride, 15s stripes, multi-ASR, reconciliation.
    """
    session_id = job_data["session_id"]
    sd = session_dir(session_id)
    logger.info(f"Starting canonical pipeline for {session_id}")

    # Check idempotency
    status = get_status(session_id)
    if status and status.get("status") in ("done", "cancelled"):
        logger.info(f"Session {session_id} already {status['status']}, skipping")
        return

    update_status(session_id, "running")
    update_progress(session_id, "initializing", 0)

    language_ctx = _session_language_context(session_id, job_data)
    language = language_ctx["forced_language"] or language_ctx["requested_language"]

    run = None
    try:
        # Create or resume pipeline run
        existing = get_canonical_run(str(sd))
        if existing and existing.status == RUN_ERROR:
            existing = None  # Restart failed runs

        if existing and existing.status != RUN_ERROR:
            run = existing
            logger.info(f"Resuming pipeline run {run.run_id}")
        else:
            run = create_canonical_run(str(sd), session_id, job_data)
            logger.info(f"Created new pipeline run {run.run_id}")

        run.start()

        # ============================================================
        # Stage 1: Normalize Audio
        # ============================================================
        if not run.is_stage_done("normalize_audio"):
            update_progress(session_id, "normalize_audio", 5)
            stage = run.start_stage("normalize_audio")
            try:
                from app.pipeline.ingest import run_ingest_stage
                ingest_result = run_ingest_stage(session_id, stage)
                audio_duration_ms = ingest_result["audio_duration_ms"]
            except Exception as e:
                stage.fail(str(e))
                raise
        else:
            # Load cached result
            norm_meta = safe_read_json(str(sd / "normalized" / "norm_meta.json")) or {}
            audio_duration_ms = norm_meta.get("normalized_duration_ms", 0)

        normalized_audio = str(sd / "normalized" / "audio.wav")

        # ============================================================
        # Stage 2: Acoustic Triage
        # ============================================================
        speech_islands = None
        if not run.is_stage_done("acoustic_triage"):
            update_progress(session_id, "acoustic_triage", 18)
            stage = run.start_stage("acoustic_triage")
            try:
                from app.pipeline.acoustic_triage import run_acoustic_triage
                triage_result = run_acoustic_triage(
                    session_id, normalized_audio, audio_duration_ms, stage
                )
                islands_data = safe_read_json(str(sd / "triage" / "speech_islands.json"))
                speech_islands = (islands_data or {}).get("islands", [])
            except Exception as e:
                stage.commit_with_fallback([], f"triage_error: {e}")
        else:
            islands_data = safe_read_json(str(sd / "triage" / "speech_islands.json"))
            if islands_data is not None:
                speech_islands = islands_data.get("islands", [])
            elif (run.get_stage("acoustic_triage").error or "").startswith("fallback:"):
                speech_islands = None
            else:
                speech_islands = []

        # ============================================================
        # Stage 3: Decode Lattice
        # ============================================================
        windows = []
        if not run.is_stage_done("decode_lattice"):
            update_progress(session_id, "decode_lattice", 28)
            stage = run.start_stage("decode_lattice")
            try:
                from app.pipeline.decode_lattice import run_decode_lattice
                lattice_result = run_decode_lattice(
                    session_id, normalized_audio, audio_duration_ms, speech_islands, stage
                )
                windows = lattice_result.get("windows", [])
            except Exception as e:
                stage.fail(str(e))
                raise
        else:
            lattice_data = safe_read_json(str(sd / "windows" / "decode_windows.json"))
            windows = (lattice_data or {}).get("windows", [])

        # ============================================================
        # Stage 4: First Pass ASR - Medium
        # ============================================================
        if not run.is_stage_done("first_pass_medium"):
            update_progress(session_id, "first_pass_medium", 36)
            stage = run.start_stage("first_pass_medium")
            stage.actual_model = DEFAULT_FIRST_PASS
            stage.routing_reason = "default_first_pass"
            try:
                from app.pipeline.asr_executor import run_asr_execution
                summary = run_asr_execution(
                    session_id, windows, [DEFAULT_FIRST_PASS],
                    language=language,
                    allowed_languages=language_ctx["allowed_languages"],
                    forced_language=language_ctx["forced_language"],
                    transcription_mode=language_ctx["transcription_mode"],
                    progress_callback=lambda p: update_progress(
                        session_id, "first_pass_medium", 36 + int(p * 0.08)
                    ),
                )
                atomic_write_json(str(stage.stage_dir / "first_pass_routing.json"), {
                    "selected_model": DEFAULT_FIRST_PASS,
                    "language_context": language_ctx,
                    "asr_summary": summary,
                })
                model_stats = (summary or {}).get("candidates_by_model", {}).get(DEFAULT_FIRST_PASS, {})
                if model_stats.get("success", 0) == 0 and model_stats.get("count", 0) > 0:
                    stage.commit_degraded(
                        f"all {model_stats['count']} windows failed for {DEFAULT_FIRST_PASS}",
                        ["first_pass_routing.json"],
                    )
                else:
                    stage.commit(["first_pass_routing.json"])
            except Exception as e:
                stage.commit_degraded(f"provider_crash: {e}")

        _unload_all_vram()

        # Propagate first-pass language detection to guide subsequent models.
        # This prevents large-v3 from mis-detecting Slavic languages (e.g. ru→pl).
        first_pass_evidence = _first_pass_language_evidence(session_id)
        detected_language = first_pass_evidence.get("dominant_language")
        effective_language = language  # user-forced or requested language takes priority
        if not effective_language and detected_language:
            effective_language = detected_language
            logger.info(
                "Propagating first-pass detected language '%s' to subsequent ASR stages",
                detected_language,
            )

        # ============================================================
        # Stage 5: Candidate ASR - Large V3
        # ============================================================
        if not run.is_stage_done("candidate_asr_large_v3"):
            update_progress(session_id, "candidate_asr_large_v3", 46)
            stage = run.start_stage("candidate_asr_large_v3")
            stage.actual_model = DEFAULT_CANDIDATE_A
            stage.routing_reason = "default_candidate_a"
            try:
                from app.pipeline.asr_executor import run_asr_execution
                summary = run_asr_execution(
                    session_id, windows, [DEFAULT_CANDIDATE_A],
                    language=effective_language,
                    allowed_languages=language_ctx["allowed_languages"],
                    forced_language=language_ctx["forced_language"],
                    transcription_mode=language_ctx["transcription_mode"],
                    progress_callback=lambda p: update_progress(
                        session_id, "candidate_asr_large_v3", 46 + int(p * 0.1)
                    ),
                )
                atomic_write_json(str(stage.stage_dir / "candidate_a_routing.json"), {
                    "selected_model": DEFAULT_CANDIDATE_A,
                    "language_context": language_ctx,
                    "asr_summary": summary,
                })
                model_stats = (summary or {}).get("candidates_by_model", {}).get(DEFAULT_CANDIDATE_A, {})
                if model_stats.get("success", 0) == 0 and model_stats.get("count", 0) > 0:
                    stage.commit_degraded(
                        f"all {model_stats['count']} windows failed for {DEFAULT_CANDIDATE_A}",
                        ["candidate_a_routing.json"],
                    )
                else:
                    stage.commit(["candidate_a_routing.json"])
            except Exception as e:
                stage.commit_degraded(f"provider_crash: {e}")

        # Targeted unload: destroy ctranslate2 state without clearing NeMo cache.
        # This avoids the CUDA allocator handle conflict between ctranslate2 and PyTorch.
        _unload_faster_whisper_only()

        # ============================================================
        # Stage 6: Candidate ASR - Secondary route
        # ============================================================
        if not run.is_stage_done("candidate_asr_secondary"):
            update_progress(session_id, "candidate_asr_secondary", 58)
            stage = run.start_stage("candidate_asr_secondary")
            try:
                from app.pipeline.asr_executor import run_asr_execution
                first_pass_evidence = _first_pass_language_evidence(session_id)
                candidate_b_model, candidate_b_reason = _select_candidate_b_model(language_ctx, session_id=session_id)
                candidate_b_language_ctx = _candidate_b_execution_language_ctx(
                    candidate_b_model,
                    candidate_b_reason,
                    language_ctx,
                )
                stage.actual_model = candidate_b_model
                stage.routing_reason = candidate_b_reason
                if candidate_b_model:
                    summary = run_asr_execution(
                        session_id,
                        windows,
                        [candidate_b_model],
                        language=candidate_b_language_ctx["language"],
                        allowed_languages=candidate_b_language_ctx["allowed_languages"],
                        forced_language=candidate_b_language_ctx["forced_language"],
                        transcription_mode=candidate_b_language_ctx["transcription_mode"],
                        progress_callback=lambda p: update_progress(
                            session_id, "candidate_asr_secondary", 58 + int(p * 0.1)
                        ),
                    )
                    atomic_write_json(str(stage.stage_dir / "candidate_b_routing.json"), {
                        "selected_model": candidate_b_model,
                        "reason": candidate_b_reason,
                        "language_context": language_ctx,
                        "effective_language_context": candidate_b_language_ctx,
                        "first_pass_language_evidence": first_pass_evidence,
                        "asr_summary": summary,
                    })
                    model_stats = (summary or {}).get("candidates_by_model", {}).get(candidate_b_model, {})
                    if model_stats.get("success", 0) == 0 and model_stats.get("count", 0) > 0:
                        stage.commit_degraded(
                            f"all {model_stats['count']} windows failed for {candidate_b_model}",
                            ["candidate_b_routing.json"],
                        )
                    else:
                        stage.commit(["candidate_b_routing.json"])
                else:
                    atomic_write_json(str(stage.stage_dir / "candidate_b_routing.json"), {
                        "selected_model": None,
                        "reason": candidate_b_reason,
                        "language_context": language_ctx,
                        "effective_language_context": candidate_b_language_ctx,
                        "first_pass_language_evidence": first_pass_evidence,
                    })
                    stage.commit_with_fallback(["candidate_b_routing.json"], candidate_b_reason)
            except Exception as e:
                stage.actual_model = stage.actual_model or "unknown"
                stage.commit_degraded(f"provider_crash: {e}")

        _unload_all_vram()

        # ============================================================
        # Post-ASR validation: fail early if zero candidates succeeded
        # ============================================================
        cand_dir = sd / "candidates"
        successful_candidates = 0
        total_candidates = 0
        if cand_dir.is_dir():
            for f in cand_dir.glob("cand_*.json"):
                c = safe_read_json(str(f))
                if c:
                    total_candidates += 1
                    if c.get("confidence_features", {}).get("success"):
                        successful_candidates += 1
        if successful_candidates == 0:
            asr_stages = ["first_pass_medium", "candidate_asr_large_v3", "candidate_asr_secondary"]
            stage_errors = []
            for sn in asr_stages:
                s = run.get_stage(sn)
                if s.error:
                    stage_errors.append(f"{sn}: {s.error}")
            error_detail = "; ".join(stage_errors) if stage_errors else "unknown"
            raise RuntimeError(
                f"All ASR models failed — zero successful candidates out of "
                f"{total_candidates} attempted. Stage errors: {error_detail}"
            )

        # ============================================================
        # Stage 7: Stripe Grouping
        # ============================================================
        stripe_packets = []
        if not run.is_stage_done("stripe_grouping"):
            update_progress(session_id, "stripe_grouping", 70)
            stage = run.start_stage("stripe_grouping")
            try:
                from app.pipeline.stripe_grouping import run_stripe_grouping
                grouping_result = run_stripe_grouping(
                    session_id, windows, audio_duration_ms, stage
                )
                stripe_packets = grouping_result.get("stripes", [])
            except Exception as e:
                stage.commit_with_fallback([], f"grouping_error: {e}")
        else:
            sp_data = safe_read_json(str(sd / "reconciliation" / "stripe_packets.json"))
            stripe_packets = (sp_data or {}).get("stripes", [])

        # ============================================================
        # Stage 8: Reconciliation
        # ============================================================
        reconciliation_result = {}
        if not run.is_stage_done("reconciliation"):
            update_progress(session_id, "reconciliation", 79)
            stage = run.start_stage("reconciliation")
            try:
                from app.pipeline.reconciliation import run_reconciliation
                reconciliation_result = run_reconciliation(
                    session_id, stripe_packets, stage
                )
            except Exception as e:
                stage.commit_with_fallback([], f"reconciliation_error: {e}")
        else:
            reconciliation_result = safe_read_json(
                str(sd / "reconciliation" / "reconciliation_result.json")
            ) or {}

        _unload_all_vram()

        # ============================================================
        # Stage 9: Canonical Assembly
        # ============================================================
        segments = []
        text = ""
        if not run.is_stage_done("canonical_assembly"):
            update_progress(session_id, "canonical_assembly", 87)
            stage = run.start_stage("canonical_assembly")
            try:
                from app.pipeline.canonical_assembly import run_canonical_assembly, build_transcript_surfaces

                # If reconciliation produced records, use them
                if reconciliation_result.get("records"):
                    assembly_result = run_canonical_assembly(
                        session_id, reconciliation_result, stripe_packets, stage
                    )
                else:
                    surfaces = build_transcript_surfaces([], session_id, stripe_decisions=[])
                    text = surfaces["text"]
                    stage.commit()
                    assembly_result = {"surfaces": surfaces, "segment_count": 0}

                if assembly_result.get("surfaces"):
                    segments = assembly_result["surfaces"].get("segments", [])
                    text = assembly_result["surfaces"].get("text", "")
            except Exception as e:
                stage.fail(str(e))
                raise
        else:
            canon_data = safe_read_json(str(sd / "canonical" / "canonical_segments.json"))
            if canon_data:
                segments = canon_data.get("segments", [])
            text_path = sd / "canonical" / "transcript.txt"
            if text_path.is_file():
                text = text_path.read_text(encoding="utf-8")

        # ============================================================
        # Stage 10: Selective Enrichment
        # ============================================================
        if not run.is_stage_done("selective_enrichment"):
            update_progress(session_id, "selective_enrichment", 93)
            stage = run.start_stage("selective_enrichment")
            try:
                from app.pipeline.selective_enrichment import run_selective_enrichment
                enrich_result = run_selective_enrichment(
                    session_id, segments, normalized_audio, stage
                )
                # Reload segments after enrichment
                canon_data = safe_read_json(str(sd / "canonical" / "canonical_segments.json"))
                if canon_data:
                    segments = canon_data.get("segments", [])
            except Exception as e:
                stage.commit_with_fallback([], f"enrichment_error: {e}")

        _unload_all_vram()

        # ============================================================
        # Stage 11: Semantic Marking
        # ============================================================
        if not run.is_stage_done("semantic_marking"):
            update_progress(session_id, "semantic_marking", 95)
            stage = run.start_stage("semantic_marking")
            try:
                from app.pipeline.semantic_marking import run_semantic_marking
                # run_semantic_marking now also emits enrichment/context_spans.json
                # (Phase 2 -- continuity-based grouping of canonical segments).
                marking_result = run_semantic_marking(session_id, segments, stage)
                logger.info(
                    "semantic_marking done for %s: markers=%s, semantic_spans=%s, context_spans=%s",
                    session_id,
                    marking_result.get("marker_count", 0),
                    marking_result.get("semantic_span_count", 0),
                    marking_result.get("context_span_count", 0),
                )
            except Exception as e:
                stage.commit_with_fallback([], f"semantic_marking_error: {e}")

        # ============================================================
        # Stage 12: Memory Graph Update
        # ============================================================
        if not run.is_stage_done("memory_graph_update"):
            update_progress(session_id, "memory_graph_update", 97)
            stage = run.start_stage("memory_graph_update")
            try:
                from app.pipeline.memory_graph import run_memory_graph_update
                marker_payload = safe_read_json(str(sd / "enrichment" / "segment_markers.json")) or {}
                markers = marker_payload.get("markers") or []
                run_memory_graph_update(session_id, markers, stage)
            except Exception as e:
                stage.commit_with_fallback([], f"memory_graph_error: {e}")

        # ============================================================
        # Stage 13: Derived Outputs
        # ============================================================
        if not run.is_stage_done("derived_outputs"):
            update_progress(session_id, "derived_outputs", 96)
            stage = run.start_stage("derived_outputs")
            try:
                from app.pipeline.derived_outputs import run_derived_outputs
                run_derived_outputs(session_id, segments, text, audio_duration_ms, stage)
            except Exception as e:
                stage.commit_with_fallback([], f"derived_error: {e}")

        # ============================================================
        # Stage 14: NoSQL Projection
        # ============================================================
        if not run.is_stage_done("nosql_projection"):
            update_progress(session_id, "nosql_projection", 98)
            stage = run.start_stage("nosql_projection")
            try:
                from app.pipeline.nosql_projection import run_nosql_projection
                nosql_result = run_nosql_projection(session_id, stage)
                logger.info(
                    "nosql_projection done for %s: %d docs",
                    session_id,
                    nosql_result.get("total_doc_count", 0),
                )
            except Exception as e:
                stage.commit_with_fallback([], f"nosql_projection_error: {e}")

        # ============================================================
        # Stage 15: Thread Linking (cross-session)
        # ============================================================
        if not run.is_stage_done("thread_linking"):
            update_progress(session_id, "thread_linking", 99)
            stage = run.start_stage("thread_linking")
            try:
                from app.pipeline.thread_linking import run_thread_linking
                thread_result = run_thread_linking(session_id, stage)
                logger.info(
                    "thread_linking done for %s: %d candidates, %d threads",
                    session_id,
                    thread_result.get("candidate_count", 0),
                    thread_result.get("thread_count", 0),
                )
            except Exception as e:
                stage.commit_with_fallback([], f"thread_linking_error: {e}")

        # ============================================================
        # Pipeline Complete
        # ============================================================
        run.complete()
        update_status(session_id, "done")
        update_progress(session_id, "done", 100)

        # Write metadata for API
        meta = get_session_meta(session_id) or {}
        meta_update = {
            "audio_duration_s": round(audio_duration_ms / 1000, 1),
            "segment_count": len(segments),
            "word_count": sum(len(s.get("text", "").split()) for s in segments),
        }
        from app.storage.session_store import update_session_meta
        update_session_meta(session_id, meta_update)

        logger.info(f"Canonical pipeline complete for {session_id}: {len(segments)} segments")

    except Exception as e:
        logger.error(f"Pipeline failed for {session_id}: {e}\n{traceback.format_exc()}")
        if run:
            run.fail(str(e))
        error_data = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "action_hint": _derive_action_hint(str(e)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(str(sd / "error.json"), error_data)
        update_status(session_id, "error", error=str(e))
    finally:
        _unload_all_vram()


def process_partial_job(job_data: dict):
    """Quick partial transcript for live preview during recording."""
    session_id = job_data["session_id"]
    sd = session_dir(session_id)
    logger.info(f"Processing partial for {session_id}")

    try:
        chunk_paths = get_chunk_paths(session_id)
        if not chunk_paths:
            return
        language_ctx = _session_language_context(session_id, job_data)

        # Merge available chunks into partial audio
        partial_audio = str(sd / "audio_partial.wav")
        from app.pipeline.ingest import merge_chunks
        merge_result = merge_chunks(chunk_paths, partial_audio)
        if not merge_result.get("success"):
            return

        # Quick transcription with base model
        from app.pipeline.asr_executor import transcribe_window
        result = transcribe_window(
            partial_audio,
            "faster-whisper:base",
            allowed_languages=language_ctx["allowed_languages"],
            forced_language=language_ctx["forced_language"],
            transcription_mode=language_ctx["transcription_mode"],
        )

        atomic_write_json(str(sd / "partial_transcript.json"), {
            "session_id": session_id,
            "provisional": True,
            "semantic_layer": "provisional_partial",
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "chunk_count_at_time": len(chunk_paths),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "degraded": result.get("degraded", False),
            "fallback_reason": result.get("error"),
        })

        from app.storage.session_store import update_session_meta
        update_session_meta(session_id, {"state": "partially_processed"})

    except Exception as e:
        logger.warning(f"Partial transcription failed for {session_id}: {e}")
    finally:
        _unload_all_vram()
        # Clean up lock
        lock = sd / "partial_pending"
        if lock.exists():
            try:
                lock.unlink()
            except OSError:
                pass


def process_basic_job(job_data: dict):
    """Basic single-model transcription (legacy v2 path)."""
    session_id = job_data["session_id"]
    sd = session_dir(session_id)
    logger.info(f"Processing basic job for {session_id}")

    update_status(session_id, "running")

    try:
        # Resolve model
        model_id = job_data.get("model_id")
        model_size = job_data.get("model_size", "small")
        language_ctx = _session_language_context(session_id, job_data)
        language = language_ctx["forced_language"] or language_ctx["requested_language"]

        if model_id:
            resolved = resolve_model_id(model_id)
        else:
            resolved = resolve_model_id(model_size)

        audio_path = str(sd / "audio.wav")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        from app.pipeline.asr_executor import transcribe_window
        result = transcribe_window(
            audio_path,
            resolved,
            language=language,
            allowed_languages=language_ctx["allowed_languages"],
            forced_language=language_ctx["forced_language"],
            transcription_mode=language_ctx["transcription_mode"],
        )

        text = result.get("text", "")
        segments = result.get("segments", [])

        # Write outputs
        atomic_write_text(str(sd / "transcript.txt"), text)
        atomic_write_json(str(sd / "transcript_timestamps.json"), {
            "segments": segments,
            "words": [{"t_ms": int(s.get("start", 0) * 1000), "speaker": "SPEAKER_00", "w": s.get("text", "")} for s in segments],
        })

        # Build canonical-compatible segments
        canon_segments = []
        for i, seg in enumerate(segments):
            canon_segments.append({
                "segment_id": f"seg_{i:06d}",
                "start_ms": int(seg.get("start", 0) * 1000),
                "end_ms": int(seg.get("end", 0) * 1000),
                "speaker": "SPEAKER_00",
                "text": seg.get("text", ""),
                "confidence": 0.5,
                "support_windows": ["FULL_SESSION_LEGACY"],
                "support_models": [resolved],
                "stabilization_state": "stabilized",
                "corruption_flags": [],
            })

        # Generate derived outputs
        from app.pipeline.derived_outputs import build_current_dir
        import wave
        with wave.open(audio_path, 'rb') as wf:
            duration_ms = int(wf.getnframes() / wf.getframerate() * 1000)
        build_current_dir(session_id, canon_segments, text, duration_ms)

        update_status(session_id, "done")
        update_progress(session_id, "done", 100)

    except Exception as e:
        logger.error(f"Basic job failed for {session_id}: {e}\n{traceback.format_exc()}")
        error_data = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "action_hint": _derive_action_hint(str(e)),
        }
        atomic_write_json(str(sd / "error.json"), error_data)
        update_status(session_id, "error", error=str(e))
    finally:
        _unload_all_vram()


def process_retranscription_job(job_data: dict):
    """Re-transcription with a different model. Writes to retranscriptions/."""
    session_id = job_data["session_id"]
    revision_id = job_data["revision_id"]
    model_id = job_data["model_id"]
    language = job_data.get("language")
    sd = session_dir(session_id)
    rev_dir = sd / "retranscriptions" / revision_id

    logger.info(f"Retranscription {revision_id} for {session_id} with {model_id}")

    rev_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(rev_dir / "status.json"), {"status": "running"})

    try:
        audio_path = str(sd / "audio.wav")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError("audio.wav not found")

        resolved = resolve_model_id(model_id)
        from app.pipeline.asr_executor import transcribe_window
        result = transcribe_window(audio_path, resolved, language=language)

        text = result.get("text", "")
        segments = result.get("segments", [])

        atomic_write_text(str(rev_dir / "transcript.txt"), text)
        atomic_write_json(str(rev_dir / "transcript_timestamps.json"), {
            "segments": segments,
        })
        atomic_write_json(str(rev_dir / "revision_meta.json"), {
            "revision_id": revision_id,
            "session_id": session_id,
            "model_id": resolved,
            "language": language,
            "status": "done",
            "text_length": len(text),
            "segment_count": len(segments),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        atomic_write_json(str(rev_dir / "status.json"), {"status": "done"})

    except Exception as e:
        logger.error(f"Retranscription failed: {e}")
        atomic_write_json(str(rev_dir / "revision_meta.json"), {
            "revision_id": revision_id,
            "session_id": session_id,
            "model_id": model_id,
            "status": "error",
            "error": str(e),
        })
        atomic_write_json(str(rev_dir / "status.json"), {"status": "error", "error": str(e)})
    finally:
        _unload_all_vram()


def main():
    """Main worker loop."""
    cfg = get_config()
    logger.info("=" * 60)
    logger.info("LiveVoiceTranscriptor Worker starting")
    logger.info(f"Sessions dir: {cfg.storage.sessions_dir}")
    logger.info(f"Redis: {cfg.redis.host}:{cfg.redis.port}")
    logger.info("=" * 60)

    # GPU probe
    gpu_info = _get_gpu_diagnostics()
    logger.info("=" * 40)
    logger.info("GPU DIAGNOSTICS")
    logger.info(f"  GPU available:      {gpu_info.get('gpu_available')}")
    logger.info(f"  GPU name:           {gpu_info.get('gpu_name')}")
    logger.info(f"  GPU reason:         {gpu_info.get('gpu_reason')}")
    logger.info(f"  CUDA version:       {gpu_info.get('cuda_version', 'N/A')}")
    logger.info(f"  VRAM total:         {gpu_info.get('memory_total_mb', 'N/A')} MB")
    logger.info(f"  Effective device:   {gpu_info.get('effective_device')}")
    logger.info(f"  Effective compute:  {gpu_info.get('effective_compute_type')}")
    logger.info(f"  Selection method:   {gpu_info.get('device_selection_method')}")
    bs = gpu_info.get("backend_status", {})
    logger.info(f"  torch CUDA:         {bs.get('torch_cuda_available')} (devices={bs.get('torch_cuda_device_count', 0)})")
    logger.info(f"  ctranslate2 CUDA:   devices={bs.get('ctranslate2_cuda_device_count', 0)}")
    logger.info(f"  faster-whisper:     {'OK' if bs.get('faster_whisper_importable') else bs.get('faster_whisper_error', 'not checked')}")
    logger.info(f"  NeMo ASR:           {'OK' if bs.get('nemo_importable') else bs.get('nemo_error', 'not checked')}")
    conf = gpu_info.get("config", {})
    logger.info(f"  Config device:      {conf.get('whisper_device')}")
    logger.info(f"  Config compute:     {conf.get('whisper_compute_type')}")
    logger.info(f"  Config fallbacks:   {conf.get('whisper_compute_type_fallbacks')}")
    logger.info(f"  Strict CUDA:        {conf.get('strict_cuda')}")
    logger.info("=" * 40)
    _write_worker_health(gpu_info)

    if cfg.worker.strict_cuda and not gpu_info.get("gpu_available"):
        logger.error("WHISPER_STRICT_CUDA=true but no GPU available. Exiting.")
        sys.exit(1)

    # Redis connect with retry
    r = None
    for attempt in range(10):
        r = _get_redis()
        if r:
            break
        logger.warning(f"Redis connection attempt {attempt + 1}/10 failed, retrying...")
        time.sleep(min(5, 2 ** attempt))

    if r is None:
        logger.error("Could not connect to Redis after 10 attempts. Exiting.")
        sys.exit(1)

    logger.info("Worker ready. Listening for jobs...")

    queue = cfg.redis.queue
    partial_queue = cfg.redis.partial_queue

    while True:
        try:
            # blpop: main queue has priority by position
            result = r.blpop([queue, partial_queue], timeout=5)
            if result is None:
                continue

            queue_name, raw_data = result
            try:
                job_data = json.loads(raw_data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in job: {raw_data[:200]}")
                continue

            job_type = job_data.get("job_type", "v2")
            session_id = job_data.get("session_id", "unknown")
            logger.info(f"Dequeued job: type={job_type} session={session_id}")

            # Check cancellation
            if job_type not in ("v2_partial",):
                status = get_status(session_id)
                if status and status.get("status") == "cancelled":
                    logger.info(f"Session {session_id} cancelled, skipping")
                    continue

            # Dispatch
            if job_type == "v2_canonical":
                process_canonical_pipeline(job_data)
            elif job_type == "v2_canonical_live":
                process_canonical_live_job(job_data)
            elif job_type == "v2_partial":
                process_partial_job(job_data)
            elif job_type == "v2_retranscribe":
                process_retranscription_job(job_data)
            else:
                process_basic_job(job_data)

        except KeyboardInterrupt:
            logger.info("Worker shutting down (keyboard interrupt)")
            break
        except Exception as e:
            logger.error(f"Worker loop error: {e}\n{traceback.format_exc()}")
            time.sleep(1)


if __name__ == "__main__":
    main()
