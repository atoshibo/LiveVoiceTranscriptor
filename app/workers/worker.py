"""
Background Worker - Redis-backed job processor.

Single-threaded to serialize GPU use. Processes:
  - v2_canonical: Full canonical pipeline
  - v2_partial: Quick partial transcript for live preview
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


def _select_candidate_b_model(language_ctx: dict) -> tuple[Optional[str], Optional[str]]:
    reg = get_registry(refresh=True)
    forced = language_ctx.get("forced_language")
    allowed = language_ctx.get("allowed_languages") or []
    transcription_mode = language_ctx.get("transcription_mode", "verbatim_multilingual")

    english_only = forced == "en" or allowed == ["en"] or (
        not forced and transcription_mode != "verbatim_multilingual" and allowed == ["en"]
    )

    if english_only:
        info = reg.get(DEFAULT_CANDIDATE_B)
        if info and info.is_usable:
            return DEFAULT_CANDIDATE_B, "english_restricted_session"
        return None, "parakeet_unavailable_for_english_session"

    canary_id = "nemo-asr:canary-1b-v2"
    canary = reg.get(canary_id)
    if canary and canary.is_usable:
        return canary_id, "multilingual_candidate_b_routed_to_canary"

    return None, "candidate_b_skipped_for_multilingual_session"


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
    """Probe GPU health."""
    result = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_reason": "not_checked",
        "backend_status": {
            "torch_cuda_available": False,
            "ctranslate2_cuda_device_count": 0,
        },
    }
    try:
        import torch
        if torch.cuda.is_available():
            result["gpu_available"] = True
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_reason"] = "cuda_available"
            props = torch.cuda.get_device_properties(0)
            result["memory_total_mb"] = round(props.total_memory / 1024 / 1024)
            result["backend_status"]["torch_cuda_available"] = True
        else:
            result["gpu_reason"] = "cuda_not_available"
    except ImportError:
        result["gpu_reason"] = "torch_not_installed"
    except Exception as e:
        result["gpu_reason"] = str(e)

    try:
        import ctranslate2

        cuda_devices = int(ctranslate2.get_cuda_device_count())
        result["backend_status"]["ctranslate2_cuda_device_count"] = cuda_devices
        if cuda_devices > 0 and not result["gpu_available"]:
            result["gpu_available"] = True
            result["gpu_reason"] = "ctranslate2_cuda_available"
            result["gpu_name"] = result.get("gpu_name") or "CUDA device via CTranslate2"
    except Exception:
        pass
    return result


def _write_worker_health(gpu_info: dict):
    """Write worker health file for the API to read."""
    cfg = get_config()
    health_path = str(Path(cfg.storage.sessions_dir).parent / "worker_health.json")
    data = {
        **gpu_info,
        "worker_started_at": datetime.now(timezone.utc).isoformat(),
        "selected_device": "cuda" if gpu_info.get("gpu_available") else "cpu",
        "selected_compute_type": "float16" if gpu_info.get("gpu_available") else "int8",
        "strict_cuda": cfg.worker.strict_cuda,
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
            try:
                from app.pipeline.asr_executor import run_asr_execution
                run_asr_execution(
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
                })
                stage.commit(["first_pass_routing.json"])
            except Exception as e:
                stage.commit_with_fallback([], f"first_pass_medium_error: {e}")

        _unload_all_vram()

        # ============================================================
        # Stage 5: Candidate ASR - Large V3
        # ============================================================
        if not run.is_stage_done("candidate_asr_large_v3"):
            update_progress(session_id, "candidate_asr_large_v3", 46)
            stage = run.start_stage("candidate_asr_large_v3")
            try:
                from app.pipeline.asr_executor import run_asr_execution
                run_asr_execution(
                    session_id, windows, [DEFAULT_CANDIDATE_A],
                    language=language,
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
                })
                stage.commit(["candidate_a_routing.json"])
            except Exception as e:
                stage.commit_with_fallback([], f"candidate_a_error: {e}")

        _unload_all_vram()

        # ============================================================
        # Stage 6: Candidate ASR - Parakeet / Canary route
        # ============================================================
        if not run.is_stage_done("candidate_asr_parakeet"):
            update_progress(session_id, "candidate_asr_parakeet", 58)
            stage = run.start_stage("candidate_asr_parakeet")
            try:
                from app.pipeline.asr_executor import run_asr_execution
                candidate_b_model, candidate_b_reason = _select_candidate_b_model(language_ctx)
                if candidate_b_model:
                    run_asr_execution(
                        session_id,
                        windows,
                        [candidate_b_model],
                        language=language,
                        allowed_languages=language_ctx["allowed_languages"],
                        forced_language=language_ctx["forced_language"],
                        transcription_mode=language_ctx["transcription_mode"],
                        progress_callback=lambda p: update_progress(
                            session_id, "candidate_asr_parakeet", 58 + int(p * 0.1)
                        ),
                    )
                    atomic_write_json(str(stage.stage_dir / "candidate_b_routing.json"), {
                        "selected_model": candidate_b_model,
                        "reason": candidate_b_reason,
                        "language_context": language_ctx,
                    })
                    stage.commit(["candidate_b_routing.json"])
                else:
                    atomic_write_json(str(stage.stage_dir / "candidate_b_routing.json"), {
                        "selected_model": None,
                        "reason": candidate_b_reason,
                        "language_context": language_ctx,
                    })
                    stage.commit_with_fallback(["candidate_b_routing.json"], candidate_b_reason)
            except Exception as e:
                stage.commit_with_fallback([], f"candidate_b_parakeet_error: {e}")

        _unload_all_vram()

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
        # Stage 11: Derived Outputs
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
    logger.info(f"GPU: {gpu_info}")
    _write_worker_health(gpu_info)

    if cfg.worker.strict_cuda and not gpu_info.get("gpu_available"):
        logger.error("STRICT_CUDA=true but no GPU available. Exiting.")
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
