"""
API v2 Router - Full backward-compatible API surface.

Preserves ALL endpoint paths, request/response schemas, auth behavior,
alias fields, status codes, and idempotent behaviors from server v0.4.2.

The application (Android/device) is NOT modified. This API must accept
all existing client requests without change.
"""
import json
import os
import shutil
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse

from app.api.auth import require_auth, require_auth_upload
from app.core.config import get_config
from app.core.atomic_io import atomic_write_json, safe_read_json
from app.storage.session_store import (
    session_dir, create_session, get_session_meta, update_session_meta,
    get_status, update_status, register_chunk, get_chunk_paths,
    session_exists, is_v2_session, delete_session, list_sessions_grouped, list_sessions,
)
from app.models.registry import (
    get_registry, resolve_model_id, get_model_info, list_all_models,
    VALID_MODEL_SIZES,
)
from app.pipeline.run import normalize_stage_name, stage_directory_candidates

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["v2"])


def _load_quality_gate(sd: Path) -> Dict[str, Any]:
    from app.pipeline.canonical_assembly import read_quality_gate
    return safe_read_json(str(sd / "canonical" / "quality_gate.json")) or read_quality_gate(sd.name)


def _load_context_spans_payload(sd: Path) -> Dict[str, Any]:
    return safe_read_json(str(sd / "enrichment" / "context_spans.json")) or {}


def _load_retrieval_payload(sd: Path) -> Optional[Dict[str, Any]]:
    derived = sd / "derived"
    current = sd / "current"
    return (
        safe_read_json(str(derived / "retrieval_index_v3.json"))
        or safe_read_json(str(current / "retrieval_index_v3.json"))
        or safe_read_json(str(derived / "retrieval_index_v2.json"))
        or safe_read_json(str(current / "retrieval_index_v2.json"))
        or safe_read_json(str(derived / "retrieval_index.json"))
        or safe_read_json(str(current / "retrieval_index.json"))
    )


def _build_diarization_summary(sd: Path, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = meta or {}
    status_payload = safe_read_json(str(sd / "canonical" / "diarization_status.json")) or {}
    speaker_turns_payload = safe_read_json(str(sd / "canonical" / "speaker_turns.json")) or {}

    policy = status_payload.get("policy")
    if policy is None:
        policy = str(meta.get("diarization_policy", "auto")).strip().lower() or "auto"

    requested = status_payload.get("requested")
    if requested is None:
        requested = bool(meta.get("run_diarization")) or policy == "forced"

    status = status_payload.get("status")
    if status is None:
        if policy == "off":
            status = "not_requested"
        elif requested:
            status = "pending"
        else:
            status = "auto_pending"

    return {
        "policy": policy,
        "requested": requested,
        "reason": status_payload.get("reason"),
        "available": status_payload.get("available"),
        "status": status,
        "speakers": (speaker_turns_payload or {}).get("speakers") or [],
        "turn_count": (speaker_turns_payload or {}).get("turn_count", 0),
    }


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
async def health():
    """Health check - no auth required."""
    cfg = get_config()
    gpu_info = _read_worker_health()
    return {
        "ok": True,
        "version": "2.5.0",
        "gpu_available": gpu_info.get("gpu_available", False),
        "gpu_reason": gpu_info.get("gpu_reason"),
        "selected_device": gpu_info.get("selected_device", "cpu"),
        "selected_compute_type": gpu_info.get("selected_compute_type", "int8"),
        "strict_cuda": gpu_info.get("strict_cuda", False),
        "server_port": cfg.server.port,
        "tls_enabled": cfg.server.tls_enabled,
        "tls_cert_file": cfg.server.tls_cert_file,
        "tls_cert_exists": Path(cfg.server.tls_cert_file).is_file(),
        "tls_key_exists": Path(cfg.server.tls_key_file).is_file(),
        "models_dir": cfg.model_paths.models_dir,
        "models_dir_exists": cfg.model_paths.models_dir_exists,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ============================================================
# SYSTEM / GPU
# ============================================================

@router.get("/system/gpu")
async def system_gpu(token: str = Depends(require_auth)):
    """GPU status and worker diagnostics."""
    gpu_info = _read_worker_health()
    r = _get_redis()

    # Active jobs
    active_jobs = []
    queue_depth = 0
    partial_queue_depth = 0
    if r:
        cfg = get_config()
        try:
            queue_depth = r.llen(cfg.redis.queue) or 0
        except Exception:
            pass
        try:
            partial_queue_depth = r.llen(cfg.redis.partial_queue) or 0
        except Exception:
            pass

    return {
        "gpu_available": gpu_info.get("gpu_available", False),
        "gpu_name": gpu_info.get("gpu_name"),
        "gpu_reason": gpu_info.get("gpu_reason"),
        "selected_device": gpu_info.get("selected_device", "cpu"),
        "selected_compute_type": gpu_info.get("selected_compute_type", "int8"),
        "strict_cuda": gpu_info.get("strict_cuda", False),
        "backend_status": gpu_info.get("backend_status", {}),
        "utilization_percent": gpu_info.get("utilization_percent"),
        "memory_used_mb": gpu_info.get("memory_used_mb"),
        "memory_total_mb": gpu_info.get("memory_total_mb"),
        "worker_started_at": gpu_info.get("worker_started_at"),
        "active_jobs": active_jobs,
        "active_job_count": len(active_jobs),
        "queue_depth": queue_depth,
        "partial_queue_depth": partial_queue_depth,
        "processing_stats": {},
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ============================================================
# MODELS
# ============================================================

@router.get("/models")
async def list_models(token: str = Depends(require_auth)):
    """List all registered models."""
    try:
        models = list_all_models()
        cfg = get_config()
        return {
            "models": [m.to_dict() for m in models],
            "count": len(models),
            "models_dir": cfg.model_paths.models_dir,
            "models_dir_exists": cfg.model_paths.models_dir_exists,
            "llm_model_available": cfg.model_paths.llm_path is not None,
        }
    except Exception as e:
        raise HTTPException(503, detail={
            "reason": "model_registry_unavailable",
            "message": str(e),
        })


@router.get("/sessions")
async def list_sessions_endpoint(token: str = Depends(require_auth)):
    sessions = list_sessions(limit=200)
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, token: str = Depends(require_auth)):
    """Get details for a specific model."""
    try:
        info = get_model_info(model_id)
    except Exception as e:
        raise HTTPException(503, detail={
            "reason": "model_registry_unavailable",
            "message": str(e),
        })
    if info is None:
        raise HTTPException(404, detail={
            "reason": "model_not_found",
            "message": f"Model '{model_id}' not found in registry.",
        })
    return info.to_dict()


# ============================================================
# SESSION CREATE
# ============================================================

@router.post("/sessions")
async def create_new_session(request: Request, token: str = Depends(require_auth)):
    """Create a new recording session.

    Preserves all alias fields:
    - sample_rate_hz OR sample_rate
    - source_type=device_import -> mode=file
    - chunk_duration_sec stored for diagnostics
    - diarization stored as diarization_hint
    """
    body = await request.json() if await request.body() else {}
    result = create_session(body)
    return result


# ============================================================
# CHUNK UPLOAD
# ============================================================

@router.post("/sessions/{session_id}/chunks")
async def upload_chunk(
    session_id: str,
    file: UploadFile = File(...),
    chunk_index: int = Form(...),
    chunk_started_ms: int = Form(0),
    chunk_duration_ms: int = Form(0),
    is_final: bool = Form(False),
    # Legacy continuity fields
    dropped_frames: int = Form(0),
    decode_failure: bool = Form(False),
    gap_before_ms: int = Form(0),
    source_degraded: bool = Form(False),
    # Android aliases
    decode_errors: int = Form(0),
    ble_gaps: int = Form(0),
    plc_frames_applied: int = Form(0),
    has_continuity_warning: bool = Form(False),
    token: str = Depends(require_auth_upload),
):
    """Upload an audio chunk.

    Preserves all continuity field aliases for Android compatibility.
    """
    # Validate session
    if not session_exists(session_id) or not is_v2_session(session_id):
        raise HTTPException(404, detail=f"Session {session_id} not found or not a V2 session.")

    meta = get_session_meta(session_id)
    state = meta.get("state", "created")

    # Reject if finalized/cancelled
    if state in ("finalized", "cancelled"):
        raise HTTPException(
            409,
            detail=f"Session {session_id} is already finalized. Create a new session for subsequent audio.",
        )

    # Check limits
    cfg = get_config()
    chunks = meta.get("chunks", [])
    if len(chunks) >= cfg.worker.max_session_chunks:
        raise HTTPException(413, detail={
            "reason": "too_many_chunks",
            "message": f"Maximum {cfg.worker.max_session_chunks} chunks per session.",
            "limit": cfg.worker.max_session_chunks,
            "current": len(chunks),
        })

    # Read file
    content = await file.read()
    if len(content) > cfg.worker.max_chunk_mb * 1024 * 1024:
        raise HTTPException(413, detail={
            "reason": "chunk_too_large",
            "message": f"Chunk exceeds {cfg.worker.max_chunk_mb}MB limit.",
            "limit_mb": cfg.worker.max_chunk_mb,
            "chunk_bytes": len(content),
        })

    # Normalize continuity fields (Android alias handling)
    actual_dropped_frames = dropped_frames + plc_frames_applied
    actual_decode_failure = decode_failure or (decode_errors > 0)
    actual_gap_before_ms = gap_before_ms if gap_before_ms > 0 else ble_gaps
    actual_source_degraded = source_degraded or has_continuity_warning

    # Save chunk file
    sd = session_dir(session_id)
    chunk_path = sd / "chunks" / f"chunk_{chunk_index:04d}.wav"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(chunk_path), "wb") as f:
        f.write(content)

    # Register chunk
    chunk_meta = {
        "chunk_index": chunk_index,
        "chunk_started_ms": chunk_started_ms,
        "chunk_duration_ms": chunk_duration_ms,
        "is_final": is_final,
        "dropped_frames": actual_dropped_frames,
        "decode_failure": actual_decode_failure,
        "gap_before_ms": actual_gap_before_ms,
        "source_degraded": actual_source_degraded,
        "file_size": len(content),
    }
    total_chunks = register_chunk(session_id, chunk_index, chunk_meta)

    # Handle is_final
    if is_final:
        meta = get_session_meta(session_id)
        if meta and meta.get("state") == "receiving":
            update_session_meta(session_id, {"state": "chunks_complete"})

    # Canonical live trigger (stream mode only, not is_final)
    if not is_final and meta.get("mode") == "stream":
        _maybe_trigger_live_canonical(session_id, total_chunks, meta)

    return {
        "accepted": True,
        "session_id": session_id,
        "chunk_index": chunk_index,
        "chunk_count": total_chunks,
        "status": "accepted",
    }


@router.get("/sessions/{session_id}/chunks")
async def list_session_chunks(session_id: str, token: str = Depends(require_auth)):
    if not session_exists(session_id):
        raise HTTPException(404, detail=f"Session {session_id} not found.")

    meta = get_session_meta(session_id) or {}
    chunk_paths = {Path(path).name: path for path in get_chunk_paths(session_id)}
    chunks = []
    for chunk in meta.get("chunks", []):
        filename = f"chunk_{int(chunk.get('chunk_index', 0)):04d}.wav"
        chunks.append({
            **chunk,
            "file_present": filename in chunk_paths,
            "file_name": filename,
        })
    return {"session_id": session_id, "count": len(chunks), "chunks": chunks}


# ============================================================
# FINALIZE
# ============================================================

@router.post("/sessions/{session_id}/finalize")
async def finalize_session(session_id: str, request: Request,
                           token: str = Depends(require_auth)):
    """Finalize a session and start processing.

    Preserves all aliases:
    - run_diarization / diarization
    - speaker_count / num_speakers
    - model_id takes priority over model_size
    - Idempotent: returns existing job_id if already finalized
    """
    if not session_exists(session_id):
        raise HTTPException(404, detail=f"Session {session_id} not found.")

    body = {}
    raw = await request.body()
    if raw:
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {}

    meta = get_session_meta(session_id)
    state = meta.get("state", "created")

    # Check cancellation
    if state == "cancelled":
        raise HTTPException(409, detail="Session was cancelled.")

    # Idempotent: already finalized
    if state == "finalized":
        status = get_status(session_id) or {}
        result = {
            "session_id": session_id,
            "job_id": session_id,
            "status_url": f"/api/v2/jobs/{session_id}",
            "enqueued": False,
            "message": "Session already finalized.",
        }
        # Re-enqueue if stuck
        if status.get("status") in ("uploaded", "pending", "created"):
            _enqueue_job(session_id, meta, body)
            result["requeued"] = True
        else:
            result["requeued"] = False
        return result

    # Validate chunks
    chunks = meta.get("chunks", [])
    if not chunks:
        raise HTTPException(400, detail="No chunks uploaded.")

    chunk_paths = get_chunk_paths(session_id)
    if not chunk_paths:
        raise HTTPException(400, detail="Chunk files missing on disk.")

    # Resolve diarization contract.
    diarization_flag_present = "run_diarization" in body or "diarization" in body
    diarization_flag = body.get("run_diarization", body.get("diarization", False))
    diarization_policy = body.get("diarization_policy")
    if diarization_policy is None:
        if diarization_flag_present:
            # A caller explicitly set run_diarization=False but without a
            # policy — respect it as the "no diarization" signal but surface
            # it as "auto" for file imports, so our selective trigger still
            # runs the work when content clearly warrants it (long audio,
            # multi-voice filename hints).  Truly disabling diarization now
            # requires sending diarization_policy="off" explicitly.
            diarization_policy = "forced" if diarization_flag else "auto"
        else:
            diarization_policy = meta.get("diarization_policy", "auto")
    diarization_policy = str(diarization_policy).strip().lower()
    if diarization_policy not in {"auto", "off", "forced"}:
        raise HTTPException(400, detail=f"Invalid diarization_policy '{diarization_policy}'. Allowed: auto, off, forced.")
    run_diarization = diarization_policy == "forced"

    # Resolve speaker count (aliases)
    speaker_count = body.get("speaker_count", body.get("num_speakers"))

    # Resolve language
    language = body.get("language", "auto")
    allowed_languages = body.get("allowed_languages")
    if not isinstance(allowed_languages, list):
        allowed_languages = meta.get("allowed_languages") or []
    allowed_languages = [str(item).strip() for item in allowed_languages if str(item).strip()]
    forced_language = body.get("forced_language", meta.get("forced_language"))
    if forced_language == "auto":
        forced_language = None
    transcription_mode = body.get("transcription_mode", meta.get("transcription_mode", "verbatim_multilingual"))

    # Resolve model
    model_id = body.get("model_id")
    model_size = body.get("model_size", "auto")

    # Validate model_size
    if model_size and model_size != "auto" and model_size not in VALID_MODEL_SIZES:
        if not model_id:
            raise HTTPException(400, detail=f"Invalid model_size '{model_size}'. Allowed: {VALID_MODEL_SIZES}")

    # Validate model_id
    if model_id:
        resolved = resolve_model_id(model_id)
        info = get_model_info(resolved)
        if info is None:
            raise HTTPException(400, detail=f"Unknown model_id '{model_id}'.")
        if not info.is_usable:
            raise HTTPException(400, detail=f"Model '{model_id}' is not usable (installed={info.installed}).")

    # Validate language_candidates
    language_candidates = body.get("language_candidates")
    language_selection_strategy = body.get("language_selection_strategy")
    if language_candidates is not None:
        if not isinstance(language_candidates, list) or len(language_candidates) == 0:
            raise HTTPException(400, detail="language_candidates must be a non-empty list.")
        if len(language_candidates) > 3:
            raise HTTPException(400, detail="language_candidates maximum is 3.")
        # Check for duplicates
        if len(set(language_candidates)) != len(language_candidates):
            raise HTTPException(400, detail="language_candidates must not contain duplicates.")
        # Check for empties
        if any(not c for c in language_candidates):
            raise HTTPException(400, detail="language_candidates must not contain empty values.")
    if language_selection_strategy and language_selection_strategy != "ordered_fallback":
        raise HTTPException(400, detail=f"Unknown language_selection_strategy '{language_selection_strategy}'.")

    # Render absolute-timeline compatibility audio
    sd = session_dir(session_id)
    audio_path = str(sd / "audio.wav")
    try:
        from app.pipeline.ingest import render_session_timeline_audio
        timeline_audio = render_session_timeline_audio(session_id, audio_path)
        if not timeline_audio.get("success"):
            raise RuntimeError(timeline_audio.get("error", "timeline render failed"))
    except Exception as e:
        raise HTTPException(500, detail=f"Timeline render failed: {e}")

    # Write metadata
    metadata = {
        "session_id": session_id,
        "language": language,
        "model_size": model_size,
        "model_id": model_id,
        "run_diarization": run_diarization,
        "diarization_policy": diarization_policy,
        "speaker_count": speaker_count,
        "force_transcribe_only": body.get("force_transcribe_only", False),
        "session_integrity": body.get("session_integrity"),
        "language_candidates": language_candidates,
        "language_selection_strategy": language_selection_strategy,
        "allowed_languages": allowed_languages,
        "forced_language": forced_language,
        "transcription_mode": transcription_mode,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(str(sd / "metadata.json"), metadata)

    # Update session state
    session_updates = {
        "state": "finalized",
        "run_diarization": run_diarization,
        "diarization_policy": diarization_policy,
        "speaker_count": speaker_count,
        "language": language,
        "allowed_languages": allowed_languages,
        "forced_language": forced_language,
        "transcription_mode": transcription_mode,
        "model_size": model_size,
        "model_id": model_id,
        "force_transcribe_only": body.get("force_transcribe_only", False),
        "session_integrity": body.get("session_integrity"),
        "language_candidates": language_candidates,
        "language_selection_strategy": language_selection_strategy,
        "finalize_requested_at": datetime.now(timezone.utc).isoformat(),
    }
    update_session_meta(session_id, session_updates)
    merged_meta = (get_session_meta(session_id) or {}).copy()
    merged_meta.update(session_updates)

    # Update status
    update_status(session_id, "uploaded")

    # Enqueue job
    _enqueue_job(session_id, merged_meta, body)

    return {
        "session_id": session_id,
        "job_id": session_id,
        "status_url": f"/api/v2/jobs/{session_id}",
        "enqueued": True,
    }


# ============================================================
# JOB STATUS
# ============================================================

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, token: str = Depends(require_auth)):
    """Get job processing status."""
    sd = session_dir(job_id)
    if not sd.is_dir():
        raise HTTPException(404, detail=f"Job {job_id} not found.")

    status_data = get_status(job_id) or {}
    meta = get_session_meta(job_id) or {}
    metadata = safe_read_json(str(sd / "metadata.json")) or {}

    raw_status = status_data.get("status", "created")
    state = _map_status_to_public(raw_status)

    # Backend outcome
    backend_outcome = _derive_backend_outcome(raw_status, sd)

    # Progress
    progress = status_data.get("progress", {"upload": 0, "processing": 0, "stage": "pending"})

    # Timing
    queued_at = status_data.get("queued_at") or metadata.get("finalized_at")
    started_at = status_data.get("started_at")
    finished_at = status_data.get("finished_at")

    # Error info
    error_info = None
    failure_category = None
    if state == "error":
        err = safe_read_json(str(sd / "error.json"))
        if err:
            error_info = err
            et = err.get("error_type", "").lower()
            em = err.get("error_message", "").lower()
            if "cuda" in et or "cuda" in em:
                failure_category = "gpu_error"
            elif "audio" in em or "wav" in em:
                failure_category = "audio_missing"
            elif "model" in em:
                failure_category = "model_error"
            else:
                failure_category = "internal_error"
        else:
            error_info = {"error_type": "unknown", "error_message": status_data.get("error", "")}

    # Transcript presence
    transcript_present = _has_transcript(sd)

    # Partial
    partial_available = (sd / "partial_transcript.json").is_file()
    partial_preview = status_data.get("partial_preview") if state == "running" else None

    diarization = _build_diarization_summary(sd, metadata)
    quality_gate = _load_quality_gate(sd)
    context_spans_payload = _load_context_spans_payload(sd)

    # Build meta
    job_meta = {
        "language_requested": metadata.get("language"),
        "model_size": metadata.get("model_size"),
        "speaker_count": metadata.get("speaker_count"),
        "diarization_enabled": bool(diarization.get("requested")),
        "diarization": diarization,
        "quality_gate": quality_gate,
        "context_span_count": (context_spans_payload or {}).get("span_count", 0),
    }

    if metadata.get("language_candidates"):
        job_meta["language_candidates"] = metadata["language_candidates"]
        job_meta["language_selection_strategy"] = metadata.get("language_selection_strategy")

    # Add post-processing meta
    if transcript_present:
        quality = safe_read_json(str(sd / "current" / "quality_report.json")) or safe_read_json(str(sd / "quality_report.json"))
        if quality:
            job_meta["audio_duration_s"] = quality.get("audio_duration_s")
            job_meta["segment_count"] = quality.get("segment_count")
            job_meta["transcript_last_end_s"] = quality.get("transcript_last_end_s")
            job_meta["transcript_coverage_ratio"] = quality.get("transcript_coverage_ratio")
            if quality.get("transcript_coverage_ratio") and quality["transcript_coverage_ratio"] < 0.5:
                job_meta["coverage_warning"] = True

        classification = safe_read_json(str(sd / "current" / "classification.json")) or safe_read_json(str(sd / "classification.json"))
        if classification:
            job_meta["classification"] = classification.get("category")
            job_meta["classification_confidence"] = classification.get("confidence")

    result = {
        "job_id": job_id,
        "state": state,
        "backend_outcome": backend_outcome,
        "transcript_present": transcript_present,
        "failure_category": failure_category,
        "partial_transcript_available": partial_available,
        "progress": progress,
        "message": _state_message(state),
        "partial_preview": partial_preview,
        "queued_at": queued_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "diarization": diarization,
        "quality_gate": quality_gate,
        "context_span_count": (context_spans_payload or {}).get("span_count", 0),
        "meta": job_meta,
    }

    if error_info and state == "error":
        result["error"] = error_info

    return result


@router.get("/jobs/{job_id}/error")
async def get_job_error(job_id: str, token: str = Depends(require_auth)):
    """Get detailed error info for a failed job."""
    sd = session_dir(job_id)
    if not sd.is_dir():
        raise HTTPException(404, detail="Job not found.")
    err_path = sd / "error.json"
    if not err_path.is_file():
        raise HTTPException(404, detail="Error details not available for this job.")
    return safe_read_json(str(err_path)) or {}


# ============================================================
# SESSION STATUS
# ============================================================

@router.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str, token: str = Depends(require_auth)):
    """Full session status for polling during live recording."""
    if not session_exists(session_id):
        raise HTTPException(404, detail=f"Session {session_id} not found.")

    meta = get_session_meta(session_id) or {}
    status_data = get_status(session_id) or {}
    sd = session_dir(session_id)
    metadata = safe_read_json(str(sd / "metadata.json")) or {}

    # Derive state
    session_state = meta.get("state", "created")
    worker_status = status_data.get("status", "")
    if worker_status in ("done", "error", "cancelled"):
        state = _map_status_to_public(worker_status)
    elif worker_status in ("running", "processing"):
        state = "running"
    elif session_state == "finalized" and worker_status in ("uploaded", "pending"):
        state = "queued"
    else:
        state = session_state

    chunks = meta.get("chunks", [])
    chunk_count = len(chunks)

    # Partial info
    partial_data = safe_read_json(str(sd / "partial_transcript.json"))
    stabilized_partial = safe_read_json(str(sd / "current" / "stabilized_partial.json")) or safe_read_json(str(sd / "stabilized_partial.json"))
    final_transcript = safe_read_json(str(sd / "current" / "final_transcript.json")) or safe_read_json(str(sd / "final_transcript.json"))
    partial_available = partial_data is not None
    partial_preview = status_data.get("partial_preview")
    partial_updated_at = partial_data.get("generated_at") if partial_data else None

    error_message = None
    if state == "error":
        error_message = status_data.get("error")
        if not error_message:
            err = safe_read_json(str(sd / "error.json"))
            if err:
                error_message = err.get("error_message", "Unknown error")

    diarization = _build_diarization_summary(sd, metadata or meta)
    quality_gate = _load_quality_gate(sd)
    context_spans_payload = _load_context_spans_payload(sd)

    return {
        "session_id": session_id,
        "state": state,
        "backend_outcome": _derive_backend_outcome(status_data.get("status", ""), sd),
        "transcript_present": _has_transcript(sd),
        "error_message": error_message,
        "chunk_count": chunk_count,
        "chunks_received": chunk_count,  # Android compat alias
        "last_chunk_index": chunks[-1].get("chunk_index") if chunks else None,
        "total_audio_ms": sum(c.get("chunk_duration_ms", 0) for c in chunks),
        "partial_transcript_available": partial_available,
        "provisional_partial_available": partial_available,
        "stabilized_partial_available": stabilized_partial is not None,
        "final_transcript_available": final_transcript is not None,
        "partial_preview": partial_preview,
        "partial_updated_at": partial_updated_at,
        "last_processed_chunk_index": None,
        "created_at": meta.get("created_at"),
        "first_chunk_at": meta.get("first_chunk_at"),
        "last_chunk_at": meta.get("last_chunk_at"),
        "finalize_requested_at": meta.get("finalize_requested_at"),
        "queued_at": status_data.get("queued_at") or meta.get("finalize_requested_at"),
        "started_at": status_data.get("started_at"),
        "finished_at": status_data.get("finished_at"),
        "progress": status_data.get("progress", {"upload": 0, "processing": 0, "stage": "pending"}),
        "mode": meta.get("mode", "stream"),
        "device_id": meta.get("device_id"),
        "language": metadata.get("language", meta.get("language", "auto")),
        "model_size": metadata.get("model_size", meta.get("model_size", "auto")),
        "original_filename": meta.get("original_filename"),
        "diarization_enabled": bool(diarization.get("requested")),
        "diarization": diarization,
        "quality_gate": quality_gate,
        "context_span_count": (context_spans_payload or {}).get("span_count", 0),
    }


# ============================================================
# TRANSCRIPTS
# ============================================================

@router.get("/sessions/{session_id}/transcript/partial")
async def get_partial_transcript(session_id: str, token: str = Depends(require_auth)):
    """Get provisional partial transcript."""
    sd = session_dir(session_id)
    partial_path = sd / "partial_transcript.json"
    if not partial_path.is_file():
        raise HTTPException(404, detail={
            "message": "No partial transcript available yet.",
            "hint": "Partial transcripts are generated periodically during recording.",
        })
    data = safe_read_json(str(partial_path))
    if data is None:
        raise HTTPException(500, detail="Failed to read partial transcript.")
    return {
        "session_id": session_id,
        "provisional": True,
        "semantic_layer": data.get("semantic_layer", "provisional_partial"),
        "text": data.get("text", ""),
        "segments": data.get("segments", []),
        "chunk_count_at_time": data.get("chunk_count_at_time", 0),
        "generated_at": data.get("generated_at"),
        "partial_updated_at": data.get("generated_at"),
        "degraded": data.get("degraded", False),
        "fallback_reason": data.get("fallback_reason"),
    }


@router.get("/sessions/{session_id}/transcript")
async def get_transcript(session_id: str, token: str = Depends(require_auth)):
    """Get final transcript. Returns 202 if still processing.

    Reads from current/ directory. Preserves all backward-compatible fields.
    """
    sd = session_dir(session_id)
    if not sd.is_dir():
        raise HTTPException(404, detail=f"Session {session_id} not found.")

    status_data = get_status(session_id) or {}
    if status_data.get("status") != "done":
        return JSONResponse(
            status_code=202,
            content={
                "message": "Transcript not ready yet.",
                "state": _map_status_to_public(status_data.get("status", "pending")),
                "progress": status_data.get("progress", {}),
            },
        )

    # Canonical spec 16.1: prefer canonical/ and derived/ over compatibility
    # copies in current/.  current/ is the last-resort read layer, kept for
    # older clients.
    canonical = sd / "canonical"
    derived = sd / "derived"
    current = sd / "current"

    final_surface = (
        safe_read_json(str(canonical / "final_transcript.json"))
        or safe_read_json(str(derived / "final_transcript.json"))
        or safe_read_json(str(current / "final_transcript.json"))
        or safe_read_json(str(sd / "final_transcript.json"))
    )
    stabilized_surface = (
        safe_read_json(str(canonical / "stabilized_partial.json"))
        or safe_read_json(str(derived / "stabilized_partial.json"))
        or safe_read_json(str(current / "stabilized_partial.json"))
        or safe_read_json(str(sd / "stabilized_partial.json"))
    )
    provisional_surface = (
        safe_read_json(str(canonical / "provisional_partial.json"))
        or safe_read_json(str(derived / "provisional_partial.json"))
        or safe_read_json(str(current / "provisional_partial.json"))
        or safe_read_json(str(sd / "provisional_partial.json"))
    )

    # Resolve read layer for derived artifacts (clean, quality, etc.).
    # Prefer derived/, then current/, then session root.
    if _has_any_transcript_file(derived):
        read_dir = derived
    elif _has_any_transcript_file(current):
        read_dir = current
    else:
        read_dir = sd

    # Load text
    text = (final_surface or {}).get("text", "")
    segments = (final_surface or {}).get("segments", [])
    if not text:
        text_path = read_dir / "transcript.txt"
        if text_path.is_file():
            text = text_path.read_text(encoding="utf-8").strip()
    if not segments:
        segments = _load_segments(read_dir)

    # If no text but segments exist, reconstruct
    if not text and segments:
        text = " ".join(s.get("text", "") for s in segments if s.get("text"))

    # Load other artifacts
    clean = safe_read_json(str(read_dir / "clean_transcript.json"))
    quality = safe_read_json(str(read_dir / "quality_report.json"))
    classification = safe_read_json(str(read_dir / "classification.json"))
    timestamps = safe_read_json(str(read_dir / "transcript_timestamps.json"))

    # Speaker timestamped text
    speaker_text = None
    speaker_txt_path = read_dir / "transcript_by_speaker.txt"
    if speaker_txt_path.is_file():
        speaker_text = speaker_txt_path.read_text(encoding="utf-8")
    elif segments:
        speaker_text = _synthesize_speaker_timestamped(segments)

    # Words
    words = []
    if timestamps and "words" in timestamps:
        words = timestamps["words"]
    elif segments:
        words = [{"t_ms": s["start_ms"], "speaker": s.get("speaker", "SPEAKER_00"), "w": s.get("text", "")} for s in segments]

    # Source integrity
    source_integrity = None
    if quality and "source_integrity" in quality:
        source_integrity = quality["source_integrity"]
    else:
        meta = get_session_meta(session_id) or {}
        si = meta.get("session_integrity")
        if si:
            source_integrity = {
                "hints_available": True,
                "session_degraded": si.get("session_degraded", False),
                "total_dropped_frames": si.get("total_dropped_frames", 0),
                "chunks_with_decode_failure": None,
                "chunks_with_gaps": None,
                "integrity_note": si.get("integrity_note", ""),
            }

    # Reading text
    reading_text = None
    if quality:
        reading_text = quality.get("reading_text")
    if not reading_text:
        reading_text = text

    # Semantic markers + retrieval grounding (enrichment layer).  Canonical
    # spec 16.1: markers/retrieval are part of the transcript read contract,
    # not a hidden sidecar.
    markers_payload = safe_read_json(str(sd / "enrichment" / "segment_markers.json"))
    markers = (markers_payload or {}).get("markers") or []
    spans_payload = safe_read_json(str(sd / "enrichment" / "semantic_spans.json"))
    context_spans_payload = _load_context_spans_payload(sd)
    retrieval_payload = _load_retrieval_payload(sd)
    diarization = _build_diarization_summary(sd, get_session_meta(session_id) or {})
    quality_gate = _load_quality_gate(sd)

    return {
        "session_id": session_id,
        "text": text,
        "raw_text": text,
        "clean_text": clean.get("clean_text") if clean else None,
        "paragraphs": clean.get("paragraphs") if clean else None,
        "dropped_segments": clean.get("dropped_segments") if clean else None,
        "reading_text": reading_text,
        "speaker_timestamped": speaker_text,
        "segments": segments,
        "words": words,
        "semantic_layer": "final_transcript",
        "semantic_layers": {
            "provisional_partial_available": provisional_surface is not None,
            "stabilized_partial_available": stabilized_surface is not None,
            "final_transcript_available": final_surface is not None,
        },
        "markers": markers,
        "semantic_spans": (spans_payload or {}).get("spans") or [],
        "context_spans": (context_spans_payload or {}).get("spans") or [],
        "retrieval_summary": {
            "entry_count": (retrieval_payload or {}).get("entry_count", 0),
            "excluded_count": (retrieval_payload or {}).get("excluded_count", 0),
            "version": (retrieval_payload or {}).get("version"),
            "source": (retrieval_payload or {}).get("source"),
            "url": f"/api/v2/sessions/{session_id}/retrieval",
        },
        "diarization": diarization,
        "quality_gate": quality_gate,
        "formats": {
            "srt_url": f"/api/v2/sessions/{session_id}/subtitle.srt",
            "vtt_url": f"/api/v2/sessions/{session_id}/subtitle.vtt",
            "speaker_timestamped_url": f"/api/v2/sessions/{session_id}/transcript/speaker",
            "markers_url": f"/api/v2/sessions/{session_id}/markers",
            "retrieval_url": f"/api/v2/sessions/{session_id}/retrieval",
        },
        "quality_report": quality,
        "source_integrity": source_integrity,
        "classification": classification,
    }


@router.get("/sessions/{session_id}/markers")
async def get_markers(session_id: str, token: str = Depends(require_auth)):
    """Return the enrichment layer's semantic markers + spans."""
    sd = session_dir(session_id)
    if not sd.is_dir():
        raise HTTPException(404, detail=f"Session {session_id} not found.")
    markers = safe_read_json(str(sd / "enrichment" / "segment_markers.json")) or {}
    spans = safe_read_json(str(sd / "enrichment" / "semantic_spans.json")) or {}
    context_spans = _load_context_spans_payload(sd)
    audit = safe_read_json(str(sd / "enrichment" / "marker_audit.json")) or {}
    return {
        "session_id": session_id,
        "markers": markers.get("markers") or [],
        "spans": spans.get("spans") or [],
        "semantic_spans": spans.get("spans") or [],
        "context_spans": context_spans.get("spans") or [],
        "audit": audit,
    }


@router.get("/sessions/{session_id}/retrieval")
async def get_retrieval(session_id: str, token: str = Depends(require_auth)):
    """Return the best available retrieval payload, preferring context-aware v3."""
    sd = session_dir(session_id)
    if not sd.is_dir():
        raise HTTPException(404, detail=f"Session {session_id} not found.")
    payload = _load_retrieval_payload(sd)
    if payload is None:
        raise HTTPException(404, detail="Retrieval index not available yet.")
    return payload


@router.get("/sessions/{session_id}/transcript/stabilized")
async def get_stabilized_partial(session_id: str, token: str = Depends(require_auth)):
    sd = session_dir(session_id)
    data = safe_read_json(str(sd / "current" / "stabilized_partial.json")) or safe_read_json(str(sd / "stabilized_partial.json"))
    if data is None:
        raise HTTPException(404, detail="No stabilized partial transcript available yet.")
    return data


@router.get("/sessions/{session_id}/transcript/speaker")
async def get_speaker_transcript(session_id: str, token: str = Depends(require_auth)):
    """Get speaker-annotated transcript."""
    sd = session_dir(session_id)
    status_data = get_status(session_id) or {}
    if status_data.get("status") != "done":
        return JSONResponse(status_code=202, content={
            "message": "Transcript not ready yet.",
            "state": _map_status_to_public(status_data.get("status", "pending")),
        })

    # Try current/ first
    for loc in [sd / "current" / "transcript_by_speaker.txt", sd / "transcript_by_speaker.txt"]:
        if loc.is_file():
            return {"session_id": session_id, "speaker_timestamped": loc.read_text(encoding="utf-8")}

    raise HTTPException(404, detail="Speaker transcript not available.")


@router.get("/sessions/{session_id}/subtitle.srt")
async def get_srt(session_id: str, token: str = Depends(require_auth)):
    """Download SRT subtitle file."""
    sd = session_dir(session_id)
    for loc in [sd / "current" / "subtitles.srt", sd / "subtitles.srt"]:
        if loc.is_file():
            return FileResponse(str(loc), media_type="text/plain",
                              filename=f"{session_id}.srt")
    raise HTTPException(404, detail="SRT subtitle not available (session not done yet?).")


@router.get("/sessions/{session_id}/subtitle.vtt")
async def get_vtt(session_id: str, token: str = Depends(require_auth)):
    """Download VTT subtitle file."""
    sd = session_dir(session_id)
    for loc in [sd / "current" / "subtitles.vtt", sd / "subtitles.vtt"]:
        if loc.is_file():
            return FileResponse(str(loc), media_type="text/vtt",
                              filename=f"{session_id}.vtt")
    raise HTTPException(404, detail="VTT subtitle not available (session not done yet?).")


# ============================================================
# SESSION MANAGEMENT
# ============================================================

@router.post("/sessions/{session_id}/retry")
async def retry_session(session_id: str, token: str = Depends(require_auth)):
    """Retry a failed job."""
    if not session_exists(session_id):
        raise HTTPException(404, detail=f"Session {session_id} not found.")

    meta = get_session_meta(session_id) or {}
    if meta.get("state") != "finalized":
        raise HTTPException(409, detail="Session is not finalized.")

    status_data = get_status(session_id) or {}
    prev_status = status_data.get("status", "unknown")

    if prev_status == "done":
        raise HTTPException(409, detail="Job already completed successfully.")
    if prev_status in ("processing", "running"):
        raise HTTPException(409, detail="Job is currently running.")

    sd = session_dir(session_id)
    if not (sd / "audio.wav").is_file():
        raise HTTPException(500, detail="audio.wav missing.")

    # Re-enqueue
    r = _get_redis()
    if r is None:
        raise HTTPException(503, detail="Redis unavailable.")

    try:
        update_status(session_id, "pending")
        _enqueue_job(session_id, meta, {})
    except Exception as e:
        raise HTTPException(503, detail=f"Failed to enqueue: {e}")

    return {
        "session_id": session_id,
        "job_id": session_id,
        "status_url": f"/api/v2/jobs/{session_id}",
        "message": "Job re-enqueued for retry.",
        "previous_status": prev_status,
    }


# Important: this must be registered BEFORE /sessions/{session_id}
@router.get("/sessions/grouped")
async def get_sessions_grouped(token: str = Depends(require_auth)):
    """List sessions grouped by time period."""
    return list_sessions_grouped()


@router.get("/sessions/{session_id}")
async def get_session_detail(session_id: str, token: str = Depends(require_auth)):
    """Get session details."""
    if not session_exists(session_id):
        raise HTTPException(404, detail=f"Session {session_id} not found.")

    meta = get_session_meta(session_id) or {}
    status_data = get_status(session_id) or {}
    sd = session_dir(session_id)

    state = meta.get("state", "created")
    worker_status = status_data.get("status", "")
    if worker_status in ("done", "error"):
        state = _map_status_to_public(worker_status)

    error_msg = None
    if state == "error":
        err = safe_read_json(str(sd / "error.json"))
        error_msg = err.get("error_message") if err else status_data.get("error")

    # Pipeline audit: full execution trace for canonical runs
    pipeline_audit = _build_pipeline_audit(sd)
    diarization = _build_diarization_summary(sd, meta)
    quality_gate = _load_quality_gate(sd)
    context_spans_payload = _load_context_spans_payload(sd)

    # Derive detected language from reconciliation records if available
    detected_language = None
    if pipeline_audit:
        recon_data = safe_read_json(str(sd / "reconciliation" / "reconciliation_result.json"))
        if recon_data:
            for rec in recon_data.get("records", []):
                lang = rec.get("language")
                if lang:
                    detected_language = lang
                    break

    return {
        "session_id": session_id,
        "state": state,
        "backend_outcome": _derive_backend_outcome(worker_status, sd),
        "failure_category": None,
        "created_at": meta.get("created_at"),
        "finalized_at": meta.get("finalize_requested_at"),
        "mode": meta.get("mode", "stream"),
        "language": meta.get("language", "auto"),
        "speaker_count": meta.get("speaker_count"),
        "diarization_enabled": bool(diarization.get("requested")),
        "diarization": diarization,
        "chunk_count": len(meta.get("chunks", [])),
        "model_size": meta.get("model_size", "auto"),
        "audio_duration_s": meta.get("audio_duration_s"),
        "transcript_coverage_ratio": None,
        "detected_language": detected_language,
        "transcript_present": _has_transcript(sd),
        "provisional_partial_available": (sd / "partial_transcript.json").is_file(),
        "stabilized_partial_available": (sd / "current" / "stabilized_partial.json").is_file() or (sd / "stabilized_partial.json").is_file(),
        "final_transcript_available": (sd / "current" / "final_transcript.json").is_file() or (sd / "final_transcript.json").is_file(),
        "active_override": None,
        "error": error_msg,
        "quality_gate": quality_gate,
        "context_span_count": (context_spans_payload or {}).get("span_count", 0),
        "pipeline_audit": pipeline_audit,
    }


@router.post("/sessions/{session_id}/manual-override")
async def manual_override(session_id: str, request: Request,
                          token: str = Depends(require_auth)):
    """Apply manual override to a session."""
    if not session_exists(session_id):
        raise HTTPException(404, detail=f"Session {session_id} not found.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(422, detail={
            "reason": "invalid_body",
            "message": "Request body must be valid JSON.",
        })

    target_group = body.get("target_group", "")
    if not target_group or not target_group.strip():
        raise HTTPException(422, detail={
            "reason": "missing_target_group",
            "message": "target_group is required and must be non-empty.",
        })

    override_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    override_data = {
        "override_id": override_id,
        "session_id": session_id,
        "target_group": target_group.strip(),
        "reason": body.get("reason"),
        "reprocess_requested": body.get("reprocess_requested", False),
        "created_at": now,
    }

    try:
        sd = session_dir(session_id)
        atomic_write_json(str(sd / "manual_override.json"), override_data)
    except Exception as e:
        raise HTTPException(500, detail={
            "reason": "override_write_failed",
            "message": str(e),
        })

    return override_data


@router.post("/sessions/{session_id}/retranscribe")
async def retranscribe(session_id: str, request: Request,
                       token: str = Depends(require_auth)):
    """Retranscribe with a different model."""
    if not session_exists(session_id):
        raise HTTPException(404, detail=f"Session {session_id} not found.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(422, detail={"reason": "invalid_body", "message": "Invalid JSON."})

    model_id = body.get("model_id", "")
    if not model_id:
        raise HTTPException(422, detail={"reason": "missing_model_id", "message": "model_id is required."})

    resolved = resolve_model_id(model_id)
    info = get_model_info(resolved)
    if not info or not info.selectable_for_retranscription:
        raise HTTPException(422, detail={
            "reason": "model_not_suitable",
            "message": f"Model '{model_id}' is not suitable for retranscription.",
        })

    sd = session_dir(session_id)
    if not (sd / "audio.wav").is_file():
        raise HTTPException(409, detail={"reason": "audio_missing", "message": "audio.wav not present."})

    meta = get_session_meta(session_id) or {}
    if meta.get("state") != "finalized":
        raise HTTPException(409, detail={"reason": "session_not_finalized", "message": "Session not finalized."})

    revision_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Create revision dir
    rev_dir = sd / "retranscriptions" / revision_id
    rev_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(rev_dir / "revision_meta.json"), {
        "revision_id": revision_id,
        "session_id": session_id,
        "model_id": resolved,
        "language": body.get("language"),
        "status": "queued",
        "queued_at": now,
    })

    # Enqueue
    r = _get_redis()
    if r is None:
        raise HTTPException(503, detail={"reason": "redis_unavailable", "message": "Redis not available."})

    try:
        job = {
            "job_type": "v2_retranscribe",
            "session_id": session_id,
            "revision_id": revision_id,
            "model_id": resolved,
            "language": body.get("language"),
        }
        r.rpush(get_config().redis.queue, json.dumps(job))
    except Exception as e:
        raise HTTPException(503, detail={"reason": "enqueue_failed", "message": str(e)})

    return {
        "revision_id": revision_id,
        "session_id": session_id,
        "model_id": resolved,
        "status": "queued",
        "queued_at": now,
        "status_url": f"/api/v2/sessions/{session_id}/retranscriptions/{revision_id}",
        "poll_url": f"/api/v2/jobs/{session_id}",
    }


@router.get("/sessions/{session_id}/retranscriptions/{revision_id}")
async def get_retranscription(session_id: str, revision_id: str,
                              token: str = Depends(require_auth)):
    """Get retranscription revision details."""
    sd = session_dir(session_id)
    rev_meta = sd / "retranscriptions" / revision_id / "revision_meta.json"
    if not rev_meta.is_file():
        raise HTTPException(404, detail={"reason": "revision_not_found", "message": "Revision not found."})
    return safe_read_json(str(rev_meta)) or {}


@router.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, token: str = Depends(require_auth)):
    """Delete a session and all its data."""
    if not session_exists(session_id):
        raise HTTPException(404, detail=f"Session {session_id} not found.")
    delete_session(session_id)
    return {"deleted": True, "session_id": session_id}


# ============================================================
# HELPERS
# ============================================================

def _get_redis():
    try:
        import redis
        cfg = get_config()
        r = redis.Redis(host=cfg.redis.host, port=cfg.redis.port, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


def _body_or_meta(meta: dict, body: dict, key: str, default=None):
    if key in body and body.get(key) is not None:
        return body.get(key)
    value = meta.get(key)
    return default if value is None else value


def _enqueue_job(session_id: str, meta: dict, body: dict):
    """Enqueue a transcription job to Redis."""
    r = _get_redis()
    if r is None:
        logger.warning("Redis unavailable, job not enqueued")
        return

    model_size = _body_or_meta(meta, body, "model_size", "auto")
    model_id = _body_or_meta(meta, body, "model_id")
    diarization_policy = str(_body_or_meta(meta, body, "diarization_policy", "auto")).strip().lower() or "auto"
    run_diarization = diarization_policy == "forced"

    # Determine job type: use canonical pipeline for "auto" without specific model
    if model_size == "auto" and not model_id:
        job_type = "v2_canonical"
    else:
        job_type = "v2"

    job = {
        "job_type": job_type,
        "session_id": session_id,
        "language": _body_or_meta(meta, body, "language", "auto"),
        "allowed_languages": _body_or_meta(meta, body, "allowed_languages", []),
        "forced_language": _body_or_meta(meta, body, "forced_language"),
        "transcription_mode": _body_or_meta(meta, body, "transcription_mode", "verbatim_multilingual"),
        "model_size": model_size,
        "model_id": model_id,
        "run_diarization": run_diarization,
        "diarization_policy": diarization_policy,
        "speaker_count": _body_or_meta(meta, body, "speaker_count"),
        "force_transcribe_only": _body_or_meta(meta, body, "force_transcribe_only", False),
        "session_integrity": _body_or_meta(meta, body, "session_integrity"),
        "language_candidates": _body_or_meta(meta, body, "language_candidates"),
        "language_selection_strategy": _body_or_meta(meta, body, "language_selection_strategy"),
    }

    cfg = get_config()
    try:
        r.rpush(cfg.redis.queue, json.dumps(job))
        update_status(session_id, "uploaded", queued_at=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}")


def _maybe_trigger_live_canonical(session_id: str, chunk_count: int, meta: dict):
    """Maybe enqueue an incremental canonical live job.

    We reuse the existing cadence/cooldown config that used to drive the fake
    partial preview path, but the queued work is now the real lexical canonical
    pipeline up through canonical assembly.
    """
    cfg = get_config()
    every_n = cfg.worker.partial_every_n_chunks
    if every_n <= 0:
        return
    if chunk_count % every_n != 0:
        return

    # Cooldown check
    cooldown = cfg.worker.partial_cooldown_seconds
    last_trigger = meta.get("last_live_trigger_at")
    if last_trigger and cooldown > 0:
        try:
            last_dt = datetime.fromisoformat(last_trigger.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
            if elapsed < cooldown:
                return
        except (ValueError, TypeError):
            pass

    # Check lock
    sd = session_dir(session_id)
    lock = sd / "live_canonical_pending"
    if lock.exists():
        return

    r = _get_redis()
    if r is None:
        return

    try:
        lock.touch()
        r.rpush(cfg.redis.partial_queue, json.dumps({
            "job_type": "v2_canonical_live",
            "session_id": session_id,
        }))
        update_session_meta(session_id, {
            "last_live_trigger_at": datetime.now(timezone.utc).isoformat(),
            "last_live_trigger_chunk_count": chunk_count,
        })
    except Exception as e:
        logger.warning(f"Canonical live trigger failed: {e}")
        try:
            lock.unlink()
        except OSError:
            pass


def _read_worker_health() -> dict:
    cfg = get_config()
    health_path = str(Path(cfg.storage.sessions_dir).parent / "worker_health.json")
    return safe_read_json(health_path) or {}


def _map_status_to_public(raw: str) -> str:
    mapping = {
        "created": "queued",
        "uploaded": "queued",
        "pending": "queued",
        "processing": "running",
        "running": "running",
        "done": "done",
        "error": "error",
        "cancelled": "error",
    }
    return mapping.get(raw, "queued")


def _derive_backend_outcome(raw_status: str, sd: Path) -> str:
    if raw_status == "done":
        if _has_transcript(sd):
            return "completed"
        return "completed_empty"
    elif raw_status in ("error", "cancelled"):
        return "failed"
    elif raw_status in ("processing", "running"):
        return "processing"
    elif raw_status in ("created", "uploaded", "pending"):
        return "queued"
    return "not_started"


def _state_message(state: str) -> str:
    messages = {
        "queued": "Job is queued for processing.",
        "running": "Job is currently being processed.",
        "done": "Processing complete.",
        "error": "Job failed.",
    }
    return messages.get(state, "Unknown state.")


def _has_transcript(sd: Path) -> bool:
    for loc in [sd / "current" / "transcript.txt", sd / "transcript.txt"]:
        if loc.is_file() and loc.stat().st_size > 0:
            return True
    return False


def _build_pipeline_audit(sd: Path) -> Optional[dict]:
    """Build a pipeline audit summary from on-disk run artifacts.

    Returns None when no canonical pipeline run exists (legacy sessions).
    """
    # Find canonical run id
    pointer = sd / "pipeline" / "canonical_run_id.txt"
    if not pointer.is_file():
        return None
    run_id = pointer.read_text(encoding="utf-8").strip()
    if not run_id:
        return None

    run_meta = safe_read_json(str(sd / "pipeline" / "runs" / run_id / "run_meta.json"))
    if not run_meta:
        return None

    run_dir = sd / "pipeline" / "runs" / run_id

    # --- Stage summary ---
    stages_raw = run_meta.get("stages", {})
    stages_audit = []
    for name in run_meta.get("stage_names", []):
        normalized_name = normalize_stage_name(name)
        s = stages_raw.get(name, {}) or stages_raw.get(normalized_name, {})
        entry = {
            "name": normalized_name,
            "legacy_name": name if name != normalized_name else None,
            "status": s.get("status", "pending"),
            "actual_model": s.get("actual_model"),
            "routing_reason": s.get("routing_reason"),
            "degraded": s.get("degraded", False),
            "fallback": s.get("fallback", False),
            "error": s.get("error"),
            "started_at": s.get("started_at"),
            "finished_at": s.get("finished_at"),
        }
        # Load routing detail if present
        routing_file = None
        if normalized_name == "first_pass_medium":
            routing_file = "first_pass_routing.json"
        elif normalized_name == "candidate_asr_large_v3":
            routing_file = "candidate_a_routing.json"
        elif normalized_name == "candidate_asr_secondary":
            routing_file = "candidate_b_routing.json"
        if routing_file:
            routing = None
            for stage_dir_name in stage_directory_candidates(normalized_name):
                routing = safe_read_json(str(run_dir / "stages" / stage_dir_name / routing_file))
                if routing:
                    break
            if routing:
                entry["selected_model"] = routing.get("selected_model")
                lang_ctx = routing.get("language_context", {})
                entry["forced_language"] = lang_ctx.get("forced_language")
                entry["transcription_mode"] = lang_ctx.get("transcription_mode")
        stages_audit.append(entry)

    # --- ASR summary ---
    asr_summary = safe_read_json(str(sd / "candidates" / "asr_summary.json"))
    asr_audit = None
    if asr_summary:
        by_model = asr_summary.get("candidates_by_model", {})
        asr_audit = {
            "models_attempted": asr_summary.get("model_ids", []),
            "window_count": asr_summary.get("window_count", 0),
            "total_candidates": asr_summary.get("candidate_count", 0),
            "per_model": {
                model_id: {
                    "success": stats.get("success", 0),
                    "failed": stats.get("failed", 0),
                    "total": stats.get("count", 0),
                }
                for model_id, stats in by_model.items()
            },
        }

    # --- Reconciliation summary ---
    recon = safe_read_json(str(sd / "reconciliation" / "reconciliation_result.json"))
    recon_audit = None
    if recon:
        recon_audit = {
            "stripe_count": recon.get("stripe_count", 0),
            "llm_resolved": recon.get("llm_resolved_count", 0),
            "fallback_resolved": recon.get("fallback_resolved_count", 0),
            "validation_rejected": recon.get("validation_rejected_count", 0),
            "total_chosen_chars": recon.get("total_chosen_chars", 0),
            "reconciler_status": recon.get("reconciler_status"),
            "llm_available": recon.get("llm_available", False),
        }

    # --- Decode lattice summary ---
    lattice = safe_read_json(str(sd / "windows" / "decode_windows.json"))
    lattice_audit = None
    if lattice:
        lattice_audit = {
            "total_windows": lattice.get("window_count", 0),
            "scheduled_windows": lattice.get("scheduled_count", 0),
            "bridge_windows": lattice.get("bridge_count", 0),
            "geometry": lattice.get("geometry"),
        }

    # --- Triage summary ---
    triage = safe_read_json(str(sd / "triage" / "triage_result.json"))
    triage_audit = None
    if triage:
        triage_audit = {
            "speech_ratio": triage.get("speech_ratio"),
            "island_count": len(triage.get("speech_islands", [])),
            "total_duration_ms": triage.get("total_duration_ms"),
            "speech_duration_ms": triage.get("speech_duration_ms"),
        }

    return {
        "run_id": run_id,
        "run_status": run_meta.get("status"),
        "run_type": run_meta.get("run_type", "canonical"),
        "started_at": run_meta.get("started_at"),
        "finished_at": run_meta.get("finished_at"),
        "error": run_meta.get("error"),
        "stages": stages_audit,
        "asr": asr_audit,
        "reconciliation": recon_audit,
        "decode_lattice": lattice_audit,
        "triage": triage_audit,
    }


def _has_any_transcript_file(d: Path) -> bool:
    if not d.is_dir():
        return False
    for f in ("transcript.txt", "raw_transcript.json", "transcript_timestamps.json"):
        if (d / f).is_file():
            return True
    return False


def _load_segments(read_dir: Path) -> list:
    """Load segments from available sources with fallback chain."""
    # Try raw_transcript.json
    raw = safe_read_json(str(read_dir / "raw_transcript.json"))
    if raw and raw.get("segments"):
        segs = raw["segments"]
        return _normalize_segments(segs)

    # Try transcript_by_speaker.json
    speaker = safe_read_json(str(read_dir / "transcript_by_speaker.json"))
    if speaker:
        segs = []
        if isinstance(speaker, dict) and "segments" in speaker:
            segs = speaker["segments"]
        elif isinstance(speaker, list):
            segs = speaker
        if segs:
            return _normalize_segments(segs)

    # Try transcript_timestamps.json
    timestamps = safe_read_json(str(read_dir / "transcript_timestamps.json"))
    if timestamps and timestamps.get("segments"):
        return _normalize_segments(timestamps["segments"])

    return []


def _normalize_segments(segs: list) -> list:
    """Normalize segment format, handling all legacy formats."""
    result = []
    for seg in segs:
        if not isinstance(seg, dict):
            continue

        # Resolve start_ms
        start_ms = seg.get("start_ms")
        if start_ms is None:
            start_s = seg.get("start_s") or seg.get("start") or 0
            start_ms = int(float(start_s) * 1000)

        # Resolve end_ms
        end_ms = seg.get("end_ms")
        if end_ms is None:
            end_s = seg.get("end_s") or seg.get("end") or 0
            end_ms = int(float(end_s) * 1000)

        text = seg.get("text", "").strip()
        if not text:
            continue

        result.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "speaker": seg.get("speaker"),
            "text": text,
            "corruption_flags": seg.get("corruption_flags", []),
        })

    result.sort(key=lambda s: s["start_ms"])
    return result


def _synthesize_speaker_timestamped(segments: list) -> Optional[str]:
    """Build speaker timestamped text from segments."""
    if not segments:
        return None
    lines = []
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        start_s = round(seg["start_ms"] / 1000, 1)
        end_s = round(seg["end_ms"] / 1000, 1)
        text = seg.get("text", "")
        if not text.strip():
            continue
        lines.append(f"[{speaker}] {start_s}s\u2013{end_s}s")
        lines.append(text)
        lines.append("")
    return "\n".join(lines) if lines else None
