"""
Session storage layer.

Manages the filesystem layout for sessions with the canonical pipeline
artifact hierarchy:

  sessions/{session_id}/
    chunks/             - raw uploaded transport chunks
    raw/                - original media + ingest metadata
    normalized/         - session-normalized audio timeline
    triage/             - acoustic tags, speech islands
    windows/            - synthesized decode windows
    candidates/         - per-model raw ASR outputs
    reconciliation/     - stripe packets, arbitration outputs
    canonical/          - canonical segments + final transcript surfaces
    derived/            - subtitles, quality, classification, digests
    current/            - compatibility read surface (API reads from here)
    pipeline/           - pipeline run tracking
    v2_session.json     - session metadata
    status.json         - worker status
"""
import os
import stat
import uuid
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

from app.core.config import get_config
from app.core.atomic_io import atomic_write_json, safe_read_json

logger = logging.getLogger(__name__)

# Internal storage layers (canonical spec section 11)
STORAGE_LAYERS = [
    "chunks", "raw", "normalized", "triage", "windows",
    "candidates", "reconciliation", "canonical", "derived", "current",
    "pipeline",
]


def _sessions_dir() -> Path:
    return Path(get_config().storage.sessions_dir)


def session_dir(session_id: str) -> Path:
    return _sessions_dir() / session_id


def ensure_session_dirs(session_id: str) -> Path:
    """Create all session subdirectories."""
    sd = session_dir(session_id)
    for layer in STORAGE_LAYERS:
        (sd / layer).mkdir(parents=True, exist_ok=True)
    return sd


def create_session(body: dict) -> dict:
    """Create a new session from API request body.

    Handles all alias fields for Android compatibility:
    - sample_rate_hz OR sample_rate
    - source_type=device_import -> mode=file
    - chunk_duration_sec stored for diagnostics
    - diarization stored as diarization_hint
    """
    cleanup_abandoned_draft_sessions()
    sid = str(uuid.uuid4())
    sd = ensure_session_dirs(sid)

    # Resolve aliases
    sample_rate = body.get("sample_rate_hz") or body.get("sample_rate") or 16000

    mode = body.get("mode", "stream")
    source_type = body.get("source_type")
    if source_type == "device_import":
        mode = "file"

    now = datetime.now(timezone.utc).isoformat()

    allowed_languages = body.get("allowed_languages")
    if not isinstance(allowed_languages, list):
        allowed_languages = []
    allowed_languages = [str(item).strip() for item in allowed_languages if str(item).strip()]

    forced_language = body.get("forced_language")
    if forced_language == "auto":
        forced_language = None

    session_meta = {
        "session_id": sid,
        "version": "v2",
        "state": "created",
        "device_id": body.get("device_id"),
        "client_session_id": body.get("client_session_id"),
        "started_at_utc": body.get("started_at_utc"),
        "sample_rate_hz": int(sample_rate),
        "channels": int(body.get("channels", 1)),
        "format": body.get("format", "wav"),
        "mode": mode,
        "source_type": source_type,
        "chunk_duration_sec": body.get("chunk_duration_sec"),
        "diarization_hint": bool(body.get("diarization", False)),
        "allowed_languages": allowed_languages,
        "forced_language": forced_language,
        "transcription_mode": body.get("transcription_mode", "verbatim_multilingual"),
        "diarization_policy": body.get("diarization_policy", "auto"),
        "chunks": [],
        "created_at": now,
        "first_chunk_at": None,
        "last_chunk_at": None,
        "finalize_requested_at": None,
        "last_partial_trigger_at": None,
        "run_diarization": False,
        "language": "auto",
        "model_size": "auto",
        "model_id": None,
        "session_integrity": None,
    }

    atomic_write_json(str(sd / "v2_session.json"), session_meta)
    atomic_write_json(str(sd / "status.json"), {
        "status": "created",
        "progress": {"upload": 0, "processing": 0, "stage": "created"},
    })

    return {
        "session_id": sid,
        "upload_url": f"/api/v2/sessions/{sid}/chunks",
    }


def get_session_meta(session_id: str) -> Optional[dict]:
    """Read session metadata."""
    path = session_dir(session_id) / "v2_session.json"
    return safe_read_json(str(path))


def update_session_meta(session_id: str, updates: dict) -> None:
    """Merge updates into session metadata."""
    path = session_dir(session_id) / "v2_session.json"
    meta = safe_read_json(str(path))
    if meta is None:
        meta = {}
    meta.update(updates)
    atomic_write_json(str(path), meta)


def get_status(session_id: str) -> Optional[dict]:
    """Read worker status."""
    path = session_dir(session_id) / "status.json"
    return safe_read_json(str(path))


def update_status(session_id: str, status: str, error: str = None,
                  progress: dict = None, **extra) -> None:
    """Update worker status atomically."""
    path = session_dir(session_id) / "status.json"
    data = safe_read_json(str(path)) or {}
    data["status"] = status
    if error is not None:
        data["error"] = error
    if progress is not None:
        data["progress"] = progress
    now = datetime.now(timezone.utc).isoformat()
    if status in ("running", "processing") and "started_at" not in data:
        data["started_at"] = now
    if status in ("done", "error", "cancelled"):
        data["finished_at"] = now
    data.update(extra)
    atomic_write_json(str(path), data)


def update_progress(session_id: str, stage: str, processing_pct: int) -> None:
    """Update processing progress."""
    path = session_dir(session_id) / "status.json"
    data = safe_read_json(str(path)) or {}
    data["progress"] = {
        "upload": 100,
        "processing": min(100, max(0, processing_pct)),
        "stage": stage,
    }
    atomic_write_json(str(path), data)


def register_chunk(session_id: str, chunk_index: int, chunk_meta: dict) -> int:
    """Register a chunk upload, return total chunk count."""
    meta = get_session_meta(session_id)
    if meta is None:
        raise FileNotFoundError(f"Session {session_id} not found")
    chunks = [c for c in meta.get("chunks", []) if c.get("chunk_index") != chunk_index]
    chunks.append(chunk_meta)
    chunks.sort(key=lambda c: c.get("chunk_index", 0))
    now = datetime.now(timezone.utc).isoformat()
    updates = {"chunks": chunks, "last_chunk_at": now}
    if meta.get("first_chunk_at") is None:
        updates["first_chunk_at"] = now
    if meta.get("state") == "created":
        updates["state"] = "receiving"
    update_session_meta(session_id, updates)
    return len(chunks)


def get_chunk_paths(session_id: str) -> List[str]:
    """Get ordered list of chunk file paths."""
    sd = session_dir(session_id) / "chunks"
    if not sd.is_dir():
        return []
    chunks = sorted(sd.glob("chunk_*.wav"))
    return [str(c) for c in chunks]


def session_exists(session_id: str) -> bool:
    sd = session_dir(session_id)
    return sd.is_dir() and (sd / "v2_session.json").is_file()


def is_v2_session(session_id: str) -> bool:
    meta = get_session_meta(session_id)
    return meta is not None and meta.get("version") == "v2"


def delete_session(session_id: str) -> bool:
    sd = session_dir(session_id)
    if sd.is_dir():
        _remove_tree(sd)
        return True
    return False


def list_sessions(limit: int = 100) -> List[dict]:
    """List sessions, newest first."""
    cleanup_abandoned_draft_sessions()
    sdir = _sessions_dir()
    if not sdir.is_dir():
        return []
    sessions = []
    entries = list(sdir.iterdir())
    for entry in entries:
        if not entry.is_dir():
            continue
        meta = safe_read_json(str(entry / "v2_session.json"))
        status = safe_read_json(str(entry / "status.json"))
        if meta is None:
            continue
        updated_at = _session_activity_timestamp(meta, status)
        sessions.append({
            "session_id": entry.name,
            "is_v2": meta.get("version") == "v2",
            "status": _display_session_state(meta, status),
            "raw_status": (status or {}).get("status", "unknown"),
            "updated_at": updated_at,
            "progress": (status or {}).get("progress", {}),
            "created_at": meta.get("created_at"),
            "device_id": meta.get("device_id"),
            "mode": meta.get("mode", "stream"),
        })
    sessions.sort(key=lambda item: _parse_iso_timestamp(item.get("updated_at")), reverse=True)
    return sessions[:limit]


def list_sessions_grouped() -> dict:
    """Group sessions by time period for the UI."""
    from datetime import timedelta
    sessions = list_sessions(limit=500)
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())

    groups = {
        "today": {"group_key": "today", "label": "Today", "sessions": []},
        "this_week": {"group_key": "this_week", "label": "This Week", "sessions": []},
        "older": {"group_key": "older", "label": "Older", "sessions": []},
    }

    for sess in sessions:
        status_data = get_status(sess["session_id"]) or {}
        meta = get_session_meta(sess["session_id"]) or {}
        created = meta.get("created_at", "")

        summary = {
            "session_id": sess["session_id"],
            "state": _display_session_state(meta, status_data),
            "backend_outcome": _derive_backend_outcome(status_data),
            "created_at": created,
            "finalized_at": meta.get("finalize_requested_at"),
            "mode": meta.get("mode", "stream"),
            "language": meta.get("language", "auto"),
            "speaker_count": meta.get("speaker_count"),
            "chunk_count": len(meta.get("chunks", [])),
            "model_size": meta.get("model_size", "auto"),
            "audio_duration_s": status_data.get("audio_duration_s"),
            "transcript_present": _has_transcript(sess["session_id"]),
            "active_override": None,
        }

        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if dt >= today_start:
                groups["today"]["sessions"].append(summary)
            elif dt >= week_start:
                groups["this_week"]["sessions"].append(summary)
            else:
                groups["older"]["sessions"].append(summary)
        except (ValueError, AttributeError):
            groups["older"]["sessions"].append(summary)

    result_groups = [g for g in groups.values() if g["sessions"]]
    total = sum(len(g["sessions"]) for g in result_groups)
    return {"groups": result_groups, "total_sessions": total}


def _derive_backend_outcome(status_data: dict) -> str:
    s = status_data.get("status", "")
    if s == "done":
        return "completed"
    elif s in ("error", "cancelled"):
        return "failed"
    elif s in ("processing", "running"):
        return "processing"
    elif s in ("created", "uploaded", "pending"):
        return "queued"
    return "not_started"


def cleanup_abandoned_draft_sessions(max_age_minutes: Optional[int] = None) -> dict:
    """Delete stale sessions that never progressed beyond draft creation."""
    cfg = get_config()
    if not cfg.storage.auto_cleanup_draft_sessions:
        return {
            "enabled": False,
            "deleted_count": 0,
            "deleted_session_ids": [],
        }

    ttl_minutes = max_age_minutes
    if ttl_minutes is None:
        ttl_minutes = cfg.storage.draft_session_max_age_minutes
    if ttl_minutes <= 0:
        return {
            "enabled": True,
            "deleted_count": 0,
            "deleted_session_ids": [],
        }

    now = datetime.now(timezone.utc)
    deleted_session_ids = []
    sdir = _sessions_dir()
    if not sdir.is_dir():
        return {
            "enabled": True,
            "deleted_count": 0,
            "deleted_session_ids": [],
        }

    for entry in sdir.iterdir():
        if not entry.is_dir():
            continue
        meta = safe_read_json(str(entry / "v2_session.json")) or {}
        status = safe_read_json(str(entry / "status.json")) or {}
        if not _is_abandoned_draft_session(entry, meta, status, now, ttl_minutes):
            continue
        try:
            _remove_tree(entry)
            deleted_session_ids.append(entry.name)
        except OSError as exc:
            logger.warning("Failed to delete abandoned draft session %s: %s", entry.name, exc)

    if deleted_session_ids:
        logger.info(
            "Deleted %s abandoned draft session(s): %s",
            len(deleted_session_ids),
            ", ".join(sorted(deleted_session_ids)),
        )

    return {
        "enabled": True,
        "deleted_count": len(deleted_session_ids),
        "deleted_session_ids": sorted(deleted_session_ids),
    }


def _display_session_state(meta: dict, status_data: dict) -> str:
    raw_status = (status_data or {}).get("status", "")
    session_state = (meta or {}).get("state", "")

    if raw_status == "done":
        return "done"
    if raw_status in ("error", "cancelled"):
        return "error"
    if raw_status in ("running", "processing"):
        return "running"
    if session_state == "finalized" and raw_status in ("", "created", "uploaded", "pending"):
        return "queued"
    if raw_status in ("uploaded", "pending"):
        return "queued"
    if session_state:
        return session_state
    return raw_status or "unknown"


def _session_activity_timestamp(meta: dict, status_data: dict) -> str:
    for value in [
        (status_data or {}).get("finished_at"),
        (status_data or {}).get("started_at"),
        (status_data or {}).get("queued_at"),
        (meta or {}).get("finalize_requested_at"),
        (meta or {}).get("last_chunk_at"),
        (meta or {}).get("created_at"),
    ]:
        if value:
            return value
    return ""


def _parse_iso_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _is_abandoned_draft_session(
    session_path: Path,
    meta: dict,
    status_data: dict,
    now: datetime,
    ttl_minutes: int,
) -> bool:
    if not meta:
        return False
    if meta.get("state") != "created":
        return False
    if meta.get("chunks"):
        return False
    if meta.get("first_chunk_at") or meta.get("last_chunk_at"):
        return False
    if meta.get("finalize_requested_at"):
        return False

    raw_status = (status_data or {}).get("status", "created")
    if raw_status not in ("", "created"):
        return False
    if any((status_data or {}).get(key) for key in ("queued_at", "started_at", "finished_at")):
        return False

    # Keep anything that already wrote upload or pipeline artifacts.
    if any((session_path / "chunks").glob("chunk_*.wav")):
        return False
    if (session_path / "metadata.json").is_file():
        return False
    if (session_path / "pipeline" / "canonical_run_id.txt").is_file():
        return False

    created_at = _parse_iso_timestamp(meta.get("created_at"))
    age_seconds = (now - created_at).total_seconds()
    return age_seconds >= ttl_minutes * 60


def _remove_tree(path: Path) -> None:
    def _onerror(func, target, exc_info):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            raise exc_info[1]

    shutil.rmtree(str(path), onerror=_onerror)


def _has_transcript(session_id: str) -> bool:
    sd = session_dir(session_id)
    for loc in [sd / "current" / "transcript.txt", sd / "transcript.txt"]:
        if loc.is_file() and loc.stat().st_size > 0:
            return True
    return False
