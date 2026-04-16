"""
Main application entry point.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api.api_v2 import _enqueue_job, _read_worker_health, router as api_v2_router
from app.api.auth import _extract_token
from app.api.auth import require_auth_upload
from app.core.config import get_config
from app.core.tls import ensure_tls_assets
from app.pipeline.ingest import split_file_upload_to_transport_chunks
from app.storage.session_store import (
    cleanup_abandoned_draft_sessions,
    create_session,
    delete_session,
    get_session_meta,
    list_sessions,
    register_chunk,
    session_dir,
    update_session_meta,
    update_status,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="LiveVoiceTranscriptor", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AuthMiddleware(BaseHTTPMiddleware):
    PUBLIC_PATHS = {"/", "/health", "/api/health", "/api/v2/health"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self.PUBLIC_PATHS or path.startswith("/api/v2/health"):
            return await call_next(request)
        if not path.startswith("/api/"):
            return await call_next(request)

        cfg = get_config()
        token = _extract_token(request)
        if cfg.auth.token and token != cfg.auth.token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required. Use 'Authorization: Bearer <token>' header."},
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


app.add_middleware(AuthMiddleware)


def _redis_diagnostics() -> dict:
    cfg = get_config()
    result = {
        "ok": False,
        "host": cfg.redis.host,
        "port": cfg.redis.port,
        "queue": cfg.redis.queue,
        "partial_queue": cfg.redis.partial_queue,
        "queue_len": 0,
        "partial_queue_len": 0,
        "error": None,
    }
    try:
        import redis

        client = redis.Redis(host=cfg.redis.host, port=cfg.redis.port)
        client.ping()
        result["ok"] = True
        result["queue_len"] = client.llen(cfg.redis.queue) or 0
        result["partial_queue_len"] = client.llen(cfg.redis.partial_queue) or 0
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _gpu_diagnostics() -> dict:
    worker_health = _read_worker_health()
    result = {
        "available": bool(worker_health.get("gpu_available", False)),
        "name": worker_health.get("gpu_name"),
        "reason": worker_health.get("gpu_reason"),
        "selected_device": worker_health.get("selected_device"),
        "selected_compute_type": worker_health.get("selected_compute_type"),
        "worker_started_at": worker_health.get("worker_started_at"),
        "backend_status": worker_health.get("backend_status", {}),
        "error": None,
        "server_probe": {
            "available": False,
            "name": None,
            "error": None,
        },
    }
    try:
        import torch

        result["server_probe"]["available"] = torch.cuda.is_available()
        if result["server_probe"]["available"]:
            result["server_probe"]["name"] = torch.cuda.get_device_name(0)
        if not worker_health:
            result["available"] = result["server_probe"]["available"]
            result["name"] = result["server_probe"]["name"]
            result["reason"] = "server_cuda_probe"
    except ImportError:
        result["server_probe"]["error"] = "torch not installed"
        if not worker_health:
            result["error"] = "torch not installed"
    except Exception as exc:
        result["server_probe"]["error"] = str(exc)
        if not worker_health:
            result["error"] = str(exc)
    return result


@app.on_event("startup")
async def startup():
    cfg = get_config()
    Path(cfg.storage.sessions_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Sessions dir: %s", cfg.storage.sessions_dir)
    cleanup_result = cleanup_abandoned_draft_sessions()
    if cleanup_result.get("deleted_count"):
        logger.info(
            "Cleaned up %s abandoned draft session(s) on startup",
            cleanup_result["deleted_count"],
        )
    logger.info("Server: %s:%s tls=%s", cfg.server.host, cfg.server.port, cfg.server.tls_enabled)
    if cfg.server.tls_enabled:
        tls_info = ensure_tls_assets()
        logger.info("TLS ready: %s", tls_info)
    redis_info = _redis_diagnostics()
    if redis_info["ok"]:
        logger.info("Redis connected: %s:%s", cfg.redis.host, cfg.redis.port)
    else:
        logger.warning("Redis not available: %s", redis_info["error"])


app.include_router(api_v2_router)


@app.get("/health")
async def health():
    return {"ok": True, "service": "web"}


@app.get("/api/health")
async def api_health():
    return {"ok": True, "service": "web"}


@app.get("/api/sessions")
async def legacy_list_sessions():
    return {"sessions": list_sessions(limit=100)}


@app.get("/api/diagnostics")
async def diagnostics():
    cfg = get_config()
    tls = {
        "enabled": cfg.server.tls_enabled,
        "cert_file": cfg.server.tls_cert_file,
        "key_file": cfg.server.tls_key_file,
        "auto_generate_self_signed": cfg.server.tls_auto_generate_self_signed,
        "cert_exists": Path(cfg.server.tls_cert_file).is_file(),
        "key_exists": Path(cfg.server.tls_key_file).is_file(),
    }
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "redis": _redis_diagnostics(),
        "sessions_dir": cfg.storage.sessions_dir,
        "models_dir": cfg.model_paths.models_dir,
        "models_dir_exists": cfg.model_paths.models_dir_exists,
        "server": {
            "host": cfg.server.host,
            "port": cfg.server.port,
        },
        "tls": tls,
        "gpu": _gpu_diagnostics(),
    }


@app.get("/api/selftest")
async def selftest():
    try:
        from faster_whisper import WhisperModel  # noqa: F401

        return {"ok": True, "details": "faster_whisper importable"}
    except ImportError:
        return {"ok": False, "details": "faster_whisper not installed"}
    except Exception as exc:
        return {"ok": False, "details": str(exc)}


@app.post("/api/file-upload")
@app.post("/file-upload")
async def legacy_file_upload(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    model_size: str = Form("auto"),
    diarization: bool = Form(False),
    diarization_policy: str = Form(""),
    token: str = Depends(require_auth_upload),
):
    cfg = get_config()
    session = create_session({"mode": "file", "source_type": "device_import"})
    session_id = session["session_id"]
    sd = session_dir(session_id)
    raw_dir = sd / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    raw_upload_path = raw_dir / f"uploaded{suffix}"
    upload_limit_bytes = cfg.worker.max_file_upload_mb * 1024 * 1024
    total_bytes = 0

    try:
        with raw_upload_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > upload_limit_bytes:
                    raise ValueError(
                        f"Uploaded file exceeds {cfg.worker.max_file_upload_mb}MB limit."
                    )
                out.write(chunk)

        split_result = split_file_upload_to_transport_chunks(
            str(raw_upload_path),
            str(sd / "chunks"),
            cfg.geometry.transport_chunk_ms,
        )
        if not split_result.get("success"):
            raise RuntimeError(split_result.get("error", "transport_chunk_split_failed"))

        for chunk_spec in split_result.get("chunks", []):
            register_chunk(
                session_id,
                chunk_spec["chunk_index"],
                {
                    "chunk_index": chunk_spec["chunk_index"],
                    "chunk_started_ms": chunk_spec["chunk_started_ms"],
                    "chunk_duration_ms": chunk_spec["chunk_duration_ms"],
                    "is_final": chunk_spec.get("is_final", False),
                    "file_size": chunk_spec.get("file_size", 0),
                },
            )

        resolved_diarization_policy = (
            str(diarization_policy).strip().lower()
            if str(diarization_policy).strip()
            else ("forced" if diarization else "auto")
        )
        if resolved_diarization_policy not in {"auto", "off", "forced"}:
            raise ValueError("Invalid diarization_policy. Allowed: auto, off, forced.")
        run_diarization = resolved_diarization_policy == "forced"

        update_session_meta(
            session_id,
            {
                "state": "finalized",
                "language": language,
                "model_size": model_size,
                "run_diarization": run_diarization,
                "diarization_policy": resolved_diarization_policy,
                "finalize_requested_at": datetime.now(timezone.utc).isoformat(),
                "original_filename": file.filename,
                "original_file_size": total_bytes,
                "transport_chunk_count": split_result.get("chunk_count", 0),
                "transport_chunking_method": split_result.get("method"),
            },
        )
        update_status(session_id, "uploaded")
        _enqueue_job(
            session_id,
            get_session_meta(session_id) or {},
            {
                "language": language,
                "model_size": model_size,
                "run_diarization": run_diarization,
                "diarization_policy": resolved_diarization_policy,
            },
        )
        return {
            "accepted": True,
            "session_id": session_id,
            "job_id": session_id,
            "status_url": f"/api/v2/jobs/{session_id}",
            "transport_chunk_count": split_result.get("chunk_count", 0),
        }
    except ValueError as e:
        delete_session(session_id)
        return JSONResponse(status_code=413, content={"detail": str(e)})
    except Exception as e:
        logger.exception("Whole-file upload failed for session %s", session_id)
        delete_session(session_id)
        return JSONResponse(status_code=500, content={"detail": f"File upload failed: {e}"})
    finally:
        await file.close()


@app.get("/", response_class=HTMLResponse)
async def root():
    from app.ui.dashboard import get_ui_html

    return HTMLResponse(content=get_ui_html())


@app.get("/ui/dashboard.js")
async def dashboard_js():
    ui_js_path = Path(__file__).resolve().parent / "ui" / "dashboard.js"
    return FileResponse(str(ui_js_path), media_type="application/javascript")


def _uvicorn_kwargs() -> dict:
    cfg = get_config()
    kwargs = {
        "host": cfg.server.host,
        "port": cfg.server.port,
        "reload": False,
    }
    if cfg.server.tls_enabled:
        tls_info = ensure_tls_assets()
        kwargs["ssl_certfile"] = tls_info["cert_file"]
        kwargs["ssl_keyfile"] = tls_info["key_file"]
    return kwargs


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", **_uvicorn_kwargs())


if __name__ == "__main__":
    main()
