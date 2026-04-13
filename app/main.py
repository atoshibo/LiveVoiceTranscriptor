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
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.api.api_v2 import _enqueue_job, router as api_v2_router
from app.api.auth import _extract_token
from app.api.auth import require_auth_upload
from app.core.config import get_config
from app.core.tls import ensure_tls_assets
from app.storage.session_store import create_session, get_session_meta, list_sessions, register_chunk, session_dir, update_session_meta, update_status

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
    result = {"available": False, "error": None}
    try:
        import torch

        result["available"] = torch.cuda.is_available()
        if result["available"]:
            result["name"] = torch.cuda.get_device_name(0)
    except ImportError:
        result["error"] = "torch not installed"
    except Exception as exc:
        result["error"] = str(exc)
    return result


@app.on_event("startup")
async def startup():
    cfg = get_config()
    Path(cfg.storage.sessions_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Sessions dir: %s", cfg.storage.sessions_dir)
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
    token: str = Depends(require_auth_upload),
):
    cfg = get_config()
    session = create_session({"mode": "file", "source_type": "device_import"})
    session_id = session["session_id"]
    sd = session_dir(session_id)

    content = await file.read()
    if len(content) > cfg.worker.max_chunk_mb * 1024 * 1024:
        return JSONResponse(status_code=413, content={"detail": "Uploaded file exceeds max chunk size."})

    chunk_path = sd / "chunks" / "chunk_0000.wav"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(content)
    register_chunk(
        session_id,
        0,
        {
            "chunk_index": 0,
            "chunk_started_ms": 0,
            "chunk_duration_ms": 0,
            "is_final": True,
            "file_size": len(content),
        },
    )
    update_session_meta(
        session_id,
        {
            "state": "finalized",
            "language": language,
            "model_size": model_size,
            "run_diarization": diarization,
            "finalize_requested_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    update_status(session_id, "uploaded")
    _enqueue_job(
        session_id,
        get_session_meta(session_id) or {},
        {
            "language": language,
            "model_size": model_size,
            "run_diarization": diarization,
        },
    )
    return {
        "accepted": True,
        "session_id": session_id,
        "job_id": session_id,
        "status_url": f"/api/v2/jobs/{session_id}",
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    from app.ui.dashboard import get_ui_html

    return HTMLResponse(content=get_ui_html())


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
