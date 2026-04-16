"""
Central configuration for LiveVoiceTranscriptor.

Environment variables are loaded from a project-root `.env` file first,
then overridden by the real process environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised in minimal environments
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODELS_DIR = str(PROJECT_ROOT / "models")


def _load_project_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file() and load_dotenv is not None:
        load_dotenv(env_path, override=False)
    elif env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_project_env()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int = 0) -> int:
    raw = os.environ.get(key, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    raw = os.environ.get(key, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, str(default)).strip().lower()
    return val in {"true", "1", "yes", "on"}


def _env_list(key: str) -> list[str]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_path(value: str, default: str) -> str:
    raw = value.strip() if value else default
    if not raw:
        return raw
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


@dataclass(frozen=True)
class PipelineGeometry:
    transport_chunk_ms: int = 30000
    decode_window_ms: int = 30000
    decode_stride_ms: int = 15000
    commit_stripe_ms: int = 15000

    def __post_init__(self) -> None:
        object.__setattr__(self, "transport_chunk_ms", _env_int("TRANSPORT_CHUNK_MS", 30000))
        object.__setattr__(self, "decode_window_ms", _env_int("DECODE_WINDOW_MS", 30000))
        object.__setattr__(self, "decode_stride_ms", _env_int("DECODE_STRIDE_MS", 15000))
        object.__setattr__(self, "commit_stripe_ms", _env_int("COMMIT_STRIPE_MS", 15000))


@dataclass(frozen=True)
class StorageConfig:
    sessions_dir: str = ""
    auto_cleanup_draft_sessions: bool = True
    draft_session_max_age_minutes: int = 120

    def __post_init__(self) -> None:
        value = _resolve_path(_env("SESSIONS_DIR", ""), str(PROJECT_ROOT / "data" / "sessions"))
        object.__setattr__(self, "sessions_dir", value)
        object.__setattr__(
            self,
            "auto_cleanup_draft_sessions",
            _env_bool("AUTO_CLEANUP_DRAFT_SESSIONS", True),
        )
        object.__setattr__(
            self,
            "draft_session_max_age_minutes",
            _env_int("DRAFT_SESSION_MAX_AGE_MINUTES", 120),
        )


@dataclass(frozen=True)
class RedisConfig:
    host: str = ""
    port: int = 6379
    queue: str = "transcription_jobs"
    partial_queue: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", _env("REDIS_HOST", "localhost"))
        object.__setattr__(self, "port", _env_int("REDIS_PORT", 6379))
        object.__setattr__(self, "queue", _env("REDIS_QUEUE", "transcription_jobs"))
        object.__setattr__(self, "partial_queue", _env("REDIS_PARTIAL_QUEUE", f"{self.queue}_partial"))


@dataclass(frozen=True)
class AuthConfig:
    token: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "token", _env("AUTH_TOKEN", ""))


@dataclass(frozen=True)
class RateLimitConfig:
    general_per_minute: int = 60
    upload_per_minute: int = 300

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "general_per_minute",
            _env_int("RATE_LIMIT_GENERAL_PER_MINUTE", _env_int("RATE_LIMIT_PER_MINUTE", 60)),
        )
        object.__setattr__(self, "upload_per_minute", _env_int("RATE_LIMIT_UPLOAD_PER_MINUTE", 300))


@dataclass(frozen=True)
class WorkerConfig:
    concurrency: int = 1
    partial_every_n_chunks: int = 5
    partial_cooldown_seconds: int = 120
    max_chunk_mb: int = 30
    max_file_upload_mb: int = 512
    max_session_chunks: int = 500
    strict_cuda: bool = False
    # GPU runtime configuration
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_compute_type_fallbacks: list = field(default_factory=list)
    cuda_device_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "concurrency", _env_int("WORKER_CONCURRENCY", 1))
        object.__setattr__(self, "partial_every_n_chunks", _env_int("PARTIAL_EVERY_N_CHUNKS", 5))
        object.__setattr__(self, "partial_cooldown_seconds", _env_int("PARTIAL_COOLDOWN_SECONDS", 120))
        object.__setattr__(self, "max_chunk_mb", _env_int("MAX_CHUNK_MB", 30))
        object.__setattr__(self, "max_file_upload_mb", _env_int("MAX_FILE_UPLOAD_MB", 512))
        object.__setattr__(self, "max_session_chunks", _env_int("MAX_SESSION_CHUNKS", 500))
        # STRICT_CUDA / WHISPER_STRICT_CUDA — canonical name is WHISPER_STRICT_CUDA
        strict = _env_bool("WHISPER_STRICT_CUDA", False) or _env_bool("STRICT_CUDA", False)
        object.__setattr__(self, "strict_cuda", strict)
        object.__setattr__(self, "whisper_device", _env("WHISPER_DEVICE", "cuda"))
        object.__setattr__(self, "whisper_compute_type", _env("WHISPER_COMPUTE_TYPE", "float16"))
        fallbacks_raw = _env("WHISPER_COMPUTE_TYPE_FALLBACKS", "int8_float16,int8")
        fallbacks = [f.strip() for f in fallbacks_raw.split(",") if f.strip()]
        object.__setattr__(self, "whisper_compute_type_fallbacks", fallbacks)
        object.__setattr__(self, "cuda_device_index", _env_int("CUDA_DEVICE_INDEX", 0))


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8443
    tls_enabled: bool = True
    tls_cert_file: str = ""
    tls_key_file: str = ""
    tls_auto_generate_self_signed: bool = True
    tls_cert_common_name: str = "localhost"

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", _env("SERVER_HOST", "0.0.0.0"))
        object.__setattr__(self, "port", _env_int("SERVER_PORT", 8443))
        object.__setattr__(self, "tls_enabled", _env_bool("TLS_ENABLED", True))
        object.__setattr__(
            self,
            "tls_cert_file",
            _resolve_path(_env("TLS_CERT_FILE", ""), str(PROJECT_ROOT / "data" / "certs" / "dev-cert.pem")),
        )
        object.__setattr__(
            self,
            "tls_key_file",
            _resolve_path(_env("TLS_KEY_FILE", ""), str(PROJECT_ROOT / "data" / "certs" / "dev-key.pem")),
        )
        object.__setattr__(
            self,
            "tls_auto_generate_self_signed",
            _env_bool("TLS_AUTO_GENERATE_SELF_SIGNED", True),
        )
        object.__setattr__(self, "tls_cert_common_name", _env("TLS_CERT_COMMON_NAME", "localhost"))


@dataclass(frozen=True)
class ModelPaths:
    models_dir: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "models_dir", _resolve_path(_env("MODELS_DIR", DEFAULT_MODELS_DIR), DEFAULT_MODELS_DIR))

    @property
    def models_dir_exists(self) -> bool:
        return Path(self.models_dir).is_dir()

    def resolve(self, name: str, env_override: str = "") -> Optional[str]:
        if env_override:
            override = _env(env_override, "")
            if override:
                override_path = Path(_resolve_path(override, override))
                if override_path.is_dir() or override_path.is_file():
                    return str(override_path)

        base = Path(self.models_dir)
        if not base.is_dir():
            return None

        candidate = base / name
        if candidate.is_dir() or candidate.is_file():
            return str(candidate)

        for entry in base.iterdir():
            if entry.is_dir() and name in entry.name:
                snapshots = entry / "snapshots"
                if snapshots.is_dir():
                    snapshot_dirs = sorted(
                        (p for p in snapshots.iterdir() if p.is_dir()),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if snapshot_dirs:
                        return str(snapshot_dirs[0])
                return str(entry)
        return None

    @property
    def parakeet_path(self) -> Optional[str]:
        return self.resolve("parakeet-tdt-0.6b-v3", "PARAKEET_MODEL_PATH")

    @property
    def canary_path(self) -> Optional[str]:
        return self.resolve("canary-1b-v2", "CANARY_MODEL_PATH")

    @property
    def whisper_turbo_path(self) -> Optional[str]:
        return self.resolve("whisper-large-v3-turbo-hf", "WHISPER_TURBO_MODEL_PATH")

    @property
    def diarization_path(self) -> Optional[str]:
        return self.resolve("pyannote-speaker-diarization-community-1", "DIARIZATION_MODEL_PATH")

    @property
    def llm_path(self) -> Optional[str]:
        override = _env("LLM_MODEL_PATH", "")
        if override:
            resolved_override = Path(_resolve_path(override, override))
            if resolved_override.is_file():
                return str(resolved_override)

        base = Path(self.models_dir)
        if not base.is_dir():
            return None
        for file_path in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if file_path.is_file() and file_path.suffix.lower() == ".gguf":
                return str(file_path)
        return None


@dataclass(frozen=True)
class DiarizationConfig:
    enabled_by_default: bool = False
    min_speakers: int = 1
    max_speakers: int = 10
    min_duration_on: float = 0.3
    min_duration_off: float = 0.3
    use_gpu: bool = True
    fallback_on_error: bool = True


@dataclass(frozen=True)
class ReconciliationConfig:
    min_confidence: float = 0.3
    max_tokens: int = 300
    temperature: float = 0.1
    use_llm: bool = True
    fallback_to_deterministic: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_confidence", _env_float("RECONCILIATION_MIN_CONFIDENCE", 0.3))
        object.__setattr__(self, "max_tokens", _env_int("RECONCILIATION_MAX_TOKENS", 300))
        object.__setattr__(self, "temperature", _env_float("RECONCILIATION_TEMPERATURE", 0.1))
        object.__setattr__(self, "use_llm", _env_bool("RECONCILIATION_USE_LLM", True))
        object.__setattr__(self, "fallback_to_deterministic", _env_bool("RECONCILIATION_FALLBACK_TO_DETERMINISTIC", True))


@dataclass(frozen=True)
class LanguageConfig:
    allowed_languages: list[str] = field(default_factory=list)
    forced_language: Optional[str] = None
    transcription_mode: str = "verbatim_multilingual"

    def __post_init__(self) -> None:
        forced = _env("FORCED_LANGUAGE", "").strip() or None
        object.__setattr__(self, "allowed_languages", _env_list("ALLOWED_LANGUAGES"))
        object.__setattr__(self, "forced_language", forced)
        object.__setattr__(self, "transcription_mode", _env("TRANSCRIPTION_MODE", "verbatim_multilingual"))


@dataclass
class AppConfig:
    geometry: PipelineGeometry = field(default_factory=PipelineGeometry)
    storage: StorageConfig = field(default_factory=StorageConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)


_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _load_project_env()
        _config = AppConfig()
    return _config


def reset_config() -> None:
    global _config
    _config = None
