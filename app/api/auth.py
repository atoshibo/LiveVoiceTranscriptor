"""
Authentication and rate limiting.

Preserves the old server's auth contract:
  - Authorization: Bearer <token>
  - X-Api-Token: <token>
  - Never accept token in URL query string
"""
import hashlib
import time
import logging
from typing import Optional, Dict
from fastapi import Request, HTTPException

from app.core.config import get_config

logger = logging.getLogger(__name__)

# Rate limit buckets: {bucket_key: [timestamp, ...]}
_general_buckets: Dict[str, list] = {}
_upload_buckets: Dict[str, list] = {}


def _extract_token(request: Request) -> Optional[str]:
    """Extract auth token from headers. Never from URL."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer ") and len(auth) > 7:
        return auth[7:]
    api_token = request.headers.get("x-api-token", "")
    if api_token:
        return api_token
    return None


def _validate_token(request: Request) -> str:
    """Validate token and return it. Raises on failure."""
    token = _extract_token(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Use 'Authorization: Bearer <token>' or 'X-Api-Token: <token>' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    cfg = get_config()
    if cfg.auth.token and token != cfg.auth.token:
        raise HTTPException(status_code=403, detail="Invalid token.")
    return token


def _check_rate_limit(buckets: dict, token: str, limit: int, window: int = 60) -> None:
    """Sliding-window rate limit check."""
    bucket_key = hashlib.sha256(token.encode()).hexdigest()[:16]
    now = time.time()
    cutoff = now - window

    if bucket_key not in buckets:
        buckets[bucket_key] = []

    # Clean old entries
    buckets[bucket_key] = [t for t in buckets[bucket_key] if t > cutoff]

    if len(buckets[bucket_key]) >= limit:
        return True  # Rate limited
    buckets[bucket_key].append(now)
    return False


def require_auth(request: Request) -> str:
    """FastAPI dependency for general authenticated endpoints."""
    token = _validate_token(request)
    cfg = get_config()
    if _check_rate_limit(_general_buckets, token, cfg.rate_limit.general_per_minute):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({cfg.rate_limit.general_per_minute} requests/minute).",
            headers={"Retry-After": "10"},
        )
    return token


def require_auth_upload(request: Request) -> str:
    """FastAPI dependency for upload endpoints (higher limit)."""
    token = _validate_token(request)
    cfg = get_config()
    if _check_rate_limit(_upload_buckets, token, cfg.rate_limit.upload_per_minute):
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"Upload rate limit exceeded ({cfg.rate_limit.upload_per_minute} requests/minute).",
                "reason": "upload_rate_limited",
                "limit_per_minute": cfg.rate_limit.upload_per_minute,
                "retry_after_seconds": 5,
                "endpoint_category": "upload",
            },
            headers={"Retry-After": "5"},
        )
    return token
