"""
Atomic file I/O utilities.

All JSON writes use write-to-temp + fsync + os.replace to prevent
readers from seeing partial/corrupt files.
"""
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional


def atomic_write_json(path: str, data: Any) -> None:
    """Write JSON atomically via temp file + fsync + replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        suffix=".tmp",
        prefix=".atomic_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(target))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def safe_read_json(path: str, retries: int = 10, sleep_s: float = 0.05) -> Optional[dict]:
    """Read JSON with retry for partially-written files."""
    for attempt in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    if attempt < retries - 1:
                        time.sleep(sleep_s)
                        continue
                    return None
                return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            if attempt < retries - 1:
                time.sleep(sleep_s)
                continue
            return None
    return None


def atomic_write_text(path: str, text: str) -> None:
    """Write text file atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        suffix=".tmp",
        prefix=".atomic_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(target))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
