"""
Shared test fixtures.
"""
import os
import sys
import struct
import wave
import tempfile
import shutil
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Override config before any imports
os.environ.setdefault("SESSIONS_DIR", str(Path(tempfile.mkdtemp()) / "sessions"))
os.environ.setdefault("AUTH_TOKEN", "test-token")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("MODELS_DIR", str(PROJECT_ROOT / "models"))


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config singleton between tests."""
    from app.core.config import reset_config
    reset_config()
    yield
    reset_config()


@pytest.fixture
def tmp_sessions_dir(tmp_path):
    """Provide a temporary sessions directory."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    os.environ["SESSIONS_DIR"] = str(sessions)
    from app.core.config import reset_config
    reset_config()
    yield sessions
    os.environ.pop("SESSIONS_DIR", None)
    reset_config()


@pytest.fixture
def make_wav_bytes():
    """Factory for generating WAV bytes in memory."""
    def _make(duration_s: float = 1.0, sample_rate: int = 16000,
              channels: int = 1, frequency: float = 440.0) -> bytes:
        import math
        n_samples = int(sample_rate * duration_s)
        samples = []
        for i in range(n_samples):
            t = i / sample_rate
            value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
            samples.append(value)

        raw_data = struct.pack(f'<{len(samples)}h', *samples)

        import io
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_data)
        return buf.getvalue()
    return _make


@pytest.fixture
def make_wav_file(make_wav_bytes, tmp_path):
    """Factory for generating WAV files on disk."""
    def _make(name: str = "test.wav", duration_s: float = 1.0, **kwargs) -> str:
        data = make_wav_bytes(duration_s=duration_s, **kwargs)
        path = tmp_path / name
        path.write_bytes(data)
        return str(path)
    return _make
