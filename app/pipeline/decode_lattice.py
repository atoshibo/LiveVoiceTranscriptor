"""
Stage 4 - Decode Lattice Construction

Synthesizes overlapping 30-second decode windows from the normalized timeline
using 15-second stride, independent of original transport chunk boundaries.

For a session beginning at time zero:
  W0 = [0, 30]    full
  W1 = [15, 45]   bridge
  W2 = [30, 60]   full
  W3 = [45, 75]   bridge

Each interior 15-second stripe is observed twice: once in the trailing half
of one window and once in the leading half of the next.

Output: windows/ with decode_windows.json and extracted window audio files
"""
import os
import wave
import logging
from pathlib import Path
from typing import List, Dict, Optional

from app.core.config import get_config
from app.core.atomic_io import atomic_write_json
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)


def build_decode_windows(total_duration_ms: int,
                         speech_islands: List[Dict] = None,
                         window_ms: int = None,
                         stride_ms: int = None) -> List[Dict]:
    """Build the decode window lattice from the normalized timeline.

    Args:
        total_duration_ms: Total audio duration
        speech_islands: Speech islands from triage (optional)
        window_ms: Decode window size (default from config)
        stride_ms: Decode stride (default from config)

    Returns:
        List of decode window definitions
    """
    cfg = get_config()
    window_ms = window_ms or cfg.geometry.decode_window_ms
    stride_ms = stride_ms or cfg.geometry.decode_stride_ms

    windows = []
    window_idx = 0
    start_ms = 0

    while start_ms < total_duration_ms:
        end_ms = min(start_ms + window_ms, total_duration_ms)

        # Determine window type
        # First window at each chunk boundary is "full", bridge windows span boundaries
        is_bridge = (start_ms % window_ms) != 0 if window_ms > 0 else False
        window_type = "bridge" if is_bridge else "full"

        # Determine which transport chunks this window spans
        chunk_ms = cfg.geometry.transport_chunk_ms
        source_chunks = []
        if chunk_ms > 0:
            first_chunk = start_ms // chunk_ms
            last_chunk = max(first_chunk, (end_ms - 1) // chunk_ms)
            source_chunks = [f"C{i:04d}" for i in range(first_chunk, last_chunk + 1)]

        # Check speech intersection
        speech_ratio = 1.0  # Default: assume speech if no triage
        if speech_islands:
            speech_overlap_ms = 0
            for island in speech_islands:
                overlap_start = max(start_ms, island["start_ms"])
                overlap_end = min(end_ms, island["end_ms"])
                if overlap_end > overlap_start:
                    speech_overlap_ms += overlap_end - overlap_start
            window_duration = end_ms - start_ms
            speech_ratio = speech_overlap_ms / max(1, window_duration)

        # Schedule: eligible if enough speech
        scheduled = speech_ratio >= 0.05 or window_type == "bridge"

        window = {
            "window_id": f"W{window_idx:06d}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "window_type": window_type,
            "source_chunks": source_chunks,
            "speech_intersection_ratio": round(speech_ratio, 4),
            "scheduled": scheduled,
        }
        windows.append(window)
        window_idx += 1
        start_ms += stride_ms

    return windows


def extract_window_audio(audio_path: str, window: Dict,
                         output_dir: str) -> Optional[str]:
    """Extract audio for a single decode window from the normalized timeline."""
    try:
        with wave.open(audio_path, 'rb') as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            total_frames = wf.getnframes()

            start_frame = int(window["start_ms"] / 1000 * sr)
            end_frame = min(int(window["end_ms"] / 1000 * sr), total_frames)

            if start_frame >= total_frames:
                return None

            wf.setpos(start_frame)
            frames = wf.readframes(end_frame - start_frame)

        output_path = os.path.join(output_dir, f"{window['window_id']}.wav")
        with wave.open(output_path, 'wb') as out:
            out.setnchannels(channels)
            out.setsampwidth(sampwidth)
            out.setframerate(sr)
            out.writeframes(frames)

        return output_path
    except Exception as e:
        logger.error(f"Failed to extract window audio {window['window_id']}: {e}")
        return None


def run_decode_lattice(session_id: str, audio_path: str,
                       audio_duration_ms: int, speech_islands: List[Dict],
                       stage) -> dict:
    """Execute decode lattice construction stage.

    1. Build window definitions
    2. Extract window audio files
    3. Persist to windows/
    """
    sd = session_dir(session_id)
    windows_dir = str(sd / "windows")

    # Build windows
    windows = build_decode_windows(audio_duration_ms, speech_islands)

    # Extract audio for scheduled windows
    for window in windows:
        if window["scheduled"]:
            audio_file = extract_window_audio(audio_path, window, windows_dir)
            window["audio_path"] = os.path.basename(audio_file) if audio_file else None
        else:
            window["audio_path"] = None

    # Persist
    scheduled_count = sum(1 for w in windows if w["scheduled"])
    bridge_count = sum(1 for w in windows if w["window_type"] == "bridge")

    lattice_meta = {
        "session_id": session_id,
        "total_duration_ms": audio_duration_ms,
        "window_count": len(windows),
        "scheduled_count": scheduled_count,
        "bridge_count": bridge_count,
        "geometry": {
            "window_ms": get_config().geometry.decode_window_ms,
            "stride_ms": get_config().geometry.decode_stride_ms,
        },
        "windows": windows,
    }

    atomic_write_json(str(sd / "windows" / "decode_windows.json"), lattice_meta)

    stage.commit(["decode_windows.json"] + [
        w["audio_path"] for w in windows
        if w["audio_path"]
    ])

    return {
        "window_count": len(windows),
        "scheduled_count": scheduled_count,
        "bridge_count": bridge_count,
        "windows": windows,
    }
