"""
Stage 2 - Ingest and Normalization

Receives transport chunks, decodes original media, normalizes sample rate/channels,
and places each chunk on an absolute session timeline.

The normalized timeline is the only timeline the rest of the pipeline should trust.

Output: normalized audio timeline + ingest metadata in normalized/ and raw/
"""
import os
import wave
import struct
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from app.core.atomic_io import atomic_write_json
from app.storage.session_store import session_dir, get_session_meta

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1


def normalize_audio_file(input_path: str, output_path: str,
                         target_sr: int = TARGET_SAMPLE_RATE,
                         target_channels: int = TARGET_CHANNELS) -> dict:
    """Normalize an audio file to target sample rate and channels.

    Returns metadata about the normalization.
    """
    try:
        import soundfile as sf
        import numpy as np

        data, sr = sf.read(input_path, dtype='float32')

        # Convert to mono if needed
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)

        original_sr = sr
        original_samples = len(data)
        original_duration_ms = int((len(data) / sr) * 1000)

        # Resample if needed
        if sr != target_sr:
            try:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                # Simple linear interpolation fallback
                ratio = target_sr / sr
                new_length = int(len(data) * ratio)
                indices = np.linspace(0, len(data) - 1, new_length)
                data = np.interp(indices, np.arange(len(data)), data)
            sr = target_sr

        # Write normalized WAV
        sf.write(output_path, data, sr, subtype='PCM_16')

        return {
            "original_sample_rate": original_sr,
            "original_channels": 1 if len(data.shape) == 1 else data.shape[1],
            "original_samples": original_samples,
            "original_duration_ms": original_duration_ms,
            "normalized_sample_rate": target_sr,
            "normalized_channels": target_channels,
            "normalized_samples": len(data),
            "normalized_duration_ms": int((len(data) / target_sr) * 1000),
            "success": True,
        }
    except Exception as e:
        logger.error(f"Normalization failed for {input_path}: {e}")
        # Fallback: try wave module for basic WAV
        return _normalize_wav_fallback(input_path, output_path, target_sr)


def _normalize_wav_fallback(input_path: str, output_path: str,
                            target_sr: int) -> dict:
    """Fallback normalization using only wave module."""
    try:
        with wave.open(input_path, 'rb') as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        # Convert to mono if stereo
        if channels == 2 and sampwidth == 2:
            samples = struct.unpack(f'<{n_frames * 2}h', raw_data)
            mono = [(samples[i] + samples[i + 1]) // 2 for i in range(0, len(samples), 2)]
            raw_data = struct.pack(f'<{len(mono)}h', *mono)
            channels = 1

        # Write output (no resampling in fallback)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sr)
            wf.writeframes(raw_data)

        duration_ms = int((n_frames / sr) * 1000)
        return {
            "original_sample_rate": sr,
            "original_channels": channels,
            "original_samples": n_frames,
            "original_duration_ms": duration_ms,
            "normalized_sample_rate": sr,
            "normalized_channels": 1,
            "normalized_samples": n_frames,
            "normalized_duration_ms": duration_ms,
            "success": True,
            "fallback": True,
        }
    except Exception as e:
        logger.error(f"Fallback normalization also failed: {e}")
        return {"success": False, "error": str(e)}


def merge_chunks(chunk_paths: List[str], output_path: str) -> dict:
    """Merge ordered WAV chunks into a single normalized timeline.

    Reads header from first valid chunk, concatenates frames.
    Skips incompatible chunks with warnings.
    """
    if not chunk_paths:
        return {"success": False, "error": "No chunks to merge"}

    valid_paths = [p for p in chunk_paths if os.path.isfile(p)]
    if not valid_paths:
        return {"success": False, "error": "No chunk files found on disk"}

    # Read reference header from first chunk
    try:
        with wave.open(valid_paths[0], 'rb') as ref:
            ref_sr = ref.getframerate()
            ref_ch = ref.getnchannels()
            ref_sw = ref.getsampwidth()
    except Exception as e:
        return {"success": False, "error": f"Cannot read reference chunk: {e}"}

    all_frames = bytearray()
    chunk_metas = []
    offset_ms = 0

    for i, path in enumerate(valid_paths):
        try:
            with wave.open(path, 'rb') as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                n = wf.getnframes()

                if sr != ref_sr or ch != ref_ch or sw != ref_sw:
                    logger.warning(f"Chunk {path} incompatible, skipping")
                    chunk_metas.append({
                        "index": i, "path": path, "skipped": True,
                        "reason": "incompatible_format"
                    })
                    continue

                frames = wf.readframes(n)
                duration_ms = int((n / sr) * 1000)

                chunk_metas.append({
                    "index": i,
                    "path": os.path.basename(path),
                    "offset_ms": offset_ms,
                    "duration_ms": duration_ms,
                    "frames": n,
                    "skipped": False,
                })

                all_frames.extend(frames)
                offset_ms += duration_ms
        except Exception as e:
            logger.warning(f"Error reading chunk {path}: {e}")
            chunk_metas.append({
                "index": i, "path": path, "skipped": True,
                "reason": str(e)
            })

    if not all_frames:
        return {"success": False, "error": "All chunks were empty or invalid"}

    # Write merged output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(output_path, 'wb') as out:
        out.setnchannels(ref_ch)
        out.setsampwidth(ref_sw)
        out.setframerate(ref_sr)
        out.writeframes(bytes(all_frames))

    total_frames = len(all_frames) // (ref_ch * ref_sw)
    return {
        "success": True,
        "output_path": output_path,
        "sample_rate": ref_sr,
        "channels": ref_ch,
        "sample_width": ref_sw,
        "total_frames": total_frames,
        "total_duration_ms": int((total_frames / ref_sr) * 1000),
        "chunk_count": len(valid_paths),
        "chunks": chunk_metas,
    }


def build_session_timeline(session_id: str) -> dict:
    """Build the normalized session timeline from uploaded chunks.

    Places each chunk on an absolute timeline using chunk metadata.
    Detects gaps, overlaps, and continuity issues.
    """
    meta = get_session_meta(session_id)
    if not meta:
        return {"success": False, "error": "Session not found"}

    chunks = meta.get("chunks", [])
    sd = session_dir(session_id)

    timeline_entries = []
    expected_start_ms = 0

    for i, chunk_meta in enumerate(chunks):
        chunk_start = chunk_meta.get("chunk_started_ms", expected_start_ms)
        chunk_duration = chunk_meta.get("chunk_duration_ms", 30000)
        chunk_end = chunk_start + chunk_duration

        gap_ms = chunk_start - expected_start_ms
        entry = {
            "chunk_index": chunk_meta.get("chunk_index", i),
            "start_ms": chunk_start,
            "end_ms": chunk_end,
            "duration_ms": chunk_duration,
            "gap_before_ms": max(0, gap_ms),
            "overlap_before_ms": max(0, -gap_ms),
            "continuity": {
                "dropped_frames": chunk_meta.get("dropped_frames", 0),
                "decode_failure": chunk_meta.get("decode_failure", False),
                "gap_before_ms": chunk_meta.get("gap_before_ms", 0),
                "source_degraded": chunk_meta.get("source_degraded", False),
            }
        }
        timeline_entries.append(entry)
        expected_start_ms = chunk_end

    total_duration_ms = timeline_entries[-1]["end_ms"] if timeline_entries else 0
    gaps = [e for e in timeline_entries if e["gap_before_ms"] > 0]
    overlaps = [e for e in timeline_entries if e["overlap_before_ms"] > 0]

    return {
        "success": True,
        "session_id": session_id,
        "total_duration_ms": total_duration_ms,
        "chunk_count": len(timeline_entries),
        "timeline": timeline_entries,
        "gaps": len(gaps),
        "overlaps": len(overlaps),
        "integrity": {
            "has_gaps": len(gaps) > 0,
            "has_overlaps": len(overlaps) > 0,
            "total_gap_ms": sum(e["gap_before_ms"] for e in timeline_entries),
            "total_overlap_ms": sum(e["overlap_before_ms"] for e in timeline_entries),
        }
    }


def run_ingest_stage(session_id: str, stage) -> dict:
    """Execute the full ingest + normalization stage.

    1. Merge chunks -> raw/audio.wav
    2. Normalize -> normalized/audio.wav
    3. Build timeline -> normalized/timeline.json
    """
    sd = session_dir(session_id)
    chunk_paths = sorted((sd / "chunks").glob("chunk_*.wav"))
    chunk_paths = [str(p) for p in chunk_paths]

    raw_audio = str(sd / "raw" / "audio.wav")
    normalized_audio = str(sd / "normalized" / "audio.wav")

    # Step 1: Merge chunks
    merge_result = merge_chunks(chunk_paths, raw_audio)
    atomic_write_json(str(sd / "raw" / "merge_meta.json"), merge_result)

    if not merge_result.get("success"):
        raise RuntimeError(f"Chunk merge failed: {merge_result.get('error')}")

    # Also write to session root for backward compat
    import shutil
    compat_audio = str(sd / "audio.wav")
    if os.path.isfile(raw_audio):
        shutil.copy2(raw_audio, compat_audio)

    # Step 2: Normalize
    norm_result = normalize_audio_file(raw_audio, normalized_audio)
    atomic_write_json(str(sd / "normalized" / "norm_meta.json"), norm_result)

    if not norm_result.get("success"):
        raise RuntimeError(f"Normalization failed: {norm_result.get('error')}")

    # Step 3: Build timeline
    timeline = build_session_timeline(session_id)
    atomic_write_json(str(sd / "normalized" / "timeline.json"), timeline)

    # Stage artifacts
    artifacts = ["audio.wav", "norm_meta.json", "timeline.json"]
    stage.commit(artifacts)

    return {
        "merge": merge_result,
        "normalization": norm_result,
        "timeline": timeline,
        "audio_duration_ms": norm_result.get("normalized_duration_ms", 0),
    }
