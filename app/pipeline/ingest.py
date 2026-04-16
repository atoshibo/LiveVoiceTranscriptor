"""
Stage 2 - Ingest and Normalization

Receives transport chunks, decodes original media, normalizes sample rate/channels,
and places each chunk on an absolute session timeline.

The normalized timeline is the only timeline the rest of the pipeline should trust.

Output: normalized audio timeline + ingest metadata in normalized/ and raw/
"""
import os
import subprocess
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


def _coerce_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _ms_to_frames(ms: int, sample_rate: int) -> int:
    return max(0, int(round((max(0, ms) / 1000.0) * sample_rate)))


def _frames_to_ms(frames: int, sample_rate: int) -> int:
    return int(round((max(0, frames) / max(1, sample_rate)) * 1000))


def _chunk_path_map(session_id: str) -> Dict[int, str]:
    chunk_dir = session_dir(session_id) / "chunks"
    path_map: Dict[int, str] = {}
    if not chunk_dir.is_dir():
        return path_map

    for path in sorted(chunk_dir.glob("chunk_*.wav")):
        try:
            chunk_index = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        path_map[chunk_index] = str(path)
    return path_map


def _wav_duration_ms(path: str) -> int:
    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.getnframes()
            return int(round((frames / max(1, sr)) * 1000))
    except Exception:
        return 0


def split_file_upload_to_transport_chunks(
    input_path: str,
    output_dir: str,
    transport_chunk_ms: int,
) -> dict:
    """Split a whole uploaded media file into canonical transport chunks."""
    input_file = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for stale in output_path.glob("chunk_*.wav"):
        try:
            stale.unlink()
        except OSError:
            logger.warning("Failed to remove stale transport chunk %s", stale)

    if input_file.suffix.lower() == ".wav":
        split_result = _split_wav_transport_chunks(input_file, output_path, transport_chunk_ms)
        if split_result.get("success"):
            return split_result
        logger.warning("WAV split fallback failed for %s: %s", input_file, split_result.get("error"))

    return _split_media_transport_chunks_ffmpeg(input_file, output_path, transport_chunk_ms)


def _split_wav_transport_chunks(input_path: Path, output_dir: Path, transport_chunk_ms: int) -> dict:
    try:
        with wave.open(str(input_path), "rb") as wf:
            params = wf.getparams()
            frames_per_chunk = max(1, _ms_to_frames(transport_chunk_ms, params.framerate))
            chunk_specs = []
            chunk_index = 0

            while True:
                frames = wf.readframes(frames_per_chunk)
                if not frames:
                    break

                chunk_path = output_dir / f"chunk_{chunk_index:04d}.wav"
                with wave.open(str(chunk_path), "wb") as out:
                    out.setnchannels(params.nchannels)
                    out.setsampwidth(params.sampwidth)
                    out.setframerate(params.framerate)
                    out.writeframes(frames)

                duration_ms = _wav_duration_ms(str(chunk_path))
                chunk_specs.append(
                    {
                        "chunk_index": chunk_index,
                        "path": str(chunk_path),
                        "chunk_started_ms": chunk_index * transport_chunk_ms,
                        "chunk_duration_ms": duration_ms,
                        "file_size": chunk_path.stat().st_size,
                    }
                )
                chunk_index += 1

        if not chunk_specs:
            return {"success": False, "error": "no_chunks_emitted_from_wav"}

        chunk_specs[-1]["is_final"] = True
        for chunk in chunk_specs[:-1]:
            chunk["is_final"] = False

        return {
            "success": True,
            "method": "wave_split",
            "chunk_count": len(chunk_specs),
            "chunks": chunk_specs,
        }
    except Exception as e:
        return {"success": False, "error": f"wav_split_failed:{e}"}


def _split_media_transport_chunks_ffmpeg(input_path: Path, output_dir: Path, transport_chunk_ms: int) -> dict:
    segment_seconds = max(1, int(round(transport_chunk_ms / 1000.0)))
    output_pattern = str(output_dir / "chunk_%04d.wav")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        "-c:a",
        "pcm_s16le",
        output_pattern,
    ]
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return {"success": False, "error": "ffmpeg_not_available"}
    except Exception as e:
        return {"success": False, "error": f"ffmpeg_execution_failed:{e}"}

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return {"success": False, "error": f"ffmpeg_split_failed:{detail}"}

    chunk_specs = []
    for chunk_index, chunk_path in enumerate(sorted(output_dir.glob("chunk_*.wav"))):
        duration_ms = _wav_duration_ms(str(chunk_path))
        chunk_specs.append(
            {
                "chunk_index": chunk_index,
                "path": str(chunk_path),
                "chunk_started_ms": chunk_index * transport_chunk_ms,
                "chunk_duration_ms": duration_ms,
                "file_size": chunk_path.stat().st_size,
                "is_final": False,
            }
        )

    if not chunk_specs:
        return {"success": False, "error": "ffmpeg_produced_no_chunks"}

    chunk_specs[-1]["is_final"] = True
    return {
        "success": True,
        "method": "ffmpeg_segment",
        "chunk_count": len(chunk_specs),
        "chunks": chunk_specs,
    }


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
    chunk_paths = _chunk_path_map(session_id)
    default_chunk_duration_ms = _coerce_int(meta.get("chunk_duration_sec"), 30) * 1000 or 30000

    for i, chunk_meta in enumerate(chunks):
        chunk_index = _coerce_int(chunk_meta.get("chunk_index"), i)
        chunk_path = chunk_paths.get(chunk_index)
        actual_duration_ms = _wav_duration_ms(chunk_path) if chunk_path else 0
        declared_duration_ms = _coerce_int(chunk_meta.get("chunk_duration_ms"), 0)
        chunk_duration = max(declared_duration_ms, actual_duration_ms)
        if chunk_duration <= 0:
            chunk_duration = default_chunk_duration_ms

        chunk_start = chunk_meta.get("chunk_started_ms")
        if chunk_start is None:
            chunk_start = expected_start_ms + _coerce_int(chunk_meta.get("gap_before_ms"), 0)
        chunk_start = _coerce_int(chunk_start, expected_start_ms)
        chunk_end = chunk_start + chunk_duration

        gap_ms = chunk_start - expected_start_ms
        entry = {
            "chunk_index": chunk_index,
            "chunk_path": chunk_path,
            "start_ms": chunk_start,
            "end_ms": chunk_end,
            "duration_ms": chunk_duration,
            "actual_duration_ms": actual_duration_ms,
            "gap_before_ms": max(0, gap_ms),
            "overlap_before_ms": max(0, -gap_ms),
            "continuity": {
                "dropped_frames": chunk_meta.get("dropped_frames", 0),
                "decode_failure": chunk_meta.get("decode_failure", False),
                "gap_before_ms": _coerce_int(chunk_meta.get("gap_before_ms"), max(0, gap_ms)),
                "source_degraded": chunk_meta.get("source_degraded", False),
            }
        }
        timeline_entries.append(entry)
        expected_start_ms = max(expected_start_ms, chunk_end)

    total_duration_ms = max((entry["end_ms"] for entry in timeline_entries), default=0)
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


def render_session_timeline_audio(
    session_id: str,
    output_path: str,
    timeline: Optional[dict] = None,
    target_sr: int = TARGET_SAMPLE_RATE,
    target_channels: int = TARGET_CHANNELS,
) -> dict:
    """Render the normalized absolute timeline audio used by downstream stages."""
    timeline = timeline or build_session_timeline(session_id)
    if not timeline.get("success"):
        return {"success": False, "error": timeline.get("error", "timeline_unavailable")}

    entries = timeline.get("timeline", [])
    total_duration_ms = _coerce_int(timeline.get("total_duration_ms"), 0)
    total_frames = _ms_to_frames(total_duration_ms, target_sr)
    frame_width = 2 * target_channels  # PCM_16 mono by contract
    canvas = bytearray(total_frames * frame_width)
    placements = []
    written_chunks = 0
    covered_until_frame = 0

    for entry in entries:
        chunk_path = entry.get("chunk_path")
        placement = {
            "chunk_index": entry.get("chunk_index"),
            "source_path": os.path.basename(chunk_path) if chunk_path else None,
            "start_ms": entry.get("start_ms", 0),
            "end_ms": entry.get("end_ms", 0),
            "duration_ms": entry.get("duration_ms", 0),
            "skipped": False,
        }

        if not chunk_path or not os.path.isfile(chunk_path):
            placement["skipped"] = True
            placement["reason"] = "chunk_missing_on_disk"
            placements.append(placement)
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            normalized_chunk = tmp.name

        try:
            normalized_meta = normalize_audio_file(
                chunk_path,
                normalized_chunk,
                target_sr=target_sr,
                target_channels=target_channels,
            )
            if not normalized_meta.get("success"):
                placement["skipped"] = True
                placement["reason"] = normalized_meta.get("error", "normalization_failed")
                placements.append(placement)
                continue

            with wave.open(normalized_chunk, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
                chunk_frames = wf.getnframes()

            if sr != target_sr or ch != target_channels or sw != 2:
                placement["skipped"] = True
                placement["reason"] = f"unexpected_normalized_format:{sr}/{ch}/{sw}"
                placements.append(placement)
                continue

            start_frame = _ms_to_frames(_coerce_int(entry.get("start_ms"), 0), target_sr)
            end_frame = min(start_frame + chunk_frames, total_frames)
            frames_to_write = end_frame - start_frame
            if frames_to_write <= 0:
                placement["skipped"] = True
                placement["reason"] = "outside_timeline"
                placements.append(placement)
                continue

            byte_start = start_frame * frame_width
            byte_end = byte_start + (frames_to_write * frame_width)
            canvas[byte_start:byte_end] = frames[: frames_to_write * frame_width]

            overlap_frames = max(0, covered_until_frame - start_frame)
            covered_until_frame = max(covered_until_frame, end_frame)
            written_chunks += 1

            placement.update({
                "placed_duration_ms": _frames_to_ms(frames_to_write, target_sr),
                "placed_end_ms": _frames_to_ms(end_frame, target_sr),
                "normalized_duration_ms": normalized_meta.get("normalized_duration_ms"),
                "overwrote_previous_ms": _frames_to_ms(overlap_frames, target_sr),
            })
            placements.append(placement)
        finally:
            try:
                os.unlink(normalized_chunk)
            except OSError:
                pass

    if written_chunks == 0:
        return {"success": False, "error": "No timeline chunks could be rendered"}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(output_path, "wb") as out:
        out.setnchannels(target_channels)
        out.setsampwidth(2)
        out.setframerate(target_sr)
        out.writeframes(bytes(canvas))

    return {
        "success": True,
        "output_path": output_path,
        "sample_rate": target_sr,
        "channels": target_channels,
        "sample_width": 2,
        "total_frames": total_frames,
        "normalized_duration_ms": total_duration_ms,
        "timeline_duration_ms": total_duration_ms,
        "rendered_chunk_count": written_chunks,
        "chunk_count": len(entries),
        "placements": placements,
        "gap_count": timeline.get("gaps", 0),
        "overlap_count": timeline.get("overlaps", 0),
        "overlap_strategy": "later_chunk_overwrites_earlier_audio_at_same_absolute_time",
        "integrity": timeline.get("integrity", {}),
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

    # Step 1: Merge chunks for audit/backward compatibility
    merge_result = merge_chunks(chunk_paths, raw_audio)
    atomic_write_json(str(sd / "raw" / "merge_meta.json"), merge_result)

    # Step 2: Build absolute timeline metadata
    timeline = build_session_timeline(session_id)
    atomic_write_json(str(sd / "normalized" / "timeline.json"), timeline)
    if not timeline.get("success"):
        raise RuntimeError(f"Timeline build failed: {timeline.get('error')}")

    # Step 3: Render normalized audio on the absolute session timeline
    norm_result = render_session_timeline_audio(session_id, normalized_audio, timeline=timeline)
    atomic_write_json(str(sd / "normalized" / "norm_meta.json"), norm_result)

    if not norm_result.get("success"):
        raise RuntimeError(f"Timeline render failed: {norm_result.get('error')}")

    # Prefer the absolute timeline audio as the compatibility root audio.
    import shutil
    compat_audio = str(sd / "audio.wav")
    if os.path.isfile(normalized_audio):
        shutil.copy2(normalized_audio, compat_audio)

    # Stage artifacts
    artifacts = ["audio.wav", "norm_meta.json", "timeline.json"]
    stage.commit(artifacts)

    return {
        "merge": merge_result,
        "normalization": norm_result,
        "timeline": timeline,
        "audio_duration_ms": timeline.get("total_duration_ms", norm_result.get("normalized_duration_ms", 0)),
    }
