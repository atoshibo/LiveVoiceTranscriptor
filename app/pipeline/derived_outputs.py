"""
Stage 10 - Derived Outputs

All derived outputs are produced from canonical truth, never from provisional text.

Generates:
  - transcript.txt (plain text)
  - transcript_by_speaker.json / .txt
  - subtitles.srt / subtitles.vtt
  - quality_report.json
  - raw_transcript.json
  - clean_transcript.json
  - classification.json (optional)

Output: derived/ and current/ (compatibility read surface)
"""
import os
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

from app.core.atomic_io import atomic_write_json, atomic_write_text, safe_read_json
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)


def generate_srt(segments: List[Dict]) -> str:
    """Generate SRT subtitle format from canonical segments."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _ms_to_srt_time(seg["start_ms"])
        end = _ms_to_srt_time(seg["end_ms"])
        speaker = seg.get("speaker", "")
        text = seg.get("text", "")
        label = f"[{speaker}] {text}" if speaker else text
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(label)
        lines.append("")
    return "\n".join(lines)


def generate_vtt(segments: List[Dict]) -> str:
    """Generate WebVTT subtitle format from canonical segments."""
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        start = _ms_to_vtt_time(seg["start_ms"])
        end = _ms_to_vtt_time(seg["end_ms"])
        speaker = seg.get("speaker", "")
        text = seg.get("text", "")
        label = f"[{speaker}] {text}" if speaker else text
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(label)
        lines.append("")
    return "\n".join(lines)


def _ms_to_srt_time(ms: int) -> str:
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    milli = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def _ms_to_vtt_time(ms: int) -> str:
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    milli = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"


def generate_speaker_transcript(segments: List[Dict]) -> Dict:
    """Generate speaker-annotated transcript."""
    speaker_segments = []
    text_lines = []

    for seg in segments:
        speaker = seg.get("speaker")
        start_s = round(seg["start_ms"] / 1000, 1)
        end_s = round(seg["end_ms"] / 1000, 1)
        text = seg.get("text", "")

        speaker_segments.append({
            "speaker": speaker,
            "start_s": start_s,
            "end_s": end_s,
            "text": text,
        })

        if speaker:
            text_lines.append(f"[{speaker}] {start_s}s\u2013{end_s}s")
        else:
            text_lines.append(f"{start_s}s\u2013{end_s}s")
        text_lines.append(text)
        text_lines.append("")

    return {
        "json": {"segments": speaker_segments},
        "text": "\n".join(text_lines),
    }


def generate_quality_report(segments: List[Dict], text: str,
                            audio_duration_ms: int) -> Dict:
    """Generate quality analysis report from canonical segments."""
    if not segments:
        return {
            "analysis_version": "1.1",
            "segment_count": 0,
            "total_text_length": 0,
            "issues": [],
            "reading_text": "",
            "source_integrity": None,
        }

    issues = []
    for i, seg in enumerate(segments):
        # Check for very low confidence
        conf = seg.get("confidence", 0)
        if conf < 0.3:
            issues.append({
                "type": "low_confidence",
                "confidence": "low",
                "segment_index": i,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg.get("text", "")[:100],
                "reason": f"Segment confidence {conf:.2f} below threshold",
            })

        # Check for very short segments
        duration = seg["end_ms"] - seg["start_ms"]
        if duration < 500 and seg.get("text", ""):
            issues.append({
                "type": "very_short_segment",
                "confidence": "medium",
                "segment_index": i,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg.get("text", "")[:100],
                "reason": f"Segment duration {duration}ms very short",
            })

    # Coverage
    transcript_end = max((s["end_ms"] for s in segments), default=0)
    coverage_ratio = transcript_end / max(1, audio_duration_ms)

    return {
        "analysis_version": "1.1",
        "segment_count": len(segments),
        "total_text_length": len(text),
        "audio_duration_s": round(audio_duration_ms / 1000, 1),
        "transcript_last_end_s": round(transcript_end / 1000, 1),
        "transcript_coverage_ratio": round(coverage_ratio, 4),
        "issues": issues,
        "issue_count": len(issues),
        "reading_text": text,
        "source_integrity": None,
    }


def generate_raw_transcript(segments: List[Dict], text: str) -> Dict:
    """Generate raw transcript with corruption flags."""
    return {
        "raw_text": text,
        "segments": [
            {
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "speaker": seg.get("speaker"),
                "text": seg.get("text", ""),
                "corruption_flags": seg.get("corruption_flags", []),
            }
            for seg in segments
        ],
    }


def generate_clean_transcript(segments: List[Dict], text: str) -> Dict:
    """Generate clean transcript with paragraphs."""
    # Build paragraphs by grouping adjacent segments
    paragraphs = []
    current_para = {"text": "", "segment_indices": []}

    for i, seg in enumerate(segments):
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue

        if current_para["text"]:
            current_para["text"] += " " + seg_text
        else:
            current_para["text"] = seg_text
        current_para["segment_indices"].append(i)

        # Start new paragraph on speaker change or large time gap
        next_seg = segments[i + 1] if i + 1 < len(segments) else None
        if next_seg:
            gap = next_seg["start_ms"] - seg["end_ms"]
            speaker_change = next_seg.get("speaker") != seg.get("speaker")
            if gap > 5000 or speaker_change:
                if current_para["text"]:
                    paragraphs.append(current_para)
                current_para = {"text": "", "segment_indices": []}

    if current_para["text"]:
        paragraphs.append(current_para)

    return {
        "clean_text": text,
        "paragraphs": paragraphs,
    }


def _normalized_retrieval_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _extract_keywords(text: str, limit: int = 8) -> List[str]:
    keywords = []
    for token in _normalized_retrieval_text(text).split():
        token = "".join(ch for ch in token if ch.isalnum())
        if len(token) < 3:
            continue
        if token in keywords:
            continue
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


def generate_retrieval_index(session_id: str, segments: List[Dict]) -> Dict:
    """Ground retrieval entries directly on canonical segments."""
    entries = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue
        entries.append({
            "entry_id": f"ret_{i:06d}",
            "segment_id": seg.get("segment_id"),
            "text": text,
            "normalized_text": _normalized_retrieval_text(text),
            "keywords": _extract_keywords(text),
            "language": seg.get("language"),
            "speaker": seg.get("speaker"),
            "grounding": {
                "segment_id": seg.get("segment_id"),
                "start_ms": seg.get("start_ms"),
                "end_ms": seg.get("end_ms"),
                "canonical_path": "canonical/canonical_segments.json",
            },
            "support_windows": seg.get("support_windows", []),
            "support_models": seg.get("support_models", []),
            "stabilization_state": seg.get("stabilization_state", "stabilized"),
        })

    return {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "canonical_segments",
        "entry_count": len(entries),
        "entries": entries,
    }


def _write_output_bundle(output_dir: Path, session_id: str, segments: List[Dict],
                         audio_duration_ms: int, include_retrieval: bool = False) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    final_segments = [seg for seg in segments if seg.get("stabilization_state") == "stabilized"]
    final_text = " ".join(seg.get("text", "") for seg in final_segments if seg.get("text"))

    atomic_write_text(str(output_dir / "transcript.txt"), final_text)

    speaker = generate_speaker_transcript(final_segments)
    atomic_write_json(str(output_dir / "transcript_by_speaker.json"), speaker["json"])
    atomic_write_text(str(output_dir / "transcript_by_speaker.txt"), speaker["text"])

    atomic_write_text(str(output_dir / "subtitles.srt"), generate_srt(final_segments))
    atomic_write_text(str(output_dir / "subtitles.vtt"), generate_vtt(final_segments))

    atomic_write_json(str(output_dir / "transcript_timestamps.json"), {
        "segments": [
            {
                "start": seg["start_ms"] / 1000,
                "end": seg["end_ms"] / 1000,
                "text": seg.get("text", ""),
                "speaker": seg.get("speaker"),
            }
            for seg in final_segments
        ],
        "words": [
            {
                "t_ms": seg["start_ms"],
                "speaker": seg.get("speaker"),
                "w": seg.get("text", ""),
            }
            for seg in final_segments
        ],
    })

    quality = generate_quality_report(final_segments, final_text, audio_duration_ms)
    atomic_write_json(str(output_dir / "quality_report.json"), quality)

    raw = generate_raw_transcript(final_segments, final_text)
    atomic_write_json(str(output_dir / "raw_transcript.json"), raw)

    clean = generate_clean_transcript(final_segments, final_text)
    atomic_write_json(str(output_dir / "clean_transcript.json"), clean)

    files_written = [
        "transcript.txt",
        "transcript_by_speaker.json",
        "transcript_by_speaker.txt",
        "subtitles.srt",
        "subtitles.vtt",
        "transcript_timestamps.json",
        "quality_report.json",
        "raw_transcript.json",
        "clean_transcript.json",
    ]

    if include_retrieval:
        retrieval = generate_retrieval_index(session_id, final_segments)
        atomic_write_json(str(output_dir / "retrieval_index.json"), retrieval)
        files_written.append("retrieval_index.json")

    return {
        "files_written": files_written,
        "final_segments": final_segments,
        "final_text": final_text,
    }


def build_derived_dir(session_id: str, segments: List[Dict],
                      text: str, audio_duration_ms: int) -> Dict:
    """Build the spec-facing derived/ layer, including retrieval grounding."""
    sd = session_dir(session_id)
    derived = sd / "derived"
    result = _write_output_bundle(derived, session_id, segments, audio_duration_ms, include_retrieval=True)

    canonical = sd / "canonical"
    for fname in [
        "provenance.json",
        "provisional_partial.json",
        "stabilized_partial.json",
        "final_transcript.json",
    ]:
        src = canonical / fname
        if src.is_file():
            shutil.copy2(str(src), str(derived / fname))
            result["files_written"].append(fname)

    return result


def build_current_dir(session_id: str, segments: List[Dict],
                      text: str, audio_duration_ms: int) -> Dict:
    """Build the current/ compatibility read surface.

    This is what the API reads from.
    """
    sd = session_dir(session_id)
    current = sd / "current"
    result = _write_output_bundle(current, session_id, segments, audio_duration_ms)

    # 8. Canonical layer surfaces + provenance
    for fname in [
        "provenance.json",
        "provisional_partial.json",
        "stabilized_partial.json",
        "final_transcript.json",
    ]:
        src = sd / "canonical" / fname
        if src.is_file():
            shutil.copy2(str(src), str(current / fname))
            result["files_written"].append(fname)

    # 9. Backward-compat copies to session root
    for fname in ["transcript.txt", "subtitles.srt", "subtitles.vtt",
                   "transcript_by_speaker.json", "transcript_by_speaker.txt",
                   "transcript_timestamps.json", "quality_report.json",
                   "raw_transcript.json", "clean_transcript.json",
                   "provenance.json", "final_transcript.json",
                   "provisional_partial.json", "stabilized_partial.json"]:
        src = current / fname
        if src.is_file():
            shutil.copy2(str(src), str(sd / fname))
    return result


def run_derived_outputs(session_id: str, segments: List[Dict],
                        text: str, audio_duration_ms: int, stage) -> Dict:
    """Execute derived outputs stage.

    Builds all derived files and the current/ compatibility surface.
    """
    derived_result = build_derived_dir(session_id, segments, text, audio_duration_ms)
    current_result = build_current_dir(session_id, segments, text, audio_duration_ms)
    files_written = sorted(set((derived_result.get("files_written") or []) + (current_result.get("files_written") or [])))
    stage.commit(files_written)
    return {
        "derived": derived_result,
        "current": current_result,
        "files_written": files_written,
    }
