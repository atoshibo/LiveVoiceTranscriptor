"""
Stage 12 - Derived Outputs

All derived outputs are produced from canonical truth, never from provisional text.

Generates:
  - transcript.txt (plain text)
  - transcript_by_speaker.json / .txt
  - subtitles.srt / subtitles.vtt
  - quality_report.json
  - raw_transcript.json
  - clean_transcript.json
  - retrieval_index.json / retrieval_index_v2.json
  - classification.json (optional)

Output: derived/ and current/ (compatibility read surface)
"""
import os
import re
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from app.core.atomic_io import atomic_write_json, atomic_write_text, safe_read_json
from app.pipeline.witness_diagnostics import looks_like_media_pollution
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)

# Flags that mark a canonical segment as *unreadable* for display purposes.
# A segment with any of these must not be copied verbatim into clean_transcript
# or indexed in retrieval; it appears in raw_transcript so audit still sees it.
DISPLAY_SUPPRESS_FLAGS = frozenset({
    "media_junk_suppressed",
    "no_supported_evidence",
})

_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_PUNCT_RE = re.compile(r"([.!?,;:])\1+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.!?,;:])")


def _clean_text_line(text: str) -> str:
    """Normalize whitespace and fix trivial punctuation artifacts.

    This is reading-surface cleanup only -- it does NOT rewrite content.
    Canonical truth stays in canonical/.  Used by clean_transcript and
    display subtitles so readers see tidy text without the lexical layer
    being modified.
    """
    if not text:
        return ""
    cleaned = _WHITESPACE_RE.sub(" ", text).strip()
    cleaned = _REPEATED_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned


def _segment_is_display_clean(segment: Dict) -> bool:
    """True if the segment is safe to show to a reader as-is."""
    text = (segment.get("text") or "").strip()
    if not text:
        return False
    if looks_like_media_pollution(text):
        return False
    flags = set(segment.get("corruption_flags") or [])
    if flags & DISPLAY_SUPPRESS_FLAGS:
        return False
    return True


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
    """Generate quality analysis report from canonical segments.

    Detects (beyond the original confidence/duration checks):
      - media_pollution_detected: subtitle/media hallucinations that reached
        canonical truth (e.g. "Субтитры сделал DimaTorzok")
      - corruption_flagged_segment: segments carrying any corruption flag
      - high_unsupported_token_ratio: segments whose audit showed the chosen
        text poorly matched the evidence vocabulary
      - pipeline_health: counters that downstream dashboards surface
    """
    if not segments:
        return {
            "analysis_version": "1.2",
            "segment_count": 0,
            "total_text_length": 0,
            "issues": [],
            "issue_count": 0,
            "reading_text": "",
            "source_integrity": None,
            "pipeline_health": {
                "media_pollution_count": 0,
                "corruption_flagged_count": 0,
                "low_confidence_count": 0,
            },
        }

    issues = []
    media_pollution_count = 0
    corruption_flagged_count = 0
    low_confidence_count = 0

    for i, seg in enumerate(segments):
        seg_text = seg.get("text", "")
        conf = seg.get("confidence", 0)
        duration = seg["end_ms"] - seg["start_ms"]
        corruption_flags = list(seg.get("corruption_flags") or [])
        assembly_audit = seg.get("assembly_audit") or {}

        if conf < 0.3:
            low_confidence_count += 1
            issues.append({
                "type": "low_confidence",
                "confidence": "low",
                "segment_index": i,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg_text[:100],
                "reason": f"Segment confidence {conf:.2f} below threshold",
            })

        if duration < 500 and seg_text:
            issues.append({
                "type": "very_short_segment",
                "confidence": "medium",
                "segment_index": i,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg_text[:100],
                "reason": f"Segment duration {duration}ms very short",
            })

        if looks_like_media_pollution(seg_text):
            media_pollution_count += 1
            issues.append({
                "type": "media_pollution_detected",
                "confidence": "high",
                "segment_index": i,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg_text[:200],
                "reason": "Subtitle/media hallucination reached canonical truth",
            })

        if corruption_flags:
            corruption_flagged_count += 1
            issues.append({
                "type": "corruption_flagged_segment",
                "confidence": "medium",
                "segment_index": i,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg_text[:100],
                "reason": f"Corruption flags: {', '.join(corruption_flags)}",
                "corruption_flags": corruption_flags,
            })

        unsupported = assembly_audit.get("unsupported_tokens") or []
        if unsupported and len(unsupported) >= 3:
            issues.append({
                "type": "high_unsupported_token_ratio",
                "confidence": "medium",
                "segment_index": i,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": seg_text[:100],
                "reason": f"{len(unsupported)} tokens in chosen text not found in evidence",
                "unsupported_tokens": unsupported[:10],
            })

    transcript_end = max((s["end_ms"] for s in segments), default=0)
    coverage_ratio = transcript_end / max(1, audio_duration_ms)

    return {
        "analysis_version": "1.2",
        "segment_count": len(segments),
        "total_text_length": len(text),
        "audio_duration_s": round(audio_duration_ms / 1000, 1),
        "transcript_last_end_s": round(transcript_end / 1000, 1),
        "transcript_coverage_ratio": round(coverage_ratio, 4),
        "issues": issues,
        "issue_count": len(issues),
        "reading_text": text,
        "source_integrity": None,
        "pipeline_health": {
            "media_pollution_count": media_pollution_count,
            "corruption_flagged_count": corruption_flagged_count,
            "low_confidence_count": low_confidence_count,
        },
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
    """Generate a reader-ready clean transcript.

    Cleanup rules (reading surface only -- canonical truth is untouched):
      - drop segments flagged as media pollution or that only have
        suppressed/empty evidence
      - normalize whitespace and common punctuation artifacts
      - break paragraphs on speaker change or >5s gap
      - skipped segments are reported in `dropped_segments` for auditability
    """
    paragraphs = []
    current_para = {"text": "", "segment_indices": []}
    dropped_segments = []
    kept_texts = []

    for i, seg in enumerate(segments):
        seg_text = _clean_text_line(seg.get("text", ""))
        if not seg_text:
            continue

        if not _segment_is_display_clean(seg):
            dropped_segments.append({
                "segment_index": i,
                "segment_id": seg.get("segment_id"),
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "corruption_flags": list(seg.get("corruption_flags") or []),
                "reason": "media_pollution_or_unsupported_evidence",
                "text_preview": seg.get("text", "")[:120],
            })
            # Paragraph break after dropped content
            if current_para["text"]:
                paragraphs.append(current_para)
                current_para = {"text": "", "segment_indices": []}
            continue

        if current_para["text"]:
            current_para["text"] += " " + seg_text
        else:
            current_para["text"] = seg_text
        current_para["segment_indices"].append(i)
        kept_texts.append(seg_text)

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

    clean_text = _clean_text_line(" ".join(kept_texts))
    return {
        "clean_text": clean_text,
        "paragraphs": paragraphs,
        "dropped_segments": dropped_segments,
        "dropped_count": len(dropped_segments),
        "cleanup_version": "2.0",
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


def _load_segment_markers(session_id: str) -> Dict[str, Dict]:
    sd = session_dir(session_id)
    payload = safe_read_json(str(sd / "enrichment" / "segment_markers.json")) or {}
    markers = payload.get("markers") or []
    return {
        marker.get("segment_id"): marker
        for marker in markers
        if marker.get("segment_id")
    }


def _load_context_spans(session_id: str) -> Dict:
    sd = session_dir(session_id)
    return safe_read_json(str(sd / "enrichment" / "context_spans.json")) or {}


def _ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def generate_retrieval_index(session_id: str, segments: List[Dict], marker_index: Optional[Dict[str, Dict]] = None, semantic_eligible: bool = True) -> Dict:
    """Ground retrieval entries directly on canonical segments and semantic markers.

    Segments flagged as media pollution or marked as media-junk-suppressed /
    no-supported-evidence are excluded from the index so downstream retrieval
    does not faithfully surface contaminated text.  They remain in the
    raw_transcript and quality_report artifacts for audit.

    When `semantic_eligible` is False (quality gate closed), marker data is
    ignored entirely — the index degrades to a pure lexical surface so we do
    not amplify unreliable interpretation.
    """
    if not semantic_eligible:
        marker_index = {}
    marker_index = marker_index or {}
    entries = []
    excluded = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue
        if not _segment_is_display_clean(seg):
            excluded.append({
                "segment_id": seg.get("segment_id"),
                "corruption_flags": list(seg.get("corruption_flags") or []),
                "reason": "display_unsafe_segment",
            })
            continue
        marker = marker_index.get(seg.get("segment_id"), {})
        entity_mentions = marker.get("entity_mentions") or []
        entries.append({
            "entry_id": f"ret_{i:06d}",
            "segment_id": seg.get("segment_id"),
            "text": text,
            "normalized_text": _normalized_retrieval_text(text),
            "keywords": _extract_keywords(text),
            "entity_mentions": entity_mentions,
            "entity_ids": [mention.get("entity_id") for mention in entity_mentions if mention.get("entity_id")],
            "aliases": [mention.get("surface_form") for mention in entity_mentions if mention.get("surface_form")],
            "relation_tags": marker.get("relation_tags", []),
            "topic_tags": marker.get("topic_tags", []),
            "topic_candidates": marker.get("topic_candidates", []),
            "project_tags": marker.get("project_tags", []),
            "emotion_tags": marker.get("emotion_tags", []),
            "retrieval_terms": marker.get("retrieval_terms", []) or _extract_keywords(text),
            "ambiguity_flags": marker.get("ambiguity_flags", []),
            "language": seg.get("language"),
            "speaker": seg.get("speaker"),
            "grounding": {
                "segment_id": seg.get("segment_id"),
                "start_ms": seg.get("start_ms"),
                "end_ms": seg.get("end_ms"),
                "canonical_path": "canonical/canonical_segments.json",
                "markers_path": "enrichment/segment_markers.json",
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
        "excluded_segments": excluded,
        "excluded_count": len(excluded),
        "semantic_eligible": semantic_eligible,
    }


def generate_retrieval_index_v2(session_id: str, segments: List[Dict], marker_index: Optional[Dict[str, Dict]] = None, semantic_eligible: bool = True) -> Dict:
    payload = generate_retrieval_index(session_id, segments, marker_index=marker_index, semantic_eligible=semantic_eligible)
    return {
        **payload,
        "version": 2,
        "source": "canonical_segments+segment_markers",
    }


def generate_retrieval_index_v3(
    session_id: str,
    segments: List[Dict],
    marker_index: Optional[Dict[str, Dict]] = None,
    context_spans_payload: Optional[Dict] = None,
    semantic_eligible: bool = True,
) -> Dict:
    """Ground retrieval entries on context spans while preserving canonical audit.

    Version 3 promotes the Phase 2 `context_spans` artifact to the primary
    retrieval unit.  Each entry still points back to the canonical segments
    and semantic markers that justify it.
    """
    marker_index = marker_index or {}
    context_spans_payload = context_spans_payload or {}
    spans = context_spans_payload.get("spans") or []
    segments_by_id = {
        seg.get("segment_id"): seg
        for seg in segments
        if seg.get("segment_id")
    }

    if not semantic_eligible:
        spans = []

    entries = []
    excluded = []
    for i, span in enumerate(spans):
        span_segment_ids = list(span.get("segment_ids") or [])
        span_segments = [
            segments_by_id[seg_id]
            for seg_id in span_segment_ids
            if seg_id in segments_by_id
        ]
        safe_segments = []
        excluded_segment_ids = []
        for seg in span_segments:
            if _segment_is_display_clean(seg):
                safe_segments.append(seg)
            else:
                excluded_segment_ids.append(seg.get("segment_id"))

        if not safe_segments:
            excluded.append({
                "context_id": span.get("context_id"),
                "segment_ids": span_segment_ids,
                "excluded_segment_ids": excluded_segment_ids,
                "reason": "no_display_safe_segments",
            })
            continue

        context_text = " ".join(
            _clean_text_line(seg.get("text", "").strip())
            for seg in safe_segments
            if seg.get("text")
        ).strip()
        if not context_text:
            excluded.append({
                "context_id": span.get("context_id"),
                "segment_ids": span_segment_ids,
                "excluded_segment_ids": excluded_segment_ids,
                "reason": "empty_context_text",
            })
            continue

        entity_mentions = []
        entity_seen = set()
        relation_tags: List[str] = []
        topic_tags: List[str] = []
        topic_candidates: List[str] = list(span.get("topic_candidates") or [])
        project_tags: List[str] = []
        emotion_tags: List[str] = []
        retrieval_terms: List[str] = []
        ambiguity_flags: List[str] = []

        for seg_id in span_segment_ids:
            marker = marker_index.get(seg_id, {})
            for mention in marker.get("entity_mentions") or []:
                key = (
                    mention.get("entity_id"),
                    mention.get("surface_form"),
                    mention.get("mention_type"),
                )
                if key in entity_seen:
                    continue
                entity_seen.add(key)
                entity_mentions.append(mention)
            relation_tags.extend(marker.get("relation_tags", []))
            topic_tags.extend(marker.get("topic_tags", []))
            topic_candidates.extend(marker.get("topic_candidates", []))
            project_tags.extend(marker.get("project_tags", []))
            emotion_tags.extend(marker.get("emotion_tags", []))
            retrieval_terms.extend(marker.get("retrieval_terms", []))
            ambiguity_flags.extend(marker.get("ambiguity_flags", []))

        retrieval_terms = _ordered_unique(retrieval_terms + _extract_keywords(context_text) + topic_candidates)
        entries.append({
            "entry_id": f"ctxret_{i:06d}",
            "context_id": span.get("context_id"),
            "segment_ids": span_segment_ids,
            "text": context_text,
            "normalized_text": _normalized_retrieval_text(context_text),
            "keywords": _extract_keywords(context_text),
            "entity_mentions": entity_mentions,
            "entity_ids": _ordered_unique([mention.get("entity_id") for mention in entity_mentions if mention.get("entity_id")]),
            "aliases": _ordered_unique([mention.get("surface_form") for mention in entity_mentions if mention.get("surface_form")]),
            "relation_tags": _ordered_unique(relation_tags),
            "topic_tags": _ordered_unique(topic_tags),
            "topic_candidates": _ordered_unique(topic_candidates),
            "project_tags": _ordered_unique(project_tags),
            "emotion_tags": _ordered_unique(emotion_tags),
            "retrieval_terms": retrieval_terms,
            "ambiguity_flags": _ordered_unique(ambiguity_flags),
            "language_profile": span.get("language_profile") or {},
            "speaker_ids": list(span.get("speaker_ids") or []),
            "confidence": span.get("confidence"),
            "excluded_segment_ids": excluded_segment_ids,
            "grounding": {
                "context_id": span.get("context_id"),
                "start_ms": span.get("start_ms"),
                "end_ms": span.get("end_ms"),
                "segment_ids": span_segment_ids,
                "canonical_path": "canonical/canonical_segments.json",
                "markers_path": "enrichment/segment_markers.json",
                "context_spans_path": "enrichment/context_spans.json",
            },
        })

    return {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": 3,
        "source": "context_spans+canonical_segments+segment_markers",
        "entry_count": len(entries),
        "entries": entries,
        "excluded_contexts": excluded,
        "excluded_count": len(excluded),
        "semantic_eligible": semantic_eligible,
        "context_span_count": len(spans),
        "gate_status": context_spans_payload.get("gate_status"),
    }


def _expand_display_segments(segments: List[Dict]) -> List[Dict]:
    """Build fine-grained display segments from each canonical segment.

    Canonical segments can span up to 60s to keep lexical truth coherent.
    The display surface should not inherit that coarse granularity -- the
    old server derived subtitles from ASR/stripe-level timing, and we
    restore that here by expanding each segment's assembly_decisions (one
    per reconciled stripe) into sub-segments.

    Segments that are not display-clean are dropped.  Fallback: if a
    segment has no assembly_decisions (older data), it is kept as-is so
    we never lose content.
    """
    display_segments: List[Dict] = []
    for seg in segments:
        decisions = seg.get("assembly_decisions") or []
        if not decisions:
            # Legacy path: no stripe-level timing available -- only here does
            # the segment-level display-clean filter apply, since we cannot
            # surgically drop the polluted portion.
            if not _segment_is_display_clean(seg):
                continue
            display_segments.append({
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "text": _clean_text_line(seg.get("text", "")),
                "speaker": seg.get("speaker"),
                "language": seg.get("language"),
                "source_segment_id": seg.get("segment_id"),
            })
            continue

        # Stripe-level path: drop only the polluted stripes within the segment,
        # preserving the clean ones for display.
        seg_flags = set(seg.get("corruption_flags") or [])
        if seg_flags & DISPLAY_SUPPRESS_FLAGS:
            continue
        for decision in decisions:
            sub_text = _clean_text_line(decision.get("final_text") or "")
            if not sub_text:
                continue
            if looks_like_media_pollution(sub_text):
                continue
            display_segments.append({
                "start_ms": decision.get("start_ms") or seg["start_ms"],
                "end_ms": decision.get("end_ms") or seg["end_ms"],
                "text": sub_text,
                "speaker": seg.get("speaker"),
                "language": seg.get("language"),
                "source_segment_id": seg.get("segment_id"),
                "source_stripe_id": decision.get("stripe_id"),
            })
    return display_segments


def _write_output_bundle(output_dir: Path, session_id: str, segments: List[Dict],
                         audio_duration_ms: int, include_retrieval: bool = False) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    final_segments = [seg for seg in segments if seg.get("stabilization_state") == "stabilized"]
    final_text = " ".join(seg.get("text", "") for seg in final_segments if seg.get("text"))

    # Fine-grained display segments (restored from the old server's
    # ASR/stripe-level timing path) -- used for subtitles, transcript_by_speaker,
    # and timestamps so readers get phrase granularity, not coarse canonical blocks.
    display_segments = _expand_display_segments(final_segments)

    atomic_write_text(str(output_dir / "transcript.txt"), final_text)

    speaker = generate_speaker_transcript(display_segments)
    atomic_write_json(str(output_dir / "transcript_by_speaker.json"), speaker["json"])
    atomic_write_text(str(output_dir / "transcript_by_speaker.txt"), speaker["text"])

    atomic_write_text(str(output_dir / "subtitles.srt"), generate_srt(display_segments))
    atomic_write_text(str(output_dir / "subtitles.vtt"), generate_vtt(display_segments))

    atomic_write_json(str(output_dir / "transcript_timestamps.json"), {
        "segments": [
            {
                "start": seg["start_ms"] / 1000,
                "end": seg["end_ms"] / 1000,
                "text": seg.get("text", ""),
                "speaker": seg.get("speaker"),
                "source_segment_id": seg.get("source_segment_id"),
                "source_stripe_id": seg.get("source_stripe_id"),
            }
            for seg in display_segments
        ],
        "words": [
            {
                "t_ms": seg["start_ms"],
                "speaker": seg.get("speaker"),
                "w": seg.get("text", ""),
            }
            for seg in display_segments
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
        from app.pipeline.canonical_assembly import read_quality_gate
        gate = read_quality_gate(session_id)
        semantic_eligible = bool(gate.get("semantic_eligible", True))
        marker_index = _load_segment_markers(session_id) if semantic_eligible else {}
        context_spans_payload = _load_context_spans(session_id) if semantic_eligible else {
            "span_count": 0,
            "spans": [],
            "gate_status": "suppressed_by_quality_gate",
        }
        retrieval = generate_retrieval_index(session_id, final_segments, marker_index=marker_index, semantic_eligible=semantic_eligible)
        retrieval_v2 = generate_retrieval_index_v2(session_id, final_segments, marker_index=marker_index, semantic_eligible=semantic_eligible)
        retrieval_v3 = generate_retrieval_index_v3(
            session_id,
            final_segments,
            marker_index=marker_index,
            context_spans_payload=context_spans_payload,
            semantic_eligible=semantic_eligible,
        )
        atomic_write_json(str(output_dir / "retrieval_index.json"), retrieval)
        atomic_write_json(str(output_dir / "retrieval_index_v2.json"), retrieval_v2)
        atomic_write_json(str(output_dir / "retrieval_index_v3.json"), retrieval_v3)
        files_written.append("retrieval_index.json")
        files_written.append("retrieval_index_v2.json")
        files_written.append("retrieval_index_v3.json")

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
