"""
Stage 8 - Stabilization and Canonical Assembly
"""
from __future__ import annotations

import logging
from typing import Dict, List

from app.core.atomic_io import atomic_write_json, atomic_write_text
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)


def stabilize_stripes(reconciliation_records: List[Dict], stripe_packets: List[Dict] = None) -> List[Dict]:
    stabilized = []
    packet_index = {packet.get("stripe_id"): packet for packet in (stripe_packets or [])}

    for record in reconciliation_records:
        stripe_id = record.get("stripe_id", "")
        packet = packet_index.get(stripe_id, {})
        support_count = record.get("support_window_count", packet.get("support_window_count", 1))
        state = "stabilized" if support_count >= 2 else "provisional"
        stabilized.append(
            {
                **record,
                "stabilization_state": state,
                "support_window_count": support_count,
                "support_windows": sorted(set(record.get("support_windows") or packet.get("support_windows") or [])),
                "support_models": sorted(set(record.get("support_models") or packet.get("support_models") or [])),
            }
        )
    return stabilized


def merge_into_segments(stabilized_stripes: List[Dict]) -> List[Dict]:
    if not stabilized_stripes:
        return []

    sorted_stripes = sorted(stabilized_stripes, key=lambda item: item["start_ms"])
    segments: List[Dict] = []
    current: Dict | None = None

    for stripe in sorted_stripes:
        text = stripe.get("chosen_text", "").strip()
        if not text:
            continue

        if current is None:
            current = _new_segment_from_stripe(stripe)
            continue

        can_merge = (
            stripe["start_ms"] <= current["end_ms"] + 1000
            and stripe.get("chosen_source") == current.get("source_model")
            and stripe.get("language") == current.get("language")
        )

        if can_merge:
            current["end_ms"] = stripe["end_ms"]
            current["text"] = _dedup_join(current["text"], text)
            current["confidence"] = round((current["confidence"] + stripe.get("confidence", 0.0)) / 2, 4)
            current["stripes"].append(stripe["stripe_id"])
            current["support_windows"].update(stripe.get("support_windows", []))
            current["support_models"].update(stripe.get("support_models", []))
            current["assembly_decisions"].append(_assembly_decision(stripe))
            if stripe.get("stabilization_state") == "provisional":
                current["stabilization_state"] = "provisional"
        else:
            segments.append(_finalize_segment(current))
            current = _new_segment_from_stripe(stripe)

    if current is not None:
        segments.append(_finalize_segment(current))

    for index, segment in enumerate(segments):
        segment["segment_id"] = f"seg_{index:06d}"
        segment.setdefault("speaker", None)

    return segments


def _new_segment_from_stripe(stripe: Dict) -> Dict:
    return {
        "start_ms": stripe["start_ms"],
        "end_ms": stripe["end_ms"],
        "text": stripe.get("chosen_text", "").strip(),
        "source_model": stripe.get("chosen_source", "unknown"),
        "language": stripe.get("language"),
        "confidence": round(stripe.get("confidence", 0.0), 4),
        "stabilization_state": stripe.get("stabilization_state", "provisional"),
        "support_windows": set(stripe.get("support_windows", [])),
        "support_models": set(stripe.get("support_models", [])),
        "stripes": [stripe["stripe_id"]],
        "assembly_decisions": [_assembly_decision(stripe)],
    }


def _finalize_segment(segment: Dict) -> Dict:
    segment["support_windows"] = sorted(segment["support_windows"])
    segment["support_models"] = sorted(segment["support_models"])
    return segment


def _assembly_decision(stripe: Dict) -> Dict:
    return {
        "stripe_id": stripe.get("stripe_id"),
        "start_ms": stripe.get("start_ms"),
        "end_ms": stripe.get("end_ms"),
        "method": stripe.get("method"),
        "chosen_source": stripe.get("chosen_source"),
        "confidence": stripe.get("confidence"),
        "support_windows": stripe.get("support_windows", []),
        "support_models": stripe.get("support_models", []),
        "stabilization_state": stripe.get("stabilization_state"),
        "fallback_reason": stripe.get("fallback_reason"),
    }


def _dedup_join(text_a: str, text_b: str, max_overlap_words: int = 12, min_overlap_words: int = 2) -> str:
    words_a = text_a.split()
    words_b = text_b.split()

    if not words_a or not words_b:
        return f"{text_a} {text_b}".strip()

    for overlap_len in range(min(max_overlap_words, len(words_a), len(words_b)), min_overlap_words - 1, -1):
        suffix_a = " ".join(words_a[-overlap_len:]).lower()
        prefix_b = " ".join(words_b[:overlap_len]).lower()
        if suffix_a == prefix_b:
            return text_a + " " + " ".join(words_b[overlap_len:])

    return f"{text_a} {text_b}".strip()


def build_transcript_surfaces(segments: List[Dict], session_id: str, stripe_decisions: List[Dict] | None = None) -> Dict:
    sd = session_dir(session_id)
    canonical_dir = sd / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    if stripe_decisions is None:
        existing_provenance = canonical_dir / "provenance.json"
        if existing_provenance.is_file():
            from app.core.atomic_io import safe_read_json

            prior = safe_read_json(str(existing_provenance)) or {}
            stripe_decisions = prior.get("stripe_decisions") or []

    provisional_segments = [seg for seg in segments if seg.get("text")]
    stabilized_segments = [seg for seg in provisional_segments if seg.get("stabilization_state") == "stabilized"]

    provisional_text = " ".join(seg["text"] for seg in provisional_segments)
    stabilized_text = " ".join(seg["text"] for seg in stabilized_segments)

    output_segments = [_segment_public_view(seg) for seg in provisional_segments]
    stabilized_output_segments = [_segment_public_view(seg) for seg in stabilized_segments]

    words = [
        {
            "t_ms": seg["start_ms"],
            "speaker": seg.get("speaker", "SPEAKER_00"),
            "w": seg.get("text", ""),
        }
        for seg in stabilized_segments
    ]

    provenance = {
        "session_id": session_id,
        "segment_count": len(output_segments),
        "assembly_method": "canonical_stripe_reconciliation",
        "final_transcript_source": "stabilized_canonical_segments",
        "stripe_decisions": stripe_decisions or [],
        "segments": [
            {
                "segment_id": seg["segment_id"],
                "stripes": seg.get("stripes", []),
                "source_model": seg.get("source_model"),
                "confidence": seg.get("confidence"),
                "language": seg.get("language"),
                "stabilization_state": seg.get("stabilization_state"),
                "support_windows": seg.get("support_windows", []),
                "support_models": seg.get("support_models", []),
                "assembly_decisions": seg.get("assembly_decisions", []),
            }
            for seg in provisional_segments
        ],
    }

    provisional_payload = {
        "session_id": session_id,
        "semantic_layer": "provisional_partial",
        "text": provisional_text,
        "segment_count": len(output_segments),
        "segments": output_segments,
    }
    stabilized_payload = {
        "session_id": session_id,
        "semantic_layer": "stabilized_partial",
        "text": stabilized_text,
        "segment_count": len(stabilized_output_segments),
        "segments": stabilized_output_segments,
    }
    final_payload = {
        "session_id": session_id,
        "semantic_layer": "final_transcript",
        "text": stabilized_text,
        "segment_count": len(stabilized_output_segments),
        "segments": stabilized_output_segments,
        "source": "stabilized_canonical_segments",
    }

    atomic_write_text(str(canonical_dir / "transcript.txt"), stabilized_text)
    atomic_write_json(str(canonical_dir / "canonical_segments.json"), {
        "session_id": session_id,
        "segment_count": len(output_segments),
        "segments": output_segments,
    })
    atomic_write_json(str(canonical_dir / "provisional_partial.json"), provisional_payload)
    atomic_write_json(str(canonical_dir / "stabilized_partial.json"), stabilized_payload)
    atomic_write_json(str(canonical_dir / "final_transcript.json"), final_payload)
    atomic_write_json(str(canonical_dir / "provenance.json"), provenance)

    return {
        "text": stabilized_text,
        "segments": output_segments,
        "stabilized_segments": stabilized_output_segments,
        "words": words,
        "provenance": provenance,
        "provisional_partial": provisional_payload,
        "stabilized_partial": stabilized_payload,
        "final_transcript": final_payload,
    }


def _segment_public_view(segment: Dict) -> Dict:
    return {
        "segment_id": segment["segment_id"],
        "start_ms": segment["start_ms"],
        "end_ms": segment["end_ms"],
        "speaker": segment.get("speaker"),
        "text": segment.get("text", ""),
        "language": segment.get("language"),
        "confidence": segment.get("confidence", 0.0),
        "support_windows": segment.get("support_windows", []),
        "support_models": segment.get("support_models", []),
        "stabilization_state": segment.get("stabilization_state", "provisional"),
        "corruption_flags": [],
        "stripes": segment.get("stripes", []),
        "assembly_decisions": segment.get("assembly_decisions", []),
    }


def run_canonical_assembly(session_id: str, reconciliation_result: Dict, stripe_packets: List[Dict], stage) -> Dict:
    records = reconciliation_result.get("records", [])
    stabilized = stabilize_stripes(records, stripe_packets)
    segments = merge_into_segments(stabilized)
    surfaces = build_transcript_surfaces(segments, session_id, stripe_decisions=stabilized)

    stage.commit(
        [
            "transcript.txt",
            "canonical_segments.json",
            "provenance.json",
            "provisional_partial.json",
            "stabilized_partial.json",
            "final_transcript.json",
        ]
    )

    return {
        "segment_count": len(segments),
        "stabilized_count": sum(1 for seg in segments if seg.get("stabilization_state") == "stabilized"),
        "provisional_count": sum(1 for seg in segments if seg.get("stabilization_state") == "provisional"),
        "text_length": len(surfaces["text"]),
        "surfaces": surfaces,
    }
