"""
Stage 8 - Stabilization and Canonical Assembly
"""
from __future__ import annotations

import logging
from typing import Dict, List

from app.core.atomic_io import atomic_write_json, atomic_write_text
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)

# Canonical segments are the user-facing lexical truth, but they are also the
# substrate retrieval indexes ground onto.  Collapsing dozens of stripes into a
# single multi-minute segment destroys retrieval granularity (see misconception
# about 194 stripes collapsing into 12 canonical segments).  We cap each
# canonical segment at roughly two decode windows so downstream retrieval still
# has a meaningful access unit even when stripes share source+language.
MAX_SEGMENT_MS = 60000
MAX_SEGMENT_STRIPES = 4

# Segment quality grading used by the lexical/semantic firewall (Phase 0).
# Downstream stages read `segment_quality_status` + `canonical/quality_gate.json`
# to decide whether to run semantic marking, update the memory graph, and
# enrich retrieval.  These grades stay descriptive — they do not rewrite
# canonical text.
SEGMENT_STATUS_SUPPRESSED = "suppressed"
SEGMENT_STATUS_WEAK = "weak"
SEGMENT_STATUS_GOOD = "good"

# Any one of these flags means the segment is not safe to feed to semantic
# marking as-is and must not become a source of memory updates.
_SUPPRESSED_FLAGS = frozenset({
    "media_junk_suppressed",
    "no_supported_evidence",
    "empty_output",
})

# These flags demote a segment to "weak": still canonical lexical truth, but
# semantic enrichment should not over-interpret it.
_WEAK_FLAGS = frozenset({
    "unsupported_tokens_present",
    "low_confidence",
    "single_window_support",
    "llm_selection_rejected",
    "validator_warning",
})


def stabilize_stripes(
    reconciliation_records: List[Dict],
    stripe_packets: List[Dict] = None,
    finalize_last_boundary: bool = True,
) -> List[Dict]:
    stabilized = []
    ordered_packets = list(stripe_packets or [])
    packet_index = {packet.get("stripe_id"): packet for packet in ordered_packets}
    stripe_order = {packet.get("stripe_id"): idx for idx, packet in enumerate(ordered_packets)}
    last_index = len(ordered_packets) - 1

    for record in reconciliation_records:
        stripe_id = record.get("stripe_id", "")
        packet = packet_index.get(stripe_id, {})
        support_count = record.get("support_window_count", packet.get("support_window_count", 1))
        state = _resolve_stabilization_state(
            support_count=support_count,
            stripe_index=stripe_order.get(stripe_id),
            last_index=last_index,
            finalize_last_boundary=finalize_last_boundary,
        )
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


def _resolve_stabilization_state(
    support_count: int,
    stripe_index: int | None,
    last_index: int,
    finalize_last_boundary: bool = True,
) -> str:
    if support_count >= 2:
        return "stabilized"

    # The first stripe is a true session boundary as soon as ingest begins.
    if support_count >= 1 and stripe_index == 0:
        return "stabilized"

    # The trailing stripe is only a true session boundary once the session has
    # been finalized.  During live/incremental processing we keep the tail
    # provisional so the next batch can re-evaluate the junction with fresh
    # bridge windows and yield the same final quality as a whole-session run.
    if (
        support_count >= 1
        and finalize_last_boundary
        and stripe_index is not None
        and stripe_index == last_index
    ):
        return "stabilized"

    return "provisional"


def merge_into_segments(stabilized_stripes: List[Dict]) -> List[Dict]:
    if not stabilized_stripes:
        return []

    sorted_stripes = sorted(stabilized_stripes, key=lambda item: item["start_ms"])
    segments: List[Dict] = []
    current: Dict | None = None

    for stripe in sorted_stripes:
        text = _stripe_text(stripe)
        if not text:
            continue

        if current is None:
            current = _new_segment_from_stripe(stripe)
            continue

        projected_span_ms = stripe["end_ms"] - current["start_ms"]
        would_exceed_duration = projected_span_ms > MAX_SEGMENT_MS
        would_exceed_stripes = len(current["stripes"]) >= MAX_SEGMENT_STRIPES
        can_merge = (
            stripe["start_ms"] <= current["end_ms"] + 1000
            and stripe.get("chosen_source") == current.get("source_model")
            and stripe.get("output_language", stripe.get("language")) == current.get("language")
            and not would_exceed_duration
            and not would_exceed_stripes
        )

        if can_merge:
            current["end_ms"] = stripe["end_ms"]
            current["text"] = _dedup_join(current["text"], text)
            current["confidence"] = round((current["confidence"] + stripe.get("confidence", 0.0)) / 2, 4)
            current["stripes"].append(stripe["stripe_id"])
            current["support_windows"].update(stripe.get("support_windows", []))
            current["support_models"].update(stripe.get("support_models", []))
            current["assembly_decisions"].append(_assembly_decision(stripe))
            current["corruption_flags"].update(_stripe_corruption_flags(stripe))
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
        "text": _stripe_text(stripe),
        "source_model": stripe.get("chosen_source", "unknown"),
        "language": stripe.get("output_language", stripe.get("language")),
        "confidence": round(stripe.get("confidence", 0.0), 4),
        "stabilization_state": stripe.get("stabilization_state", "provisional"),
        "support_windows": set(stripe.get("support_windows", [])),
        "support_models": set(stripe.get("support_models", [])),
        "stripes": [stripe["stripe_id"]],
        "assembly_decisions": [_assembly_decision(stripe)],
        "corruption_flags": set(_stripe_corruption_flags(stripe)),
    }


def _stripe_corruption_flags(stripe: Dict) -> List[str]:
    """Propagate stripe-level uncertainty/degradation into segment-level flags.

    Canonical spec 13.1: explicit uncertainty on weak evidence.
    Downstream retrieval and quality surfaces must be able to *see* that a
    segment inherits low-confidence or degraded ASR witnesses, instead of
    presenting it as pristine truth.
    """
    flags = set()
    for flag in stripe.get("uncertainty_flags") or []:
        if flag:
            flags.add(flag)
    if stripe.get("unsupported_tokens"):
        flags.add("unsupported_tokens_present")
    if stripe.get("validation_status") == "accepted_with_warnings":
        flags.add("validator_warning")
    if stripe.get("llm_validation_rejected"):
        flags.add("llm_selection_rejected")
    if stripe.get("fallback_reason") in {"all_empty", "no_evidence"}:
        flags.add("no_supported_evidence")
    if (stripe.get("support_window_count") or 0) < 2:
        flags.add("single_window_support")
    return sorted(flags)


def _finalize_segment(segment: Dict) -> Dict:
    segment["support_windows"] = sorted(segment["support_windows"])
    segment["support_models"] = sorted(segment["support_models"])
    segment["corruption_flags"] = sorted(segment.get("corruption_flags") or [])
    segment["assembly_audit"] = _segment_assembly_audit(segment.get("assembly_decisions", []))
    segment["segment_quality_status"] = _segment_quality_status(segment)
    return segment


def _segment_quality_status(segment: Dict) -> str:
    """Grade a canonical segment for the lexical/semantic firewall.

    The grade is descriptive: canonical text is already the lexical truth.
    What this controls is whether the segment is safe to *interpret*
    downstream (context spans, memory graph, rich retrieval).
    """
    text = (segment.get("text") or "").strip()
    if not text:
        return SEGMENT_STATUS_SUPPRESSED
    flags = set(segment.get("corruption_flags") or [])
    if flags & _SUPPRESSED_FLAGS:
        return SEGMENT_STATUS_SUPPRESSED
    if flags & _WEAK_FLAGS:
        return SEGMENT_STATUS_WEAK
    confidence = segment.get("confidence") or 0.0
    if confidence < 0.3:
        return SEGMENT_STATUS_WEAK
    return SEGMENT_STATUS_GOOD


def compute_quality_gate(segments: List[Dict]) -> Dict:
    """Aggregate segment grades into the Phase 0 lexical/semantic firewall.

    Thresholds are intentionally conservative: a session where more than 30%
    of stabilized segments were suppressed, or more than 60% were weak, is
    ruled out of semantic enrichment and memory updates.  A softer band
    (10%/40%) still allows semantic marking but blocks memory propagation.

    The artifact is the single source of truth that semantic_marking,
    memory_graph, and derived retrieval consult — callers must not re-derive
    it themselves.
    """
    stabilized = [seg for seg in segments if seg.get("stabilization_state") == "stabilized"]
    total = len(stabilized)
    if not total:
        return {
            "contract_version": "1.0",
            "session_quality_status": "insufficient_lexical_evidence",
            "semantic_eligible": False,
            "memory_update_eligible": False,
            "stabilized_segment_count": 0,
            "good_count": 0,
            "weak_count": 0,
            "suppressed_count": 0,
            "suppressed_ratio": 0.0,
            "weak_ratio": 0.0,
            "reasons": ["no_stabilized_segments"],
        }

    good = sum(1 for seg in stabilized if seg.get("segment_quality_status") == SEGMENT_STATUS_GOOD)
    weak = sum(1 for seg in stabilized if seg.get("segment_quality_status") == SEGMENT_STATUS_WEAK)
    suppressed = sum(1 for seg in stabilized if seg.get("segment_quality_status") == SEGMENT_STATUS_SUPPRESSED)

    suppressed_ratio = round(suppressed / total, 4)
    weak_ratio = round(weak / total, 4)

    reasons: List[str] = []
    semantic_eligible = True
    memory_update_eligible = True
    session_status = "healthy"

    if suppressed_ratio > 0.3 or weak_ratio > 0.6:
        semantic_eligible = False
        memory_update_eligible = False
        session_status = "unhealthy"
        if suppressed_ratio > 0.3:
            reasons.append(f"suppressed_ratio_{suppressed_ratio}_exceeds_0.3")
        if weak_ratio > 0.6:
            reasons.append(f"weak_ratio_{weak_ratio}_exceeds_0.6")
    elif suppressed_ratio > 0.1 or weak_ratio > 0.4:
        memory_update_eligible = False
        session_status = "degraded"
        if suppressed_ratio > 0.1:
            reasons.append(f"suppressed_ratio_{suppressed_ratio}_exceeds_0.1_blocks_memory")
        if weak_ratio > 0.4:
            reasons.append(f"weak_ratio_{weak_ratio}_exceeds_0.4_blocks_memory")

    return {
        "contract_version": "1.0",
        "session_quality_status": session_status,
        "semantic_eligible": semantic_eligible,
        "memory_update_eligible": memory_update_eligible,
        "stabilized_segment_count": total,
        "good_count": good,
        "weak_count": weak,
        "suppressed_count": suppressed,
        "suppressed_ratio": suppressed_ratio,
        "weak_ratio": weak_ratio,
        "reasons": reasons,
    }


def read_quality_gate(session_id: str) -> Dict:
    """Return the quality gate artifact (or a safe empty default)."""
    from app.core.atomic_io import safe_read_json
    sd = session_dir(session_id)
    payload = safe_read_json(str(sd / "canonical" / "quality_gate.json")) or {}
    if not payload:
        canonical_dir = sd / "canonical"
        canonical_artifacts_present = any(
            (canonical_dir / name).is_file()
            for name in (
                "canonical_segments.json",
                "final_transcript.json",
                "stabilized_partial.json",
                "provisional_partial.json",
                "transcript.txt",
            )
        )
        if canonical_artifacts_present:
            return {
                "contract_version": "1.0",
                "session_quality_status": "missing_after_canonical_stage",
                "semantic_eligible": False,
                "memory_update_eligible": False,
                "reasons": ["quality_gate_missing_after_canonical_stage"],
            }
        return {
            "contract_version": "1.0",
            "session_quality_status": "unverified_direct_stage",
            "semantic_eligible": True,
            "memory_update_eligible": True,
            "reasons": ["quality_gate_missing_direct_stage_defaulting_permissive"],
        }
    return payload


def _assembly_decision(stripe: Dict) -> Dict:
    return {
        "stripe_id": stripe.get("stripe_id"),
        "start_ms": stripe.get("start_ms"),
        "end_ms": stripe.get("end_ms"),
        "method": stripe.get("method"),
        "assembly_mode": stripe.get("assembly_mode"),
        "chosen_source": stripe.get("chosen_source"),
        "final_text": _stripe_text(stripe),
        "confidence": stripe.get("confidence"),
        "support_windows": stripe.get("support_windows", []),
        "support_models": stripe.get("support_models", []),
        "stabilization_state": stripe.get("stabilization_state"),
        "fallback_reason": stripe.get("fallback_reason"),
        "used_candidate_ids": stripe.get("used_candidate_ids", []),
        "unsupported_tokens": stripe.get("unsupported_tokens", []),
        "token_support_ratio": stripe.get("token_support_ratio"),
        "uncertainty_flags": stripe.get("uncertainty_flags", []),
        "validation_status": stripe.get("validation_status"),
        "evidence_notes": stripe.get("evidence_notes", []),
        "source_language": stripe.get("source_language"),
        "output_language": stripe.get("output_language"),
    }


def _dedup_join(text_a: str, text_b: str, max_overlap_words: int = 12, min_overlap_words: int = 2) -> str:
    words_a = text_a.split()
    words_b = text_b.split()

    if not words_a or not words_b:
        return f"{text_a} {text_b}".strip()

    norm_a = " ".join(words_a).strip().lower()
    norm_b = " ".join(words_b).strip().lower()
    if norm_a == norm_b:
        return text_a.strip()
    if norm_b and norm_b in norm_a:
        return text_a.strip()
    if norm_a and norm_a in norm_b:
        return text_b.strip()

    for overlap_len in range(min(max_overlap_words, len(words_a), len(words_b)), min_overlap_words - 1, -1):
        suffix_a = " ".join(words_a[-overlap_len:]).lower()
        prefix_b = " ".join(words_b[:overlap_len]).lower()
        if suffix_a == prefix_b:
            return text_a + " " + " ".join(words_b[overlap_len:])

    return f"{text_a} {text_b}".strip()


def build_transcript_surfaces(
    segments: List[Dict],
    session_id: str,
    stripe_decisions: List[Dict] | None = None,
    emit_final_surface: bool = True,
) -> Dict:
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
                "assembly_audit": seg.get("assembly_audit", {}),
            }
            for seg in provisional_segments
        ],
    }

    provisional_payload = {
        "session_id": session_id,
        "truth_layer": "lexical",
        "semantic_layer": "provisional_partial",
        "text": provisional_text,
        "segment_count": len(output_segments),
        "segments": output_segments,
    }
    stabilized_payload = {
        "session_id": session_id,
        "truth_layer": "lexical",
        "semantic_layer": "stabilized_partial",
        "text": stabilized_text,
        "segment_count": len(stabilized_output_segments),
        "segments": stabilized_output_segments,
    }
    final_payload = {
        "session_id": session_id,
        "truth_layer": "lexical",
        "semantic_layer": "final_transcript",
        "text": stabilized_text,
        "segment_count": len(stabilized_output_segments),
        "segments": stabilized_output_segments,
        "source": "stabilized_canonical_segments",
    }

    quality_gate = compute_quality_gate(provisional_segments)

    atomic_write_text(str(canonical_dir / "transcript.txt"), stabilized_text)
    atomic_write_json(str(canonical_dir / "canonical_segments.json"), {
        "session_id": session_id,
        "segment_count": len(output_segments),
        "segments": output_segments,
    })
    atomic_write_json(str(canonical_dir / "provisional_partial.json"), provisional_payload)
    atomic_write_json(str(canonical_dir / "stabilized_partial.json"), stabilized_payload)
    if emit_final_surface:
        atomic_write_json(str(canonical_dir / "final_transcript.json"), final_payload)
    atomic_write_json(str(canonical_dir / "provenance.json"), provenance)
    atomic_write_json(str(canonical_dir / "quality_gate.json"), {
        "session_id": session_id,
        **quality_gate,
    })

    return {
        "text": stabilized_text,
        "segments": output_segments,
        "stabilized_segments": stabilized_output_segments,
        "words": words,
        "provenance": provenance,
        "provisional_partial": provisional_payload,
        "stabilized_partial": stabilized_payload,
        "final_transcript": final_payload if emit_final_surface else None,
        "quality_gate": quality_gate,
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
        "corruption_flags": list(segment.get("corruption_flags") or []),
        "stripes": segment.get("stripes", []),
        "assembly_decisions": segment.get("assembly_decisions", []),
        "assembly_audit": segment.get("assembly_audit", {}),
        "segment_quality_status": segment.get("segment_quality_status") or _segment_quality_status(segment),
    }


def _stripe_text(stripe: Dict) -> str:
    return stripe.get("final_text", stripe.get("chosen_text", "")).strip()


def _segment_assembly_audit(assembly_decisions: List[Dict]) -> Dict:
    used_candidate_ids = sorted({
        candidate_id
        for decision in assembly_decisions
        for candidate_id in (decision.get("used_candidate_ids") or [])
        if candidate_id
    })
    unsupported_tokens = sorted({
        token
        for decision in assembly_decisions
        for token in (decision.get("unsupported_tokens") or [])
        if token
    })
    uncertainty_flags = sorted({
        flag
        for decision in assembly_decisions
        for flag in (decision.get("uncertainty_flags") or [])
        if flag
    })
    assembly_modes = [
        decision.get("assembly_mode")
        for decision in assembly_decisions
        if decision.get("assembly_mode")
    ]
    validation_statuses = [
        decision.get("validation_status")
        for decision in assembly_decisions
        if decision.get("validation_status")
    ]

    return {
        "assembly_mode": assembly_modes[0] if len(set(assembly_modes)) == 1 and assembly_modes else "mixed_segment_assembly",
        "used_candidate_ids": used_candidate_ids,
        "unsupported_tokens": unsupported_tokens,
        "uncertainty_flags": uncertainty_flags,
        "validation_status": validation_statuses[0] if len(set(validation_statuses)) == 1 and validation_statuses else "mixed",
    }


def run_canonical_assembly(
    session_id: str,
    reconciliation_result: Dict,
    stripe_packets: List[Dict],
    stage,
    *,
    finalize_last_boundary: bool = True,
    emit_final_surface: bool = True,
) -> Dict:
    records = reconciliation_result.get("records", [])
    stabilized = stabilize_stripes(
        records,
        stripe_packets,
        finalize_last_boundary=finalize_last_boundary,
    )
    segments = merge_into_segments(stabilized)
    surfaces = build_transcript_surfaces(
        segments,
        session_id,
        stripe_decisions=stabilized,
        emit_final_surface=emit_final_surface,
    )

    artifacts = [
        "transcript.txt",
        "canonical_segments.json",
        "provenance.json",
        "provisional_partial.json",
        "stabilized_partial.json",
        "quality_gate.json",
    ]
    if emit_final_surface:
        artifacts.append("final_transcript.json")
    stage.commit(artifacts)

    return {
        "segment_count": len(segments),
        "stabilized_count": sum(1 for seg in segments if seg.get("stabilization_state") == "stabilized"),
        "provisional_count": sum(1 for seg in segments if seg.get("stabilization_state") == "provisional"),
        "text_length": len(surfaces["text"]),
        "surfaces": surfaces,
        "quality_gate": surfaces.get("quality_gate", {}),
    }
