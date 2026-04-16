"""
Stage 11.5 - Context spans (sub-step of semantic_marking).

A `context_span` is NOT a packet of neighboring segments that happen to share
a tag.  It is a coherent block of conversation or topic, identified by the
**joint** behavior of several continuity signals:

  - temporal proximity        (gap_ms between consecutive canonical segments)
  - speaker continuity        (same diarized speaker)
  - language continuity       (same canonical language)
  - lexical continuity        (Jaccard overlap on content tokens)
  - entity continuity         (any shared entity_id from semantic markers)
  - topic_candidate continuity (any shared business taxonomy key)
  - explicit strong topic shift (no shared candidate AND temporal gap)

Output contract  -- written to ``enrichment/context_spans.json``::

    {
      "session_id": ...,
      "generated_at": ...,
      "span_count": N,
      "spans": [
        {
          "context_id": "ctx_xxxxxxxx",
          "session_id": ...,
          "start_ms": ...,
          "end_ms": ...,
          "segment_ids": [...],
          "speaker_ids": [...],
          "language_profile": {"primary": "ru", "ratio": 0.83, "languages": {...}},
          "topic_tags": [...],
          "topic_candidates": [...],   # business taxonomy
          "entity_ids": [...],
          "alias_hits": [...],         # curated_pack-resolved entity ids only
          "continuity_evidence": [
            {"between": [seg_a, seg_b],
             "signals": ["temporal_tight", "speaker", "topic"]},
            ...
          ],
          "confidence": 0.7,
          "grounding": {"canonical_path": ..., "markers_path": ...}
        }
      ]
    }

Suppressed quality gate: same as semantic_marking -- emits an empty,
annotated artifact so retrieval/audit never face a missing file.
"""
from __future__ import annotations

import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.core.atomic_io import atomic_write_json

# Continuity thresholds.  Tuned conservatively: temporal_tight is roughly the
# pause we expect inside a single conversational turn; temporal_loose marks
# the boundary at which only strong semantic continuity can keep us in the
# same span; temporal_hard_break always closes regardless of semantics.
TEMPORAL_TIGHT_MS = 8_000
TEMPORAL_LOOSE_MS = 30_000
TEMPORAL_HARD_BREAK_MS = 90_000

LEXICAL_JACCARD_THRESHOLD = 0.12
MIN_CONTINUITY_SCORE_LOOSE_GAP = 2  # signals required when gap ∈ (tight, loose]
MIN_CONTINUITY_SCORE_FAR_GAP = 3    # signals required when gap > loose
LEXICAL_TOKEN_MIN_LEN = 4

CONTRACT_VERSION = "1.0"

_CONTENT_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _content_tokens(text: str) -> Set[str]:
    """Return the bag of content tokens used for lexical continuity."""
    return {
        token.lower()
        for token in _CONTENT_TOKEN_RE.findall(text or "")
        if len(token) >= LEXICAL_TOKEN_MIN_LEN
    }


def _entity_ids_from_marker(marker: Dict) -> Set[str]:
    return {
        mention.get("entity_id")
        for mention in (marker.get("entity_mentions") or [])
        if mention.get("entity_id")
    }


def _alias_hits_from_marker(marker: Dict) -> List[Dict]:
    return [
        {
            "entity_id": mention.get("entity_id"),
            "surface_form": mention.get("surface_form"),
            "canonical_name": mention.get("canonical_name"),
        }
        for mention in (marker.get("entity_mentions") or [])
        if mention.get("source") == "curated_pack" and mention.get("entity_id")
    ]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _signals_between(
    prev: Dict,
    curr: Dict,
    gap_ms: int,
) -> Tuple[List[str], int]:
    """Return (signals, score) for the boundary between prev and curr."""
    signals: List[str] = []

    if gap_ms <= TEMPORAL_TIGHT_MS:
        signals.append("temporal_tight")
    elif gap_ms <= TEMPORAL_LOOSE_MS:
        signals.append("temporal_loose")

    prev_speaker = prev.get("speaker")
    curr_speaker = curr.get("speaker")
    if prev_speaker and curr_speaker and prev_speaker == curr_speaker:
        signals.append("speaker")

    prev_lang = prev.get("language")
    curr_lang = curr.get("language")
    if prev_lang and curr_lang and prev_lang == curr_lang:
        signals.append("language")

    if _jaccard(prev["_tokens"], curr["_tokens"]) >= LEXICAL_JACCARD_THRESHOLD:
        signals.append("lexical")

    if prev["_entities"] & curr["_entities"]:
        signals.append("entity")

    if prev["_topic_candidates"] & curr["_topic_candidates"]:
        signals.append("topic")

    return signals, len(signals)


def _should_close_span(prev: Dict, curr: Dict) -> Tuple[bool, List[str], str]:
    """Decide whether the boundary between prev and curr closes the span.

    Returns (close, signals, reason).
    """
    gap_ms = max(0, curr.get("start_ms", 0) - prev.get("end_ms", 0))
    signals, score = _signals_between(prev, curr, gap_ms)

    if gap_ms > TEMPORAL_HARD_BREAK_MS:
        return True, signals, "hard_temporal_break"

    has_topic_overlap = "topic" in signals
    has_entity_overlap = "entity" in signals

    # Strong topic shift: gap > loose AND no semantic glue at all.
    if gap_ms > TEMPORAL_LOOSE_MS and not has_topic_overlap and not has_entity_overlap:
        return True, signals, "strong_topic_shift"

    if gap_ms <= TEMPORAL_TIGHT_MS:
        # Still in the tight zone -- only close if the topic_candidates set
        # changed completely AND we have no entity / lexical glue.
        if (
            prev["_topic_candidates"]
            and curr["_topic_candidates"]
            and not has_topic_overlap
            and not has_entity_overlap
            and "lexical" not in signals
        ):
            return True, signals, "intra_turn_topic_pivot"
        return False, signals, "continuity_within_turn"

    if gap_ms <= TEMPORAL_LOOSE_MS:
        if score < MIN_CONTINUITY_SCORE_LOOSE_GAP:
            return True, signals, "weak_continuity_loose_gap"
        return False, signals, "continuity_loose_gap"

    # Beyond loose, only close if we don't have enough semantic glue.
    if score < MIN_CONTINUITY_SCORE_FAR_GAP:
        return True, signals, "weak_continuity_far_gap"
    return False, signals, "semantic_continuity_far_gap"


def _enriched_segment(segment: Dict, marker: Optional[Dict]) -> Dict:
    """Pre-compute everything continuity scoring needs, once per segment."""
    text = (segment.get("text") or "").strip()
    return {
        "segment_id": segment.get("segment_id"),
        "start_ms": int(segment.get("start_ms") or 0),
        "end_ms": int(segment.get("end_ms") or 0),
        "speaker": segment.get("speaker"),
        "language": segment.get("language"),
        "_tokens": _content_tokens(text),
        "_entities": _entity_ids_from_marker(marker or {}),
        "_topic_candidates": set((marker or {}).get("topic_candidates") or []),
        "_topic_tags": set((marker or {}).get("topic_tags") or []),
        "_alias_hits": _alias_hits_from_marker(marker or {}),
        "_marker": marker or {},
    }


def _language_profile(span_segments: List[Dict]) -> Dict:
    languages = [seg["language"] for seg in span_segments if seg.get("language")]
    if not languages:
        return {"primary": None, "ratio": 0.0, "languages": {}}
    counter = Counter(languages)
    primary, primary_count = counter.most_common(1)[0]
    total = sum(counter.values())
    return {
        "primary": primary,
        "ratio": round(primary_count / total, 3) if total else 0.0,
        "languages": dict(counter),
    }


def _serialize_span(
    session_id: str,
    span_segments: List[Dict],
    boundary_signals: List[Dict],
) -> Dict:
    start_ms = span_segments[0]["start_ms"]
    end_ms = span_segments[-1]["end_ms"]

    speaker_ids = sorted({seg["speaker"] for seg in span_segments if seg.get("speaker")})
    entity_ids: Set[str] = set()
    topic_candidates: Counter = Counter()
    topic_tags: Counter = Counter()
    alias_hits: Dict[Tuple[str, str], Dict] = {}

    for seg in span_segments:
        entity_ids |= seg["_entities"]
        for cand in seg["_topic_candidates"]:
            topic_candidates[cand] += 1
        for tag in seg["_topic_tags"]:
            topic_tags[tag] += 1
        for hit in seg["_alias_hits"]:
            key = (hit.get("entity_id") or "", hit.get("surface_form") or "")
            alias_hits.setdefault(key, hit)

    # Confidence = blend of signal density and semantic richness.  Bounded
    # so an empty span never claims high confidence.
    semantic_signal_density = (
        bool(entity_ids) + bool(topic_candidates) + bool(topic_tags) + bool(alias_hits)
    ) / 4.0
    boundary_strength = (
        sum(len(b["signals"]) for b in boundary_signals)
        / max(1, len(boundary_signals) * 6)
        if boundary_signals
        else 0.5
    )
    confidence = round(min(0.95, 0.4 + 0.3 * semantic_signal_density + 0.3 * boundary_strength), 3)

    return {
        "context_id": f"ctx_{uuid.uuid4().hex[:10]}",
        "session_id": session_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": max(0, end_ms - start_ms),
        "segment_ids": [seg["segment_id"] for seg in span_segments],
        "segment_count": len(span_segments),
        "speaker_ids": speaker_ids,
        "language_profile": _language_profile(span_segments),
        "topic_tags": [tag for tag, _ in topic_tags.most_common()],
        "topic_candidates": [cand for cand, _ in topic_candidates.most_common()],
        "entity_ids": sorted(entity_ids),
        "alias_hits": list(alias_hits.values()),
        "continuity_evidence": boundary_signals,
        "confidence": confidence,
        "grounding": {
            "canonical_path": "canonical/canonical_segments.json",
            "markers_path": "enrichment/segment_markers.json",
        },
    }


def build_context_spans(
    session_id: str,
    segments: List[Dict],
    markers: List[Dict],
) -> Dict:
    """Compute context spans from canonical segments + segment_markers.

    Pure function (no IO): callers persist the returned payload themselves so
    this is testable and reusable.
    """
    markers_by_segment = {marker.get("segment_id"): marker for marker in markers or []}

    enriched: List[Dict] = []
    for segment in segments or []:
        if segment.get("stabilization_state") and segment.get("stabilization_state") != "stabilized":
            continue
        if segment.get("segment_quality_status") == "suppressed":
            continue
        enriched.append(
            _enriched_segment(segment, markers_by_segment.get(segment.get("segment_id")))
        )

    if not enriched:
        return {
            "contract_version": CONTRACT_VERSION,
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "span_count": 0,
            "spans": [],
        }

    spans_serialized: List[Dict] = []
    current: List[Dict] = [enriched[0]]
    current_boundaries: List[Dict] = []

    for prev, curr in zip(enriched, enriched[1:]):
        close, signals, reason = _should_close_span(prev, curr)
        if close:
            spans_serialized.append(_serialize_span(session_id, current, current_boundaries))
            current = [curr]
            current_boundaries = []
        else:
            current.append(curr)
            current_boundaries.append({
                "between": [prev["segment_id"], curr["segment_id"]],
                "gap_ms": max(0, curr["start_ms"] - prev["end_ms"]),
                "signals": signals,
                "reason": reason,
            })

    if current:
        spans_serialized.append(_serialize_span(session_id, current, current_boundaries))

    return {
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "span_count": len(spans_serialized),
        "spans": spans_serialized,
    }


def empty_payload(session_id: str, reason: str, gate_reasons: Optional[List[str]] = None) -> Dict:
    """Annotated empty payload for the gate-suppressed path."""
    return {
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "span_count": 0,
        "spans": [],
        "gate_status": reason,
        "gate_reasons": gate_reasons or [],
    }


def write_context_spans(session_dir_path: Path, payload: Dict) -> Path:
    """Write the payload to enrichment/context_spans.json atomically."""
    enrichment_dir = session_dir_path / "enrichment"
    enrichment_dir.mkdir(parents=True, exist_ok=True)
    out_path = enrichment_dir / "context_spans.json"
    atomic_write_json(str(out_path), payload)
    return out_path
