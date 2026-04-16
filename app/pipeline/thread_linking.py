"""
Stage 15 - Thread Linking (cross-session).

Groups context_spans across sessions into threads based on:
  - shared entities (entity_ids in common)
  - similar topics (topic_candidates overlap)
  - recurring vocabulary (lexical Jaccard on retrieval_terms)
  - compatible language profiles
  - compatible relation tags
  - optional temporal proximity

A thread is a cluster of context_spans from one or more sessions that
collectively discuss the same topic/entity constellation.  Threads enable
cross-session retrieval: "find everything ever said about X".

Output: ``derived/thread_candidates.json`` (per session)

This is the authoritative thread contract.  Phase 5 (nosql_projection) does
NOT emit a threads collection -- downstream ingestion reads
thread_candidates.json directly.

Design rules:
  - Thread candidates are proposals, not facts.
  - Each candidate carries a similarity_score and match evidence.
  - The stage never mutates upstream artifacts.
  - Sessions with suppressed quality gates (via read_quality_gate()) produce
    empty candidates.
  - Target sessions are also filtered through read_quality_gate() so
    gate-suppressed sessions never pollute cross-session linking.
  - Both source and target spans are enriched with retrieval_terms from
    segment_markers before similarity scoring, so lexical evidence is
    based on real retrieval text on both sides.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.core.atomic_io import atomic_write_json, safe_read_json
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "1.1"

# Similarity thresholds for thread candidate generation
MIN_ENTITY_OVERLAP = 1          # at least 1 shared entity_id
MIN_TOPIC_OVERLAP = 1           # at least 1 shared topic_candidate
MIN_LEXICAL_JACCARD = 0.10      # vocabulary overlap threshold
MIN_THREAD_SCORE = 0.25         # minimum combined score to emit a candidate
MAX_CANDIDATES_PER_SPAN = 10    # cap candidates per source span

_CONTENT_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
TOKEN_MIN_LEN = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sessions_root() -> Path:
    from app.core.config import get_config
    return Path(get_config().storage.sessions_dir)


def _content_tokens(terms: List[str]) -> Set[str]:
    """Extract content tokens from retrieval terms."""
    tokens = set()
    for term in terms or []:
        for tok in _CONTENT_TOKEN_RE.findall(term):
            if len(tok) >= TOKEN_MIN_LEN:
                tokens.add(tok.lower())
    return tokens


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _language_compatible(profile_a: Dict, profile_b: Dict) -> bool:
    """Check if two language profiles are compatible (share primary language)."""
    lang_a = profile_a.get("primary")
    lang_b = profile_b.get("primary")
    if not lang_a or not lang_b:
        return True  # Unknown language is compatible with anything
    return lang_a == lang_b


def _retrieval_terms_for_span(
    span: Dict,
    marker_index: Dict[str, Dict],
) -> List[str]:
    """Collect retrieval_terms from segment_markers for all segments in a span.

    This produces the same vocabulary the retrieval_index_v3 and
    nosql_projection retrieval_docs use, so lexical evidence in thread
    linking is based on real retrieval text rather than just the sparse
    topic_tags / topic_candidates that context_spans carry natively.
    """
    terms: List[str] = []
    seen: Set[str] = set()
    for seg_id in span.get("segment_ids") or []:
        marker = marker_index.get(seg_id, {})
        for term in marker.get("retrieval_terms") or []:
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def _enrich_spans_with_retrieval_terms(
    spans: List[Dict],
    marker_index: Dict[str, Dict],
) -> List[Dict]:
    """Stamp retrieval_terms onto each span from its segments' markers.

    Returns new dicts (does not mutate the originals).
    """
    enriched = []
    for span in spans:
        span_copy = dict(span)
        span_copy["retrieval_terms"] = _retrieval_terms_for_span(span, marker_index)
        enriched.append(span_copy)
    return enriched


def _extract_span_signature(span: Dict) -> Dict:
    """Extract the matching-relevant fields from a context span."""
    retrieval_terms = list(span.get("retrieval_terms") or [])
    return {
        "context_id": span.get("context_id"),
        "session_id": span.get("session_id"),
        "entity_ids": set(span.get("entity_ids") or []),
        "topic_candidates": set(span.get("topic_candidates") or []),
        "topic_tags": set(span.get("topic_tags") or []),
        "language_profile": span.get("language_profile") or {},
        "speaker_ids": span.get("speaker_ids") or [],
        "start_ms": span.get("start_ms"),
        "end_ms": span.get("end_ms"),
        "confidence": span.get("confidence"),
        "retrieval_terms": retrieval_terms,
        "_tokens": _content_tokens(
            retrieval_terms
            + list(span.get("topic_candidates") or [])
            + list(span.get("topic_tags") or [])
        ),
    }


def _compute_similarity(source: Dict, target: Dict) -> Tuple[float, Dict]:
    """Compute similarity between two span signatures.

    Returns (score, evidence) where score is in [0, 1].
    """
    evidence = {}
    scores = []

    # Entity overlap
    entity_overlap = source["entity_ids"] & target["entity_ids"]
    if entity_overlap:
        entity_score = min(1.0, len(entity_overlap) / max(1, min(len(source["entity_ids"]), len(target["entity_ids"]))))
        scores.append(("entity", entity_score, 0.35))
        evidence["shared_entities"] = sorted(entity_overlap)
    else:
        scores.append(("entity", 0.0, 0.35))

    # Topic candidate overlap
    topic_overlap = source["topic_candidates"] & target["topic_candidates"]
    if topic_overlap:
        topic_score = min(1.0, len(topic_overlap) / max(1, min(len(source["topic_candidates"]), len(target["topic_candidates"]))))
        scores.append(("topic", topic_score, 0.30))
        evidence["shared_topics"] = sorted(topic_overlap)
    else:
        scores.append(("topic", 0.0, 0.30))

    # Lexical overlap (retrieval terms / vocabulary)
    lexical_jaccard = _jaccard(source["_tokens"], target["_tokens"])
    scores.append(("lexical", min(1.0, lexical_jaccard / 0.3) if lexical_jaccard > 0 else 0.0, 0.20))
    if lexical_jaccard >= MIN_LEXICAL_JACCARD:
        evidence["lexical_jaccard"] = round(lexical_jaccard, 4)
        shared_tokens = source["_tokens"] & target["_tokens"]
        evidence["shared_vocabulary"] = sorted(list(shared_tokens)[:10])

    # Language compatibility
    lang_compat = _language_compatible(source["language_profile"], target["language_profile"])
    scores.append(("language", 1.0 if lang_compat else 0.0, 0.10))
    if lang_compat:
        evidence["language_compatible"] = True

    # Relation / topic_tags overlap (secondary signal)
    tag_overlap = source["topic_tags"] & target["topic_tags"]
    if tag_overlap:
        tag_score = min(1.0, len(tag_overlap) / max(1, min(len(source["topic_tags"]), len(target["topic_tags"]))))
        scores.append(("relation", tag_score, 0.05))
        evidence["shared_tags"] = sorted(tag_overlap)
    else:
        scores.append(("relation", 0.0, 0.05))

    # Weighted score
    total_score = sum(score * weight for _, score, weight in scores)
    total_score = round(min(1.0, total_score), 4)

    return total_score, evidence


def _load_other_session_spans(current_session_id: str) -> List[Dict]:
    """Load context spans from all other sessions for cross-session matching.

    Uses read_quality_gate() per session so gate-suppressed sessions are
    excluded with the same logic the pipeline uses, not raw JSON loads.
    Target spans are enriched with retrieval_terms from their segment_markers
    so lexical similarity is based on real retrieval text.
    """
    from app.pipeline.canonical_assembly import read_quality_gate

    sessions_root = _sessions_root()
    if not sessions_root.is_dir():
        return []

    all_spans = []
    for session_path in sessions_root.iterdir():
        if not session_path.is_dir():
            continue
        other_session_id = session_path.name
        if other_session_id == current_session_id:
            continue

        # Use read_quality_gate() -- handles missing gate, missing-after-
        # canonical, and explicit suppression identically to the pipeline.
        gate = read_quality_gate(other_session_id)
        if not gate.get("semantic_eligible", True):
            continue

        spans_payload = safe_read_json(str(session_path / "enrichment" / "context_spans.json")) or {}
        spans = spans_payload.get("spans") or []
        if not spans:
            continue

        # Build marker index for this target session so we can enrich its
        # spans with real retrieval_terms (same as we do for source spans).
        marker_payload = safe_read_json(str(session_path / "enrichment" / "segment_markers.json")) or {}
        target_marker_index = {
            m.get("segment_id"): m
            for m in (marker_payload.get("markers") or [])
            if m.get("segment_id")
        }

        for span in spans:
            span_copy = dict(span)
            span_copy["session_id"] = other_session_id
            span_copy["retrieval_terms"] = _retrieval_terms_for_span(span_copy, target_marker_index)
            all_spans.append(span_copy)

    return all_spans


def build_thread_candidates(
    session_id: str,
    source_spans: List[Dict],
    target_spans: List[Dict],
) -> Dict:
    """Find thread candidates by comparing source spans against target spans.

    Pure function -- no IO.  Both source and target spans must already carry
    retrieval_terms (enriched by the caller).

    Args:
        session_id: The current session being processed.
        source_spans: Context spans from the current session (enriched).
        target_spans: Context spans from other sessions (enriched).

    Returns:
        Thread candidates payload.
    """
    if not source_spans or not target_spans:
        return {
            "contract_version": CONTRACT_VERSION,
            "session_id": session_id,
            "generated_at": _utc_now(),
            "candidate_count": 0,
            "candidates": [],
            "source_span_count": len(source_spans),
            "target_session_count": len({s.get("session_id") for s in target_spans}),
        }

    source_sigs = [_extract_span_signature(s) for s in source_spans]
    target_sigs = [_extract_span_signature(s) for s in target_spans]

    candidates = []
    for src_sig in source_sigs:
        src_candidates = []
        for tgt_sig in target_sigs:
            if src_sig["context_id"] == tgt_sig["context_id"]:
                continue  # Same span, skip

            score, evidence = _compute_similarity(src_sig, tgt_sig)
            if score < MIN_THREAD_SCORE:
                continue

            # Must have at least one strong signal
            has_entity = bool(evidence.get("shared_entities"))
            has_topic = bool(evidence.get("shared_topics"))
            has_lexical = bool(evidence.get("shared_vocabulary"))
            if not (has_entity or has_topic or has_lexical):
                continue

            src_candidates.append({
                "source_context_id": src_sig["context_id"],
                "source_session_id": session_id,
                "target_context_id": tgt_sig["context_id"],
                "target_session_id": tgt_sig["session_id"],
                "similarity_score": score,
                "match_signals": [
                    k for k in ["shared_entities", "shared_topics", "shared_vocabulary",
                                "language_compatible", "shared_tags"]
                    if k in evidence
                ],
                "evidence": evidence,
            })

        # Cap per-source-span to avoid explosion
        src_candidates.sort(key=lambda c: c["similarity_score"], reverse=True)
        candidates.extend(src_candidates[:MAX_CANDIDATES_PER_SPAN])

    # Deduplicate and sort by score
    seen = set()
    unique_candidates = []
    for c in sorted(candidates, key=lambda x: x["similarity_score"], reverse=True):
        key = (c["source_context_id"], c["target_context_id"])
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(c)

    target_sessions = {s.get("session_id") for s in target_spans}

    return {
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "generated_at": _utc_now(),
        "candidate_count": len(unique_candidates),
        "candidates": unique_candidates,
        "source_span_count": len(source_spans),
        "target_session_count": len(target_sessions),
        "target_sessions": sorted(target_sessions - {None}),
    }


def _group_candidates_into_threads(candidates: List[Dict]) -> List[Dict]:
    """Group related candidates into thread clusters.

    Two candidates belong to the same thread if they share a source or
    target context and have overlapping entity/topic evidence.
    """
    if not candidates:
        return []

    # Build adjacency by shared entities/topics
    threads: List[Dict] = []
    assigned: Set[int] = set()

    for i, cand in enumerate(candidates):
        if i in assigned:
            continue

        thread_contexts = {
            (cand["source_context_id"], cand["source_session_id"]),
            (cand["target_context_id"], cand["target_session_id"]),
        }
        thread_entities = set(cand.get("evidence", {}).get("shared_entities") or [])
        thread_topics = set(cand.get("evidence", {}).get("shared_topics") or [])
        thread_candidates_idx = {i}
        assigned.add(i)

        # Greedily absorb related candidates
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(candidates):
                if j in assigned:
                    continue
                other_contexts = {
                    (other["source_context_id"], other["source_session_id"]),
                    (other["target_context_id"], other["target_session_id"]),
                }
                other_entities = set(other.get("evidence", {}).get("shared_entities") or [])
                other_topics = set(other.get("evidence", {}).get("shared_topics") or [])

                has_context_overlap = bool(thread_contexts & other_contexts)
                has_entity_overlap = bool(thread_entities & other_entities) if thread_entities and other_entities else False
                has_topic_overlap = bool(thread_topics & other_topics) if thread_topics and other_topics else False

                if has_context_overlap and (has_entity_overlap or has_topic_overlap):
                    thread_contexts |= other_contexts
                    thread_entities |= other_entities
                    thread_topics |= other_topics
                    thread_candidates_idx.add(j)
                    assigned.add(j)
                    changed = True

        thread_cands = [candidates[idx] for idx in sorted(thread_candidates_idx)]
        all_sessions = set()
        all_context_ids = []
        for tc in thread_cands:
            all_sessions.add(tc["source_session_id"])
            all_sessions.add(tc["target_session_id"])
            all_context_ids.append((tc["source_session_id"], tc["source_context_id"]))
            all_context_ids.append((tc["target_session_id"], tc["target_context_id"]))

        # Deduplicate context references
        seen_ctx = set()
        unique_contexts = []
        for sid, cid in all_context_ids:
            key = (sid, cid)
            if key not in seen_ctx:
                seen_ctx.add(key)
                unique_contexts.append({"session_id": sid, "context_id": cid})

        avg_score = sum(tc["similarity_score"] for tc in thread_cands) / len(thread_cands)

        threads.append({
            "thread_entity_ids": sorted(thread_entities),
            "thread_topic_candidates": sorted(thread_topics),
            "session_ids": sorted(all_sessions - {None}),
            "session_count": len(all_sessions - {None}),
            "context_refs": unique_contexts,
            "context_count": len(unique_contexts),
            "candidate_count": len(thread_cands),
            "avg_similarity": round(avg_score, 4),
            "max_similarity": round(max(tc["similarity_score"] for tc in thread_cands), 4),
        })

    threads.sort(key=lambda t: t["max_similarity"], reverse=True)
    return threads


def empty_candidates(session_id: str, reason: str, gate_reasons: Optional[List[str]] = None) -> Dict:
    """Annotated empty payload for gate-suppressed sessions."""
    return {
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "generated_at": _utc_now(),
        "candidate_count": 0,
        "candidates": [],
        "threads": [],
        "source_span_count": 0,
        "target_session_count": 0,
        "gate_status": reason,
        "gate_reasons": gate_reasons or [],
    }


def write_thread_candidates(session_dir_path: Path, payload: Dict) -> Path:
    """Persist thread candidates to derived/thread_candidates.json."""
    derived_dir = session_dir_path / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    out_path = derived_dir / "thread_candidates.json"
    atomic_write_json(str(out_path), payload)
    return out_path


def run_thread_linking(session_id: str, stage) -> Dict:
    """Execute the thread linking stage.

    Uses read_quality_gate() exclusively for both the current session
    and all target sessions.  Source spans are enriched with
    retrieval_terms from segment_markers before scoring.
    """
    sd = session_dir(session_id)

    from app.pipeline.canonical_assembly import read_quality_gate
    gate = read_quality_gate(session_id)

    if not gate.get("semantic_eligible", True):
        payload = empty_candidates(
            session_id,
            gate.get("session_quality_status", "suppressed_by_quality_gate"),
            gate.get("reasons") or [],
        )
        write_thread_candidates(sd, payload)
        stage.commit(["thread_candidates.json"])
        return {
            "candidate_count": 0,
            "thread_count": 0,
            "gate_status": payload["gate_status"],
        }

    # Load current session's context spans
    spans_payload = safe_read_json(str(sd / "enrichment" / "context_spans.json")) or {}
    source_spans = spans_payload.get("spans") or []

    if not source_spans:
        payload = empty_candidates(session_id, "no_source_spans")
        write_thread_candidates(sd, payload)
        stage.commit(["thread_candidates.json"])
        return {
            "candidate_count": 0,
            "thread_count": 0,
            "reason": "no_source_spans",
        }

    # Enrich source spans with retrieval_terms from this session's markers
    marker_payload = safe_read_json(str(sd / "enrichment" / "segment_markers.json")) or {}
    source_marker_index = {
        m.get("segment_id"): m
        for m in (marker_payload.get("markers") or [])
        if m.get("segment_id")
    }
    source_spans = _enrich_spans_with_retrieval_terms(source_spans, source_marker_index)

    # Load other sessions' context spans (already enriched inside the loader)
    target_spans = _load_other_session_spans(session_id)

    # Build raw candidates
    candidates_payload = build_thread_candidates(session_id, source_spans, target_spans)

    # Group into thread clusters
    threads = _group_candidates_into_threads(candidates_payload.get("candidates") or [])
    candidates_payload["threads"] = threads
    candidates_payload["thread_count"] = len(threads)

    write_thread_candidates(sd, candidates_payload)

    logger.info(
        "thread_linking done for %s: %d candidates, %d threads across %d target sessions",
        session_id,
        candidates_payload["candidate_count"],
        len(threads),
        candidates_payload.get("target_session_count", 0),
    )

    stage.commit(["thread_candidates.json"])
    return {
        "candidate_count": candidates_payload["candidate_count"],
        "thread_count": len(threads),
        "target_session_count": candidates_payload.get("target_session_count", 0),
    }
