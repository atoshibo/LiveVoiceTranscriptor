"""
Stage 14 - NoSQL Projection.

Transforms the session artifact tree into stable, collection-oriented NoSQL
documents ready for database ingestion.  The projection is *read-only* with
respect to all upstream artifacts -- it never rewrites canonical, enrichment,
or memory data.

Output: ``derived/nosql_projection.json``

Target collections:

  sessions          -- one doc per session
  segments          -- one doc per canonical segment
  speaker_turns     -- one doc per diarized speaker turn (from canonical/speaker_turns.json)
  context_spans     -- one doc per context span
  entities          -- one doc per entity
  aliases           -- one doc per alias resolution
  context_entity_links -- one doc per entity-context binding
  retrieval_docs    -- one doc per retrievable context block

Design rules:
  - retrieval_doc points first to context_span, then to underlying segment_ids
  - Every doc carries session_id + generated_at for cross-session traceability
  - IDs are deterministic (derived from existing artifact IDs, not random)
  - Quality-gate-suppressed sessions emit annotated empty collections via
    read_quality_gate() -- never raw JSON loads
  - speaker_turns are read from canonical/speaker_turns.json (real diarization);
    when absent or diarization was skipped, zero turn docs are emitted with
    explicit diarization_status rather than fake None-speaker turns
  - segment_count and word_count are computed directly from canonical segments,
    not from v2_session.json metadata (which may not be written yet)
  - Thread linking is a separate contract (derived/thread_candidates.json);
    this stage does not claim a threads collection
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.core.atomic_io import atomic_write_json, safe_read_json
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "1.1"

# All collection names this stage produces.
COLLECTION_NAMES = [
    "sessions", "segments", "speaker_turns", "context_spans",
    "entities", "aliases", "context_entity_links", "retrieval_docs",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_artifact(sd: Path, *parts: str) -> Dict:
    return safe_read_json(str(sd.joinpath(*parts))) or {}


def _project_session(
    session_id: str,
    meta: Dict,
    gate: Dict,
    segments: List[Dict],
) -> Dict:
    """Project the session-level document.

    segment_count and word_count are computed from canonical segments directly
    so we never depend on the metadata write that happens after the pipeline
    finishes.
    """
    segment_count = len(segments)
    word_count = sum(len((s.get("text") or "").split()) for s in segments)

    return {
        "_collection": "sessions",
        "_id": session_id,
        "session_id": session_id,
        "state": meta.get("state"),
        "device_id": meta.get("device_id"),
        "created_at": meta.get("created_at"),
        "finalize_requested_at": meta.get("finalize_requested_at"),
        "sample_rate_hz": meta.get("sample_rate_hz"),
        "channels": meta.get("channels"),
        "diarization_policy": meta.get("diarization_policy"),
        "allowed_languages": meta.get("allowed_languages") or [],
        "forced_language": meta.get("forced_language"),
        "transcription_mode": meta.get("transcription_mode"),
        "audio_duration_s": meta.get("audio_duration_s"),
        "segment_count": segment_count,
        "word_count": word_count,
        "quality_gate": {
            "semantic_eligible": gate.get("semantic_eligible", True),
            "memory_update_eligible": gate.get("memory_update_eligible", True),
            "reasons": gate.get("reasons") or [],
        },
        "generated_at": _utc_now(),
    }


def _project_segments(session_id: str, segments: List[Dict]) -> List[Dict]:
    """Project canonical segments into NoSQL docs."""
    docs = []
    for seg in segments:
        seg_id = seg.get("segment_id")
        if not seg_id:
            continue
        docs.append({
            "_collection": "segments",
            "_id": f"{session_id}:{seg_id}",
            "session_id": session_id,
            "segment_id": seg_id,
            "start_ms": seg.get("start_ms"),
            "end_ms": seg.get("end_ms"),
            "duration_ms": max(0, (seg.get("end_ms") or 0) - (seg.get("start_ms") or 0)),
            "text": seg.get("text", ""),
            "language": seg.get("language"),
            "confidence": seg.get("confidence"),
            "speaker": seg.get("speaker"),
            "stabilization_state": seg.get("stabilization_state"),
            "segment_quality_status": seg.get("segment_quality_status"),
            "corruption_flags": seg.get("corruption_flags") or [],
            "source_model": seg.get("source_model"),
            "support_models": seg.get("support_models") or [],
            "support_windows": seg.get("support_windows") or [],
            "generated_at": _utc_now(),
        })
    return docs


def _project_speaker_turns(
    session_id: str,
    speaker_turns_payload: Dict,
    diarization_status_payload: Dict,
) -> tuple[List[Dict], Dict]:
    """Project speaker turns from canonical/speaker_turns.json.

    Returns (docs, status) where status carries the diarization provenance.
    When speaker_turns.json is absent or diarization was skipped, returns
    zero docs with explicit status -- never synthesizes fake turns.
    """
    turns = speaker_turns_payload.get("turns") or []
    diar_status = diarization_status_payload.get("status") or "unknown"
    diar_reason = diarization_status_payload.get("reason") or "unknown"
    diar_requested = diarization_status_payload.get("requested", False)

    status = {
        "diarization_status": diar_status,
        "diarization_reason": diar_reason,
        "diarization_requested": diar_requested,
    }

    if not turns:
        return [], status

    docs = []
    for i, turn in enumerate(turns):
        turn_id = f"turn_{i:06d}"
        speaker = turn.get("speaker")
        start = turn.get("start")
        end = turn.get("end")
        docs.append({
            "_collection": "speaker_turns",
            "_id": f"{session_id}:{turn_id}",
            "session_id": session_id,
            "turn_id": turn_id,
            "turn_index": i,
            "speaker": speaker,
            "start_s": start,
            "end_s": end,
            "start_ms": int(start * 1000) if start is not None else None,
            "end_ms": int(end * 1000) if end is not None else None,
            "generated_at": _utc_now(),
        })
    return docs, status


def _project_context_spans(session_id: str, spans_payload: Dict) -> List[Dict]:
    """Project context spans into NoSQL docs."""
    docs = []
    for span in spans_payload.get("spans") or []:
        ctx_id = span.get("context_id")
        if not ctx_id:
            continue
        docs.append({
            "_collection": "context_spans",
            "_id": f"{session_id}:{ctx_id}",
            "session_id": session_id,
            "context_id": ctx_id,
            "start_ms": span.get("start_ms"),
            "end_ms": span.get("end_ms"),
            "duration_ms": span.get("duration_ms") or max(0, (span.get("end_ms") or 0) - (span.get("start_ms") or 0)),
            "segment_ids": span.get("segment_ids") or [],
            "segment_count": span.get("segment_count") or len(span.get("segment_ids") or []),
            "speaker_ids": span.get("speaker_ids") or [],
            "language_profile": span.get("language_profile") or {},
            "topic_tags": span.get("topic_tags") or [],
            "topic_candidates": span.get("topic_candidates") or [],
            "entity_ids": span.get("entity_ids") or [],
            "alias_hits": span.get("alias_hits") or [],
            "confidence": span.get("confidence"),
            "continuity_evidence": span.get("continuity_evidence") or [],
            "generated_at": _utc_now(),
        })
    return docs


def _project_entities(session_id: str, registry: Dict) -> List[Dict]:
    """Project entity registry into NoSQL docs."""
    docs = []
    for entity in registry.get("entities") or []:
        eid = entity.get("entity_id")
        if not eid:
            continue
        docs.append({
            "_collection": "entities",
            "_id": f"{session_id}:{eid}",
            "session_id": session_id,
            "entity_id": eid,
            "entity_type": entity.get("entity_type"),
            "display_name": entity.get("display_name"),
            "aliases": entity.get("aliases") or [],
            "roles": entity.get("roles") or [],
            "origin": entity.get("origin"),
            "status": entity.get("status"),
            "confidence": entity.get("confidence"),
            "pack_id": entity.get("pack_id"),
            "first_seen_segment": entity.get("first_seen_segment"),
            "last_seen_segment": entity.get("last_seen_segment"),
            "mention_count": entity.get("mention_count"),
            "generated_at": _utc_now(),
        })
    return docs


def _project_aliases(session_id: str, alias_graph: Dict) -> List[Dict]:
    """Project alias graph into NoSQL docs."""
    docs = []
    for entry in alias_graph.get("aliases") or []:
        alias = entry.get("alias")
        if not alias:
            continue
        entity_ids = entry.get("entity_ids") or []
        docs.append({
            "_collection": "aliases",
            "_id": f"{session_id}:alias:{alias}",
            "session_id": session_id,
            "alias": alias,
            "entity_ids": entity_ids,
            "ambiguous": entry.get("ambiguous", len(entity_ids) > 1),
            "generated_at": _utc_now(),
        })
    return docs


def _project_context_entity_links(
    session_id: str,
    spans_payload: Dict,
    proposals: Dict,
) -> List[Dict]:
    """Project entity-context bindings from context spans + graph proposals."""
    docs = []
    seen = set()

    # Direct links from context spans (entity_ids inside each span)
    for span in spans_payload.get("spans") or []:
        ctx_id = span.get("context_id")
        for eid in span.get("entity_ids") or []:
            key = (ctx_id, eid)
            if key in seen:
                continue
            seen.add(key)
            alias_surfaces = [
                hit.get("surface_form")
                for hit in (span.get("alias_hits") or [])
                if hit.get("entity_id") == eid and hit.get("surface_form")
            ]
            docs.append({
                "_collection": "context_entity_links",
                "_id": f"{session_id}:{ctx_id}:{eid}",
                "session_id": session_id,
                "context_id": ctx_id,
                "entity_id": eid,
                "source": "context_span_direct",
                "alias_surfaces": alias_surfaces,
                "confidence": span.get("confidence"),
                "generated_at": _utc_now(),
            })

    # Proposed links from memory graph proposals
    for prop in proposals.get("proposals") or []:
        if prop.get("kind") != "context_entity_link":
            continue
        ctx_id = prop.get("context_id")
        eid = prop.get("entity_id")
        if not ctx_id or not eid:
            continue
        key = (ctx_id, eid)
        if key in seen:
            continue
        seen.add(key)
        docs.append({
            "_collection": "context_entity_links",
            "_id": f"{session_id}:{ctx_id}:{eid}",
            "session_id": session_id,
            "context_id": ctx_id,
            "entity_id": eid,
            "source": "graph_proposal",
            "alias_surfaces": prop.get("alias_surfaces") or [],
            "confidence": prop.get("confidence"),
            "proposal_status": prop.get("status"),
            "generated_at": _utc_now(),
        })

    return docs


def _project_retrieval_docs(
    session_id: str,
    spans_payload: Dict,
    segments: List[Dict],
    marker_index: Dict[str, Dict],
) -> List[Dict]:
    """Project retrieval docs grounded on context_spans -> segment_ids.

    Rule: retrieval_doc points first to context_span, then to the
    canonical segment_ids underneath.
    """
    docs = []
    segments_by_id = {s.get("segment_id"): s for s in segments if s.get("segment_id")}

    for i, span in enumerate(spans_payload.get("spans") or []):
        ctx_id = span.get("context_id")
        seg_ids = span.get("segment_ids") or []
        span_segs = [segments_by_id[sid] for sid in seg_ids if sid in segments_by_id]

        text_parts = []
        all_entity_ids = set(span.get("entity_ids") or [])
        all_topic_tags = list(span.get("topic_tags") or [])
        all_topic_candidates = list(span.get("topic_candidates") or [])
        all_retrieval_terms = []

        for seg in span_segs:
            seg_text = (seg.get("text") or "").strip()
            if seg_text:
                text_parts.append(seg_text)
            marker = marker_index.get(seg.get("segment_id"), {})
            for mention in marker.get("entity_mentions") or []:
                eid = mention.get("entity_id")
                if eid:
                    all_entity_ids.add(eid)
            all_retrieval_terms.extend(marker.get("retrieval_terms") or [])

        context_text = " ".join(text_parts).strip()
        if not context_text:
            continue

        # Deduplicate retrieval_terms while preserving order
        seen_terms = set()
        unique_terms = []
        for term in all_retrieval_terms:
            if term and term not in seen_terms:
                seen_terms.add(term)
                unique_terms.append(term)

        docs.append({
            "_collection": "retrieval_docs",
            "_id": f"{session_id}:ret:{ctx_id}",
            "session_id": session_id,
            "context_id": ctx_id,
            "segment_ids": seg_ids,
            "text": context_text,
            "entity_ids": sorted(all_entity_ids),
            "topic_tags": all_topic_tags,
            "topic_candidates": all_topic_candidates,
            "retrieval_terms": unique_terms,
            "speaker_ids": span.get("speaker_ids") or [],
            "language_profile": span.get("language_profile") or {},
            "confidence": span.get("confidence"),
            "start_ms": span.get("start_ms"),
            "end_ms": span.get("end_ms"),
            "grounding": {
                "context_span": ctx_id,
                "segment_ids": seg_ids,
                "canonical_path": "canonical/canonical_segments.json",
                "context_spans_path": "enrichment/context_spans.json",
                "markers_path": "enrichment/segment_markers.json",
            },
            "generated_at": _utc_now(),
        })

    return docs


def build_nosql_projection(session_id: str, gate: Dict) -> Dict:
    """Build the full NoSQL projection from all session artifacts.

    The caller must pass the gate obtained via read_quality_gate().
    Pure assembly -- reads upstream artifacts, never writes upstream.
    Returns the projection payload for the caller to persist.
    """
    sd = session_dir(session_id)
    semantic_eligible = gate.get("semantic_eligible", True)

    meta = _load_artifact(sd, "v2_session.json")
    canon = _load_artifact(sd, "canonical", "canonical_segments.json")
    segments = canon.get("segments") or []
    spans_payload = _load_artifact(sd, "enrichment", "context_spans.json")
    marker_payload = _load_artifact(sd, "enrichment", "segment_markers.json")
    registry = _load_artifact(sd, "memory", "entity_registry.json")
    alias_graph = _load_artifact(sd, "memory", "alias_graph.json")
    proposals = _load_artifact(sd, "memory", "graph_update_proposals.json")

    # Real diarization artifacts
    speaker_turns_payload = _load_artifact(sd, "canonical", "speaker_turns.json")
    diarization_status_payload = _load_artifact(sd, "canonical", "diarization_status.json")

    marker_index = {
        m.get("segment_id"): m
        for m in (marker_payload.get("markers") or [])
        if m.get("segment_id")
    }

    turn_docs, diarization_meta = _project_speaker_turns(
        session_id, speaker_turns_payload, diarization_status_payload,
    )

    collections: Dict[str, List[Dict]] = {
        "sessions": [_project_session(session_id, meta, gate, segments)],
        "segments": _project_segments(session_id, segments),
        "speaker_turns": turn_docs,
        "context_spans": _project_context_spans(session_id, spans_payload) if semantic_eligible else [],
        "entities": _project_entities(session_id, registry) if semantic_eligible else [],
        "aliases": _project_aliases(session_id, alias_graph) if semantic_eligible else [],
        "context_entity_links": (
            _project_context_entity_links(session_id, spans_payload, proposals)
            if semantic_eligible else []
        ),
        "retrieval_docs": (
            _project_retrieval_docs(session_id, spans_payload, segments, marker_index)
            if semantic_eligible else []
        ),
    }

    total_docs = sum(len(docs) for docs in collections.values())

    return {
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "generated_at": _utc_now(),
        "semantic_eligible": semantic_eligible,
        "diarization_status": diarization_meta,
        "collection_counts": {name: len(docs) for name, docs in collections.items()},
        "total_doc_count": total_docs,
        "collections": collections,
    }


def empty_projection(session_id: str, reason: str, gate_reasons: Optional[List[str]] = None) -> Dict:
    """Annotated empty projection for gate-suppressed sessions."""
    empty_collections = {name: [] for name in COLLECTION_NAMES}
    return {
        "contract_version": CONTRACT_VERSION,
        "session_id": session_id,
        "generated_at": _utc_now(),
        "semantic_eligible": False,
        "gate_status": reason,
        "gate_reasons": gate_reasons or [],
        "collection_counts": {name: 0 for name in COLLECTION_NAMES},
        "total_doc_count": 0,
        "collections": empty_collections,
    }


def write_nosql_projection(session_dir_path: Path, payload: Dict) -> Path:
    """Persist the projection to derived/nosql_projection.json."""
    derived_dir = session_dir_path / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    out_path = derived_dir / "nosql_projection.json"
    atomic_write_json(str(out_path), payload)
    return out_path


def run_nosql_projection(session_id: str, stage) -> Dict:
    """Execute the NoSQL projection stage.

    Uses read_quality_gate() exclusively -- never reads quality_gate.json
    directly.  Suppressed / missing-after-canonical sessions get
    empty_projection().
    """
    sd = session_dir(session_id)

    from app.pipeline.canonical_assembly import read_quality_gate
    gate = read_quality_gate(session_id)

    if not gate.get("semantic_eligible", True):
        projection = empty_projection(
            session_id,
            gate.get("session_quality_status", "suppressed_by_quality_gate"),
            gate.get("reasons") or [],
        )
        write_nosql_projection(sd, projection)
        stage.commit(["nosql_projection.json"])
        return {
            "total_doc_count": 0,
            "collection_counts": projection["collection_counts"],
            "semantic_eligible": False,
            "gate_status": projection["gate_status"],
        }

    projection = build_nosql_projection(session_id, gate)
    write_nosql_projection(sd, projection)

    logger.info(
        "nosql_projection done for %s: %d total docs across %d collections",
        session_id,
        projection["total_doc_count"],
        len(projection["collections"]),
    )

    stage.commit(["nosql_projection.json"])
    return {
        "total_doc_count": projection["total_doc_count"],
        "collection_counts": projection["collection_counts"],
        "semantic_eligible": projection["semantic_eligible"],
    }
