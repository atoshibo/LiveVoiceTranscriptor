"""
Stage 11 - Memory graph update (canonical spec section 9.11).

This stage is downstream of canonical lexical truth AND semantic marking.
It maintains the user-level structured memory model:

  memory/
    entity_registry.json      -- canonical entity identities + aliases
    context_packs/*.json      -- versioned curated context packs
    graph_updates.json        -- append-only audit of graph changes

Hard rules:
  - Memory layer never rewrites canonical transcript text.
  - Every graph update is versioned and confidence-scored.
  - Inferred entities are separated from curated entities.
  - Graph updates are reversible (append-only log with `status` fields).

The implementation here is deliberately conservative: we build the minimum
auditable memory artifacts the canonical specification requires, seeded
from semantic markers.  A richer alias-resolution engine and external
context-pack import loop can be plugged in later without changing the on-disk
contract.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.core.atomic_io import atomic_write_json, safe_read_json
from app.storage.session_store import session_dir

ENTITY_REGISTRY_VERSION = "1.0"
CONTEXT_PACK_VERSION = 1
GRAPH_CONTRACT_VERSION = "1.0"

_SURFACE_ENTITY_ID_RE = re.compile(r"[^\w]+", re.UNICODE)


def _slug(text: str) -> str:
    slug = _SURFACE_ENTITY_ID_RE.sub("_", (text or "").strip().lower())
    return slug.strip("_") or "anon"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_entity_type(mention: Dict) -> str:
    mention_type = mention.get("mention_type", "")
    if mention_type == "email":
        return "contact_email"
    if mention_type == "url":
        return "reference_url"
    if mention_type == "handle":
        return "contact_handle"
    if mention_type == "acronym":
        return "acronym_or_project"
    if mention_type == "capitalized_phrase":
        return "person_or_entity"
    return "unknown"


def _load_curated_packs(sd: Path) -> List[Dict]:
    curated_dir = sd / "memory" / "curated_packs"
    if not curated_dir.is_dir():
        return []
    packs: List[Dict] = []
    for path in sorted(curated_dir.glob("*.json")):
        data = safe_read_json(str(path))
        if isinstance(data, dict):
            packs.append(data)
    return packs


def _seed_inferred_entities(markers: List[Dict]) -> Dict[str, Dict]:
    entities: Dict[str, Dict] = {}
    for marker in markers:
        segment_id = marker.get("segment_id")
        for mention in marker.get("entity_mentions", []) or []:
            surface = (mention.get("surface_form") or "").strip()
            if not surface:
                continue
            entity_id = mention.get("entity_id") or f"ent_{_slug(surface)}"
            confidence = float(mention.get("confidence") or 0.5)
            entity_type = _infer_entity_type(mention)
            entity = entities.setdefault(
                entity_id,
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "display_name": surface,
                    "aliases": [],
                    "origin": "inferred_from_markers",
                    "status": "candidate",
                    "confidence": confidence,
                    "first_seen_segment": segment_id,
                    "last_seen_segment": segment_id,
                    "mention_count": 0,
                    "updated_at": _utc_now(),
                },
            )
            if surface and surface not in entity["aliases"]:
                entity["aliases"].append(surface)
            entity["mention_count"] += 1
            entity["last_seen_segment"] = segment_id or entity["last_seen_segment"]
            # Soft-raise confidence as we see more mentions, capped so inferred
            # entities can never appear more confident than curated ones.
            entity["confidence"] = round(min(0.7, max(entity["confidence"], 0.3 + 0.05 * entity["mention_count"])), 3)
    return entities


def _merge_curated_entities(
    inferred: Dict[str, Dict],
    curated_packs: List[Dict],
) -> Dict[str, Dict]:
    merged: Dict[str, Dict] = dict(inferred)
    for pack in curated_packs:
        for entity in pack.get("entities", []) or []:
            entity_id = entity.get("entity_id") or f"ent_{_slug(entity.get('display_name', ''))}"
            if not entity_id:
                continue
            previous = merged.get(entity_id, {})
            aliases = list(dict.fromkeys([*(previous.get("aliases") or []), *(entity.get("aliases") or [])]))
            merged[entity_id] = {
                **previous,
                "entity_id": entity_id,
                "entity_type": entity.get("entity_type") or previous.get("entity_type") or "unknown",
                "display_name": entity.get("display_name") or previous.get("display_name") or entity_id,
                "aliases": aliases,
                "roles": entity.get("roles") or previous.get("roles") or [],
                "origin": "curated_pack",
                "status": entity.get("status") or "active",
                "confidence": float(entity.get("confidence") or 0.9),
                "pack_id": pack.get("pack_id"),
                "updated_at": _utc_now(),
            }
    return merged


def _build_alias_graph(entities: Dict[str, Dict]) -> Dict:
    alias_to_entity: Dict[str, List[str]] = {}
    for entity in entities.values():
        for alias in entity.get("aliases", []) or []:
            key = alias.strip().lower()
            if not key:
                continue
            alias_to_entity.setdefault(key, []).append(entity["entity_id"])
    return {
        "contract_version": GRAPH_CONTRACT_VERSION,
        "aliases": [
            {
                "alias": alias,
                "entity_ids": sorted(set(entity_ids)),
                "ambiguous": len(set(entity_ids)) > 1,
            }
            for alias, entity_ids in sorted(alias_to_entity.items())
        ],
    }


def _graph_updates_from_entities(
    session_id: str,
    entities: Dict[str, Dict],
    markers: List[Dict],
) -> Dict:
    updates = []
    for entity in entities.values():
        updates.append(
            {
                "update_id": f"upd_{uuid.uuid4().hex[:10]}",
                "session_id": session_id,
                "timestamp": _utc_now(),
                "entity_id": entity["entity_id"],
                "kind": "entity_upsert",
                "origin": entity.get("origin"),
                "confidence": entity.get("confidence"),
                "status": entity.get("status"),
                "reversible": True,
                "supporting_segments": [
                    marker.get("segment_id")
                    for marker in markers
                    if any(
                        (mention.get("entity_id") == entity["entity_id"])
                        or (mention.get("surface_form", "").lower() in {a.lower() for a in entity.get("aliases", [])})
                        for mention in (marker.get("entity_mentions") or [])
                    )
                ],
            }
        )
    return {
        "contract_version": GRAPH_CONTRACT_VERSION,
        "session_id": session_id,
        "generated_at": _utc_now(),
        "update_count": len(updates),
        "updates": updates,
    }


def _build_graph_update_proposals(
    session_id: str,
    context_links_payload: Dict,
    entities: Dict[str, Dict],
) -> Dict:
    links = context_links_payload.get("links") or []
    proposals = []

    resolved_alias_groups: Dict[tuple, List[Dict]] = {}
    context_entity_groups: Dict[tuple, List[Dict]] = {}
    unresolved_groups: Dict[tuple, List[Dict]] = {}

    for link in links:
        memory_kind = link.get("memory_kind")
        context_id = link.get("context_id")
        entity_id = link.get("entity_id")
        alias_surface = (link.get("alias_surface") or "").strip()
        if memory_kind == "resolved_alias_to_known_entity" and context_id and entity_id:
            resolved_alias_groups.setdefault(
                (context_id, entity_id, alias_surface.lower()),
                [],
            ).append(link)
            context_entity_groups.setdefault((context_id, entity_id), []).append(link)
        elif memory_kind == "context_local_unresolved_mention":
            unresolved_groups.setdefault(
                (context_id, alias_surface.lower()),
                [],
            ).append(link)

    for (context_id, entity_id), group in sorted(context_entity_groups.items()):
        entity = entities.get(entity_id, {})
        proposals.append({
            "proposal_id": f"prop_{uuid.uuid4().hex[:10]}",
            "session_id": session_id,
            "kind": "context_entity_link",
            "status": "proposed",
            "reversible": True,
            "context_id": context_id,
            "entity_id": entity_id,
            "canonical_name": entity.get("display_name") or entity.get("canonical_name"),
            "confidence": round(max(float(link.get("confidence") or 0.0) for link in group), 3),
            "alias_surfaces": sorted({link.get("alias_surface") for link in group if link.get("alias_surface")}),
            "mention_texts": sorted({link.get("mention_text") for link in group if link.get("mention_text")}),
            "supporting_segments": sorted({link.get("segment_id") for link in group if link.get("segment_id")}),
            "supporting_links": [link.get("link_id") for link in group if link.get("link_id")],
            "support_texts": [link.get("support_text") for link in group if link.get("support_text")][:3],
            "rationale": "resolved alias observed in context span",
        })

    for (context_id, entity_id, alias_key), group in sorted(resolved_alias_groups.items()):
        entity = entities.get(entity_id, {})
        proposals.append({
            "proposal_id": f"prop_{uuid.uuid4().hex[:10]}",
            "session_id": session_id,
            "kind": "alias_resolution_observation",
            "status": "proposed",
            "reversible": True,
            "context_id": context_id,
            "entity_id": entity_id,
            "canonical_name": entity.get("display_name") or entity.get("canonical_name"),
            "alias_surface": group[0].get("alias_surface"),
            "mention_texts": sorted({link.get("mention_text") for link in group if link.get("mention_text")}),
            "confidence": round(max(float(link.get("confidence") or 0.0) for link in group), 3),
            "supporting_segments": sorted({link.get("segment_id") for link in group if link.get("segment_id")}),
            "supporting_links": [link.get("link_id") for link in group if link.get("link_id")],
            "support_texts": [link.get("support_text") for link in group if link.get("support_text")][:3],
            "rationale": "alias surface resolved to known entity",
        })

    for (_context_id, alias_key), group in sorted(unresolved_groups.items()):
        alias_surface = group[0].get("alias_surface") or group[0].get("mention_text")
        proposals.append({
            "proposal_id": f"prop_{uuid.uuid4().hex[:10]}",
            "session_id": session_id,
            "kind": "unresolved_context_alias_candidate",
            "status": "proposed",
            "reversible": True,
            "context_id": group[0].get("context_id"),
            "entity_id": None,
            "proposed_entity_id": f"ent_{_slug(alias_surface or alias_key)}" if (alias_surface or alias_key) else None,
            "alias_surface": alias_surface,
            "mention_texts": sorted({link.get("mention_text") for link in group if link.get("mention_text")}),
            "confidence": round(max(float(link.get("confidence") or 0.0) for link in group), 3),
            "supporting_segments": sorted({link.get("segment_id") for link in group if link.get("segment_id")}),
            "supporting_links": [link.get("link_id") for link in group if link.get("link_id")],
            "support_texts": [link.get("support_text") for link in group if link.get("support_text")][:3],
            "rationale": "unresolved mention observed in local context span",
        })

    return {
        "contract_version": GRAPH_CONTRACT_VERSION,
        "session_id": session_id,
        "generated_at": _utc_now(),
        "proposal_count": len(proposals),
        "proposals": proposals,
        "source": "context_links",
    }


def _auto_context_pack(
    session_id: str,
    markers: List[Dict],
    entities: Dict[str, Dict],
    context_spans_payload: Optional[Dict] = None,
) -> Dict:
    topics: Dict[str, int] = {}
    topic_candidates: Dict[str, int] = {}
    projects: Dict[str, int] = {}
    for marker in markers:
        for tag in marker.get("topic_tags") or []:
            topics[tag] = topics.get(tag, 0) + 1
        for cand in marker.get("topic_candidates") or []:
            topic_candidates[cand] = topic_candidates.get(cand, 0) + 1
        for tag in marker.get("project_tags") or []:
            projects[tag] = projects.get(tag, 0) + 1

    context_summaries = []
    for span in (context_spans_payload or {}).get("spans") or []:
        context_summaries.append({
            "context_id": span.get("context_id"),
            "start_ms": span.get("start_ms"),
            "end_ms": span.get("end_ms"),
            "segment_count": span.get("segment_count"),
            "speaker_ids": list(span.get("speaker_ids") or []),
            "language_profile": span.get("language_profile") or {},
            "topic_tags": list(span.get("topic_tags") or []),
            "topic_candidates": list(span.get("topic_candidates") or []),
            "entity_ids": list(span.get("entity_ids") or []),
            "alias_hits": list(span.get("alias_hits") or []),
            "confidence": span.get("confidence"),
        })

    return {
        "pack_id": "session_auto_context",
        "version": CONTEXT_PACK_VERSION,
        "scope": "session_local",
        "origin": "auto_generated",
        "curator_type": "assistant",
        "session_id": session_id,
        "generated_at": _utc_now(),
        "entities": [
            {
                "entity_id": e["entity_id"],
                "display_name": e.get("display_name"),
                "aliases": list(e.get("aliases") or []),
                "entity_type": e.get("entity_type"),
                "confidence": e.get("confidence"),
                "origin": e.get("origin"),
            }
            for e in entities.values()
        ],
        "topic_frequency": topics,
        "topic_candidate_frequency": topic_candidates,
        "project_frequency": projects,
        "context_span_count": len(context_summaries),
        "context_summaries": context_summaries,
        "negative_rules": [],
    }


def run_memory_graph_update(
    session_id: str,
    markers: List[Dict],
    stage,
) -> Dict:
    """Persist entity registry, alias graph, curated packs index and graph updates.

    This stage is safe when no markers exist — it emits the empty contract
    instead of skipping, so downstream retrieval surfaces can always point at
    a real memory layer file.
    """
    sd = session_dir(session_id)
    memory_dir = sd / "memory"
    context_packs_dir = memory_dir / "context_packs"
    memory_dir.mkdir(parents=True, exist_ok=True)
    context_packs_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0 firewall: do not let a degraded session pollute the global
    # memory graph.  We still write every artifact (empty, annotated) so
    # audit never finds missing files -- we just refuse to commit inferred
    # entities or graph updates.
    from app.pipeline.canonical_assembly import read_quality_gate
    gate = read_quality_gate(session_id)
    if not gate.get("memory_update_eligible", True):
        empty_registry = {
            "contract_version": ENTITY_REGISTRY_VERSION,
            "session_id": session_id,
            "generated_at": _utc_now(),
            "entity_count": 0,
            "entities": [],
            "gate_status": "suppressed_by_quality_gate",
            "gate_reasons": gate.get("reasons") or [],
        }
        empty_alias_graph = {
            "contract_version": GRAPH_CONTRACT_VERSION,
            "aliases": [],
            "gate_status": "suppressed_by_quality_gate",
        }
        empty_updates = {
            "contract_version": GRAPH_CONTRACT_VERSION,
            "session_id": session_id,
            "generated_at": _utc_now(),
            "update_count": 0,
            "updates": [],
            "gate_status": "suppressed_by_quality_gate",
            "gate_reasons": gate.get("reasons") or [],
        }
        empty_proposals = {
            "contract_version": GRAPH_CONTRACT_VERSION,
            "session_id": session_id,
            "generated_at": _utc_now(),
            "proposal_count": 0,
            "proposals": [],
            "gate_status": "suppressed_by_quality_gate",
            "gate_reasons": gate.get("reasons") or [],
        }
        atomic_write_json(str(memory_dir / "entity_registry.json"), empty_registry)
        atomic_write_json(str(memory_dir / "alias_graph.json"), empty_alias_graph)
        atomic_write_json(str(memory_dir / "graph_updates.json"), empty_updates)
        atomic_write_json(str(memory_dir / "graph_update_proposals.json"), empty_proposals)
        atomic_write_json(str(memory_dir / "context_pack_summary.json"), {
            "contract_version": GRAPH_CONTRACT_VERSION,
            "session_id": session_id,
            "generated_at": _utc_now(),
            "pack_ids": [],
            "curated_count": 0,
            "auto_pack_id": None,
            "gate_status": "suppressed_by_quality_gate",
        })
        stage.commit([
            "entity_registry.json",
            "alias_graph.json",
            "graph_updates.json",
            "graph_update_proposals.json",
            "context_pack_summary.json",
        ])
        return {
            "entity_count": 0,
            "inferred_entity_count": 0,
            "curated_pack_count": 0,
            "alias_count": 0,
            "graph_update_count": 0,
            "proposal_count": 0,
            "gate_status": "suppressed_by_quality_gate",
            "gate_reasons": gate.get("reasons") or [],
        }

    curated_packs = _load_curated_packs(sd)
    inferred_entities = _seed_inferred_entities(markers)
    entities = _merge_curated_entities(inferred_entities, curated_packs)
    alias_graph = _build_alias_graph(entities)
    graph_updates = _graph_updates_from_entities(session_id, entities, markers)
    context_spans_payload = safe_read_json(str(sd / "enrichment" / "context_spans.json")) or {}
    context_links_payload = safe_read_json(str(sd / "enrichment" / "context_links.json")) or {}
    graph_update_proposals = _build_graph_update_proposals(session_id, context_links_payload, entities)
    auto_pack = _auto_context_pack(session_id, markers, entities, context_spans_payload=context_spans_payload)

    registry = {
        "contract_version": ENTITY_REGISTRY_VERSION,
        "session_id": session_id,
        "generated_at": _utc_now(),
        "entity_count": len(entities),
        "entities": sorted(entities.values(), key=lambda e: e["entity_id"]),
    }

    atomic_write_json(str(memory_dir / "entity_registry.json"), registry)
    atomic_write_json(str(memory_dir / "alias_graph.json"), alias_graph)
    atomic_write_json(str(memory_dir / "graph_updates.json"), graph_updates)
    atomic_write_json(str(memory_dir / "graph_update_proposals.json"), graph_update_proposals)
    atomic_write_json(str(context_packs_dir / "session_auto_context.json"), auto_pack)

    # Re-expose curated packs into the canonical layout so downstream retrieval
    # can load them without reading user-specific curated_packs/ directly.
    for pack in curated_packs:
        pack_id = pack.get("pack_id") or f"pack_{uuid.uuid4().hex[:8]}"
        atomic_write_json(str(context_packs_dir / f"{pack_id}.json"), {
            **pack,
            "origin": pack.get("origin", "curated"),
            "imported_at": _utc_now(),
        })

    pack_summary = {
        "contract_version": GRAPH_CONTRACT_VERSION,
        "session_id": session_id,
        "generated_at": _utc_now(),
        "pack_ids": sorted({path.stem for path in context_packs_dir.glob("*.json")}),
        "curated_count": len(curated_packs),
        "auto_pack_id": auto_pack["pack_id"],
    }
    atomic_write_json(str(memory_dir / "context_pack_summary.json"), pack_summary)

    result = {
        "entity_count": len(entities),
        "inferred_entity_count": len(inferred_entities),
        "curated_pack_count": len(curated_packs),
        "alias_count": len(alias_graph["aliases"]),
        "graph_update_count": graph_updates["update_count"],
        "proposal_count": graph_update_proposals["proposal_count"],
    }

    stage.commit([
        "entity_registry.json",
        "alias_graph.json",
        "graph_updates.json",
        "graph_update_proposals.json",
        "context_pack_summary.json",
        "context_packs/session_auto_context.json",
    ])
    return result
