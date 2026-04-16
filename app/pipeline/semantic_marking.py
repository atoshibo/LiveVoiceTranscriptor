"""
Stage 10 - Semantic marking.

This stage is downstream of canonical lexical truth and never rewrites it.
The MVP implementation stays conservative and heuristic-driven so every marker
remains grounded in the canonical segment text or nearby lexical context.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Tuple

from app.core.atomic_io import atomic_write_json, safe_read_json
from app.storage.session_store import session_dir

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,32}\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,8}(?:-[A-Z0-9]{1,8})?\b")
TITLE_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
PROJECT_REF_RE = re.compile(
    r"\b(?:project|repo|repository|ticket|issue|branch|pr)\s+([A-Za-z0-9._/-]{2,40})",
    re.IGNORECASE,
)
PRONOUN_ONLY_RE = re.compile(r"\b(he|she|they|him|her|them|il|elle|ils|elles|on|она|он|они)\b", re.IGNORECASE)
KEYWORD_RE = re.compile(r"\w+", re.UNICODE)

# Phase 3: business-domain taxonomy.  These produce `topic_candidates` --
# distinct from the generic heuristic `topic_tags` -- so retrieval and memory
# can reason about *why* a span matters to the user, not just what generic
# bucket it falls into.  The taxonomy is overridable per-pack via
# `pack["domain_taxonomy"]` or `pack["ontology"]["topic_candidates"]`.
DEFAULT_DOMAIN_TAXONOMY: Dict[str, List[str]] = {
    "relationship/partner": [
        "partner", "boyfriend", "girlfriend", "wife", "husband",
        "copain", "copine", "mari", "femme", "épouse",
        "муж", "жена", "парень", "девушка", "бойфренд",
    ],
    "money_help": [
        "loan", "borrow", "lend", "owe", "debt", "repay", "pay back", "send money",
        "wire transfer", "transfer money",
        "argent", "prêt", "prêter", "rembourser", "dette", "emprunt",
        "деньги", "займ", "одолжить", "одолжи", "долг", "оплат", "перевод",
    ],
    "travel_logistics": [
        "flight", "airport", "boarding", "hotel", "train", "taxi", "uber",
        "visa", "passport", "itinerary", "layover",
        "vol", "aéroport", "embarquement", "hôtel", "billet", "réservation",
        "рейс", "аэропорт", "посадка", "отель", "поезд", "виза", "паспорт", "бронь",
    ],
    "project_live_transcriptor": [
        "live voice transcriptor", "livevoicetranscriptor", "transcriptor",
        "canonical pipeline", "asr pipeline", "decode lattice",
        "stripe grouping", "semantic marking",
    ],
    "banking": [
        "bank account", "iban", "swift", "wire transfer", "bank branch",
        "savings account", "checking account",
        "banque", "compte bancaire", "virement", "agence bancaire",
        "банк", "расчётный счет", "счёт", "счет в банке", "перевод в банк", "сбер",
    ],
    "work_observability": [
        "dynatrace", "grafana", "datadog", "splunk", "prometheus",
        "opentelemetry", "jaeger", "tracing", "metric alert",
        "log alert", "incident", "sli", "slo", "on-call", "pagerduty",
    ],
}

TAG_RULES = {
    "topic_tags": {
        "relationship": ["relationship", "partner", "boyfriend", "girlfriend", "wife", "husband", "copain", "copine", "mari", "wife", "love", "dating", "отнош", "муж", "жена"],
        "meeting": ["meeting", "call", "standup", "sync", "rendez", "appel", "meeting", "встреч"],
        "travel": ["airport", "flight", "hotel", "train", "taxi", "travel", "voyage", "trip", "avion", "поезд", "путеше"],
        "health": ["doctor", "hospital", "medicine", "therapy", "sick", "stress", "health", "docteur", "malade", "боль", "врач"],
        "work": ["client", "deadline", "deploy", "release", "bug", "feature", "meeting", "ticket", "project", "repo", "branch", "merge"],
        "family_context": ["mother", "father", "mom", "dad", "brother", "sister", "family", "maman", "papa", "frere", "soeur", "семья", "мама", "папа"],
    },
    "relation_tags": {
        "romantic_context": ["boyfriend", "girlfriend", "wife", "husband", "partner", "dating", "love", "mari", "copine", "copain", "муж", "жена", "отнош"],
        "family_context": ["mother", "father", "mom", "dad", "brother", "sister", "family", "maman", "papa", "frere", "soeur", "семья", "мама", "папа"],
        "work_collaboration": ["client", "manager", "team", "project", "repo", "ticket", "branch", "colleague", "coworker"],
    },
    "emotion_tags": {
        "anxiety": ["anxious", "worried", "stress", "stressed", "panic", "afraid", "concerned", "angoiss", "inquiet", "трев", "волную"],
        "joy": ["happy", "excited", "glad", "great", "amazing", "content", "heureux", "super", "рад"],
        "sadness": ["sad", "upset", "depressed", "cry", "triste", "pleur", "груст", "печал"],
        "anger": ["angry", "mad", "furious", "annoyed", "colere", "furieux", "зл", "бесит"],
    },
}


def run_semantic_marking(session_id: str, segments: List[Dict], stage) -> Dict:
    sd = session_dir(session_id)
    enrichment_dir = sd / "enrichment"
    enrichment_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0 gate: if the canonical layer says the session is unhealthy,
    # we still emit every enrichment artifact -- with empty payloads and a
    # gate_reason -- so downstream readers never face missing files.  This is
    # the "audit only" mode called out in the plan.
    from app.pipeline.canonical_assembly import read_quality_gate
    gate = read_quality_gate(session_id)
    if not gate.get("semantic_eligible", True):
        empty_payload = {
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "marker_count": 0,
            "markers": [],
            "gate_status": "suppressed_by_quality_gate",
            "gate_reasons": gate.get("reasons") or [],
        }
        atomic_write_json(str(enrichment_dir / "segment_markers.json"), empty_payload)
        atomic_write_json(str(enrichment_dir / "semantic_spans.json"), {
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "span_count": 0,
            "spans": [],
            "gate_status": "suppressed_by_quality_gate",
            "gate_reasons": gate.get("reasons") or [],
        })
        atomic_write_json(str(enrichment_dir / "marker_audit.json"), {
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "marker_count": 0,
            "gate_status": "suppressed_by_quality_gate",
            "gate_reasons": gate.get("reasons") or [],
            "guardrails": {
                "markers_must_not_rewrite_canonical_text": True,
                "semantic_eligible": False,
            },
        })
        atomic_write_json(str(enrichment_dir / "context_links.json"), {
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "link_count": 0,
            "links": [],
            "gate_status": "suppressed_by_quality_gate",
            "gate_reasons": gate.get("reasons") or [],
        })
        # Phase 2: emit an annotated empty context_spans payload too, so
        # retrieval/audit tooling never sees a missing file even when the
        # quality gate suppresses everything downstream.
        from app.pipeline.context_spans import empty_payload as _empty_context_spans
        atomic_write_json(
            str(enrichment_dir / "context_spans.json"),
            _empty_context_spans(
                session_id,
                "suppressed_by_quality_gate",
                gate.get("reasons") or [],
            ),
        )
        stage.commit([
            "segment_markers.json",
            "semantic_spans.json",
            "marker_audit.json",
            "context_spans.json",
            "context_links.json",
        ])
        return {
            **empty_payload,
            "semantic_span_count": 0,
            "context_span_count": 0,
            "context_link_count": 0,
        }

    alias_index, domain_ontology, domain_taxonomy = _load_context_pack_indices(sd)

    # Only semantic-eligible segments feed marking.  Suppressed segments are
    # still in canonical truth for audit, but we refuse to interpret them.
    stabilized_segments = [
        segment
        for segment in segments
        if segment.get("stabilization_state") == "stabilized"
        and segment.get("segment_quality_status") != "suppressed"
    ]
    markers = []

    for index, segment in enumerate(stabilized_segments):
        prev_text = stabilized_segments[index - 1].get("text", "") if index > 0 else ""
        next_text = stabilized_segments[index + 1].get("text", "") if index + 1 < len(stabilized_segments) else ""
        markers.append(
            _build_marker(
                segment, prev_text, next_text,
                alias_index, domain_ontology, domain_taxonomy,
            )
        )

    payload = {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "marker_count": len(markers),
        "markers": markers,
        "gate_status": gate.get("session_quality_status", "unknown"),
    }
    atomic_write_json(str(enrichment_dir / "segment_markers.json"), payload)

    spans_payload = _build_semantic_spans(session_id, stabilized_segments, markers)
    atomic_write_json(str(enrichment_dir / "semantic_spans.json"), spans_payload)

    # Phase 2: real context spans grouped by continuity signals (temporal,
    # speaker, language, lexical, entity, topic).  This is the artifact that
    # retrieval should prefer over `semantic_spans.json`; the latter is kept
    # for backward compatibility with older readers.
    from app.pipeline.context_spans import build_context_spans
    context_spans_payload = build_context_spans(session_id, stabilized_segments, markers)
    atomic_write_json(str(enrichment_dir / "context_spans.json"), context_spans_payload)
    context_links_payload = _build_context_links(session_id, stabilized_segments, markers, context_spans_payload)
    atomic_write_json(str(enrichment_dir / "context_links.json"), context_links_payload)

    marker_audit = _marker_audit(session_id, markers)
    atomic_write_json(str(enrichment_dir / "marker_audit.json"), marker_audit)

    stage.commit([
        "segment_markers.json",
        "semantic_spans.json",
        "context_spans.json",
        "context_links.json",
        "marker_audit.json",
    ])
    return {
        **payload,
        "semantic_span_count": spans_payload.get("span_count", 0),
        "context_span_count": context_spans_payload.get("span_count", 0),
        "context_link_count": context_links_payload.get("link_count", 0),
    }


def _build_semantic_spans(session_id: str, segments: List[Dict], markers: List[Dict]) -> Dict:
    """Group adjacent canonical segments into higher-level semantic spans.

    Spans are grouping units for retrieval and dialogue-level meaning
    (canonical spec 7: SemanticSpan M_k).  They do not replace canonical
    segments — they point back to them.
    """
    if not segments or not markers:
        return {
            "session_id": session_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "span_count": 0,
            "spans": [],
        }

    markers_by_segment = {marker.get("segment_id"): marker for marker in markers}
    spans: List[Dict] = []
    current: Dict | None = None

    def _signature(marker: Dict) -> tuple:
        return (
            tuple(sorted(marker.get("topic_tags") or [])),
            tuple(sorted(marker.get("project_tags") or [])),
        )

    for segment in segments:
        marker = markers_by_segment.get(segment.get("segment_id"))
        if not marker:
            if current is not None:
                spans.append(current)
                current = None
            continue
        sig = _signature(marker)
        if current is None:
            current = _new_span(segment, marker, sig)
            continue
        gap_ms = segment.get("start_ms", 0) - current["end_ms"]
        if sig == current["_signature"] and gap_ms <= 45000:
            current["end_ms"] = segment.get("end_ms", current["end_ms"])
            current["segments"].append(segment.get("segment_id"))
            current["retrieval_terms"].update(marker.get("retrieval_terms") or [])
            current["entity_ids"].update(
                mention.get("entity_id") or mention.get("surface_form")
                for mention in (marker.get("entity_mentions") or [])
                if mention.get("entity_id") or mention.get("surface_form")
            )
        else:
            spans.append(current)
            current = _new_span(segment, marker, sig)

    if current is not None:
        spans.append(current)

    serialized = []
    for index, span in enumerate(spans):
        serialized.append({
            "span_id": f"span_{index:06d}",
            "start_ms": span["start_ms"],
            "end_ms": span["end_ms"],
            "segment_ids": list(span["segments"]),
            "segment_count": len(span["segments"]),
            "topic_tags": list(span["_signature"][0]),
            "project_tags": list(span["_signature"][1]),
            "retrieval_terms": sorted(term for term in span["retrieval_terms"] if term),
            "entity_ids": sorted(ent for ent in span["entity_ids"] if ent),
            "grounding": {
                "canonical_path": "canonical/canonical_segments.json",
                "markers_path": "enrichment/segment_markers.json",
            },
        })

    return {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "span_count": len(serialized),
        "spans": serialized,
    }


def _new_span(segment: Dict, marker: Dict, signature: tuple) -> Dict:
    return {
        "start_ms": segment.get("start_ms", 0),
        "end_ms": segment.get("end_ms", 0),
        "segments": [segment.get("segment_id")],
        "retrieval_terms": set(marker.get("retrieval_terms") or []),
        "entity_ids": {
            mention.get("entity_id") or mention.get("surface_form")
            for mention in (marker.get("entity_mentions") or [])
            if mention.get("entity_id") or mention.get("surface_form")
        },
        "_signature": signature,
    }


def _marker_audit(session_id: str, markers: List[Dict]) -> Dict:
    ambiguous = [marker for marker in markers if marker.get("ambiguity_flags")]
    pronoun_only = sum(
        1
        for marker in markers
        if "pronoun_without_grounded_entity" in (marker.get("ambiguity_flags") or [])
    )
    entity_link_count = sum(
        1
        for marker in markers
        for mention in marker.get("entity_mentions") or []
        if mention.get("entity_id")
    )
    return {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "marker_count": len(markers),
        "ambiguous_marker_count": len(ambiguous),
        "pronoun_without_entity_count": pronoun_only,
        "resolved_entity_link_count": entity_link_count,
        "guardrails": {
            "markers_must_not_rewrite_canonical_text": True,
            "weak_entities_remain_ambiguous": True,
            "imported_packs_are_separate": True,
        },
    }


def _build_marker(
    segment: Dict,
    prev_text: str,
    next_text: str,
    alias_index: Optional[Dict[str, Dict]] = None,
    domain_ontology: Optional[Dict[str, List[str]]] = None,
    domain_taxonomy: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    text = (segment.get("text") or "").strip()
    context_text = " ".join(part for part in [prev_text, text, next_text] if part).strip()

    entity_mentions = _entity_mentions(text)
    # Augment with alias-based entity hits from the curated context packs.
    # This is what lets tags like `Omar_husband`, `Dynatrace_tool`, or
    # `Bangkok_place` appear -- heuristic regex alone cannot produce them.
    alias_hits = _alias_based_mentions(text, alias_index or {})
    entity_mentions = _merge_entity_mentions(entity_mentions, alias_hits)

    topic_tags = _tags_for("topic_tags", context_text)
    relation_tags = _tags_for("relation_tags", context_text)
    emotion_tags = _tags_for("emotion_tags", context_text)
    project_tags = _project_tags(text)
    # Fold in domain ontology tags (e.g. `tool_tag:dynatrace`,
    # `topic_tag:observability`) from any loaded curated pack.
    ontology_tags = _ontology_tags(context_text, domain_ontology or {})
    topic_tags = sorted(set(topic_tags) | set(ontology_tags.get("topic_tags", [])))
    relation_tags = sorted(set(relation_tags) | set(ontology_tags.get("relation_tags", [])))
    project_tags = sorted(set(project_tags) | set(ontology_tags.get("project_tags", [])))

    # Phase 3: business-domain topic_candidates, distinct from heuristic topic_tags.
    topic_candidates = _topic_candidates(context_text, domain_taxonomy or {})

    retrieval_terms = _retrieval_terms(
        text, entity_mentions, topic_tags, relation_tags, project_tags, topic_candidates,
    )

    signal_count = sum(
        1
        for value in [
            entity_mentions, topic_tags, relation_tags, project_tags,
            emotion_tags, retrieval_terms, topic_candidates,
        ]
        if value
    )
    marker_confidence = round(min(0.9, 0.2 + signal_count * 0.12), 3) if signal_count else 0.15
    ambiguity_flags = _ambiguity_flags(text, entity_mentions, retrieval_terms, marker_confidence)

    if ambiguity_flags and marker_confidence > 0.2:
        marker_confidence = round(max(0.2, marker_confidence - 0.1), 3)

    marker_source = "heuristic_segment_text"
    if alias_hits:
        marker_source = "heuristic+curated_alias_resolution"

    return {
        "segment_id": segment.get("segment_id"),
        "entity_mentions": entity_mentions,
        "topic_tags": topic_tags,
        "topic_candidates": topic_candidates,
        "relation_tags": relation_tags,
        "project_tags": project_tags,
        "emotion_tags": emotion_tags,
        "retrieval_terms": retrieval_terms,
        "ambiguity_flags": ambiguity_flags,
        "marker_confidence": marker_confidence,
        "marker_source": marker_source,
    }


def _topic_candidates(text: str, domain_taxonomy: Dict[str, List[str]]) -> List[str]:
    """Match business-domain triggers against the segment context.

    Returns sorted unique topic-candidate keys (e.g. ``money_help``,
    ``relationship/partner``).  Triggers are matched as substrings on the
    lowercased context text -- cheap, deterministic, and language-agnostic
    given the multilingual trigger lists in ``DEFAULT_DOMAIN_TAXONOMY``.
    """
    if not text or not domain_taxonomy:
        return []
    haystack = text.lower()
    matched: List[str] = []
    for key, needles in domain_taxonomy.items():
        for needle in needles:
            if needle and needle in haystack:
                matched.append(key)
                break
    return sorted(set(matched))


def _entity_mentions(text: str) -> List[Dict]:
    mentions = []
    seen = set()

    for surface in _ordered_matches([EMAIL_RE, URL_RE, HANDLE_RE, ACRONYM_RE, TITLE_PHRASE_RE], text):
        key = surface.lower()
        if key in seen:
            continue
        seen.add(key)
        mention_type = _mention_type(surface)
        mentions.append({
            "surface_form": surface,
            "observed_surface_form": surface,
            "alias_surface": surface,
            "entity_id": None,
            "mention_type": mention_type,
            "confidence": _mention_confidence(mention_type),
            "source": "heuristic_surface",
            "observation_kind": "surface_alias_seen_in_text",
            "resolution_status": "context_local_unresolved",
        })

    return mentions


def _mention_type(surface: str) -> str:
    if EMAIL_RE.fullmatch(surface):
        return "email"
    if URL_RE.fullmatch(surface):
        return "url"
    if HANDLE_RE.fullmatch(surface):
        return "handle"
    if ACRONYM_RE.fullmatch(surface):
        return "acronym"
    return "capitalized_phrase"


def _mention_confidence(mention_type: str) -> float:
    return {
        "email": 0.92,
        "url": 0.92,
        "handle": 0.88,
        "acronym": 0.72,
        "capitalized_phrase": 0.58,
    }.get(mention_type, 0.5)


def _ordered_matches(patterns: Iterable[re.Pattern], text: str) -> List[str]:
    matches = []
    for pattern in patterns:
        for match in pattern.finditer(text or ""):
            value = match.group(0).strip(".,;:!?()[]{}")
            if value:
                matches.append((match.start(), value))
    matches.sort(key=lambda item: item[0])
    return [value for _, value in matches]


def _tags_for(group: str, text: str) -> List[str]:
    haystack = (text or "").lower()
    results = []
    for tag, needles in TAG_RULES[group].items():
        if any(needle.lower() in haystack for needle in needles):
            results.append(tag)
    return sorted(set(results))


def _project_tags(text: str) -> List[str]:
    tags = []
    for match in PROJECT_REF_RE.finditer(text or ""):
        value = match.group(1).strip(".,;:!?()[]{}")
        if len(value) >= 2:
            tags.append(value)
    return sorted(set(tags))


def _retrieval_terms(
    text: str,
    entity_mentions: List[Dict],
    topic_tags: List[str],
    relation_tags: List[str],
    project_tags: List[str],
    topic_candidates: Optional[List[str]] = None,
) -> List[str]:
    terms = []
    for mention in entity_mentions:
        terms.append(mention.get("surface_form", ""))
    terms.extend(project_tags)
    terms.extend(topic_tags)
    terms.extend(topic_candidates or [])
    terms.extend(relation_tags)

    for token in KEYWORD_RE.findall((text or "").lower()):
        clean = token.strip()
        if len(clean) < 4:
            continue
        terms.append(clean)

    deduped = []
    seen = set()
    for term in terms:
        normalized = term.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if len(deduped) >= 12:
            break
    return deduped


def _ambiguity_flags(
    text: str,
    entity_mentions: List[Dict],
    retrieval_terms: List[str],
    marker_confidence: float,
) -> List[str]:
    flags = []
    if not entity_mentions and PRONOUN_ONLY_RE.search(text or ""):
        flags.append("pronoun_without_grounded_entity")
    if not retrieval_terms:
        flags.append("no_retrieval_terms")
    if marker_confidence < 0.35:
        flags.append("weak_signal")
    return flags


def _load_context_pack_indices(
    sd: Path,
) -> Tuple[Dict[str, Dict], Dict[str, List[List[str]]], Dict[str, List[str]]]:
    """Load curated context packs + domain ontology + business taxonomy.

    Returns (alias_index, domain_ontology, domain_taxonomy).

    alias_index maps lowercased alias -> {entity_id, entity_type, canonical_name, confidence}.
    domain_ontology maps group (topic_tags / relation_tags / project_tags)
    to a list of [tag, pattern_string] pairs -- the legacy heuristic ontology
    surfaced in `topic_tags`.
    domain_taxonomy maps business-topic key -> list of trigger phrases used
    to populate the new `topic_candidates` field (Phase 3).  Defaults seed
    from `DEFAULT_DOMAIN_TAXONOMY` and can be extended/overridden by curated
    packs via `pack["domain_taxonomy"]` or `pack["ontology"]["topic_candidates"]`.

    Packs live under `<session>/memory/curated_packs/*.json` or a global
    `memory/curated_packs/` directory.  Schema (flexible, all keys optional):

      {
        "entities": [{"entity_id": "person_omar", "canonical_name": "Omar",
                       "aliases": ["omar", "мой муж", "husband"],
                       "type": "person", "confidence": 0.95}],
        "ontology": {
          "topic_tags": {"observability": ["dynatrace", "grafana", "traces"]},
          "project_tags": {"livevoicetranscriptor": ["live voice", "transcriptor"]},
          "topic_candidates": {"banking": ["sber", "iban"]}
        },
        "domain_taxonomy": {"money_help": ["airtm", "wise"]}
      }
    """
    alias_index: Dict[str, Dict] = {}
    domain_ontology: Dict[str, List[Tuple[str, str]]] = {}
    # Seed business taxonomy with the in-code defaults so a session with no
    # curated packs still surfaces meaningful topic_candidates.
    domain_taxonomy: Dict[str, List[str]] = {
        key: [needle.lower() for needle in needles]
        for key, needles in DEFAULT_DOMAIN_TAXONOMY.items()
    }

    curated_dirs = [sd / "memory" / "curated_packs", Path("memory") / "curated_packs"]
    seen_paths = set()
    for curated_dir in curated_dirs:
        if not curated_dir.is_dir():
            continue
        for pack_path in sorted(curated_dir.glob("*.json")):
            if pack_path in seen_paths:
                continue
            seen_paths.add(pack_path)
            pack = safe_read_json(str(pack_path))
            if not isinstance(pack, dict):
                continue
            for entity in pack.get("entities") or []:
                entity_id = entity.get("entity_id")
                if not entity_id:
                    continue
                entity_info = {
                    "entity_id": entity_id,
                    "entity_type": entity.get("type") or "curated_entity",
                    "canonical_name": entity.get("canonical_name") or entity_id,
                    "confidence": float(entity.get("confidence") or 0.9),
                    "source": "curated_pack",
                }
                aliases = list(entity.get("aliases") or [])
                if entity.get("canonical_name"):
                    aliases.append(entity["canonical_name"])
                for alias in aliases:
                    if not isinstance(alias, str):
                        continue
                    key = alias.strip().lower()
                    if not key:
                        continue
                    # First pack wins on conflict, keeping behavior deterministic.
                    alias_index.setdefault(key, entity_info)

            for group, tag_map in (pack.get("ontology") or {}).items():
                if not isinstance(tag_map, dict):
                    continue
                if group == "topic_candidates":
                    # Folded into the business taxonomy bucket, NOT into the
                    # generic ontology -- keeps Phase 3 separation clean.
                    for tag, needles in tag_map.items():
                        if not isinstance(needles, list):
                            continue
                        bucket = domain_taxonomy.setdefault(tag, [])
                        for needle in needles:
                            if isinstance(needle, str) and needle.strip():
                                bucket.append(needle.strip().lower())
                    continue
                bucket = domain_ontology.setdefault(group, [])
                for tag, needles in tag_map.items():
                    if not isinstance(needles, list):
                        continue
                    for needle in needles:
                        if not isinstance(needle, str) or not needle.strip():
                            continue
                        bucket.append((tag, needle.strip().lower()))

            # Top-level `domain_taxonomy` overrides/extends the in-code defaults.
            for tag, needles in (pack.get("domain_taxonomy") or {}).items():
                if not isinstance(needles, list):
                    continue
                bucket = domain_taxonomy.setdefault(tag, [])
                for needle in needles:
                    if isinstance(needle, str) and needle.strip():
                        bucket.append(needle.strip().lower())

    # Deduplicate taxonomy needles deterministically.
    domain_taxonomy = {
        tag: sorted(set(needles)) for tag, needles in domain_taxonomy.items() if needles
    }

    # Convert tuple list to plain lists for a clean cross-call contract.
    flat_ontology: Dict[str, List[List[str]]] = {
        group: [[tag, needle] for tag, needle in entries]
        for group, entries in domain_ontology.items()
    }
    return alias_index, flat_ontology, domain_taxonomy


# Common Russian inflectional endings, longest first so 3-char endings strip
# before single-char ones.  Used by `_simple_stem` so curated aliases written
# in nominative form ("мой муж") still resolve when the text uses an inflected
# form ("моим мужем", "о моём муже").  Latin tokens never end in these cyrillic
# strings so the stemmer is a no-op for English/French aliases.
_RU_ENDINGS = (
    "ого", "его", "ому", "ему", "ыми", "ими", "ами", "ями",
    "ой", "ей", "ом", "ем", "им", "ым", "их", "ых",
    "ам", "ям", "ах", "ях", "ов", "ев",
    "ую", "юю", "ою", "ею",
    "ая", "яя", "ое", "ее", "ие", "ый", "ий",
    "ье", "ья", "ью",
    "у", "ю", "е", "ы", "и", "а", "я", "о", "ь", "й",
)


def _simple_stem(token: str) -> str:
    """Strip a single Russian inflectional ending if removing it keeps >=2 chars."""
    if not token:
        return token
    for ending in _RU_ENDINGS:
        if token.endswith(ending) and len(token) - len(ending) >= 2:
            return token[: -len(ending)]
    return token


def _tokenize_for_alias(text: str) -> List[str]:
    return [part["normalized"] for part in _tokenize_alias_parts(text)]


def _tokenize_alias_parts(text: str) -> List[Dict[str, str]]:
    parts = []
    for match in KEYWORD_RE.finditer(text or ""):
        raw = match.group(0)
        normalized = raw.lower()
        parts.append({
            "raw": raw,
            "normalized": normalized,
            "stem": _simple_stem(normalized),
        })
    return parts


def _alias_based_mentions(text: str, alias_index: Dict[str, Dict]) -> List[Dict]:
    """Find curated-pack alias hits in the text.

    Matching is token-based with a light Russian stemmer so an alias written in
    nominative form ("мой муж") still resolves when the speaker uses an
    inflected case ("моим мужем").  We do NOT pull in a real morphological
    analyzer -- the stemmer is intentionally minimal and deterministic.
    """
    if not text or not alias_index:
        return []
    text_parts = _tokenize_alias_parts(text)
    text_stems = [part["stem"] for part in text_parts]

    hits: List[Dict] = []
    seen_entities: set[str] = set()
    # Longer aliases first so "мой муж" wins over "муж".
    for alias in sorted(alias_index.keys(), key=len, reverse=True):
        if not alias:
            continue
        info = alias_index[alias]
        if info["entity_id"] in seen_entities:
            continue
        alias_tokens = _tokenize_for_alias(alias)
        if not alias_tokens:
            continue
        alias_stems = [_simple_stem(tok) for tok in alias_tokens]
        n = len(alias_stems)
        matched_at = None
        for i in range(0, len(text_stems) - n + 1):
            if text_stems[i:i + n] == alias_stems:
                matched_at = i
                break
        if matched_at is not None:
            observed_surface = " ".join(
                part["raw"] for part in text_parts[matched_at:matched_at + n]
            ).strip()
            seen_entities.add(info["entity_id"])
            hits.append({
                "surface_form": observed_surface or alias,
                "observed_surface_form": observed_surface or alias,
                "alias_surface": alias,
                "entity_id": info["entity_id"],
                "mention_type": info.get("entity_type", "curated_entity"),
                "confidence": info.get("confidence", 0.9),
                "canonical_name": info.get("canonical_name"),
                "source": info.get("source", "curated_pack"),
                "observation_kind": "surface_alias_seen_in_text",
                "resolution_status": "resolved_known_entity",
            })
    return hits


def _merge_entity_mentions(
    heuristic: List[Dict],
    alias_hits: List[Dict],
) -> List[Dict]:
    if not alias_hits:
        return heuristic
    merged = list(heuristic)
    seen_surface = {(mention.get("surface_form") or "").lower() for mention in merged}
    for hit in alias_hits:
        key = (hit.get("surface_form") or "").lower()
        if key in seen_surface:
            # Upgrade an existing heuristic mention with curated entity_id
            # when the surface form already matched.
            for mention in merged:
                if (mention.get("surface_form") or "").lower() == key:
                    mention["entity_id"] = mention.get("entity_id") or hit.get("entity_id")
                    mention["canonical_name"] = hit.get("canonical_name")
                    mention["source"] = "curated_pack"
                    mention["alias_surface"] = hit.get("alias_surface") or mention.get("alias_surface")
                    mention["observed_surface_form"] = hit.get("observed_surface_form") or mention.get("observed_surface_form")
                    mention["observation_kind"] = "surface_alias_seen_in_text"
                    mention["resolution_status"] = "resolved_known_entity"
                    mention["confidence"] = max(float(mention.get("confidence") or 0.0), float(hit.get("confidence") or 0.0))
                    break
            continue
        merged.append(hit)
        seen_surface.add(key)
    return merged


def _support_text_excerpt(text: str, mention_text: str, limit: int = 140) -> str:
    source = (text or "").strip()
    if not source:
        return ""
    needle = (mention_text or "").strip()
    if not needle:
        return source[:limit]
    idx = source.lower().find(needle.lower())
    if idx < 0:
        return source[:limit]
    start = max(0, idx - 40)
    end = min(len(source), idx + len(needle) + 40)
    excerpt = source[start:end].strip()
    if start > 0:
        excerpt = f"...{excerpt}"
    if end < len(source):
        excerpt = f"{excerpt}..."
    return excerpt[:limit]


def _context_link_payload(
    session_id: str,
    context_id: Optional[str],
    segment: Dict,
    mention: Dict,
    link_index: int,
) -> Dict:
    mention_text = (
        mention.get("observed_surface_form")
        or mention.get("surface_form")
        or mention.get("canonical_name")
        or ""
    )
    alias_surface = mention.get("alias_surface") or mention.get("surface_form") or mention_text
    resolved = bool(mention.get("entity_id")) and mention.get("source") == "curated_pack"
    memory_kind = "resolved_alias_to_known_entity" if resolved else "context_local_unresolved_mention"
    resolution_status = mention.get("resolution_status") or (
        "resolved_known_entity" if resolved else "context_local_unresolved"
    )
    return {
        "link_id": f"ctxlink_{link_index:06d}",
        "session_id": session_id,
        "context_id": context_id,
        "segment_id": segment.get("segment_id"),
        "mention_text": mention_text,
        "alias_surface": alias_surface,
        "entity_id": mention.get("entity_id"),
        "canonical_name": mention.get("canonical_name"),
        "mention_type": mention.get("mention_type"),
        "observation_kind": mention.get("observation_kind") or "surface_alias_seen_in_text",
        "memory_kind": memory_kind,
        "resolution_status": resolution_status,
        "confidence": round(float(mention.get("confidence") or 0.0), 3),
        "source": mention.get("source") or "heuristic_surface",
        "support_text": _support_text_excerpt(segment.get("text", ""), mention_text or alias_surface),
        "start_ms": segment.get("start_ms"),
        "end_ms": segment.get("end_ms"),
        "language": segment.get("language"),
        "speaker": segment.get("speaker"),
    }


def _build_context_links(
    session_id: str,
    segments: List[Dict],
    markers: List[Dict],
    context_spans_payload: Dict,
) -> Dict:
    segments_by_id = {
        segment.get("segment_id"): segment
        for segment in segments
        if segment.get("segment_id")
    }
    context_by_segment = {}
    for span in (context_spans_payload or {}).get("spans") or []:
        for segment_id in span.get("segment_ids") or []:
            context_by_segment[segment_id] = span.get("context_id")

    links = []
    for marker in markers:
        segment_id = marker.get("segment_id")
        segment = segments_by_id.get(segment_id)
        if not segment:
            continue
        context_id = context_by_segment.get(segment_id)
        for mention in marker.get("entity_mentions") or []:
            links.append(
                _context_link_payload(
                    session_id=session_id,
                    context_id=context_id,
                    segment=segment,
                    mention=mention,
                    link_index=len(links),
                )
            )

    return {
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "link_count": len(links),
        "links": links,
        "source": "context_spans+segment_markers",
    }


def _ontology_tags(text: str, domain_ontology: Dict[str, List[List[str]]]) -> Dict[str, List[str]]:
    if not text or not domain_ontology:
        return {}
    haystack = text.lower()
    result: Dict[str, set] = {}
    for group, entries in domain_ontology.items():
        for entry in entries:
            if not entry or len(entry) != 2:
                continue
            tag, needle = entry
            if needle and needle in haystack:
                result.setdefault(group, set()).add(tag)
    return {group: sorted(tags) for group, tags in result.items()}
