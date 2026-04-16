# Canonical Pipeline Specification
## for the Next-Generation Wearable Audio Memory System

**Version 2.0**  •  **Canonical engineering document**  •  **14 April 2026**

**Purpose:** define the frozen server-side pipeline, truth layers, LLM contracts, semantic marking model, context-pack architecture, storage layout, and retrieval grounding model for the next application generation.

| Status | Canonical implementation reference for the next major server generation |
| --- | --- |
| Primary domain | Continuous multilingual audio capture, transcription, semantic marking, and grounded personal memory retrieval |
| Locked acoustic center | 30-second transport chunks at ingest; 30-second decode windows with 15-second stride; 15-second stabilization stripes |
| Locked semantic center | Lexical truth is assembled locally from ASR evidence; semantic meaning is attached after lexical stabilization and never replaces lexical truth |
| Audience | Application architect, backend engineer, pipeline engineer, ASR engineer, LLM engineer, retrieval engineer, QA |

> **Canonical sentence**  
> The application uploads 30-second transport chunks, but the server’s user-facing truth is built from overlap-aware 30-second decode windows with 15-second stride, locally synthesized stripe-by-stripe from multiple ASR witnesses, then semantically marked into a retrieval-ready memory layer without ever replacing the grounded lexical transcript.

---

# 1. Executive summary

This document defines the ideal canonical pipeline for a personal audio memory system that must do more than generate a transcript. The system must produce two separate but connected outputs:

1. **Grounded lexical truth** — the best recoverable version of what was actually spoken, with timestamps, provenance, uncertainty, and local language fidelity.
2. **Semantic memory truth** — markers, entities, aliases, relation links, project tags, emotional/context flags, and retrieval terms that make the transcript useful for later search, clustering, summarization, and external reasoning.

The system is designed for constrained local hardware, multilingual speech, long recordings, overlapping speech boundaries, and future NoSQL-backed retrieval. It explicitly rejects a false choice between “simple transcription” and “giant memory AI.” The correct design is layered:

- transport on the client,
- acoustic evidence building on the server,
- bounded lexical synthesis by local LLM plus deterministic safeguards,
- semantic marking after lexical stabilization,
- optional external high-level synthesis through imported context packs.

The central design decision is that **transcript truth and memory meaning are not the same object**. If they are collapsed into one layer, the system will either become a brittle keyword store or a hallucinating life-logger.

---

# 2. Core problem statement

The previous generation of the system aimed mainly to produce a final transcript from multiple ASR models. That is necessary but insufficient.

The new generation must solve a larger problem:

- preserve multilingual spoken content locally and faithfully,
- recover words damaged by chunk boundaries,
- compare multiple ASR witnesses on the same acoustic slice,
- allow bounded LLM synthesis from incomplete or conflicting evidence,
- attach retrieval markers to each stabilized text region,
- maintain a user-specific memory graph that can resolve indirect references,
- support future semantic search over people, projects, relationships, themes, and recurring concerns,
- remain auditable and safe on limited VRAM.

This is not a generic transcription pipeline. It is a **grounded personal memory pipeline**.

---

# 3. Design goals

## 3.1 Primary goals

- Preserve what was actually spoken, in the language(s) actually spoken, with no default translation.
- Protect word boundaries, speaker turns, and language switches at transport chunk edges.
- Use multiple ASR witnesses on shared decode geometry.
- Let the LLM synthesize the most probable local lexical truth from bounded evidence, not just choose one candidate verbatim.
- Attach semantic markers that make every text region retrievable later by person, alias, project, topic, relation, emotion, or user-defined domain.
- Keep all semantic enrichment grounded on canonical stabilized segments.
- Make every canonical segment traceable back to concrete model outputs and time spans.
- Make every semantic marker traceable back to concrete canonical segments and context-pack facts.
- Support partial live output without confusing provisional text with stabilized truth.
- Support constrained VRAM by keeping acoustic geometry fixed and model scheduling sequential.
- Support a future NoSQL or vector-backed memory layer without redesigning the canonical transcript model.

## 3.2 Secondary goals

- Enable human-in-the-loop or ChatGPT-assisted curation of aliases, entities, packs, and disambiguation rules.
- Support offline export/import of context bundles and curated memory packs.
- Make retrieval useful even when exact names are not spoken literally.
- Keep the mobile app dumb and stable.

---

# 4. Explicit non-goals

- No fixed-length transport chunk may be treated as the final transcription unit.
- No single dominant language may be forced across an entire long multilingual session.
- No whole-day ASR pass may run before acoustic filtering and segmentation.
- No giant whole-transcript LLM rewrite may define canonical lexical truth.
- No semantic marker may silently overwrite transcript text.
- No summary may become the only searchable memory artifact.
- No diarization may run blindly on every second of audio.
- No external reasoning system may inject unsupported words into canonical transcript truth.
- No retrieval layer may index only summaries while losing traceability to grounded segments.

---

# 5. Frozen architectural decisions

| Decision | Frozen value | Reason |
| --- | --- | --- |
| Transport chunk length | 30 seconds | Stable upload size; practical buffering and resend behavior on device/app |
| Decode window length | 30 seconds | Shared bounded geometry across ASR models |
| Decode stride | 15 seconds | Protects chunk boundaries and gives interior stripes two supporting windows |
| Canonical commit unit | 15-second stabilization stripe | Final lexical truth is committed stripe-by-stripe, not chunk-by-chunk |
| Acoustic scheduling position | Before expensive ASR | Avoid wasting compute on obvious non-speech |
| Lexical synthesis scope | Local and bounded | Prevents giant hallucinating rewrites |
| Semantic marking position | After lexical stabilization | Prevents enrichment from becoming hidden transcript source |
| Retrieval grounding unit | Canonical segment plus marker bundle | Retrieval must point back to stabilized lexical truth |
| Personal memory source of truth | Context packs + entity graph + segment markers | Personal meaning must be explicit, versioned, and auditable |
| Model execution policy | Sequential by model family | Required for constrained VRAM |
| Candidate stage naming | Generic by role, not by model brand | Stage names must remain truthful when routing swaps models |

---

# 6. The four truth layers

The system is built on four separate layers. This separation is mandatory.

## 6.1 Layer A — Acoustic evidence truth

This is the raw evidence from normalized audio, decode windows, ASR candidates, timestamps, language hints, and trust zones.

This layer answers:
- what raw witnesses exist,
- where they came from,
- how reliable they are locally.

## 6.2 Layer B — Canonical lexical truth

This is the stabilized text assembled from local ASR evidence and bounded context.

This layer answers:
- what was most probably said,
- where it was said on the session timeline,
- what evidence supports it,
- what uncertainty remains.

## 6.3 Layer C — Semantic marker truth

This is the semantic annotation attached to stabilized lexical segments.

This layer answers:
- who or what the segment may refer to,
- what themes or projects it belongs to,
- what relation or emotional flags should be indexed,
- what retrieval terms should be searchable later.

## 6.4 Layer D — Personal memory graph truth

This is the user-level graph of entities, aliases, roles, project domains, recurring topics, disambiguation rules, and curated context packs.

This layer answers:
- who Mattiew is in the user’s memory system,
- what aliases can refer to Omar,
- what “WebLogic memory leak” means in the user’s work context,
- what knowledge should be loaded when certain topics recur.

### Mandatory rule

Layers B, C, and D are connected, but none of them may silently replace the layer below it.

- Layer C may not rewrite Layer B text.
- Layer D may not fabricate Layer B text.
- Layer B may reference Layer A evidence but may not claim unsupported certainty.

---

# 7. Time abstractions

Three time abstractions remain mandatory, and a fourth semantic abstraction is added.

| Object | Definition | Example | Owner |
| --- | --- | --- | --- |
| TransportChunk C_n | Uploaded client payload placed on absolute timeline | C0=[0,30], C1=[30,60] | App → Server |
| DecodeWindow W_k | 30-second server-side window cut from normalized timeline with 15-second stride | W1=[15,45] | Server ASR scheduler |
| CommitStripe S_k | 15-second stabilization unit committed only after enough evidence exists | S2=[30,45] | Lexical synthesizer |
| SemanticSpan M_k | Optional higher-level grouping of adjacent canonical segments for retrieval or dialogue-level meaning | M3=[seg_12, seg_13, seg_14] | Marker/retrieval layer |

The new semantic span abstraction does **not** replace stripes or segments. It is a downstream grouping unit for retrieval and later memory assembly.

---

# 8. End-to-end canonical pipeline

## Stage 1 — Capture

The wearable or device records continuously. The application performs capture, buffering, resend, timestamp preservation, and transport upload only.

## Stage 2 — Ingest and normalization

The server receives transport chunks, decodes original media, normalizes sample rate/channels, and places each chunk on an absolute session timeline.

## Stage 3 — Acoustic triage

The server marks speech, non-speech, noise, music/media, mixed, or uncertain regions and builds speech islands.

## Stage 4 — Decode lattice construction

The server synthesizes overlapping 30-second decode windows with 15-second stride from normalized timeline audio, independent of original transport chunk boundaries.

## Stage 5 — Multi-ASR execution

For each eligible decode window, the server runs the configured ASR witnesses sequentially and persists raw candidate evidence.

## Stage 6 — Candidate normalization and witness audit

The server converts candidate outputs into a shared witness schema and adds language/script/task-drift diagnostics.

## Stage 7 — Stripe packet construction

The server regroups window candidates by 15-second stabilization stripe, with neighboring lexical context and local trust-zone metadata.

## Stage 8 — Bounded lexical synthesis

The local LLM receives all candidate witnesses for the stripe plus bounded context and produces a supported lexical synthesis result. Deterministic validators then audit the output.

## Stage 9 — Stabilization and canonical assembly

The server stabilizes stripes when enough support exists and merges them into canonical segments with provenance.

## Stage 10 — Semantic marking

The server attaches entities, aliases, relation tags, project tags, topic tags, emotion/context tags, ambiguity flags, and retrieval terms to stabilized segments.

## Stage 11 — Memory graph update

The server updates entity graph, alias graph, project graph, and context-pack usage metadata. This stage never rewrites canonical text.

## Stage 12 — Derived outputs and retrieval surfaces

The server builds transcript surfaces, retrieval indexes, semantic spans, summaries, and export bundles, all grounded on canonical segments and semantic markers.

## Stage 13 — Optional external synthesis loop

The system may export bounded context bundles to a stronger external reasoning system and later import curated context packs, alias maps, or graph corrections. Imported semantic knowledge is versioned and auditable and does not silently mutate historical lexical truth.

---

# 9. Detailed stage specification

## 9.1 Stage 1 — Capture and transport

The application is intentionally dumb. This is a design requirement.

Application responsibilities are limited to:
- continuous capture,
- buffer management,
- resend on failure,
- preservation of capture timestamps,
- upload of 30-second transport chunks,
- metadata envelope attachment.

The application must not perform language selection, semantic labeling, transcript repair, diarization, or memory reasoning.

### Required metadata envelope

| Field | Type | Meaning |
| --- | --- | --- |
| session_id | string | Stable session identifier |
| chunk_index | integer | Monotonic transport chunk number |
| capture_start_ms | integer | Absolute session-relative start timestamp |
| capture_end_ms | integer | Absolute session-relative end timestamp |
| codec_container | string | Original media format |
| device_clock_info | object | Clock origin/drift metadata if available |
| allowed_languages | array<string> | Optional allowlist configured by user/session |
| forced_language | string \| null | Optional hard override |
| transcription_mode | string | For example `verbatim_multilingual` |
| diarization_policy | string | For example `auto`, `off`, `forced` |
| pack_hints | array<string> | Optional request hints such as `relationships`, `work_dynatrace` |

## 9.2 Stage 2 — Ingest and normalization

Each transport chunk is decoded and converted into a server-side normalized representation. The normalized timeline is the only timeline downstream stages should trust.

Normalization responsibilities:
- decode source media honestly,
- resample and channel-normalize consistently,
- preserve absolute timeline placement,
- record gaps or overlaps without silent collapse,
- preserve original media for audit.

Outputs:
- `normalized/audio.wav`
- `normalized/timeline.json`
- ingest audit metadata

## 9.3 Stage 3 — Acoustic triage

Acoustic triage happens before expensive ASR.

Recommended tags:
- `speech`
- `non_speech`
- `music_media`
- `noise`
- `mixed`
- `uncertain`

Adjacent speech-bearing regions are merged conservatively into speech islands so short pauses do not fragment a single utterance.

### Scheduling rule

A decode window is eligible for expensive ASR when it intersects speech strongly enough to justify compute. However, bridge windows remain mandatory when speech crosses a chunk boundary, even if the speech density is modest.

## 9.4 Stage 4 — Decode lattice construction

The server constructs 30-second decode windows with 15-second stride.

For a session beginning at time zero:
- `W0 = [0,30]`
- `W1 = [15,45]`
- `W2 = [30,60]`
- `W3 = [45,75]`

This geometry remains frozen because it protects word boundaries, speaker changes, and language switches at transport chunk edges. The client still uploads `C0=[0,30]` and `C1=[30,60]`, but the server does not treat those chunks as final truth units. The bridge window `W1=[15,45]` is mandatory when speech crosses the boundary. This locked acoustic design remains the center of the architecture. fileciteturn0file0

## 9.5 Stage 5 — Multi-ASR execution

For each scheduled window, the server runs all configured ASR witnesses on the exact same audio geometry.

### Required witness policy

- Models are witnesses, not authorities.
- Model families run sequentially under VRAM control.
- Each candidate is persisted immediately.
- One model’s language guess must not silently poison the others.
- Candidate stage names must describe role, not a hard-coded model brand.

### Canonical stage naming

Use role-based names such as:
- `first_pass_asr`
- `candidate_asr_primary`
- `candidate_asr_secondary`
- `candidate_asr_tertiary`

Do **not** name a stage after a specific model if routing can swap that model at runtime.

## 9.6 Stage 6 — Candidate normalization and witness audit

Every raw ASR output must be converted into a common witness schema and audited.

### Candidate schema requirements

Each candidate must retain:
- model provenance,
- exact time span,
- raw text,
- segments/words when available,
- local language hints,
- decode metadata,
- diagnostic flags.

### Mandatory witness diagnostics

The system must compute local flags such as:
- `language_mismatch`
- `script_mismatch`
- `possible_translation`
- `semantic_drift`
- `statement_question_drift`
- `repetition_anomaly`
- `edge_truncation_suspected`
- `empty_candidate`
- `provider_degraded`

These flags do not automatically discard a candidate, but they affect trust scoring and later reasoning.

## 9.7 Stage 7 — Stripe packet construction

The server does not reconcile windows directly into final transcript. It first regroups evidence by 15-second stabilization stripe.

For example:
- `S1=[15,30]` is supported by `W0` right half and `W1` left half.
- `S2=[30,45]` is supported by `W1` right half and `W2` left half.

Each stripe packet may include:
- all candidate witnesses that overlap the stripe,
- trust-zone metadata,
- edge penalties,
- neighboring stabilized stripes,
- local language hints,
- relevant context-pack facts,
- candidate diagnostics.

## 9.8 Stage 8 — Bounded lexical synthesis

This stage is the biggest conceptual upgrade from the previous design.

The LLM is **not** merely a candidate selector. It is a bounded lexical synthesis engine.

### LLM role in lexical synthesis

For each stripe, the LLM may:
- compare all ASR candidate witnesses,
- merge supported fragments from multiple candidates,
- use neighboring stabilized context to resolve cut words or local ambiguity,
- preserve code-switching,
- preserve uncertainty when evidence remains weak.

The LLM may **not**:
- globally rewrite the transcript,
- summarize,
- improve style,
- invent unsupported facts,
- translate by default,
- inject knowledge from the personal memory graph as transcript content.

### Key distinction

The LLM may synthesize the most probable text from evidence, but it may not fabricate transcript content from biography.

### Validation rule

Every lexical synthesis output must be audited against evidence.

The validator must compute at least:
- `used_candidates`
- `unsupported_tokens`
- `token_support_ratio`
- `language_consistency`
- `script_consistency`
- `assembly_mode` (`exact_candidate`, `whitespace_normalized`, `multi_candidate_synthesis`, `uncertain_empty`)
- `validation_status`

If unsupported content exceeds policy threshold, the stripe must either:
- fall back to deterministic assembly, or
- remain provisional with explicit uncertainty.

## 9.9 Stage 9 — Stabilization and canonical assembly

A stripe becomes canonical only when enough support exists.

When only one supporting window exists, the stripe may be emitted as provisional for live UX. When a second supporting window arrives, the stripe becomes stabilizable and may be committed with greater confidence.

Adjacent compatible stabilized stripes are merged into canonical segments.

### Canonical segment principles

A canonical segment must include:
- time span,
- text,
- local language,
- speaker if justified,
- confidence/support summary,
- support windows and models,
- stabilization state,
- synthesis audit metadata.

## 9.10 Stage 10 — Semantic marking

Semantic marking is downstream of canonical lexical truth.

This stage exists because transcripts alone are a weak memory substrate.

### Semantic marking responsibilities

For each stabilized canonical segment, compute:
- entity mentions,
- alias resolutions,
- relation tags,
- topic tags,
- project/domain tags,
- emotion/context tags,
- privacy flags,
- ambiguity flags,
- retrieval terms.

### Semantic marking inputs

- stabilized canonical segment text,
- neighboring canonical segments,
- speaker if available,
- relevant context packs,
- entity graph and alias graph,
- local session metadata.

### Semantic marking outputs

The stage must generate a `segment_markers` artifact separate from canonical transcript text.

## 9.11 Stage 11 — Memory graph update

The memory graph is not the transcript. It is the structured user-specific context model that future retrieval and reasoning can load dynamically.

This graph may include:
- people,
- aliases,
- roles,
- project domains,
- recurring issues,
- preferred vocabulary,
- negative disambiguation rules,
- pack usage telemetry.

### Update policy

Graph updates must be versioned, confidence-scored, and reversible.

Auto-generated graph updates must never silently become hard truth without auditability.

## 9.12 Stage 12 — Derived outputs and retrieval surfaces

All derived outputs must be grounded on canonical segments and semantic markers.

Derived outputs may include:
- plain transcript surfaces,
- speaker transcript surfaces,
- retrieval index,
- topic spans,
- person spans,
- daily digests,
- unresolved threads,
- export bundles.

### Retrieval principle

Retrieval must work on:
- literal text,
- normalized text,
- aliases,
- entity IDs,
- relation tags,
- topic/project tags,
- time,
- support and confidence metadata.

## 9.13 Stage 13 — Optional external synthesis loop

A stronger external assistant may help refine semantic memory structure when the local system is compute-limited.

### Safe export/import model

The server may export bounded review bundles containing:
- selected canonical segments,
- existing markers,
- unresolved aliases,
- ambiguous links,
- candidate graph updates.

An external system may return:
- curated context packs,
- alias mappings,
- entity merges/splits,
- disambiguation rules,
- retrieval heuristics.

Imported knowledge must be:
- versioned,
- attributable,
- auditable,
- separated from lexical truth.

---

# 10. LLM role model

The system uses LLMs in three distinct roles.

## 10.1 Role A — Lexical synthesizer

Purpose: recover the best supported local transcript text.

Scope: one stripe plus bounded neighbors.

Hard rule: no unsupported hallucination.

## 10.2 Role B — Semantic marker

Purpose: tag canonical segments for retrieval and memory use.

Scope: one segment or a short segment neighborhood plus relevant packs.

Hard rule: never overwrite canonical text.

## 10.3 Role C — Pack curator

Purpose: help build or revise entity graph, alias graph, project packs, and relation packs from accumulated evidence.

Scope: offline or batch.

Hard rule: curated knowledge remains separate from transcript truth and must retain provenance.

---

# 11. Context-pack architecture

Context packs are the controlled interface between user-specific memory and future interpretation.

## 11.1 Why context packs exist

Users often refer to the same person or topic indirectly:
- “my husband” may refer to Omar,
- “the addict from Bangkok” may refer to Mattiew,
- “WebLogic memory leak” may belong to a work-learning context tied to Dynatrace or observability.

Literal transcript text alone is too weak to resolve these consistently.

## 11.2 Pack categories

Recommended pack families:
- `people_pack`
- `relationships_pack`
- `work_domains_pack`
- `projects_pack`
- `health_or_mood_pack`
- `negative_disambiguation_pack`
- `recent_context_pack`

## 11.3 Pack rules

- Packs are semantic aids, not transcript sources.
- Packs must be versioned.
- Packs must be confidence-scored.
- Packs must support manual curation.
- Packs may be selectively loaded by topic, session, or retrieval query.

## 11.4 Pack loading policy

Do not load the entire user memory blindly for every stripe.

Load only relevant pack fragments based on:
- segment topic hints,
- session pack hints,
- recent retrieval usage,
- explicit user request.

---

# 12. Data contracts and object definitions

## 12.1 SessionManifest

```json
{
  "session_id": "uuid",
  "allowed_languages": ["ru", "fr", "en"],
  "forced_language": null,
  "transcription_mode": "verbatim_multilingual",
  "diarization_policy": "auto",
  "capture_mode": "continuous",
  "transport_chunk_ms": 30000,
  "decode_window_ms": 30000,
  "decode_stride_ms": 15000,
  "commit_stripe_ms": 15000,
  "context_pack_hints": ["relationships", "work_dynatrace"]
}
```

## 12.2 DecodeWindow

```json
{
  "window_id": "W000123",
  "start_ms": 15000,
  "end_ms": 45000,
  "window_type": "bridge",
  "source_chunks": ["C0000", "C0001"],
  "speech_intersection_ratio": 0.73,
  "scheduled": true,
  "bridge_required": true
}
```

## 12.3 CandidateWitness

```json
{
  "candidate_id": "cand_9d3f",
  "session_id": "uuid",
  "model_id": "nemo-asr:canary-1b-v2",
  "window_id": "W000123",
  "window_type": "bridge",
  "window_start_ms": 15000,
  "window_end_ms": 45000,
  "raw_text": "...",
  "segments": [...],
  "language_evidence": [...],
  "confidence_features": {
    "avg_logprob": -0.24,
    "repetition_ratio": 0.03,
    "anomaly_flags": []
  },
  "candidate_flags": [
    "possible_translation"
  ],
  "decode_metadata": {
    "task": "transcribe",
    "requested_language": "ru",
    "segment_timestamp_unit": "seconds"
  }
}
```

## 12.4 StripeEvidencePacket

```json
{
  "stripe_id": "S0007",
  "start_ms": 105000,
  "end_ms": 120000,
  "support_windows": ["W000008", "W000009"],
  "support_models": [
    "faster-whisper:large-v3",
    "faster-whisper:medium",
    "nemo-asr:canary-1b-v2"
  ],
  "neighbor_context": {
    "prev_stabilized_text": "...",
    "next_stabilized_text": null
  },
  "evidence": [...],
  "loaded_context_facts": [
    {
      "pack_id": "relationships",
      "fact_id": "f_001",
      "text": "The phrase 'my husband' usually refers to Omar",
      "confidence": 0.91
    }
  ]
}
```

## 12.5 LexicalSynthesisResult

```json
{
  "stripe_id": "S0007",
  "final_text": "...",
  "assembly_mode": "multi_candidate_synthesis",
  "used_candidates": ["cand_a", "cand_b"],
  "unsupported_tokens": [],
  "token_support_ratio": 0.93,
  "language": "ru",
  "confidence": 0.84,
  "uncertainty_flags": [],
  "validation_status": "accepted"
}
```

## 12.6 CanonicalSegment

```json
{
  "segment_id": "seg_000231",
  "start_ms": 30210,
  "end_ms": 34490,
  "speaker": "S1",
  "language": "fr",
  "text": "...",
  "confidence": 0.87,
  "support_windows": ["W000123", "W000124"],
  "support_models": ["faster-whisper:large-v3", "nemo-asr:canary-1b-v2"],
  "stabilization_state": "stabilized",
  "assembly_audit": {
    "assembly_mode": "multi_candidate_synthesis",
    "used_candidates": ["cand_1", "cand_4"],
    "unsupported_tokens": []
  }
}
```

## 12.7 SegmentMarkers

```json
{
  "segment_id": "seg_000231",
  "entities": [
    {
      "entity_id": "person_mattiew",
      "surface_form": "наркоман из Бангкока",
      "link_type": "indirect_alias",
      "confidence": 0.78
    }
  ],
  "relation_tags": ["romantic_context", "past_relationship"],
  "topic_tags": ["rumination", "relationship"],
  "project_tags": [],
  "emotion_tags": ["anxiety"],
  "privacy_flags": [],
  "ambiguity_flags": [],
  "retrieval_terms": ["Mattiew", "наркоман из Бангкока", "отношения"],
  "marker_confidence": 0.76,
  "marker_sources": [
    {
      "type": "context_pack",
      "pack_id": "relationships",
      "fact_id": "f_004"
    }
  ]
}
```

## 12.8 EntityRegistry

```json
{
  "entities": [
    {
      "entity_id": "person_omar",
      "entity_type": "person",
      "display_name": "Omar",
      "aliases": ["Omar", "мой муж", "my husband"],
      "roles": ["spouse"],
      "confidence": 0.93,
      "status": "active"
    }
  ]
}
```

## 12.9 ContextPack

```json
{
  "pack_id": "relationships",
  "version": 3,
  "scope": "user_global",
  "facts": [
    {
      "fact_id": "f_001",
      "text": "The phrase 'мой муж' usually refers to Omar",
      "confidence": 0.91,
      "entity_ids": ["person_omar"]
    }
  ],
  "negative_rules": [
    {
      "rule_id": "n_001",
      "text": "Do not map every mention of 'he' to Omar without nearby supporting relation cues"
    }
  ]
}
```

## 12.10 RetrievalEntryV2

```json
{
  "entry_id": "ret_000091",
  "segment_id": "seg_000231",
  "text": "...",
  "normalized_text": "...",
  "entity_ids": ["person_mattiew"],
  "aliases": ["наркоман из Бангкока", "Mattiew"],
  "relation_tags": ["past_relationship"],
  "topic_tags": ["relationship", "rumination"],
  "project_tags": [],
  "emotion_tags": ["anxiety"],
  "retrieval_terms": ["Mattiew", "отношения"],
  "grounding": {
    "segment_id": "seg_000231",
    "start_ms": 30210,
    "end_ms": 34490,
    "canonical_path": "canonical/canonical_segments.json",
    "markers_path": "enrichment/segment_markers.json"
  }
}
```

---

# 13. Validation and guardrails

## 13.1 Lexical guardrails

The lexical synthesis layer must enforce:
- no default translation,
- no style improvement,
- no summarization,
- no unsupported content injection,
- no global rewrite,
- explicit uncertainty on weak evidence.

## 13.2 Semantic guardrails

The semantic marking layer must enforce:
- markers do not mutate canonical text,
- weak entity links remain ambiguous instead of overconfident,
- imported packs remain separate from inferred packs,
- negative rules are honored.

## 13.3 External-pack guardrails

Imported context packs must carry:
- source,
- version,
- creation timestamp,
- confidence,
- pack scope,
- curator type (`human`, `assistant`, `hybrid`).

---

# 14. Storage and artifact layout

Recommended storage layers:

- `raw/` — original uploaded media and ingest metadata
- `normalized/` — normalized timeline artifacts
- `triage/` — speech maps, speech islands, scheduler decisions
- `windows/` — synthesized decode windows and queue state
- `candidates/` — per-model raw ASR witness outputs
- `reconciliation/` — stripe packets, lexical synthesis outputs, validation audits
- `canonical/` — canonical segments and transcript surfaces
- `enrichment/` — segment markers, semantic spans, marker audits
- `memory/` — entity registry, alias graph, context packs, graph updates
- `derived/` — retrieval surfaces, summaries, exports, QA reports
- `exports/` — bounded review bundles for external reasoning
- `imports/` — imported curated packs and change manifests

The application should read mainly from `canonical/` and selected `derived/` or `enrichment/` surfaces, never directly from candidate or reconciliation internals.

---

# 15. API surface principles

## 15.1 Transcript API split

Expose at least:
- `provisional_partial`
- `stabilized_partial`
- `final_transcript`

This distinction already exists as a core idea in the earlier canonical specification and remains mandatory in V2. fileciteturn0file0

## 15.2 Semantic retrieval API split

Expose separate retrieval-oriented surfaces such as:
- `segment_markers`
- `retrieval_index_v2`
- `entity_registry`
- `context_pack_summary`
- `export_review_bundle`

These are not transcript replacements.

---

# 16. Metrics and evaluation

A serious pipeline must measure both transcript quality and memory usefulness.

## 16.1 Acoustic and lexical metrics

- ASR candidate disagreement ratio per stripe
- edge-repair rate
- duplicate suppression rate
- empty-window ratio after triage
- language-switch preservation incidents
- lexical synthesis usage rate
- deterministic fallback rate
- unsupported token ratio
- validation rejection rate
- canonical stabilization delay

## 16.2 Semantic metrics

- entity-link acceptance rate
- indirect-alias hit rate
- ambiguity rate
- project-tag precision audit
- emotion-tag disagreement rate
- retrieval grounding rate
- pack-fact usage rate

## 16.3 External-loop metrics

- pack import acceptance rate
- graph correction rate
- human override rate
- semantic drift after import

Evaluation must be performed on real multilingual field recordings, not just clean benchmark audio. The earlier canonical document already established that the retrieval layer must point back to canonical segments; V2 extends that requirement to semantic markers and packs as well. fileciteturn0file0

---

# 17. Canonical acceptance criteria

The V2 system is accepted only if all of the following are true:

1. No final transcript path reads directly from transport chunks as final truth.
2. Every interior 15-second stripe is reconcilable from at least two supporting windows whenever those windows exist.
3. Every canonical segment retains support provenance.
4. The lexical synthesis stage may combine multiple witnesses, but every accepted output is validated against evidence support.
5. Semantic markers are stored separately from canonical transcript text.
6. Retrieval indexes include grounded semantic markers, not only keywords.
7. Entity graph and context packs are versioned and auditable.
8. Imported external knowledge cannot silently rewrite historical lexical truth.
9. Language attribution remains local and may vary within one session.
10. The system degrades gracefully under missing models, missing packs, or failed LLM synthesis without fabricating transcript text.

---

# 18. Migration rules from previous server generation

## 18.1 Keep

Keep these foundations:
- 30s ingest / 30s decode / 15s stride / 15s stripe geometry,
- candidate persistence,
- stripe grouping,
- canonical segments as user-facing lexical truth,
- partial vs stabilized vs final transcript split.

These were already correctly established in the earlier canonical specification. fileciteturn0file0

## 18.2 Rename

Rename role-misleading stages and outputs such as model-branded candidate stage names.

Example:
- `candidate_asr_parakeet` → `candidate_asr_secondary`

## 18.3 Add

Add these missing artifacts:
- `reconciliation/lexical_synthesis_result.json`
- `reconciliation/validation_audit.json`
- `enrichment/segment_markers.json`
- `enrichment/semantic_spans.json`
- `memory/entities.json`
- `memory/context_packs/*.json`
- `derived/retrieval_index_v2.json`
- `exports/review_bundle_*.json`
- `imports/import_manifest_*.json`

## 18.4 Remove as canonical assumptions

Remove the assumption that reconciliation means “choose the best candidate string.”

In V2, reconciliation means **bounded lexical synthesis with evidence audit**.

---

# 19. MVP implementation sequence

| Phase | Deliverable | What must be true before moving on |
| --- | --- | --- |
| Phase A | Ingest + normalization + timeline | Chunk reconstruction and timestamps are trustworthy |
| Phase B | Acoustic triage + speech islands | Speech-bearing vs non-speech regions are inspectable |
| Phase C | Decode lattice + one ASR | Windows are synthesized correctly and persisted |
| Phase D | Multi-ASR witnesses + sequential scheduler | All witness models process the same windows |
| Phase E | Candidate diagnostics + stripe packets | Each stripe packet contains auditable evidence and local flags |
| Phase F | Bounded lexical synthesis | LLM can synthesize locally with evidence validation |
| Phase G | Canonical assembly + transcript API split | Provisional, stabilized, and final surfaces are distinct |
| Phase H | Semantic marking | Segment markers exist separately from canonical text |
| Phase I | Memory graph + context packs | Entities, aliases, roles, and packs are versioned |
| Phase J | Retrieval V2 + export/import loop | The system can search meaningfully and import curated memory knowledge |

---

# 20. Practical example

## 20.1 Acoustic layer

Transport uploaded by app:
- `C0 = [0,30]`
- `C1 = [30,60]`

Server decode windows:
- `W0 = [0,30]` full
- `W1 = [15,45]` bridge
- `W2 = [30,60]` full

Commit stripes:
- `S0 = [0,15]`
- `S1 = [15,30]`
- `S2 = [30,45]`
- `S3 = [45,60]`

## 20.2 Lexical layer

Candidates for `S2`:
- Witness A: “...чтобы проверить насколько сервер работает”
- Witness B: “...чтобы проверить, насколько сервер работает”
- Witness C: translated drift output in English

Lexical synthesizer result:
- final text uses A and B,
- C is down-weighted due to language mismatch,
- audit records exact witnesses used.

## 20.3 Semantic layer

Stabilized segment text:
- “Я опять думал про наркомана из Бангкока.”

Semantic marker result:
- entity link: `person_mattiew`
- topic tags: `relationship`, `rumination`
- retrieval terms: `Mattiew`, `наркоман из Бангкока`

The semantic layer does **not** rewrite the transcript into “I was thinking about Mattiew.” It preserves the original spoken phrase and attaches meaning separately.

---

# 21. What this architecture prevents

- boundary clipping being mistaken for transcript truth,
- duplicate text from naïve overlap concatenation,
- one model’s bad edge decode dominating all final output,
- silent translation drift being treated as equivalent witness evidence,
- giant transcript rewrites by an unconstrained LLM,
- semantic markers corrupting lexical transcript truth,
- memory retrieval depending only on exact keyword matches,
- imported assistant knowledge silently altering historical speech,
- building a fake “memory AI” that cannot point back to grounded audio evidence.

---

# 22. Final implementation mandate

> Build the next application around a dumb client and an intelligent, layered server.  
> Treat 30-second uploads as transport only.  
> Decode on a 30-second / 15-second overlap lattice.  
> Synthesize lexical truth locally by 15-second stripes from multiple ASR witnesses.  
> Validate that synthesis against evidence.  
> Commit canonical stabilized segments as user-facing lexical truth.  
> Attach semantic memory markers in a separate layer.  
> Maintain versioned context packs and entity graphs for future retrieval.  
> Let retrieval point back to grounded segments, not only summaries.
