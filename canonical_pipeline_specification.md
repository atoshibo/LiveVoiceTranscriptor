# Canonical Pipeline Specification
for the Next-Generation Wearable Audio Memory System

Version 1.0  •  Canonical engineering document  •  12 April 2026
Purpose: define the frozen server-side pipeline, data contracts, stabilization logic, and retrieval model for the next application generation.

| Status | Canonical reference specification for implementation |
| --- | --- |
| Primary domain | Continuous multilingual audio capture, transcription, enrichment, and memory retrieval |
| Locked architectural center | 30-second transport chunks at ingest; 30-second decode windows with 15-second stride; 15-second stabilization stripes |
| Audience | Application architect, backend engineer, pipeline engineer, ASR/LLM engineer, data engineer, QA |

| Canonical sentence<br>The app uploads 30-second transport chunks, but the server’s final transcript truth is built from overlap-aware 30-second decode windows with 15-second stride, reconciled stripe-by-stripe with explicit provenance across models, full windows, and bridge windows. |
| --- |

# 1. Executive summary

This document defines the perfect canonical pipeline for the next version of the system. It is designed for a wearable or device that records continuously, an application layer that performs only transport duties, and a server that performs all heavy reasoning, filtering, transcription, stabilization, enrichment, and retrieval.

The central design choice is that transport shape and transcription truth are different things. The client sends 30-second chunks because this is practical for buffering, resend, and timestamping. The server must not treat those chunks as final truth. Instead, the server builds a continuous normalized timeline, creates overlapping decode windows, reconciles them locally, and commits only stabilized canonical segments.

This architecture is specifically intended to avoid the mistakes that break long-form multilingual audio systems: fixed chunks as final truth, one dominant language for an entire long session, full-stream ASR before triage, global summarization too early, giant whole-transcript rewrites, and diarization applied blindly to all audio regardless of content.

# 2. Scope, design goals, and non-goals

## 2.1 Scope

The scope of this specification covers continuous audio ingestion, normalization, acoustic triage, speech island formation, overlap-aware multi-ASR decoding, bounded LLM reconciliation, selective diarization, canonical segment assembly, derived transcript production, and memory retrieval indexing.

## 2.2 Design goals

- Preserve what was actually spoken, in the language(s) actually spoken, with no default translation.
- Protect word boundaries, speaker turns, and language switches at transport chunk edges.
- Use bounded local reasoning instead of giant whole-transcript rewriting.
- Keep the application simple: capture, buffer, resend, timestamp, upload.
- Move all intelligence to the server so behavior is consistent across devices and sessions.
- Make every canonical segment traceable back to concrete model outputs and concrete time spans.
- Support partial live output without confusing provisional text with final truth.
- Scale to constrained VRAM by standardizing decode geometry and scheduling models sequentially.
## 2.3 Explicit non-goals

- No fixed-length transport chunk may be treated as the final transcription unit.
- No single dominant language may be forced across an entire long session.
- No whole-day ASR pass may run before acoustic filtering and segmentation.
- No global summary may become the basis of canonical transcript truth.
- No giant whole-transcript LLM rewrite is allowed to replace local evidence-based reconciliation.
- No diarization should run on every second of audio by default.
# 3. Frozen architectural decisions

| Decision | Frozen value | Reason |
| --- | --- | --- |
| Transport chunk length | 30 seconds | Stable upload size; practical buffering and resend behavior on device/app. |
| Decode window length | 30 seconds | Common bounded decode geometry across models; manageable on limited VRAM. |
| Decode stride | 15 seconds | Creates bridge context at boundaries and gives each interior stripe two supporting windows. |
| Canonical commit unit | 15-second stabilization stripe | Final truth is committed stripe-by-stripe, not chunk-by-chunk. |
| Acoustic triage position | Before expensive ASR | Avoid wasting VRAM and latency on noise, TV, music, or non-speech. |
| Reconciliation scope | Local and bounded | Prevents hallucinated whole-transcript rewrites and preserves provenance. |
| Diarization position | Selective, after stabilization or on justified regions | Apply only where multi-speaker evidence exists. |
| Memory retrieval base | Canonical stabilized segments | Retrieval must index grounded transcript truth, not only summaries. |

# 4. Core conceptual model

The pipeline is built on three different time abstractions. Their separation is mandatory.

First, the transport chunk is the upload unit used by the application. Second, the decode window is the overlap-aware unit used by the ASR engines. Third, the stabilization stripe is the commit unit used by the reconciler and canonical assembler. If these three abstractions are collapsed into one, the system will produce edge truncation, duplicate text, contradictory repairs, and bad final assembly.

| Object | Definition | Canonical example | Owner |
| --- | --- | --- | --- |
| TransportChunk C_n | Uploaded client payload placed on absolute timeline | C0=[0,30], C1=[30,60] | Application → Server |
| DecodeWindow W_k | 30-second server-side window cut from normalized timeline with 15-second stride | W0=[0,30], W1=[15,45], W2=[30,60] | Server ASR scheduler |
| CommitStripe S_k | 15-second stabilization unit committed only after enough evidence exists | S1=[15,30], S2=[30,45] | Reconciler / Assembler |
| SpeechIsland | Merged adjacent speech region on normalized timeline | example: [12.4, 53.1] | Acoustic triage |
| Candidate | One model’s output for one decode window | Whisper on W1; Parakeet on W1 | ASR layer |
| CanonicalSegment | Final stabilized segment with text, timestamps, provenance, and optional speaker attribution | segment_id=seg_000231 | Assembler |

# 5. End-to-end canonical pipeline

## Stage 1 — Capture

The wearable or device records continuously. The application performs capture, buffering, resend, timestamp preservation, and transport upload only. No language decisions, summarization, diarization, or semantic inference belong here.

## Stage 2 — Ingest and normalization

The server receives transport chunks, decodes original media, normalizes sample rate/channels, and places each chunk on an absolute session timeline.

## Stage 3 — Acoustic triage

The server classifies time regions as speech, non-speech, music/media, noise, mixed, or uncertain. It merges adjacent speech-bearing regions into speech islands.

## Stage 4 — Decode lattice construction

The server synthesizes overlapping 30-second decode windows from the normalized timeline using 15-second stride, independent of the original transport chunk boundaries.

## Stage 5 — Multi-ASR execution

For each speech-bearing decode window, the server runs the configured ASR models sequentially under VRAM control and persists raw candidate evidence.

## Stage 6 — Stripe-level evidence grouping

Candidate outputs are regrouped by 15-second stabilization stripe so each stripe sees evidence from its supporting windows and supporting models.

## Stage 7 — Bounded LLM reconciliation

A local reconciler compares model outputs and adjacent-window evidence, resolves contradictions, detects edge truncation, and proposes the best stripe text without rewriting beyond the local packet.

## Stage 8 — Stabilization and canonical assembly

Stripes remain provisional until enough support exists. Once stabilized, they are assembled into canonical segments with timestamps and provenance.

## Stage 9 — Selective enrichment

Diarization, entity extraction, relationship markers, and uncertainty/confidence are applied only where justified by the stabilized speech regions.

## Stage 10 — Derived outputs and retrieval

Plain transcript text, digests, repeated themes, unresolved threads, and search indexes are all derived from canonical segments rather than from provisional or rewritten text.

# 6. Detailed stage specification

## 6.1 Stage 1 — Capture and transport

The application is intentionally dumb. This is not a weakness; it is a design requirement. If the application starts performing intelligent segmentation, language selection, or summarization, behavior will diverge across clients and debugging will become unreliable.

Application responsibilities are limited to:
• continuous capture
• buffer management
• resend on failure
• preservation of capture timestamps
• upload of 30-second transport chunks
• metadata envelope attachment

The application must not decide what is speech, what language dominates, whether diarization is needed, or what the final transcript should say.

Required metadata envelope

| Field | Type | Meaning |
| --- | --- | --- |
| session_id | string | Stable session identifier. |
| chunk_index | integer | Monotonic transport chunk number. |
| capture_start_ms | integer | Absolute session-relative start timestamp. |
| capture_end_ms | integer | Absolute session-relative end timestamp. |
| codec/container | string | Original media transport format. |
| device_clock_info | object | Clock origin or drift metadata if available. |
| allowed_languages | array<string> | Optional language allowlist configured by user or session. |
| forced_language | string|null | Optional hard override; use sparingly. |
| transcription_mode | string | For example verbatim_multilingual. |
| diarization_policy | string | For example auto, off, forced. |

## 6.2 Stage 2 — Ingest and normalization

Each transport chunk is decoded and converted into a server-side normalized representation. The normalized timeline is the only timeline the rest of the pipeline should trust. The ingest stage may preserve original media for audit purposes, but downstream windowing must operate on normalized audio.

Normalization responsibilities include consistent sample rate, channel handling, timestamp placement, and gap/overlap accounting. The output of this stage is not yet transcript text. It is a growing session timeline plus ingest metadata.

## 6.3 Stage 3 — Acoustic triage

Acoustic triage happens before expensive ASR. This stage determines whether a region is worth transcription and what caution flags should follow it. The purpose is not to remove difficult audio from the system. The purpose is to separate speech-bearing work from obvious waste and to mark mixed conditions explicitly.

Recommended tags are: speech, non_speech, music_media, noise, mixed, uncertain. Adjacent speech regions should be merged into speech islands using conservative merge logic so short pauses do not fragment a single utterance into many tiny units.

Scheduling rule after triage

A decode window is eligible for expensive ASR when it intersects a speech island strongly enough to justify compute. Windows that are overwhelmingly non-speech may be skipped or downgraded. Bridge windows remain mandatory whenever a speech island crosses the chunk boundary, even if the amount of speech in that bridge window is modest.

## 6.4 Stage 4 — Decode lattice construction

This stage is the center of the architecture. The server constructs 30-second decode windows with 15-second stride from the normalized timeline.

For a session beginning at time zero, the first windows are:
W0 = [0,30]
W1 = [15,45]
W2 = [30,60]
W3 = [45,75]

This geometry means each interior 15-second stripe is observed twice: once near the trailing half of one window and once near the leading half of the next. That is exactly what protects the system from boundary damage.

| Why this geometry is locked<br>The app still uploads C0=[0,30] and C1=[30,60], but the server does not decode only C0 then C1 as final truth. It also synthesizes W1=[15,45], which contains the last 15 seconds of C0 and the first 15 seconds of C1. Without that bridge, boundary words, speaker turns, and language switches are regularly mangled or duplicated. |
| --- |

## 6.5 Stage 5 — Multi-ASR execution

For each decode window that survives scheduling, the server runs the configured ASR models on the exact same audio geometry. This is extremely important. Different models may have different native capabilities, but for canonical reconciliation they must be compared on the same evidence slice.

On constrained VRAM, models must be scheduled sequentially, not simultaneously. The scheduler must load one model family, drain the queue it is responsible for, persist outputs, free memory, then move to the next model.

Recommended execution policy

- Standardize decode geometry across all ASR models to 30-second windows with 15-second stride.
- Run models sequentially under a bounded scheduler.
- Persist each candidate immediately after decoding to avoid re-running windows after failure.
- Treat each model as a comparable witness, not as an unquestioned authority.
- Do not let one model’s language guess pre-filter another model’s decode unless the policy explicitly requires it.
## 6.6 Stage 6 — Candidate persistence and provenance

Every raw ASR output must be persisted as a first-class artifact. The reconciler may never operate on anonymous text strings. Each candidate must retain model provenance, exact time span, and window type.

| Candidate field | Meaning |
| --- | --- |
| candidate_id | Stable unique identifier. |
| session_id | Owning session. |
| model_id | For example whisper_large_v3 or parakeet. |
| window_id | Owning decode window. |
| window_start_ms / window_end_ms | Absolute time span on normalized timeline. |
| window_type | full or bridge. |
| raw_text | Verbatim model text for that window. |
| segments / words | Model-native segment or word timestamps when available. |
| language_evidence | Per-window or per-segment language hints, not global session truth. |
| confidence_features | Raw confidence, logprob, no-speech score, repetition score, anomaly markers. |
| decode_metadata | Prompt state, runtime flags, tokenizer mode, model revision, etc. |

## 6.7 Stage 7 — Stripe-level evidence grouping

The server does not reconcile windows directly into a final transcript. It first regroups evidence by 15-second stabilization stripe.

For example:
S1=[15,30] is supported by W0 right half and W1 left half.
S2=[30,45] is supported by W1 right half and W2 left half.

If two models are configured, each stripe packet will ideally contain four local witnesses: two models × two supporting windows.

## 6.8 Stage 8 — Trust zones inside each decode window

Window edges are less reliable than window centers. The assembler and reconciler must know that. A practical rule is to define an edge trust penalty for the first and last 5 seconds of each 30-second decode window, leaving the middle 20 seconds as the highest-trust zone.

The exact threshold may later be tuned, but the concept is fixed: text aligned closer to the center of a supporting window is generally more trustworthy than text aligned at the very edge.

## 6.9 Stage 9 — Bounded LLM reconciliation

The LLM is used as a local evidence arbiter, not as a transcript writer. Its job is to choose the most plausible stripe text from bounded local evidence and to preserve uncertainty where the evidence remains weak.

The LLM input packet for one stripe should contain:
• stripe absolute time span
• left and right supporting windows
• model IDs and window types
• raw candidate text snippets for the stripe region
• local timestamps and trust-zone information
• neighboring stripe context when available
• language hints and anomaly markers

Hard rules for the reconciler:
• do not translate by default
• do not summarize
• do not improve style
• do not globally rewrite content
• preserve code-switching
• prefer supported wording over polished wording
• emit uncertainty rather than inventing content

## 6.10 Stage 10 — Stabilization logic

A stripe becomes canonical only when enough supporting evidence exists. That means the system must separate provisional text from stabilized text.

Example:
When only C0 has arrived, W0=[0,30] can be decoded. The stripe S1=[15,30] exists, but it has only one supporting window. It may be surfaced as provisional for live UX, but it is not yet fully stabilized.

When C1 arrives, the server can synthesize W1=[15,45] and W2=[30,60]. Now S1 and S2 each have two supporting windows and can be stabilized more safely.

This means the server should intentionally carry a one-window stabilization delay rather than pretending early chunk text is final truth.

## 6.11 Stage 11 — Canonical segment assembly

After stripe stabilization, adjacent compatible stripes are merged into canonical segments. Canonical segments are the lowest user-facing truth unit. Everything else is derived from them.

| Canonical segment field | Meaning |
| --- | --- |
| segment_id | Stable segment identifier. |
| start_ms / end_ms | Absolute time span on normalized session timeline. |
| speaker | Optional speaker label if justified by selective diarization. |
| text | Final stabilized text. |
| language | Local language attribution if available; may change across segments. |
| confidence | Aggregate confidence or support score. |
| support_windows | Set of windows that supported this segment. |
| support_models | Set of models that supported this segment. |
| stabilization_state | provisional, stabilized, or corrected. |

## 6.12 Stage 12 — Selective enrichment

Enrichment is downstream of canonical lexical truth. It must never become the hidden source of transcript words.

Selective enrichment includes:
• diarization where multi-speaker evidence is likely
• entity extraction
• relationship and context markers
• uncertainty/confidence projection
• topic or project markers for retrieval

Diarization is selective, not universal. It should run where the content or prior classifier indicates a real chance of more than one speaker, overlapping turns, or conversation-like structure.

## 6.13 Stage 13 — Derived outputs and memory retrieval

Derived outputs include user-readable transcript text, daily digests, repeated themes, unresolved threads, person-centric recall, and project/topic recall. All of these are downstream products. None of them may overwrite canonical segments.

Memory retrieval must index canonical segments plus enriched metadata. It may also index embeddings of those segments and of higher-level aggregates, but retrieval must always be able to point back to grounded segments.

# 7. Data contracts and object definitions

## 7.1 SessionManifest

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
"commit_stripe_ms": 15000
}

## 7.2 DecodeWindow

{
"window_id": "W000123",
"start_ms": 15000,
"end_ms": 45000,
"window_type": "bridge",
"source_chunks": ["C0000", "C0001"],
"speech_intersection_ratio": 0.73,
"scheduled": true
}

## 7.3 Candidate

{
"candidate_id": "cand_9d3f",
"session_id": "uuid",
"model_id": "whisper_large_v3",
"window_id": "W000123",
"window_type": "bridge",
"start_ms": 15000,
"end_ms": 45000,
"raw_text": "...",
"segments": [...],
"language_evidence": [...],
"confidence_features": {
"avg_logprob": -0.24,
"repetition_ratio": 0.03,
"anomaly_flags": []
}
}

## 7.4 CanonicalSegment

{
"segment_id": "seg_000231",
"start_ms": 30210,
"end_ms": 34490,
"speaker": "S1",
"language": "fr",
"text": "...",
"confidence": 0.87,
"support_windows": ["W000123", "W000124"],
"support_models": ["whisper_large_v3", "parakeet"],
"stabilization_state": "stabilized"
}

# 8. Partial output policy versus final truth

The user interface may display partial or near-live text, but that text must be explicitly treated as provisional. A safe contract is:

• provisional_partial: produced from current available support only
• stabilized_partial: produced from stripe(s) that already have the minimum support for stabilization
• final_transcript: built only from canonical stabilized segments

This distinction prevents a common product failure where users see unstable early text, assume it is final, and later interpret corrections as bugs.

# 9. Multi-language behavior

Language handling is local and time-varying. The system must not force one dominant language onto a long multilingual session. Allowed languages may constrain routing, but language attribution must remain local to windows, stripes, or final segments.

Implications:
• no session-wide language poisoning
• no translation by default
• code-switching is preserved
• entity extraction and retrieval may use language-aware normalization downstream without changing canonical segment text

# 10. Failure handling and resilience

| Failure mode | Required behavior |
| --- | --- |
| Transport chunk missing or delayed | Mark timeline gap; do not silently collapse time. Continue session when possible. |
| Normalization failure for one chunk | Persist error artifact, retain original media, flag session, and avoid fake continuity. |
| ASR model crash on one window | Persist failure for that model/window pair; continue with remaining models/windows. |
| One model unavailable | Allow degraded reconciliation with remaining candidates; mark support deficit. |
| LLM reconciliation failure | Fall back to deterministic assembly policy using support weighting and trust zones. |
| Low-confidence stripe | Emit uncertainty marker or hold as provisional rather than invent text. |
| Diarization model unavailable | Keep lexical transcript canonical and omit speaker labels rather than guessing. |

# 11. Storage and artifact layout

A clean implementation should persist pipeline artifacts by stage while exposing a small canonical read surface for the application.

Recommended storage layers:
• raw/: original uploaded transport chunks and ingest metadata
• normalized/: session-normalized audio timeline artifacts
• triage/: acoustic tags, speech islands, scheduler decisions
• windows/: synthesized decode windows and queue state
• candidates/: per-model raw ASR outputs
• reconciliation/: stripe packets, arbitration outputs, uncertainty markers
• canonical/: canonical segments and final transcript surfaces
• derived/: digests, indexes, themes, unresolved threads, subtitles, QA reports

The application should read mainly from canonical/ and selected derived/ artifacts, not from intermediate stage folders.

# 12. Metrics and validation

A serious pipeline needs operational metrics that reveal whether the architecture is helping or harming quality. Recommended metrics include:
• ASR candidate disagreement ratio per stripe
• edge-repair rate: how often bridge support changed the preferred text
• duplicate suppression rate
• empty-window ratio after acoustic triage
• language-switch preservation incidents
• LLM rejection or uncertainty rate
• canonical stabilization delay
• diarization damage rate on stabilized text
• retrieval grounding rate: percentage of retrieval outputs traceable to canonical segments

Evaluation must be performed against real multilingual field recordings, not only clean benchmark audio.

# 13. Canonical acceptance criteria

- No final transcript path may read directly from transport chunks as final truth.
- Every interior 15-second stripe must be reconcilable from at least two supporting windows whenever those windows exist.
- Every canonical segment must retain support provenance.
- The reconciler must operate on bounded local packets and must not perform whole-transcript stylistic rewriting.
- The system must separate provisional and stabilized outputs in both storage and API semantics.
- Language attribution must remain local and may vary over time within one session.
- The retrieval layer must point back to canonical segments.
- The system must gracefully degrade under single-model or failed-window conditions without fabricating transcript text.
# 14. MVP implementation sequence

| Phase | Deliverable | What must be true before moving on |
| --- | --- | --- |
| Phase A | Ingest + normalization + absolute timeline | Transport chunks are reconstructed correctly and timestamp placement is trustworthy. |
| Phase B | Acoustic triage + speech islands | Speech-bearing vs non-speech regions are visible and scheduler decisions are inspectable. |
| Phase C | Decode lattice + one ASR | Windows are synthesized correctly and persisted with provenance. |
| Phase D | Second ASR + sequential scheduler | Both models process the same windows under VRAM constraints. |
| Phase E | Stripe grouping + deterministic alignment | The system can compare supporting windows and suppress obvious duplicates without LLM help. |
| Phase F | Bounded LLM reconciliation | The system improves local quality without global rewrite behavior. |
| Phase G | Canonical assembly + partial/final API split | Users and downstream services can distinguish provisional from stabilized truth. |
| Phase H | Selective enrichment + retrieval index | Search and memory features are grounded on canonical segments. |

# 15. Practical example: first 60 seconds of a session

Transport uploaded by app:
C0 = [0,30]
C1 = [30,60]

Server decode windows:
W0 = [0,30]   full
W1 = [15,45]  bridge
W2 = [30,60]  full

Commit stripes:
S0 = [0,15]   supported mainly by W0
S1 = [15,30]  supported by W0 and W1
S2 = [30,45]  supported by W1 and W2
S3 = [45,60]  supported mainly by W2 until the next bridge window exists

Interpretation:
• S1 is where chunk-boundary protection begins.
• S2 is where bridge support is essential because it spans the transport boundary directly.
• S3 may remain provisional until W3=[45,75] becomes available, if the session continues.

Transport chunks:   C0=[0,30]             C1=[30,60]
Decode windows:       W0=[0,30]
W1=[15,45]
W2=[30,60]
Commit stripes:       S0=[0,15] S1=[15,30] S2=[30,45] S3=[45,60]

# 16. What this architecture prevents

- Boundary word clipping being mistaken for real transcript content.
- Duplicates created by naïvely concatenating overlapping windows.
- One model’s bad edge decode dominating the final transcript.
- Session-wide dominant language contamination over local multilingual speech.
- Wasted compute on TV, music, pure noise, or non-speech regions.
- LLM hallucination caused by asking for a whole transcript rewrite instead of a bounded local decision.
- Speaker diarization corrupting lexical truth by running in regions where it is not justified.
- Memory retrieval indexing summaries or rewritten text that cannot be traced back to audio-grounded transcript units.
# 17. Final implementation mandate

| Frozen implementation mandate<br>Build the next application around a dumb client and an intelligent server.<br>Treat 30-second client uploads as transport only.<br>Decode on a 30-second / 15-second overlap lattice.<br>Reconcile locally by 15-second stripes.<br>Commit canonical stabilized segments as user-facing truth. |
| --- |
