"""
Stage 7 - Bounded lexical synthesis preparation.

The reconciliation layer now emits a lexical-truth contract rather than a
bare "pick one string" result. This iteration stays conservative:

- LLM output is still validated against local evidence
- deterministic fallback still prefers an exact supported candidate
- the artifact shape now supports future bounded multi-witness synthesis

Hard rules:
  - do not translate by default
  - do not summarize
  - do not improve style
  - do not globally rewrite content
  - preserve code-switching
  - prefer supported wording over polished wording
  - emit uncertainty rather than inventing content

Outputs:
  - reconciliation/reconciliation_result.json
  - reconciliation/lexical_synthesis_result.json
  - reconciliation/validation_audit.json
"""
import json
import logging
import re
import time
from collections import Counter
from typing import List, Dict, Optional, Tuple

from app.core.config import get_config
from app.core.atomic_io import atomic_write_json
from app.models.registry import DEFAULT_CANDIDATE_A, DEFAULT_CANDIDATE_B, DEFAULT_FIRST_PASS, resolve_model_id
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)

ASSEMBLY_MODE_EXACT = "exact_candidate"
ASSEMBLY_MODE_NORMALIZED = "normalized_candidate"
ASSEMBLY_MODE_SYNTHESIZED = "synthesized_from_multiple_candidates"
ASSEMBLY_MODE_FALLBACK = "deterministic_fallback"
ASSEMBLY_MODE_SUPPRESSED = "suppressed_junk"
LEXICAL_CONTRACT_VERSION = "2.1"
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Candidate priority order for deterministic fallback
# (higher priority = preferred when no LLM available)
MODEL_PRIORITY = {
    resolve_model_id(DEFAULT_CANDIDATE_A): 3,
    resolve_model_id(DEFAULT_CANDIDATE_B): 2,
    resolve_model_id(DEFAULT_FIRST_PASS): 1,
}

# Flags that discredit a candidate during fallback scoring.  Any single one of
# these is enough to demote the candidate below a clean sibling.
CORRUPTION_PENALTY_FLAGS = {
    "language_mismatch": 1.4,
    "script_mismatch": 1.4,
    "possible_translation": 0.9,
    "repetition_anomaly": 1.1,
    "semantic_drift": 0.7,
    "empty_candidate": 2.0,
    "provider_degraded": 0.3,
    "edge_truncation_suspected": 0.15,
    "statement_question_drift": 0.2,
    "media_pollution_suspected": 2.0,
}

# Known subtitle / media pollution patterns (case insensitive).  These are the
# exact strings (or near-exact) that slip into ASR output when a model
# hallucinates on silence, music, or training-set contamination.  Matching any
# of these suppresses the candidate before it can reach canonical truth.
_JUNK_PATTERNS = [
    # Russian / CIS YouTube subtitle credits
    r"субтитры\s+сделал",
    r"субтитры\s+подготовил",
    r"корректор\s+\w+",
    r"редактор\s+\w+",
    r"dimatorzok",
    r"продолжение\s+следует",
    r"подписывайтесь",
    r"спасибо\s+за\s+просмотр",
    # English generic YouTube / media pollution
    r"thanks?\s+for\s+watching",
    r"subscribe\s+to\s+(my|our)\s+channel",
    r"like\s+and\s+subscribe",
    r"^\s*\[music\]\s*$",
    r"^\s*\(music\)\s*$",
    r"^\s*\[applause\]\s*$",
    # French media pollution
    r"sous[- ]titres?\s+réalisés?\s+par",
    r"merci\s+d(?:'|e\s+)avoir\s+regardé",
]
_JUNK_REGEX = re.compile("|".join(_JUNK_PATTERNS), re.IGNORECASE | re.UNICODE)


def looks_like_media_junk(text: str) -> bool:
    """Return True if the text looks like subtitle/media pollution.

    This is the layer-B firewall: canonical lexical truth must not include
    known training-set contamination strings.  Used by both stripe-level
    suppression and downstream retrieval filtering.
    """
    if not text or not text.strip():
        return False
    return bool(_JUNK_REGEX.search(text))


def _candidate_corruption_penalty(item: Dict) -> float:
    """Total corruption penalty from candidate_flags.  0.0 = clean."""
    flags = item.get("candidate_flags") or []
    penalty = 0.0
    for flag in flags:
        penalty += CORRUPTION_PENALTY_FLAGS.get(flag, 0.0)
    if looks_like_media_junk(item.get("text", "")):
        penalty += 1.0
    return penalty

RECONCILIATION_PROMPT_TEMPLATE = """You are a bounded lexical reconciliation engine.

TIME SEGMENT: {start_ms}ms - {end_ms}ms

CANDIDATES:
{candidates_text}

RULES:
1. Stay evidence-bound. Do NOT translate, summarize, or improve style.
2. If one candidate is clearly best, return that supported text.
3. Preserve the original language and any code-switching.
4. If evidence is weak, prefer uncertainty or empty text over invention.
5. Respond only with supported wording from the local evidence in this iteration.

Respond with ONLY a JSON object:
{{"text": "best supported text", "source_model": "model_id", "confidence": 0.0-1.0, "reason": "brief reason"}}
"""


def _load_llm():
    """Load the local LLM for reconciliation."""
    cfg = get_config()
    if not cfg.reconciliation.use_llm:
        return None, "llm_disabled_by_config"

    llm_path = cfg.model_paths.llm_path
    if not llm_path:
        logger.info("No LLM model path configured, using deterministic fallback")
        return None, "llm_model_missing"

    try:
        from llama_cpp import Llama
        llm = Llama(
            model_path=llm_path,
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False,
        )
        return llm, "llm_loaded"
    except ImportError as e:
        logger.warning("llama_cpp not installed, using deterministic fallback: %s", e)
        return None, f"llama_cpp_missing:{e}"
    except Exception as e:
        logger.warning("Failed to load LLM: %s", e)
        return None, f"llm_load_failed:{e}"


def _call_llm(llm, prompt: str, max_tokens: int = 300) -> Optional[str]:
    """Call the LLM with the reconciliation prompt."""
    try:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            stop=["\n\n"],
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return None


def _parse_llm_response(raw: str) -> Optional[Dict]:
    """Parse LLM JSON response with tolerance for markdown/preamble."""
    if not raw:
        return None

    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "text" in data:
            return data
    except json.JSONDecodeError:
        pass

    try:
        start = text.index("{")
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            char = text[i]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        data = json.loads(text[start:i + 1])
                        if isinstance(data, dict) and "text" in data:
                            return data
                        break
    except (ValueError, json.JSONDecodeError):
        pass

    return None


SYNTHESIS_TOKEN_SUPPORT_MIN = 0.85


def _validate_llm_selection(parsed: Dict, candidates: List[Dict]) -> Tuple[bool, str]:
    """Audited synthesis validator.

    Accepts three forms of LLM output, in order:
      1. exact match of one candidate's text -> accepted as exact selection
      2. normalized match of one candidate's text -> accepted as normalized
      3. synthesized text whose tokens are >=85% supported by the combined
         evidence vocabulary -> accepted as bounded synthesis

    The third mode is the spec's 'bounded lexical synthesis' contract.
    Rejecting it (as the old validator did) is why the fallback swamped the
    LLM path in production.
    """
    if not parsed or not parsed.get("text"):
        return False, "empty_llm_text"

    selected_text = _normalize_text(parsed["text"])
    if not selected_text:
        return False, "empty_llm_text"

    # Reject hallucinated media/subtitle pollution, regardless of source model.
    if looks_like_media_junk(parsed["text"]):
        return False, "llm_emitted_media_junk"

    source_model = parsed.get("source_model", "")

    # Tier 1/2: exact or normalized match to a single candidate.
    for cand in candidates:
        if source_model and cand.get("model_id") != source_model:
            continue
        candidate_text = _normalize_text(cand.get("text", ""))
        if candidate_text and selected_text == candidate_text:
            return True, "exact_candidate_match"

    for cand in candidates:
        candidate_text = _normalize_text(cand.get("text", ""))
        if candidate_text and selected_text == candidate_text:
            return True, "normalized_candidate_match"

    # Tier 3: bounded synthesis -- tokens must come from evidence vocabulary.
    llm_tokens = _tokenize(parsed["text"])
    if not llm_tokens:
        return False, "no_tokens_in_llm_text"

    evidence_vocab = set()
    for cand in candidates:
        evidence_vocab.update(_tokenize(cand.get("text", "")))

    if not evidence_vocab:
        return False, "no_evidence_vocabulary"

    supported = sum(1 for tok in llm_tokens if tok in evidence_vocab)
    ratio = supported / len(llm_tokens)
    if ratio >= SYNTHESIS_TOKEN_SUPPORT_MIN:
        return True, f"bounded_synthesis:{ratio:.2f}"

    return False, f"unsupported_synthesis:{ratio:.2f}"


def _select_fallback(evidence: List[Dict]) -> Dict:
    """Deterministic fallback selector.

    Scoring: priority * trust - corruption_penalty.  Candidates whose text is
    a known media/subtitle hallucination (e.g. "Субтитры сделал DimaTorzok")
    are suppressed entirely rather than promoted to canonical truth, even if
    they are the only non-empty candidate for a stripe.
    """
    if not evidence:
        return {
            "chosen_text": "",
            "chosen_source": "none",
            "confidence": 0.0,
            "method": "fallback",
            "fallback_reason": "no_evidence",
        }

    non_empty = [item for item in evidence if item.get("text", "").strip()]
    if not non_empty:
        return {
            "chosen_text": "",
            "chosen_source": "none",
            "confidence": 0.0,
            "method": "fallback",
            "fallback_reason": "all_empty",
        }

    clean = [item for item in non_empty if not looks_like_media_junk(item.get("text", ""))]
    junk_suppressed = len(non_empty) - len(clean)

    # If every candidate is media junk, do not canonicalize it -- return empty
    # text and let the uncertainty layer mark the stripe as suppressed.
    if not clean:
        return {
            "chosen_text": "",
            "chosen_source": "none",
            "confidence": 0.0,
            "method": "fallback",
            "fallback_reason": "all_candidates_media_junk",
            "suppressed_count": junk_suppressed,
        }

    best = None
    best_score = -1e9
    for item in clean:
        priority = MODEL_PRIORITY.get(resolve_model_id(item.get("model_id", "")), 0)
        trust = item.get("trust_score", 0.5)
        penalty = _candidate_corruption_penalty(item)
        score = priority * trust - penalty
        if score > best_score:
            best_score = score
            best = item

    reason = "deterministic_selection"
    if junk_suppressed:
        reason = f"deterministic_selection_after_suppressing_{junk_suppressed}_junk"

    return {
        "chosen_text": best["text"].strip(),
        "chosen_source": best.get("model_id", "unknown"),
        "confidence": max(0.0, min(1.0, best_score / 3.0)),
        "method": "fallback",
        "fallback_reason": reason,
        "suppressed_count": junk_suppressed,
    }


def reconcile_stripe(stripe_packet: Dict, llm=None) -> Dict:
    """Reconcile a single stripe using LLM or deterministic fallback."""
    evidence = stripe_packet.get("evidence", [])
    start_ms = stripe_packet["start_ms"]
    end_ms = stripe_packet["end_ms"]
    support_windows = stripe_packet.get("support_windows") or sorted({
        item.get("window_id") for item in evidence if item.get("window_id")
    })
    support_models = stripe_packet.get("support_models") or sorted({
        item.get("model_id") for item in evidence if item.get("model_id")
    })

    start_time = time.time()
    llm_rejection_reason = None

    if llm and evidence:
        candidates_text = ""
        for i, item in enumerate(evidence):
            candidates_text += f"\n[{i + 1}] Model: {item.get('model_id', 'unknown')}"
            candidates_text += f" | Window: {item.get('window_id', '?')}"
            candidates_text += f" | Trust: {item.get('trust_score', 0.5)}"
            candidates_text += f"\nText: {item.get('text', '')}\n"

        prompt = RECONCILIATION_PROMPT_TEMPLATE.format(
            start_ms=start_ms,
            end_ms=end_ms,
            candidates_text=candidates_text,
        )

        raw_response = _call_llm(llm, prompt)
        parsed = _parse_llm_response(raw_response)

        if parsed:
            candidates_for_validation = [
                {"model_id": item.get("model_id", ""), "text": item.get("text", "")}
                for item in evidence
            ]
            accepted, validation_reason = _validate_llm_selection(parsed, candidates_for_validation)
            if accepted:
                confidence = parsed.get("confidence", 0.5)
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = 0.5
                confidence = max(0.0, min(1.0, confidence))

                min_conf = get_config().reconciliation.min_confidence
                if confidence >= min_conf:
                    elapsed = time.time() - start_time
                    return _build_lexical_record(
                        stripe_packet=stripe_packet,
                        evidence=evidence,
                        final_text=parsed["text"],
                        chosen_source=parsed.get("source_model", "llm"),
                        confidence=confidence,
                        method="llm",
                        fallback_reason=None,
                        raw_llm_response=raw_response,
                        elapsed_s=round(elapsed, 3),
                        support_windows=support_windows,
                        support_models=support_models,
                        validation_reason=validation_reason,
                    )
                llm_rejection_reason = "llm_confidence_below_minimum"
            else:
                llm_rejection_reason = validation_reason or "unsupported_llm_output"
        elif raw_response:
            llm_rejection_reason = "llm_response_unparseable"

    fallback = _select_fallback(evidence)
    elapsed = time.time() - start_time
    return _build_lexical_record(
        stripe_packet=stripe_packet,
        evidence=evidence,
        final_text=fallback.get("chosen_text", ""),
        chosen_source=fallback.get("chosen_source"),
        confidence=fallback.get("confidence", 0.0),
        method=fallback.get("method", "fallback"),
        fallback_reason=fallback.get("fallback_reason"),
        raw_llm_response=None,
        elapsed_s=round(elapsed, 3),
        support_windows=support_windows,
        support_models=support_models,
        llm_rejection_reason=llm_rejection_reason,
    )


def _build_lexical_record(
    stripe_packet: Dict,
    evidence: List[Dict],
    final_text: str,
    chosen_source: Optional[str],
    confidence: float,
    method: str,
    fallback_reason: Optional[str],
    raw_llm_response: Optional[str],
    elapsed_s: float,
    support_windows: List[str],
    support_models: List[str],
    llm_rejection_reason: Optional[str] = None,
    validation_reason: Optional[str] = None,
) -> Dict:
    source_language = _derive_language(stripe_packet, chosen_source)
    used_candidate_ids, exact_match_count, normalized_match_count = _matched_candidate_ids(
        evidence,
        final_text,
        chosen_source,
    )
    unsupported_tokens, token_support_ratio = _unsupported_tokens(final_text, evidence)
    suppressed_junk = fallback_reason == "all_candidates_media_junk"
    assembly_mode = _determine_assembly_mode(
        final_text=final_text,
        method=method,
        exact_match_count=exact_match_count,
        normalized_match_count=normalized_match_count,
        suppressed_junk=suppressed_junk,
        validation_reason=validation_reason,
    )
    uncertainty_flags = _uncertainty_flags(
        final_text=final_text,
        confidence=confidence,
        unsupported_tokens=unsupported_tokens,
        support_windows=support_windows,
        llm_rejection_reason=llm_rejection_reason,
        suppressed_junk=suppressed_junk,
    )
    evidence_notes = _evidence_notes(
        method=method,
        fallback_reason=fallback_reason,
        llm_rejection_reason=llm_rejection_reason,
        used_candidate_ids=used_candidate_ids,
        chosen_source=chosen_source,
        validation_reason=validation_reason,
    )
    if suppressed_junk:
        validation_status = "suppressed_media_junk"
    elif not unsupported_tokens:
        validation_status = "accepted"
    else:
        validation_status = "accepted_with_warnings"

    return {
        "stripe_id": stripe_packet["stripe_id"],
        "start_ms": stripe_packet["start_ms"],
        "end_ms": stripe_packet["end_ms"],
        "final_text": final_text,
        "chosen_text": final_text,
        "chosen_source": chosen_source or "unknown",
        "confidence": confidence,
        "method": method,
        "fallback_reason": fallback_reason,
        "raw_llm_response": raw_llm_response,
        "elapsed_s": elapsed_s,
        "support_windows": support_windows,
        "support_models": support_models,
        "language": source_language,
        "source_language": source_language,
        "output_language": source_language,
        "support_window_count": stripe_packet.get("support_window_count", len(support_windows)),
        "support_model_count": len(support_models),
        "evidence_count": len(evidence),
        "assembly_mode": assembly_mode,
        "used_candidate_ids": used_candidate_ids,
        "used_candidates": used_candidate_ids,
        "unsupported_tokens": unsupported_tokens,
        "token_support_ratio": token_support_ratio,
        "uncertainty_flags": uncertainty_flags,
        "evidence_notes": evidence_notes,
        "validation_status": validation_status,
        "llm_validation_rejected": bool(llm_rejection_reason),
    }


def _determine_assembly_mode(
    final_text: str,
    method: str,
    exact_match_count: int,
    normalized_match_count: int,
    suppressed_junk: bool = False,
    validation_reason: Optional[str] = None,
) -> str:
    if suppressed_junk:
        return ASSEMBLY_MODE_SUPPRESSED
    if method != "llm":
        return ASSEMBLY_MODE_FALLBACK
    if not final_text:
        return ASSEMBLY_MODE_FALLBACK
    if exact_match_count > 0:
        return ASSEMBLY_MODE_EXACT
    if normalized_match_count > 0:
        return ASSEMBLY_MODE_NORMALIZED
    # Bounded synthesis -- token-supported but no single matching candidate.
    return ASSEMBLY_MODE_SYNTHESIZED


def _matched_candidate_ids(
    evidence: List[Dict],
    final_text: str,
    chosen_source: Optional[str],
) -> Tuple[List[str], int, int]:
    final_exact = (final_text or "").strip()
    final_normalized = _normalize_text(final_text)
    exact = []
    normalized = []

    for item in evidence:
        candidate_id = item.get("candidate_id")
        if not candidate_id:
            continue
        if chosen_source and item.get("model_id") != chosen_source:
            continue
        candidate_text = item.get("text", "")
        if candidate_text.strip() == final_exact and final_exact:
            exact.append(candidate_id)
        elif _normalize_text(candidate_text) == final_normalized and final_normalized:
            normalized.append(candidate_id)

    used = exact or normalized
    if not used and chosen_source:
        used = [
            item.get("candidate_id")
            for item in evidence
            if item.get("candidate_id") and item.get("model_id") == chosen_source
        ]

    return sorted(set(used)), len(exact), len(normalized)


def _unsupported_tokens(final_text: str, evidence: List[Dict]) -> Tuple[List[str], float]:
    tokens = _tokenize(final_text)
    if not tokens:
        return [], 1.0

    evidence_vocab = set()
    for item in evidence:
        evidence_vocab.update(_tokenize(item.get("text", "")))

    unsupported = sorted({token for token in tokens if token not in evidence_vocab})
    supported_count = len(tokens) - len(unsupported)
    ratio = round(supported_count / max(1, len(tokens)), 3)
    return unsupported, ratio


def _evidence_notes(
    method: str,
    fallback_reason: Optional[str],
    llm_rejection_reason: Optional[str],
    used_candidate_ids: List[str],
    chosen_source: Optional[str],
    validation_reason: Optional[str] = None,
) -> List[str]:
    notes = []
    if method == "llm":
        notes.append("llm_output_validated_against_local_evidence")
        if validation_reason:
            notes.append(f"llm_validation:{validation_reason}")
    else:
        notes.append("deterministic_candidate_fallback")
    if fallback_reason:
        notes.append(f"fallback_reason:{fallback_reason}")
    if llm_rejection_reason:
        notes.append(f"llm_rejection:{llm_rejection_reason}")
    if used_candidate_ids:
        notes.append(f"used_candidate_count:{len(used_candidate_ids)}")
    if chosen_source and chosen_source != "none":
        notes.append(f"chosen_source:{chosen_source}")
    return notes


def _uncertainty_flags(
    final_text: str,
    confidence: float,
    unsupported_tokens: List[str],
    support_windows: List[str],
    llm_rejection_reason: Optional[str],
    suppressed_junk: bool = False,
) -> List[str]:
    flags = []
    if suppressed_junk:
        flags.append("media_junk_suppressed")
    if not final_text:
        flags.append("empty_output")
    if confidence < 0.5:
        flags.append("low_confidence")
    if len(support_windows) < 2:
        flags.append("single_window_support")
    if unsupported_tokens:
        flags.append("unsupported_tokens_present")
    if llm_rejection_reason:
        flags.append("llm_selection_rejected")
    return flags


def _derive_language(stripe_packet: Dict, chosen_source: Optional[str]) -> Optional[str]:
    for evidence in stripe_packet.get("evidence", []):
        if chosen_source and evidence.get("model_id") != chosen_source:
            continue
        detected = (evidence.get("language_evidence") or {}).get("detected_language")
        if detected:
            return detected
    for evidence in stripe_packet.get("evidence", []):
        detected = (evidence.get("language_evidence") or {}).get("detected_language")
        if detected:
            return detected
    return None


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall((text or "").lower()) if token.strip()]


def _validation_audit(
    session_id: str,
    records: List[Dict],
    llm_status: str,
    llm_available: bool,
    stripe_packets: Optional[List[Dict]] = None,
) -> Dict:
    assembly_mode_counts = Counter(record.get("assembly_mode") for record in records)
    validation_status_counts = Counter(record.get("validation_status") for record in records)
    uncertainty_flag_counts = Counter()
    for record in records:
        uncertainty_flag_counts.update(record.get("uncertainty_flags") or [])

    llm_resolved = sum(1 for record in records if record.get("method") == "llm")
    fallback_resolved = sum(1 for record in records if record.get("method") != "llm")
    validation_rejected = sum(1 for record in records if record.get("llm_validation_rejected"))
    stripe_count = len(records) or 1
    single_support = sum(
        1
        for record in records
        if (record.get("support_window_count") or len(record.get("support_windows") or [])) < 2
    )
    total_support_windows = sum(
        record.get("support_window_count") or len(record.get("support_windows") or []) or 0
        for record in records
    )
    avg_support = round(total_support_windows / stripe_count, 3)

    witness_flag_counts = _witness_flag_counts(stripe_packets or [])
    disagreement_stripes = _disagreement_stripes(stripe_packets or [])

    return {
        "session_id": session_id,
        "lexical_contract_version": LEXICAL_CONTRACT_VERSION,
        "stripe_count": len(records),
        "llm_available": llm_available,
        "reconciler_status": llm_status,
        "assembly_mode_counts": dict(assembly_mode_counts),
        "validation_status_counts": dict(validation_status_counts),
        "uncertainty_flag_counts": dict(uncertainty_flag_counts),
        "usage_rates": {
            "llm_resolved_rate": round(llm_resolved / stripe_count, 3),
            "fallback_resolved_rate": round(fallback_resolved / stripe_count, 3),
            "validation_rejected_rate": round(validation_rejected / stripe_count, 3),
            "single_window_support_rate": round(single_support / stripe_count, 3),
            "avg_support_window_count": avg_support,
        },
        "witness_flag_counts": witness_flag_counts,
        "disagreement_stripe_count": len(disagreement_stripes),
        "records": [
            {
                "stripe_id": record.get("stripe_id"),
                "assembly_mode": record.get("assembly_mode"),
                "validation_status": record.get("validation_status"),
                "unsupported_tokens": record.get("unsupported_tokens", []),
                "token_support_ratio": record.get("token_support_ratio"),
                "uncertainty_flags": record.get("uncertainty_flags", []),
                "used_candidate_ids": record.get("used_candidate_ids", []),
                "evidence_notes": record.get("evidence_notes", []),
            }
            for record in records
        ],
    }


def _witness_flag_counts(stripe_packets: List[Dict]) -> Dict[str, int]:
    counts: Counter = Counter()
    for packet in stripe_packets:
        for evidence in packet.get("evidence") or []:
            for flag in evidence.get("candidate_flags") or []:
                counts[flag] += 1
    return dict(counts)


def _disagreement_stripes(stripe_packets: List[Dict]) -> List[str]:
    """Stripe ids where supporting witnesses disagree on the text.

    Canonical spec 9.6: witness disagreement is audited, not hidden. This
    surface exists so tests (and dashboards) can prove the pipeline is
    routing conflicting evidence to the synthesis stage rather than
    silently picking one and discarding the rest.
    """
    stripes = []
    for packet in stripe_packets:
        texts = {
            (evidence.get("text") or "").strip().lower()
            for evidence in packet.get("evidence") or []
            if (evidence.get("text") or "").strip()
        }
        if len(texts) > 1:
            stripes.append(packet.get("stripe_id"))
    return stripes


def run_reconciliation(session_id: str, stripe_packets: List[Dict], stage) -> Dict:
    """Execute reconciliation on all stripe packets."""
    sd = session_dir(session_id)

    llm = None
    cfg = get_config()
    llm_status = "deterministic_fallback_only"
    if cfg.reconciliation.use_llm:
        llm, llm_status = _load_llm()

    records = []
    llm_resolved = 0
    fallback_resolved = 0
    validation_rejected = 0
    total_chars = 0

    for stripe_packet in stripe_packets:
        record = reconcile_stripe(stripe_packet, llm)
        records.append(record)

        if record["method"] == "llm":
            llm_resolved += 1
        else:
            fallback_resolved += 1
        if record.get("llm_validation_rejected"):
            validation_rejected += 1
        total_chars += len(record.get("final_text", ""))

    lexical_result = {
        "session_id": session_id,
        "lexical_contract_version": LEXICAL_CONTRACT_VERSION,
        "truth_layer": "lexical",
        "stripe_count": len(records),
        "llm_resolved_count": llm_resolved,
        "fallback_resolved_count": fallback_resolved,
        "validation_rejected_count": validation_rejected,
        "total_final_chars": total_chars,
        "records": records,
        "reconciler_status": llm_status,
        "llm_available": llm is not None,
    }
    validation_audit = _validation_audit(
        session_id,
        records,
        llm_status,
        llm is not None,
        stripe_packets=stripe_packets,
    )
    compatibility_result = {
        **lexical_result,
        "total_chosen_chars": total_chars,
    }

    atomic_write_json(str(sd / "reconciliation" / "lexical_synthesis_result.json"), lexical_result)
    atomic_write_json(str(sd / "reconciliation" / "validation_audit.json"), validation_audit)
    atomic_write_json(str(sd / "reconciliation" / "reconciliation_result.json"), compatibility_result)
    stage.commit([
        "reconciliation_result.json",
        "lexical_synthesis_result.json",
        "validation_audit.json",
    ])

    del llm
    return compatibility_result
