"""
Stage 7 - Bounded LLM Reconciliation

The LLM is used as a local evidence arbiter, not as a transcript writer.
Its job is to choose the most plausible stripe text from bounded local evidence.

Hard rules:
  - do not translate by default
  - do not summarize
  - do not improve style
  - do not globally rewrite content
  - preserve code-switching
  - prefer supported wording over polished wording
  - emit uncertainty rather than inventing content

Primary path: bounded local LLM arbitration
Fallback path: deterministic selection using confidence/trust weighting

Output: reconciliation/ with reconciliation_result.json
"""
import json
import logging
import time
from typing import List, Dict, Optional, Tuple

from app.core.config import get_config
from app.core.atomic_io import atomic_write_json
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)

# Candidate priority order for deterministic fallback
# (higher priority = preferred when no LLM available)
MODEL_PRIORITY = {
    "faster-whisper:large-v3": 3,
    "nemo-asr:parakeet-tdt-0.6b-v3": 2,  # Parakeet replaces turbo
    "faster-whisper:medium": 1,
}

RECONCILIATION_PROMPT_TEMPLATE = """You are a transcript reconciliation engine. Your ONLY job is to select the best transcript text for a specific time segment.

TIME SEGMENT: {start_ms}ms - {end_ms}ms

CANDIDATES:
{candidates_text}

RULES:
1. Select the BEST candidate text. Do NOT rewrite, translate, summarize, or improve it.
2. If candidates agree, pick any. If they disagree, pick the most complete and coherent one.
3. Preserve the original language. Do NOT translate.
4. Preserve code-switching between languages.
5. If uncertain, prefer the candidate with more context words.
6. If all candidates are empty or garbage, return empty text with low confidence.

Respond with ONLY a JSON object:
{{"text": "the chosen text", "source_model": "model_id", "confidence": 0.0-1.0, "reason": "brief reason"}}
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
            n_gpu_layers=-1,  # Use GPU if available
            verbose=False,
        )
        return llm, "llm_loaded"
    except ImportError as e:
        logger.warning("llama_cpp not installed, using deterministic fallback: %s", e)
        return None, f"llama_cpp_missing:{e}"
    except Exception as e:
        logger.warning(f"Failed to load LLM: {e}")
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
        logger.warning(f"LLM call failed: {e}")
        return None


def _parse_llm_response(raw: str) -> Optional[Dict]:
    """Parse LLM JSON response with tolerance for markdown/preamble."""
    if not raw:
        return None

    # Strip markdown code fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "text" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Extract first JSON object with brace matching
    try:
        start = text.index("{")
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\':
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
            if not in_string:
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        data = json.loads(text[start:i + 1])
                        if isinstance(data, dict) and "text" in data:
                            return data
                        break
    except (ValueError, json.JSONDecodeError):
        pass

    return None


def _validate_llm_selection(parsed: Dict, candidates: List[Dict]) -> bool:
    """Validate that the LLM selected actual candidate text, not hallucinated content."""
    if not parsed or not parsed.get("text"):
        return False

    source_model = parsed.get("source_model", "")
    selected_text = parsed["text"].strip().lower()

    # Build allowlist
    for cand in candidates:
        if cand["model_id"] == source_model:
            candidate_text = cand["text"].strip().lower()
            if candidate_text and selected_text == candidate_text:
                return True
            # Allow minor whitespace differences
            if candidate_text and " ".join(selected_text.split()) == " ".join(candidate_text.split()):
                return True

    return False


def _select_fallback(evidence: List[Dict]) -> Dict:
    """Deterministic fallback: select best candidate by priority and trust score."""
    if not evidence:
        return {
            "chosen_text": "",
            "chosen_source": "none",
            "confidence": 0.0,
            "method": "fallback",
            "fallback_reason": "no_evidence",
        }

    # Filter non-empty evidence
    non_empty = [e for e in evidence if e.get("text", "").strip()]
    if not non_empty:
        return {
            "chosen_text": "",
            "chosen_source": "none",
            "confidence": 0.0,
            "method": "fallback",
            "fallback_reason": "all_empty",
        }

    # Score by model priority * trust score
    best = None
    best_score = -1
    for e in non_empty:
        priority = MODEL_PRIORITY.get(e.get("model_id", ""), 0)
        trust = e.get("trust_score", 0.5)
        score = priority * trust
        if score > best_score:
            best_score = score
            best = e

    return {
        "chosen_text": best["text"].strip(),
        "chosen_source": best.get("model_id", "unknown"),
        "confidence": min(1.0, best_score / 3.0),
        "method": "fallback",
        "fallback_reason": "deterministic_selection",
    }


def reconcile_stripe(stripe_packet: Dict, llm=None) -> Dict:
    """Reconcile a single stripe using LLM or deterministic fallback."""
    evidence = stripe_packet.get("evidence", [])
    start_ms = stripe_packet["start_ms"]
    end_ms = stripe_packet["end_ms"]
    support_windows = stripe_packet.get("support_windows") or sorted({item.get("window_id") for item in evidence if item.get("window_id")})
    support_models = stripe_packet.get("support_models") or sorted({item.get("model_id") for item in evidence if item.get("model_id")})

    start_time = time.time()

    # Try LLM path
    if llm and evidence:
        # Build prompt
        candidates_text = ""
        for i, e in enumerate(evidence):
            candidates_text += f"\n[{i+1}] Model: {e.get('model_id', 'unknown')}"
            candidates_text += f" | Window: {e.get('window_id', '?')}"
            candidates_text += f" | Trust: {e.get('trust_score', 0.5)}"
            candidates_text += f"\nText: {e.get('text', '')}\n"

        prompt = RECONCILIATION_PROMPT_TEMPLATE.format(
            start_ms=start_ms,
            end_ms=end_ms,
            candidates_text=candidates_text,
        )

        raw_response = _call_llm(llm, prompt)
        parsed = _parse_llm_response(raw_response)

        if parsed:
            candidates_for_validation = [
                {"model_id": e.get("model_id", ""), "text": e.get("text", "")}
                for e in evidence
            ]
            if _validate_llm_selection(parsed, candidates_for_validation):
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
                    return {
                        "stripe_id": stripe_packet["stripe_id"],
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "chosen_text": parsed["text"],
                        "chosen_source": parsed.get("source_model", "llm"),
                        "confidence": confidence,
                        "method": "llm",
                        "fallback_reason": None,
                        "raw_llm_response": raw_response,
                        "elapsed_s": round(elapsed, 3),
                        "support_windows": support_windows,
                        "support_models": support_models,
                        "language": _derive_language(stripe_packet, parsed.get("source_model")),
                        "support_window_count": stripe_packet.get("support_window_count", len(support_windows)),
                        "support_model_count": len(support_models),
                        "evidence_count": len(evidence),
                    }

    # Fallback path
    fallback = _select_fallback(evidence)
    elapsed = time.time() - start_time

    return {
        "stripe_id": stripe_packet["stripe_id"],
        "start_ms": start_ms,
        "end_ms": end_ms,
        **fallback,
        "raw_llm_response": None,
        "elapsed_s": round(elapsed, 3),
        "support_windows": support_windows,
        "support_models": support_models,
        "language": _derive_language(stripe_packet, fallback.get("chosen_source")),
        "support_window_count": stripe_packet.get("support_window_count", len(support_windows)),
        "support_model_count": len(support_models),
        "evidence_count": len(evidence),
    }


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


def run_reconciliation(session_id: str, stripe_packets: List[Dict],
                       stage) -> Dict:
    """Execute reconciliation on all stripe packets."""
    sd = session_dir(session_id)

    # Try to load LLM
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
        total_chars += len(record.get("chosen_text", ""))

    result = {
        "session_id": session_id,
        "stripe_count": len(records),
        "llm_resolved_count": llm_resolved,
        "fallback_resolved_count": fallback_resolved,
        "validation_rejected_count": validation_rejected,
        "total_chosen_chars": total_chars,
        "records": records,
        "reconciler_status": llm_status,
        "llm_available": llm is not None,
    }

    atomic_write_json(str(sd / "reconciliation" / "reconciliation_result.json"), result)
    stage.commit(["reconciliation_result.json"])

    # Free LLM
    del llm

    return result
