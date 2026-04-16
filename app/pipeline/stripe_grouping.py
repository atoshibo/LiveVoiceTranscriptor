"""
Stage 6 - Stripe-Level Evidence Grouping

Regroups candidate outputs by 15-second stabilization stripe so each stripe
sees evidence from its supporting windows and supporting models.

For example:
  S1=[15,30] is supported by W0 right half and W1 left half.
  S2=[30,45] is supported by W1 right half and W2 left half.

If two models are configured, each stripe packet will ideally contain
four local witnesses: two models x two supporting windows.

Trust zones: text near window edges (first/last 5s) gets a trust penalty.

Output: reconciliation/ with stripe_packets.json
"""
import logging
from typing import List, Dict, Tuple

from app.core.config import get_config
from app.core.atomic_io import atomic_write_json, safe_read_json
from app.storage.session_store import session_dir, get_session_meta

logger = logging.getLogger(__name__)

EDGE_TRUST_ZONE_MS = 5000  # First/last 5s of each 30s window


def build_stripes(total_duration_ms: int, stripe_ms: int = None) -> List[Dict]:
    """Build 15-second commit stripes across the session."""
    cfg = get_config()
    stripe_ms = stripe_ms or cfg.geometry.commit_stripe_ms

    stripes = []
    start = 0
    idx = 0
    while start < total_duration_ms:
        end = min(start + stripe_ms, total_duration_ms)
        stripes.append({
            "stripe_id": f"S{idx:04d}",
            "start_ms": start,
            "end_ms": end,
            "duration_ms": end - start,
        })
        start = end
        idx += 1
    return stripes


def _compute_trust_score(segment_start_ms: float, segment_end_ms: float,
                         window_start_ms: int, window_end_ms: int) -> float:
    """Compute trust score based on position within window.

    Center of window = highest trust (1.0)
    Edges (first/last 5s) = reduced trust (0.5-0.8)
    """
    window_dur = window_end_ms - window_start_ms
    seg_mid = (segment_start_ms + segment_end_ms) / 2
    relative_pos = (seg_mid - window_start_ms) / max(1, window_dur)

    # Distance from center (0 = center, 0.5 = edge)
    dist_from_center = abs(relative_pos - 0.5)

    if dist_from_center < 0.25:
        return 1.0  # Center zone
    else:
        # Linear falloff in edge zone
        return max(0.5, 1.0 - (dist_from_center - 0.25) * 2)


def group_evidence_by_stripe(stripes: List[Dict], windows: List[Dict],
                              candidates: List[Dict],
                              allowed_languages: List[str] = None,
                              forced_language: str = None,
                              transcription_mode: str = "verbatim_multilingual") -> List[Dict]:
    """Group candidate evidence by stripe.

    Each stripe packet contains all candidate text from all models
    for all windows that support this stripe.
    """
    # Index candidates by window_id and model_id
    cand_index: Dict[Tuple[str, str], Dict] = {}
    for c in candidates:
        key = (c.get("window_id", ""), c.get("model_id", ""))
        cand_index[key] = c

    stripe_packets = []

    for stripe in stripes:
        # Find supporting windows
        supporting_windows = []
        for window in windows:
            # Window supports this stripe if they overlap
            overlap_start = max(stripe["start_ms"], window["start_ms"])
            overlap_end = min(stripe["end_ms"], window["end_ms"])
            if overlap_end > overlap_start:
                overlap_ms = overlap_end - overlap_start
                # Determine which half of the window
                window_mid = (window["start_ms"] + window["end_ms"]) / 2
                stripe_mid = (stripe["start_ms"] + stripe["end_ms"]) / 2
                position = "left_half" if stripe_mid < window_mid else "right_half"

                supporting_windows.append({
                    "window_id": window["window_id"],
                    "window_type": window["window_type"],
                    "overlap_ms": overlap_ms,
                    "position": position,
                })

        # Collect evidence from candidates for each supporting window
        evidence = []
        model_ids_seen = set()

        for sw in supporting_windows:
            for c in candidates:
                if c.get("window_id") == sw["window_id"]:
                    if not _candidate_allowed_for_stripe(c, allowed_languages, forced_language):
                        continue
                    # Extract text relevant to this stripe's time range
                    stripe_text = _extract_stripe_text(
                        c, stripe["start_ms"], stripe["end_ms"]
                    )
                    trust_score = _compute_trust_score(
                        stripe["start_ms"], stripe["end_ms"],
                        c.get("window_start_ms", 0), c.get("window_end_ms", 30000)
                    )
                    evidence.append({
                        "candidate_id": c.get("candidate_id"),
                        "model_id": c.get("model_id"),
                        "window_id": sw["window_id"],
                        "window_type": sw["window_type"],
                        "position": sw["position"],
                        "text": stripe_text,
                        "trust_score": round(trust_score, 3),
                        "language_evidence": c.get("language_evidence", {}),
                        "candidate_success": c.get("confidence_features", {}).get("success", False),
                        "candidate_degraded": c.get("confidence_features", {}).get("degraded", False),
                        "candidate_flags": list(c.get("candidate_flags") or []),
                        "witness_audit": c.get("witness_audit") or {},
                    })
                    model_ids_seen.add(c.get("model_id"))

        stripe_packet = {
            **stripe,
            "support_window_count": len(supporting_windows),
            "support_windows": sorted({w["window_id"] for w in supporting_windows}),
            "support_models": sorted(model_ids_seen),
            "supporting_windows": supporting_windows,
            "evidence_count": len(evidence),
            "evidence": evidence,
            "model_ids": sorted(model_ids_seen),
            "stabilization_state": "provisional" if len(supporting_windows) < 2 else "stabilizable",
            "language_policy": {
                "allowed_languages": allowed_languages or [],
                "forced_language": forced_language,
                "transcription_mode": transcription_mode,
            },
        }
        stripe_packets.append(stripe_packet)

    return stripe_packets


def _extract_stripe_text(candidate: Dict, stripe_start_ms: int,
                         stripe_end_ms: int) -> str:
    """Extract text from candidate that falls within stripe time range.

    Uses segment timestamps when available, falls back to full text.
    """
    segments = candidate.get("segments", [])
    if not segments:
        return candidate.get("raw_text", "")

    relevant_parts = []
    for seg in segments:
        seg_start_ms, seg_end_ms = _segment_bounds_ms(candidate, seg)

        # Add window offset
        window_start = candidate.get("window_start_ms", 0)
        seg_start_ms += window_start
        seg_end_ms += window_start

        # Check overlap with stripe
        overlap_start = max(stripe_start_ms, seg_start_ms)
        overlap_end = min(stripe_end_ms, seg_end_ms)
        if overlap_end > overlap_start:
            text = seg.get("text", "").strip()
            if text:
                relevant_parts.append(text)

    if relevant_parts:
        return " ".join(relevant_parts)

    # Fallback: return full text
    return candidate.get("raw_text", "")


def _segment_bounds_ms(candidate: Dict, segment: Dict) -> Tuple[float, float]:
    """Normalize provider segment timestamps to window-relative milliseconds.

    New candidate artifacts persist an explicit unit in decode_metadata.
    Older artifacts fall back to inference based on the decode window
    duration, which is much safer than a fixed numeric threshold.
    """
    if "start_ms" in segment or "end_ms" in segment:
        return float(segment.get("start_ms", 0.0)), float(segment.get("end_ms", 0.0))

    seg_start = _coerce_float(segment.get("start", 0.0))
    seg_end = _coerce_float(segment.get("end", seg_start))
    unit = _segment_timestamp_unit(candidate, seg_start, seg_end)
    multiplier = 1000.0 if unit == "seconds" else 1.0
    return seg_start * multiplier, seg_end * multiplier


def _segment_timestamp_unit(candidate: Dict, seg_start: float, seg_end: float) -> str:
    decode_metadata = candidate.get("decode_metadata") or {}
    explicit_unit = str(decode_metadata.get("segment_timestamp_unit") or "").strip().lower()
    if explicit_unit in {"seconds", "milliseconds"}:
        return explicit_unit

    window_start_ms = _coerce_float(candidate.get("window_start_ms", 0.0))
    window_end_ms = _coerce_float(candidate.get("window_end_ms", window_start_ms))
    window_duration_ms = max(1.0, window_end_ms - window_start_ms)
    window_duration_s = window_duration_ms / 1000.0
    max_value = max(abs(seg_start), abs(seg_end))

    # Legacy fallback: provider-local timestamps that fit inside the decode
    # window span are treated as seconds; otherwise we assume milliseconds.
    if max_value <= window_duration_s + 5.0:
        return "seconds"
    return "milliseconds"


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_allowed_for_stripe(candidate: Dict, allowed_languages: List[str], forced_language: str) -> bool:
    if forced_language:
        detected = candidate.get("language_evidence", {}).get("detected_language")
        return detected in (None, "", forced_language)
    if not allowed_languages:
        return True
    detected = candidate.get("language_evidence", {}).get("detected_language")
    return detected in (None, "", *allowed_languages)


def run_stripe_grouping(session_id: str, windows: List[Dict],
                        audio_duration_ms: int, stage) -> Dict:
    """Execute stripe grouping stage.

    1. Build stripes
    2. Load all candidates
    3. Group evidence by stripe
    4. Persist stripe packets
    """
    sd = session_dir(session_id)

    # Build stripes
    stripes = build_stripes(audio_duration_ms)

    # Load all candidates
    candidates = []
    cand_dir = sd / "candidates"
    if cand_dir.is_dir():
        for f in cand_dir.glob("cand_*.json"):
            c = safe_read_json(str(f))
            if c:
                candidates.append(c)

    # Group evidence
    meta = get_session_meta(session_id) or {}
    stripe_packets = group_evidence_by_stripe(
        stripes,
        windows,
        candidates,
        allowed_languages=meta.get("allowed_languages") or [],
        forced_language=meta.get("forced_language"),
        transcription_mode=meta.get("transcription_mode", "verbatim_multilingual"),
    )

    # Stats
    stabilizable = sum(1 for sp in stripe_packets if sp["stabilization_state"] == "stabilizable")
    provisional = sum(1 for sp in stripe_packets if sp["stabilization_state"] == "provisional")

    result = {
        "session_id": session_id,
        "stripe_count": len(stripe_packets),
        "stabilizable": stabilizable,
        "provisional": provisional,
        "stripes": stripe_packets,
    }

    atomic_write_json(str(sd / "reconciliation" / "stripe_packets.json"), result)
    stage.commit(["stripe_packets.json"])

    return result
