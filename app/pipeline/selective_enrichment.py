"""
Stage 10 - Selective Enrichment

Enrichment is downstream of canonical lexical truth. It must never become
the hidden source of transcript words.

Selective enrichment includes:
  - diarization where multi-speaker evidence is likely
  - entity extraction
  - uncertainty/confidence projection

Diarization is SELECTIVE, not universal:
  - Run only where content indicates a real chance of multiple speakers
  - If diarization is unavailable, keep lexical transcript canonical
  - Enrichment never overwrites canonical text

Output: canonical/ segments updated with speaker labels (when justified)
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from app.core.config import get_config
from app.core.atomic_io import atomic_write_json, safe_read_json
from app.storage.session_store import session_dir, get_session_meta

logger = logging.getLogger(__name__)


def _diarization_decision(session_meta: dict, policy: str) -> Tuple[bool, str]:
    """Return (requested, reason).

    The reason string is persisted in diarization_status.json so audit can
    see *why* we ran (or skipped) without re-running the heuristic.
    """
    if policy == "off":
        return False, "policy_off"
    if policy == "forced":
        return True, "policy_forced"

    if session_meta.get("run_diarization"):
        return True, "caller_hint_run_diarization"
    if session_meta.get("diarization_hint"):
        return True, "caller_hint_diarization_hint"

    speaker_count = session_meta.get("speaker_count", 1)
    if speaker_count and speaker_count > 1:
        return True, f"speaker_count_declared:{speaker_count}"

    duration_ms = session_meta.get("duration_ms") or session_meta.get("audio_duration_ms") or 0
    if duration_ms and duration_ms >= 45000:
        return True, f"auto_duration_ms:{int(duration_ms)}"

    filename_hint = " ".join(
        str(session_meta.get(key) or "").lower()
        for key in ("original_filename", "filename", "source_filename", "title")
    )
    for token in (
        "2voices", "3voix", "3voices", "voices", "voix",
        "interview", "phonecall", "phone_call", "call",
        "meeting", "conversation", "dialogue", "discussion",
        "panel", "podcast",
    ):
        if token in filename_hint:
            return True, f"auto_filename_hint:{token}"

    return False, "auto_short_monologue"


def should_run_diarization(session_meta: dict, segments: List[Dict],
                           policy: str = "auto") -> bool:
    """Determine if diarization should run.  See `_diarization_decision` for
    the reason-returning sibling used by the status-persistence path."""
    requested, _reason = _diarization_decision(session_meta, policy)
    return requested


def run_diarization(audio_path: str, session_meta: dict) -> Tuple[Optional[List[Dict]], str]:
    """Run speaker diarization on the audio.

    Returns (turns, status). `status` is one of:
      - "done": turns were produced
      - "no_model_path": cfg.model_paths.diarization_path is unset
      - "package_missing": pyannote.audio isn't installed
      - "failed:<err>": pyannote loaded but raised an exception
    Callers are expected to persist this status so "diarization was requested
    but unavailable" is visible to audit tools instead of silently becoming
    'skipped'.
    """
    cfg = get_config()
    model_path = cfg.model_paths.diarization_path

    if not model_path:
        logger.info("No diarization model path configured")
        return None, "no_model_path"

    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() and cfg.diarization.use_gpu else "cpu")

        pipeline = PyannotePipeline.from_pretrained(
            model_path,
            use_auth_token=False,
        )
        pipeline = pipeline.to(device)

        params = {}
        speaker_count = session_meta.get("speaker_count")
        if speaker_count:
            params["num_speakers"] = speaker_count
        else:
            params["min_speakers"] = cfg.diarization.min_speakers
            params["max_speakers"] = cfg.diarization.max_speakers

        diarization = pipeline(audio_path, **params)

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "speaker": speaker,
                "start_s": round(turn.start, 3),
                "end_s": round(turn.end, 3),
            })

        return turns, "done"

    except ImportError:
        logger.warning("pyannote.audio not available, skipping diarization")
        return None, "package_missing"
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        if cfg.diarization.fallback_on_error:
            return None, f"failed:{type(e).__name__}"
        raise


def assign_speakers_to_segments(segments: List[Dict],
                                speaker_turns: List[Dict]) -> List[Dict]:
    """Assign speaker labels to canonical segments based on diarization.

    Uses overlap-based assignment: the speaker with the most temporal
    overlap with a segment wins.
    """
    if not speaker_turns:
        # Default: assign SPEAKER_00 to all
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return segments

    for seg in segments:
        seg_start_s = seg["start_ms"] / 1000
        seg_end_s = seg["end_ms"] / 1000
        seg_dur = seg_end_s - seg_start_s

        # Accumulate overlap per speaker
        speaker_overlap: Dict[str, float] = {}
        for turn in speaker_turns:
            overlap_start = max(seg_start_s, turn["start_s"])
            overlap_end = min(seg_end_s, turn["end_s"])
            if overlap_end > overlap_start:
                sp = turn["speaker"]
                speaker_overlap[sp] = speaker_overlap.get(sp, 0) + (overlap_end - overlap_start)

        if speaker_overlap:
            best_speaker = max(speaker_overlap, key=speaker_overlap.get)
            best_overlap = speaker_overlap[best_speaker]
            # Require at least 10% overlap
            if best_overlap / max(0.001, seg_dur) >= 0.1:
                seg["speaker"] = best_speaker
            else:
                seg["speaker"] = "SPEAKER_UNKNOWN"
        else:
            seg["speaker"] = "SPEAKER_00"

    return segments


def run_selective_enrichment(session_id: str, segments: List[Dict],
                             audio_path: str, stage) -> Dict:
    """Execute selective enrichment stage.

    1. Check if diarization should run
    2. If yes, run diarization and assign speakers
    3. Update canonical segments
    """
    sd = session_dir(session_id)
    meta = get_session_meta(session_id) or safe_read_json(str(sd / "v2_session.json")) or {}

    diarization_policy = str(meta.get("diarization_policy", "auto")).strip().lower() or "auto"
    if diarization_policy not in {"auto", "off", "forced"}:
        diarization_policy = "auto"
    if meta.get("run_diarization") and diarization_policy == "auto":
        diarization_policy = "forced"

    speaker_turns = None
    diarization_status = "skipped"
    diarization_requested, diarization_reason = _diarization_decision(meta, diarization_policy)

    if diarization_requested:
        logger.info(f"Running selective diarization for {session_id}: {diarization_reason}")
        speaker_turns, diarization_status = run_diarization(audio_path, meta)
        if speaker_turns:
            atomic_write_json(str(sd / "canonical" / "speaker_turns.json"), {
                "session_id": session_id,
                "turns": speaker_turns,
                "turn_count": len(speaker_turns),
                "speakers": sorted(set(t["speaker"] for t in speaker_turns)),
            })
    else:
        logger.info(f"Diarization not justified for {session_id}: {diarization_reason}")

    # "available" captures whether the model could run at all for this
    # request.  It is distinct from "status" -- a skipped auto run is
    # available but not requested; a package_missing run is unavailable.
    available = True
    if diarization_status in {"no_model_path", "package_missing"} or diarization_status.startswith("failed:"):
        available = False

    # Always persist an explicit diarization status so "requested but
    # unavailable" is visible in audit, not silently hidden behind 'skipped'.
    diarization_payload = {
        "session_id": session_id,
        "policy": diarization_policy,
        "requested": diarization_requested,
        "reason": diarization_reason,
        "available": available,
        "status": diarization_status,
        "turn_count": len(speaker_turns) if speaker_turns else 0,
    }
    atomic_write_json(str(sd / "canonical" / "diarization_status.json"), diarization_payload)

    # Only attach speakers when selective diarization produced real evidence.
    if speaker_turns:
        segments = assign_speakers_to_segments(segments, speaker_turns)
    else:
        for seg in segments:
            seg.setdefault("speaker", None)

    # Update canonical segments file
    atomic_write_json(str(sd / "canonical" / "canonical_segments.json"), {
        "session_id": session_id,
        "segment_count": len(segments),
        "segments": segments,
    })
    try:
        from app.pipeline.canonical_assembly import build_transcript_surfaces

        build_transcript_surfaces(segments, session_id)
    except Exception as exc:
        logger.warning("Failed to refresh canonical transcript surfaces after enrichment: %s", exc)

    result = {
        "diarization_policy": diarization_policy,
        "diarization_status": diarization_status,
        "diarization_requested": diarization_requested,
        "diarization_reason": diarization_reason,
        "diarization_available": available,
        "speaker_count": len({s.get("speaker") for s in segments if s.get("speaker")}),
        "turn_count": len(speaker_turns) if speaker_turns else 0,
    }

    artifacts = [
        "canonical_segments.json",
        "transcript.txt",
        "provenance.json",
        "provisional_partial.json",
        "stabilized_partial.json",
        "final_transcript.json",
        "diarization_status.json",
    ]
    if speaker_turns:
        artifacts.append("speaker_turns.json")
    stage.commit(artifacts)
    return result
