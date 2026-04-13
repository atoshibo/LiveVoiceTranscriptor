"""
Stage 9 - Selective Enrichment

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
from typing import List, Dict, Optional

from app.core.config import get_config
from app.core.atomic_io import atomic_write_json, safe_read_json
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)


def should_run_diarization(session_meta: dict, segments: List[Dict],
                           policy: str = "auto") -> bool:
    """Determine if diarization should run.

    Policy options:
      - "off": never run
      - "forced": always run
      - "auto": run if evidence suggests multiple speakers
    """
    if policy == "off":
        return False
    if policy == "forced":
        return True

    # Auto: check hints
    if session_meta.get("run_diarization"):
        return True
    if session_meta.get("diarization_hint"):
        return True

    speaker_count = session_meta.get("speaker_count", 1)
    if speaker_count and speaker_count > 1:
        return True

    return False


def run_diarization(audio_path: str, session_meta: dict) -> Optional[List[Dict]]:
    """Run speaker diarization on the audio.

    Returns list of speaker turns: [{speaker, start_s, end_s}]
    """
    cfg = get_config()
    model_path = cfg.model_paths.diarization_path

    if not model_path:
        logger.info("No diarization model path configured")
        return None

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

        return turns

    except ImportError:
        logger.warning("pyannote.audio not available, skipping diarization")
        return None
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        if cfg.diarization.fallback_on_error:
            return None
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
    meta = safe_read_json(str(sd / "v2_session.json")) or {}

    diarization_policy = "auto"
    if meta.get("run_diarization"):
        diarization_policy = "forced"

    speaker_turns = None
    diarization_status = "skipped"

    if should_run_diarization(meta, segments, diarization_policy):
        logger.info(f"Running selective diarization for {session_id}")
        speaker_turns = run_diarization(audio_path, meta)
        if speaker_turns:
            diarization_status = "done"
            # Persist speaker turns
            atomic_write_json(str(sd / "canonical" / "speaker_turns.json"), {
                "session_id": session_id,
                "turns": speaker_turns,
                "turn_count": len(speaker_turns),
                "speakers": sorted(set(t["speaker"] for t in speaker_turns)),
            })
        else:
            diarization_status = "failed"
    else:
        logger.info(f"Diarization not justified for {session_id}")

    # Assign speakers
    segments = assign_speakers_to_segments(segments, speaker_turns or [])

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
        "diarization_status": diarization_status,
        "speaker_count": len(set(s.get("speaker", "") for s in segments)),
        "turn_count": len(speaker_turns) if speaker_turns else 0,
    }

    stage.commit(["canonical_segments.json", "transcript.txt", "provenance.json", "final_transcript.json"])
    return result
