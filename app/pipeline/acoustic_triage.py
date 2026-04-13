"""
Stage 3 - Acoustic Triage

Classifies time regions as speech, non_speech, music_media, noise, mixed, uncertain.
Merges adjacent speech regions into speech islands.
Purpose: avoid wasting ASR compute on non-speech content.

Output: triage/ with speech_map.json and speech_islands.json
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple

from app.core.atomic_io import atomic_write_json
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)

# Region types
SPEECH = "speech"
NON_SPEECH = "non_speech"
MUSIC_MEDIA = "music_media"
NOISE = "noise"
MIXED = "mixed"
UNCERTAIN = "uncertain"

# Merge parameters
SPEECH_MERGE_GAP_MS = 2000  # Merge speech regions separated by < 2s
MIN_SPEECH_ISLAND_MS = 1000  # Minimum speech island duration


def classify_regions(audio_path: str, total_duration_ms: int,
                     region_size_ms: int = 1000) -> List[Dict]:
    """Classify audio regions using energy-based VAD.

    For each region_size_ms chunk, compute energy and classify.
    This is a lightweight triage - not a full ASR pass.
    """
    regions = []

    try:
        import soundfile as sf
        import numpy as np

        data, sr = sf.read(audio_path, dtype='float32')
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        samples_per_region = int(sr * region_size_ms / 1000)

        for start_sample in range(0, len(data), samples_per_region):
            end_sample = min(start_sample + samples_per_region, len(data))
            chunk = data[start_sample:end_sample]

            start_ms = int(start_sample / sr * 1000)
            end_ms = int(end_sample / sr * 1000)

            # Energy-based classification
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            zero_crossings = float(np.sum(np.abs(np.diff(np.signbit(chunk))))) / len(chunk)

            # Simple heuristic classification
            if rms < 0.005:
                tag = NON_SPEECH
            elif rms < 0.01 and zero_crossings < 0.05:
                tag = NOISE
            elif rms > 0.02:
                tag = SPEECH
            elif zero_crossings > 0.15:
                tag = MIXED
            else:
                tag = UNCERTAIN

            regions.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "tag": tag,
                "rms": round(rms, 6),
                "zero_crossing_rate": round(zero_crossings, 6),
            })

    except ImportError:
        # Fallback: assume all speech (conservative - won't skip anything)
        logger.warning("soundfile/numpy not available, assuming all speech")
        n_regions = max(1, total_duration_ms // region_size_ms)
        for i in range(n_regions):
            regions.append({
                "start_ms": i * region_size_ms,
                "end_ms": min((i + 1) * region_size_ms, total_duration_ms),
                "tag": SPEECH,
                "rms": 0.05,
                "zero_crossing_rate": 0.1,
            })

    return regions


def build_speech_islands(regions: List[Dict],
                         merge_gap_ms: int = SPEECH_MERGE_GAP_MS,
                         min_duration_ms: int = MIN_SPEECH_ISLAND_MS) -> List[Dict]:
    """Merge adjacent speech-bearing regions into speech islands.

    Conservative merge: short pauses don't fragment utterances.
    """
    speech_regions = [
        r for r in regions
        if r["tag"] in (SPEECH, MIXED, UNCERTAIN)
    ]

    if not speech_regions:
        return []

    # Sort by start
    speech_regions.sort(key=lambda r: r["start_ms"])

    # Merge adjacent regions
    islands = []
    current = {
        "start_ms": speech_regions[0]["start_ms"],
        "end_ms": speech_regions[0]["end_ms"],
        "regions": [speech_regions[0]],
    }

    for region in speech_regions[1:]:
        gap = region["start_ms"] - current["end_ms"]
        if gap <= merge_gap_ms:
            current["end_ms"] = max(current["end_ms"], region["end_ms"])
            current["regions"].append(region)
        else:
            if current["end_ms"] - current["start_ms"] >= min_duration_ms:
                islands.append(current)
            current = {
                "start_ms": region["start_ms"],
                "end_ms": region["end_ms"],
                "regions": [region],
            }

    # Don't forget the last island
    if current["end_ms"] - current["start_ms"] >= min_duration_ms:
        islands.append(current)

    # Add metadata
    for i, island in enumerate(islands):
        island["island_id"] = f"SI_{i:04d}"
        island["duration_ms"] = island["end_ms"] - island["start_ms"]
        island["region_count"] = len(island["regions"])
        del island["regions"]  # Don't persist individual regions in islands

    return islands


def run_acoustic_triage(session_id: str, audio_path: str,
                        audio_duration_ms: int, stage) -> dict:
    """Execute acoustic triage stage.

    1. Classify regions
    2. Build speech islands
    3. Persist to triage/
    """
    sd = session_dir(session_id)

    # Classify
    regions = classify_regions(audio_path, audio_duration_ms)
    atomic_write_json(str(sd / "triage" / "speech_map.json"), {
        "session_id": session_id,
        "total_duration_ms": audio_duration_ms,
        "region_count": len(regions),
        "regions": regions,
    })

    # Build islands
    islands = build_speech_islands(regions)
    atomic_write_json(str(sd / "triage" / "speech_islands.json"), {
        "session_id": session_id,
        "island_count": len(islands),
        "islands": islands,
        "total_speech_ms": sum(i["duration_ms"] for i in islands),
        "speech_ratio": sum(i["duration_ms"] for i in islands) / max(1, audio_duration_ms),
    })

    # Tag counts
    tag_counts = {}
    for r in regions:
        tag_counts[r["tag"]] = tag_counts.get(r["tag"], 0) + 1

    result = {
        "region_count": len(regions),
        "island_count": len(islands),
        "tag_counts": tag_counts,
        "total_speech_ms": sum(i["duration_ms"] for i in islands),
        "speech_ratio": sum(i["duration_ms"] for i in islands) / max(1, audio_duration_ms),
    }

    stage.commit(["speech_map.json", "speech_islands.json"])
    return result
