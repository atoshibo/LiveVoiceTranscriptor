from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from app.core.atomic_io import atomic_write_json, safe_read_json
from app.pipeline.canonical_assembly import compute_quality_gate
from app.pipeline.derived_outputs import run_derived_outputs
from app.pipeline.memory_graph import run_memory_graph_update
from app.pipeline.semantic_marking import run_semantic_marking
from app.pipeline.selective_enrichment import run_selective_enrichment
from app.storage.session_store import session_dir


class _StageStub:
    def __init__(self, name: str) -> None:
        self.name = name
        self.artifacts: List[str] = []
        self.fallback_reason: Optional[str] = None

    def commit(self, artifacts=None):
        self.artifacts = list(artifacts or [])

    def commit_with_fallback(self, artifacts=None, reason: str = ""):
        self.artifacts = list(artifacts or [])
        self.fallback_reason = reason


def _canonical_segments(session_id: str) -> List[dict]:
    payload = safe_read_json(str(session_dir(session_id) / "canonical" / "canonical_segments.json")) or {}
    return payload.get("segments") or []


def _canonical_text(session_id: str, segments: List[dict]) -> str:
    payload = safe_read_json(str(session_dir(session_id) / "canonical" / "final_transcript.json")) or {}
    text = (payload.get("text") or "").strip()
    if text:
        return text
    return " ".join((seg.get("text") or "").strip() for seg in segments if seg.get("text")).strip()


def _audio_duration_ms(session_id: str, segments: List[dict]) -> int:
    sd = session_dir(session_id)
    meta = safe_read_json(str(sd / "v2_session.json")) or {}
    for key in ("audio_duration_ms", "duration_ms"):
        value = meta.get(key)
        if value:
            return int(value)
    seconds = meta.get("audio_duration_s")
    if seconds:
        return int(float(seconds) * 1000)
    quality = safe_read_json(str(sd / "derived" / "quality_report.json")) or {}
    if quality.get("audio_duration_s"):
        return int(float(quality["audio_duration_s"]) * 1000)
    return max((int(seg.get("end_ms") or 0) for seg in segments), default=0)


def _resolve_audio_path(session_id: str) -> Optional[str]:
    sd = session_dir(session_id)
    for candidate in (
        sd / "normalized" / "audio.wav",
        sd / "audio.wav",
        sd / "raw" / "audio.wav",
        sd / "raw" / "uploaded.wav",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def backfill_session(session_id: str, rerun_diarization: bool = False) -> dict:
    sd = session_dir(session_id)
    segments = _canonical_segments(session_id)
    if not segments:
        return {"session_id": session_id, "status": "skipped", "reason": "canonical_segments_missing"}

    canonical_dir = sd / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    quality_gate = compute_quality_gate(segments)
    atomic_write_json(str(canonical_dir / "quality_gate.json"), {
        "session_id": session_id,
        **quality_gate,
    })

    if rerun_diarization:
        audio_path = _resolve_audio_path(session_id)
        if audio_path:
            enrich_stage = _StageStub("selective_enrichment")
            run_selective_enrichment(session_id, segments, audio_path, enrich_stage)
            # run_selective_enrichment mutates and persists canonical segments in place;
            # reload from disk so downstream steps always read the persisted truth.
            segments = _canonical_segments(session_id) or segments

    semantic_stage = _StageStub("semantic_marking")
    marking_result = run_semantic_marking(session_id, segments, semantic_stage)

    marker_payload = safe_read_json(str(sd / "enrichment" / "segment_markers.json")) or {}
    markers = marker_payload.get("markers") or []
    memory_stage = _StageStub("memory_graph_update")
    memory_result = run_memory_graph_update(session_id, markers, memory_stage)

    derived_stage = _StageStub("derived_outputs")
    derived_result = run_derived_outputs(
        session_id,
        segments,
        _canonical_text(session_id, segments),
        _audio_duration_ms(session_id, segments),
        derived_stage,
    )

    return {
        "session_id": session_id,
        "status": "done",
        "quality_gate": quality_gate,
        "marker_count": marking_result.get("marker_count", 0),
        "context_span_count": marking_result.get("context_span_count", 0),
        "graph_update_count": memory_result.get("graph_update_count", 0),
        "files_written": derived_result.get("files_written", []),
    }


def _iter_target_sessions(args) -> Iterable[str]:
    if args.session_ids:
        for session_id in args.session_ids:
            yield session_id
        return

    root = Path(args.sessions_root)
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill phase 0-3 artifacts on existing sessions.")
    parser.add_argument("session_ids", nargs="*", help="Specific session ids to backfill. Defaults to all sessions.")
    parser.add_argument("--sessions-root", default="data/sessions", help="Sessions root directory.")
    parser.add_argument(
        "--rerun-diarization",
        action="store_true",
        help="Also rerun selective enrichment when a local audio file is available.",
    )
    args = parser.parse_args()

    results = [
        backfill_session(session_id, rerun_diarization=args.rerun_diarization)
        for session_id in _iter_target_sessions(args)
    ]
    done = sum(1 for result in results if result.get("status") == "done")
    skipped = sum(1 for result in results if result.get("status") == "skipped")
    print({
        "done": done,
        "skipped": skipped,
        "results": results,
    })


if __name__ == "__main__":
    main()
