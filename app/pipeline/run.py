"""
Pipeline Run - Filesystem-based execution tracking for the canonical pipeline.

Manages:
  pipeline/
    canonical_run_id.txt         -- atomic pointer to active run
    runs/{run_id}/
      run_meta.json              -- PipelineRun status + stage summaries
      config_snapshot.json       -- frozen job_data at creation
      stages/{stage_name}/
        stage_meta.json          -- StageRun status, timestamps, artifacts
        <artifact files>

Canonical stages aligned to the pipeline specification:
  1. normalize_audio
  2. acoustic_triage
  3. decode_lattice
  4. first_pass_medium
  5. candidate_asr_large_v3
  6. candidate_asr_secondary
  7. stripe_grouping
  8. reconciliation
  9. canonical_assembly
  10. selective_enrichment
  11. semantic_marking
  12. memory_graph_update
  13. derived_outputs
  14. nosql_projection
  15. thread_linking
"""
import os
import uuid
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.core.atomic_io import atomic_write_json, safe_read_json

logger = logging.getLogger(__name__)

# Stage statuses
STAGE_PENDING = "pending"
STAGE_RUNNING = "running"
STAGE_DONE = "done"
STAGE_DEGRADED = "degraded"   # completed but provider crashed / all witnesses failed
STAGE_ERROR = "error"
STAGE_SKIPPED = "skipped"

# Run statuses
RUN_PENDING = "pending"
RUN_RUNNING = "running"
RUN_DONE = "done"
RUN_ERROR = "error"

# Canonical stages - full-session ASR is intentionally excluded before triage.
CANONICAL_V1_STAGES = [
    "normalize_audio",
    "acoustic_triage",
    "decode_lattice",
    "first_pass_medium",
    "candidate_asr_large_v3",
    "candidate_asr_secondary",
    "stripe_grouping",
    "reconciliation",
    "canonical_assembly",
    "selective_enrichment",
    "semantic_marking",
    "memory_graph_update",
    "derived_outputs",
    "nosql_projection",
    "thread_linking",
]

# Legacy stages kept for backward-compat detection
LEGACY_STAGES = [
    "transcribe", "diarization", "quality", "clean",
    "classify", "coverage", "subtitles",
    "candidate_asr_parakeet",
    "candidate_asr_turbo_hf",  # Old turbo stage name
]

STAGE_NAME_ALIASES = {
    "candidate_asr_parakeet": "candidate_asr_secondary",
    "candidate_asr_turbo_hf": "candidate_asr_secondary",
}


def normalize_stage_name(name: str) -> str:
    return STAGE_NAME_ALIASES.get(name, name)


def stage_directory_candidates(name: str) -> List[str]:
    normalized = normalize_stage_name(name)
    candidates = [normalized]
    for legacy_name, current_name in STAGE_NAME_ALIASES.items():
        if current_name == normalized and legacy_name not in candidates:
            candidates.append(legacy_name)
    return candidates


def _canonicalize_stage_names(stage_names: Optional[List[str]]) -> List[str]:
    names = list(stage_names or CANONICAL_V1_STAGES)
    normalized = [normalize_stage_name(name) for name in names]
    merged: List[str] = []

    for name in CANONICAL_V1_STAGES:
        if name not in merged:
            merged.append(name)

    for name in normalized:
        if name not in merged:
            merged.append(name)

    return merged


def migrate_run_stage_layout(run_dir: Path) -> None:
    stages_root = run_dir / "stages"
    if not stages_root.is_dir():
        return

    for legacy_name, current_name in STAGE_NAME_ALIASES.items():
        legacy_dir = stages_root / legacy_name
        current_dir = stages_root / current_name
        if legacy_dir.is_dir() and not current_dir.exists():
            legacy_dir.rename(current_dir)


class StageRun:
    """Tracks execution of a single pipeline stage."""

    def __init__(self, name: str, run_dir: Path):
        self.name = normalize_stage_name(name)
        self.run_dir = run_dir
        self.stage_dir = self._resolve_stage_dir()
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.status = STAGE_PENDING
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.error: Optional[str] = None
        self.artifacts: List[str] = []
        self.actual_model: Optional[str] = None   # what model actually ran
        self.routing_reason: Optional[str] = None  # why this model was chosen

    def start(self) -> "StageRun":
        self.status = STAGE_RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return self

    def commit(self, artifacts: List[str] = None) -> None:
        self.status = STAGE_DONE
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if artifacts:
            self.artifacts = artifacts
        else:
            # Auto-discover artifacts
            self.artifacts = [
                f.name for f in self.stage_dir.iterdir()
                if f.name != "stage_meta.json"
            ]
        self._save()

    def commit_with_fallback(self, artifacts: List[str] = None, reason: str = "") -> None:
        """Mark as done but record fallback reason for auditability.

        Use this for expected graceful degradation (e.g. candidate B skipped
        because the session is non-English).
        """
        self.status = STAGE_DONE
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.error = f"fallback: {reason}" if reason else "fallback: unknown"
        if artifacts:
            self.artifacts = artifacts
        else:
            self.artifacts = [
                f.name for f in self.stage_dir.iterdir()
                if f.name != "stage_meta.json"
            ]
        # Write fallback marker
        atomic_write_json(
            str(self.stage_dir / "_fallback.json"),
            {"reason": reason, "timestamp": self.finished_at}
        )
        self._save()

    def commit_degraded(self, reason: str, artifacts: List[str] = None) -> None:
        """Mark stage as degraded: provider crashed or all witnesses failed.

        Pipeline continues but the failure is visible in audit.  This is
        NOT hidden behind 'done'.
        """
        self.status = STAGE_DEGRADED
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.error = reason
        if artifacts:
            self.artifacts = artifacts
        else:
            self.artifacts = [
                f.name for f in self.stage_dir.iterdir()
                if f.name != "stage_meta.json"
            ]
        atomic_write_json(
            str(self.stage_dir / "_degraded.json"),
            {"reason": reason, "timestamp": self.finished_at}
        )
        self._save()

    def fail(self, error: str) -> None:
        self.status = STAGE_ERROR
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.error = error
        self._save()

    def skip(self, reason: str = "") -> None:
        self.status = STAGE_SKIPPED
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.error = f"skipped: {reason}" if reason else None
        self._save()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "artifacts": self.artifacts,
            "fallback": self.error.startswith("fallback:") if self.error else False,
            "degraded": self.status == STAGE_DEGRADED,
            "actual_model": self.actual_model,
            "routing_reason": self.routing_reason,
        }

    def _save(self) -> None:
        atomic_write_json(str(self.stage_dir / "stage_meta.json"), self.to_dict())

    def _resolve_stage_dir(self) -> Path:
        stages_root = self.run_dir / "stages"
        normalized_dir = stages_root / self.name
        if normalized_dir.exists():
            return normalized_dir

        for candidate in stage_directory_candidates(self.name)[1:]:
            legacy_dir = stages_root / candidate
            if legacy_dir.exists():
                return legacy_dir

        return normalized_dir

    @classmethod
    def from_disk(cls, name: str, run_dir: Path) -> "StageRun":
        stage = cls(normalize_stage_name(name), run_dir)
        meta_path = stage.stage_dir / "stage_meta.json"
        if meta_path.is_file():
            data = safe_read_json(str(meta_path))
            if data:
                stage.name = normalize_stage_name(data.get("name", stage.name))
                stage.status = data.get("status", STAGE_PENDING)
                stage.started_at = data.get("started_at")
                stage.finished_at = data.get("finished_at")
                stage.error = data.get("error")
                stage.artifacts = data.get("artifacts", [])
                stage.actual_model = data.get("actual_model")
                stage.routing_reason = data.get("routing_reason")
        return stage


class PipelineRun:
    """Tracks execution of a full canonical pipeline run."""

    def __init__(self, session_dir: str, run_id: str,
                 run_type: str = "canonical",
                 stage_names: List[str] = None):
        self.session_dir = Path(session_dir)
        self.run_id = run_id
        self.run_type = run_type
        migrate_run_stage_layout(self.run_dir)
        self._stage_names = _canonicalize_stage_names(stage_names)
        self.status = RUN_PENDING
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.error: Optional[str] = None
        self._stages: Dict[str, StageRun] = {}
        for name in self._stage_names:
            self._stages[name] = StageRun(name, self.run_dir)

    @property
    def run_dir(self) -> Path:
        return self.session_dir / "pipeline" / "runs" / self.run_id

    @property
    def config_snapshot_path(self) -> Path:
        return self.run_dir / "config_snapshot.json"

    @property
    def meta_path(self) -> Path:
        return self.run_dir / "run_meta.json"

    def initialize(self, config_snapshot: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(self.config_snapshot_path), config_snapshot)
        self._save()
        # Point canonical_run_id at this run immediately so interrupted runs are
        # discoverable by resume/audit tooling.  complete() re-writes the same
        # value on success, which is safe and idempotent.
        self._set_canonical_run_id()

    def start(self) -> None:
        self.status = RUN_RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._save()

    def complete(self) -> None:
        self.status = RUN_DONE
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self._save()
        self._set_canonical_run_id()

    def fail(self, error: str) -> None:
        self.status = RUN_ERROR
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.error = error
        self._save()

    def get_stage(self, name: str) -> StageRun:
        normalized = normalize_stage_name(name)
        if normalized not in self._stages:
            raise ValueError(f"Unknown stage: {name}. Valid stages: {self._stage_names}")
        return self._stages[normalized]

    def start_stage(self, name: str) -> StageRun:
        stage = self.get_stage(name)
        if stage.status == STAGE_DONE:
            return stage  # Idempotent for resumability
        return stage.start()

    def is_stage_done(self, name: str) -> bool:
        stage = self.get_stage(name)
        return stage.status in (STAGE_DONE, STAGE_SKIPPED, STAGE_DEGRADED)

    def skip_stage(self, name: str, reason: str = "") -> None:
        self.get_stage(name).skip(reason)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "status": self.status,
            "stage_names": self._stage_names,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "stages": {name: s.to_dict() for name, s in self._stages.items()},
        }

    def _save(self) -> None:
        atomic_write_json(str(self.meta_path), self.to_dict())

    def _set_canonical_run_id(self) -> None:
        pointer = self.session_dir / "pipeline" / "canonical_run_id.txt"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(self.run_id, encoding="utf-8")

    @classmethod
    def from_disk(cls, session_dir: str, run_id: str) -> Optional["PipelineRun"]:
        sd = Path(session_dir)
        meta_path = sd / "pipeline" / "runs" / run_id / "run_meta.json"
        data = safe_read_json(str(meta_path))
        if data is None:
            return None
        run = cls(
            session_dir=session_dir,
            run_id=run_id,
            run_type=data.get("run_type", "canonical"),
            stage_names=data.get("stage_names", CANONICAL_V1_STAGES),
        )
        run.status = data.get("status", RUN_PENDING)
        run.started_at = data.get("started_at")
        run.finished_at = data.get("finished_at")
        run.error = data.get("error")
        # Restore stages from disk
        for name in run._stage_names:
            run._stages[name] = StageRun.from_disk(name, run.run_dir)
        return run


def create_canonical_run(session_dir: str, session_id: str,
                        config_snapshot: dict,
                        stage_names: List[str] = None) -> PipelineRun:
    run_id = str(uuid.uuid4())
    run = PipelineRun(
        session_dir=session_dir,
        run_id=run_id,
        stage_names=stage_names,
    )
    config_snapshot["session_id"] = session_id
    run.initialize(config_snapshot)
    return run


def get_canonical_run_id(session_dir: str) -> Optional[str]:
    pointer = Path(session_dir) / "pipeline" / "canonical_run_id.txt"
    if pointer.is_file():
        run_id = pointer.read_text(encoding="utf-8").strip()
        if run_id:
            return run_id
    # Fallback: pointer missing (older runs or interrupted before initialize
    # wrote the pointer).  Pick the most recently started run_meta.json under
    # pipeline/runs/ so resume/audit tooling can still find orphaned runs.
    runs_dir = Path(session_dir) / "pipeline" / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "run_meta.json"
        if not meta_path.is_file():
            continue
        data = safe_read_json(str(meta_path)) or {}
        started = data.get("started_at") or ""
        candidates.append((started, run_dir.name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def get_canonical_run(session_dir: str) -> Optional[PipelineRun]:
    run_id = get_canonical_run_id(session_dir)
    if run_id is None:
        return None
    return PipelineRun.from_disk(session_dir, run_id)
