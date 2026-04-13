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

Canonical V1 stages (turbo replaced by parakeet):
  1. normalize_audio
  2. first_pass_medium
  3. speaker_diarization
  4. acoustic_triage
  5. decode_lattice
  6. candidate_asr_large_v3
  7. candidate_asr_parakeet       <-- replaces candidate_asr_turbo_hf
  8. stripe_grouping
  9. reconciliation
  10. canonical_assembly
  11. selective_enrichment
  12. derived_outputs
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
STAGE_ERROR = "error"
STAGE_SKIPPED = "skipped"

# Run statuses
RUN_PENDING = "pending"
RUN_RUNNING = "running"
RUN_DONE = "done"
RUN_ERROR = "error"

# Canonical V1 stages - Parakeet replaces Turbo
CANONICAL_V1_STAGES = [
    "normalize_audio",
    "first_pass_medium",
    "speaker_diarization",
    "acoustic_triage",
    "decode_lattice",
    "candidate_asr_large_v3",
    "candidate_asr_parakeet",      # Was: candidate_asr_turbo_hf
    "stripe_grouping",
    "reconciliation",
    "canonical_assembly",
    "selective_enrichment",
    "derived_outputs",
]

# Legacy stages kept for backward-compat detection
LEGACY_STAGES = [
    "transcribe", "diarization", "quality", "clean",
    "classify", "coverage", "subtitles",
    "candidate_asr_turbo_hf",  # Old turbo stage name
]


class StageRun:
    """Tracks execution of a single pipeline stage."""

    def __init__(self, name: str, run_dir: Path):
        self.name = name
        self.run_dir = run_dir
        self.stage_dir = run_dir / "stages" / name
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.status = STAGE_PENDING
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.error: Optional[str] = None
        self.artifacts: List[str] = []

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
        """Mark as done but record fallback reason for auditability."""
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
        }

    def _save(self) -> None:
        atomic_write_json(str(self.stage_dir / "stage_meta.json"), self.to_dict())

    @classmethod
    def from_disk(cls, name: str, run_dir: Path) -> "StageRun":
        stage = cls(name, run_dir)
        meta_path = stage.stage_dir / "stage_meta.json"
        if meta_path.is_file():
            data = safe_read_json(str(meta_path))
            if data:
                stage.status = data.get("status", STAGE_PENDING)
                stage.started_at = data.get("started_at")
                stage.finished_at = data.get("finished_at")
                stage.error = data.get("error")
                stage.artifacts = data.get("artifacts", [])
        return stage


class PipelineRun:
    """Tracks execution of a full canonical pipeline run."""

    def __init__(self, session_dir: str, run_id: str,
                 run_type: str = "canonical",
                 stage_names: List[str] = None):
        self.session_dir = Path(session_dir)
        self.run_id = run_id
        self.run_type = run_type
        self._stage_names = stage_names or list(CANONICAL_V1_STAGES)
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
        if name not in self._stages:
            raise ValueError(f"Unknown stage: {name}. Valid stages: {self._stage_names}")
        return self._stages[name]

    def start_stage(self, name: str) -> StageRun:
        stage = self.get_stage(name)
        if stage.status == STAGE_DONE:
            return stage  # Idempotent for resumability
        return stage.start()

    def is_stage_done(self, name: str) -> bool:
        stage = self.get_stage(name)
        return stage.status in (STAGE_DONE, STAGE_SKIPPED)

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
        return pointer.read_text(encoding="utf-8").strip()
    return None


def get_canonical_run(session_dir: str) -> Optional[PipelineRun]:
    run_id = get_canonical_run_id(session_dir)
    if run_id is None:
        return None
    return PipelineRun.from_disk(session_dir, run_id)
