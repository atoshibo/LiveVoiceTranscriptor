"""
Stage 5 - Multi-ASR Execution

For each decode window that survives scheduling, runs the configured ASR models
on the exact same audio geometry. Models are scheduled sequentially under VRAM
control. Each candidate is persisted immediately after decoding.

Default models (turbo replaced by parakeet):
  - first pass: faster-whisper:medium
  - candidate A: faster-whisper:large-v3
  - candidate B: nemo-asr:parakeet-tdt-0.6b-v3

Output: candidates/ with per-model per-window candidate files
"""
import gc
import os
import uuid
import logging
import time
import wave
from pathlib import Path
from typing import List, Dict, Optional, Callable

from app.core.atomic_io import atomic_write_json, safe_read_json
from app.core.config import get_config
from app.models.registry import (
    get_model_info, resolve_model_id, is_model_safe_for_language,
    DEFAULT_FIRST_PASS, DEFAULT_CANDIDATE_A, DEFAULT_CANDIDATE_B,
)
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)

# Model caches for sequential VRAM management
_faster_whisper_model = None
_nemo_model_cache: Dict[str, object] = {}
_hf_model_cache: Dict[str, tuple] = {}


def _normalized_allowed_languages(allowed_languages: Optional[List[str]]) -> List[str]:
    return [str(item).strip() for item in (allowed_languages or []) if str(item).strip()]


def _session_language_is_english_only(
    forced_language: Optional[str],
    allowed_languages: Optional[List[str]],
    transcription_mode: str,
) -> bool:
    allowed = _normalized_allowed_languages(allowed_languages)
    if forced_language:
        return forced_language == "en"
    if transcription_mode == "verbatim_multilingual" and len(allowed) > 1:
        return False
    return allowed == ["en"]


def _requested_language_hint(
    forced_language: Optional[str],
    allowed_languages: Optional[List[str]],
) -> Optional[str]:
    if forced_language:
        return forced_language
    allowed = _normalized_allowed_languages(allowed_languages)
    if allowed == ["en"]:
        return "en"
    return None


def _skip_reason_for_model(
    model_id: str,
    forced_language: Optional[str],
    allowed_languages: Optional[List[str]],
    transcription_mode: str,
) -> Optional[str]:
    resolved = resolve_model_id(model_id)
    info = get_model_info(resolved)
    if info is None:
        return f"unknown_model:{resolved}"

    if "parakeet" in resolved.lower():
        if not _session_language_is_english_only(forced_language, allowed_languages, transcription_mode):
            return "parakeet_skipped_for_non_english_or_multilingual_session"

    if forced_language and not is_model_safe_for_language(resolved, forced_language):
        return f"forced_language_not_supported:{forced_language}"

    if allowed_languages:
        unsupported = [lang for lang in allowed_languages if not is_model_safe_for_language(resolved, lang)]
        if unsupported and len(unsupported) == len(allowed_languages):
            return f"allowed_languages_not_supported:{','.join(unsupported)}"

    return None


def _fallback_result(
    model_id: str,
    reason: str,
    requested_language: Optional[str],
    window_ms: Optional[int] = None,
) -> Dict:
    end_s = round((window_ms or 0) / 1000, 3)
    return {
        "text": "",
        "segments": [],
        "detection_info": {
            "requested_language": requested_language,
            "detected_language": None,
            "language_probability": None,
            "fallback_reason": reason,
        },
        "success": False,
        "error": reason,
        "degraded": True,
        "model_id": model_id,
        "fallback_segments": [{"start": 0.0, "end": end_s, "text": ""}] if end_s else [],
        "segment_timestamp_unit": "seconds",
    }


def unload_all_models():
    """Free all cached models and VRAM."""
    global _faster_whisper_model
    _faster_whisper_model = None
    _nemo_model_cache.clear()
    _hf_model_cache.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _transcribe_faster_whisper(audio_path: str, model_size: str,
                                language: str = None) -> Dict:
    """Transcribe using faster-whisper."""
    global _faster_whisper_model
    try:
        from faster_whisper import WhisperModel

        # faster-whisper runs on CTranslate2, so probe that backend first.
        device = "cpu"
        compute_type = "int8"
        try:
            import ctranslate2

            if int(ctranslate2.get_cuda_device_count()) > 0:
                device = "cuda"
                compute_type = "float16"
        except Exception:
            pass

        if device == "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
            except ImportError:
                pass
            except Exception:
                pass

        try:
            if device == "cuda":
                logger.info("Loading faster-whisper:%s on GPU", model_size)
        except Exception:
            pass

        if _faster_whisper_model is None or getattr(_faster_whisper_model, '_model_size', '') != model_size:
            unload_all_models()
            _faster_whisper_model = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
            _faster_whisper_model._model_size = model_size

        kwargs = {"language": language} if language and language != "auto" else {}
        kwargs["task"] = "transcribe"  # NEVER translate
        kwargs["beam_size"] = 5

        segments_iter, info = _faster_whisper_model.transcribe(audio_path, **kwargs)
        segments = []
        full_text_parts = []

        for seg in segments_iter:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
            full_text_parts.append(seg.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "segments": segments,
            "detection_info": {
                "detected_language": info.language,
                "language_probability": info.language_probability,
                "requested_language": language,
            },
            "success": True,
            "segment_timestamp_unit": "seconds",
        }
    except ImportError as e:
        logger.warning("faster-whisper not installed; using degraded fallback")
        return _fallback_result(f"faster-whisper:{model_size}", f"provider_unavailable:{e}", language)
    except Exception as e:
        logger.error(f"faster-whisper transcription failed: {e}")
        return _fallback_result(f"faster-whisper:{model_size}", str(e), language)


def _transcribe_nemo(audio_path: str, model_path: str,
                     model_id: str, language: str = None) -> Dict:
    """Transcribe using NeMo ASR (Parakeet or Canary).

    IMPORTANT: Parakeet is English-only. We do NOT fake multilingual capability.
    If language is non-English and model is Parakeet, we proceed but mark
    low confidence and provenance degradation.
    """
    try:
        import nemo.collections.asr as nemo_asr
        from app.pipeline.ingest import normalize_audio_file
        import tempfile

        # Always normalize to mono 16kHz for NeMo
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            normalize_audio_file(audio_path, tmp_path)

            if model_path not in _nemo_model_cache:
                # Find .nemo file
                nemo_file = None
                for f in Path(model_path).iterdir():
                    if f.suffix == ".nemo" and f.stat().st_size > 1_000_000:
                        nemo_file = str(f)
                        break
                if nemo_file is None:
                    return _fallback_result(model_id, "no_nemo_file_found", language)

                model = nemo_asr.models.ASRModel.restore_from(nemo_file)
                _nemo_model_cache[model_path] = model

            model = _nemo_model_cache[model_path]

            # Determine model type
            is_canary = "MultiTask" in type(model).__name__

            if is_canary:
                # Canary: transcribe only (source_lang == target_lang)
                lang = language if language and language != "auto" else "en"
                try:
                    result = model.transcribe(
                        [tmp_path],
                        source_lang=lang,
                        target_lang=lang,
                        pnc="yes",
                    )
                except TypeError:
                    result = model.transcribe(
                        [tmp_path],
                        source_lang=lang,
                        target_lang=lang,
                    )
            else:
                # Parakeet TDT / other RNNT/CTC
                result = model.transcribe([tmp_path], timestamps=True)

            # Extract text and segments
            if hasattr(result, '__iter__') and not isinstance(result, str):
                if hasattr(result[0], 'text'):
                    text = result[0].text
                    segments = []
                    # Try to get word timestamps
                    if hasattr(result[0], 'timestamp') and result[0].timestamp:
                        ts = result[0].timestamp
                        if isinstance(ts, dict) and "word" in ts:
                            for w in ts["word"]:
                                segments.append({
                                    "start": w.get("start", 0.0),
                                    "end": w.get("end", 0.0),
                                    "text": w.get("word", ""),
                                })
                    if not segments and text:
                        segments = [{"start": 0.0, "end": 30.0, "text": text}]
                else:
                    text = str(result[0]) if result else ""
                    segments = [{"start": 0.0, "end": 30.0, "text": text}] if text else []
            else:
                text = str(result) if result else ""
                segments = [{"start": 0.0, "end": 30.0, "text": text}] if text else []

            # Honest language handling for Parakeet
            language_note = None
            if "parakeet" in model_id.lower() and language and language not in ("en", "auto", None):
                language_note = f"Parakeet is English-only; requested language '{language}' may produce degraded results"

            return {
                "text": text.strip(),
                "segments": segments,
                "detection_info": {
                    "detected_language": "en" if "parakeet" in model_id.lower() else language,
                    "language_probability": None,
                    "requested_language": language,
                    "language_note": language_note,
                },
                "success": True,
                "segment_timestamp_unit": "seconds",
            }
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except ImportError as e:
        logger.warning("NeMo ASR not installed; using degraded fallback for %s", model_id)
        return _fallback_result(model_id, f"provider_unavailable:{e}", language)
    except Exception as e:
        logger.error(f"NeMo transcription failed for {model_id}: {e}")
        return _fallback_result(model_id, str(e), language)


def transcribe_window(audio_path: str, model_id: str,
                      language: str = None,
                      allowed_languages: Optional[List[str]] = None,
                      forced_language: Optional[str] = None,
                      transcription_mode: str = "verbatim_multilingual") -> Dict:
    """Dispatch transcription to the appropriate provider."""
    resolved = resolve_model_id(model_id)
    info = get_model_info(resolved)
    requested_language = language or _requested_language_hint(forced_language, allowed_languages)
    skip_reason = _skip_reason_for_model(
        resolved,
        forced_language=forced_language,
        allowed_languages=allowed_languages,
        transcription_mode=transcription_mode,
    )

    if skip_reason:
        logger.info("Skipping %s: %s", resolved, skip_reason)
        return _fallback_result(resolved, skip_reason, requested_language)

    if info is None:
        return _fallback_result(resolved, f"unknown_model:{resolved}", requested_language)

    if info.provider == "faster_whisper":
        size = resolved.split(":")[-1]
        return _transcribe_faster_whisper(audio_path, size, requested_language)
    elif info.provider == "nemo_asr":
        if not info.model_path or not Path(info.model_path).is_dir():
            return _fallback_result(resolved, f"model_path_not_found:{info.model_path}", requested_language)
        return _transcribe_nemo(audio_path, info.model_path, resolved, requested_language)
    else:
        return _fallback_result(resolved, f"unsupported_provider:{info.provider}", requested_language)


def persist_candidate(session_id: str, window: Dict, model_id: str,
                      result: Dict) -> Dict:
    """Persist a single ASR candidate to candidates/."""
    sd = session_dir(session_id)
    candidate_id = f"cand_{uuid.uuid4().hex[:8]}"

    candidate = {
        "candidate_id": candidate_id,
        "session_id": session_id,
        "model_id": model_id,
        "window_id": window["window_id"],
        "window_start_ms": window["start_ms"],
        "window_end_ms": window["end_ms"],
        "window_type": window["window_type"],
        "raw_text": result.get("text", ""),
        "segments": result.get("segments", []),
        "language_evidence": result.get("detection_info", {}),
        "confidence_features": {
            "success": result.get("success", False),
            "error": result.get("error"),
            "degraded": result.get("degraded", False),
        },
        "decode_metadata": {
            "model_id": model_id,
            "transcription_mode": result.get("transcription_mode"),
            "segment_timestamp_unit": result.get("segment_timestamp_unit"),
        },
    }

    filename = f"{candidate_id}_{window['window_id']}_{model_id.replace(':', '_')}.json"
    atomic_write_json(str(sd / "candidates" / filename), candidate)

    return candidate


def run_asr_execution(session_id: str, windows: List[Dict],
                      model_ids: List[str], language: str = None,
                      allowed_languages: Optional[List[str]] = None,
                      forced_language: Optional[str] = None,
                      transcription_mode: str = "verbatim_multilingual",
                      stage_map: Dict = None,
                      progress_callback: Callable = None) -> Dict:
    """Execute multi-ASR on all scheduled windows.

    Runs models sequentially for VRAM management.
    Persists each candidate immediately.
    """
    sd = session_dir(session_id)
    windows_dir = str(sd / "windows")

    scheduled = [w for w in windows if w.get("scheduled", True)]
    total_work = len(scheduled) * len(model_ids)
    completed = 0
    all_candidates = []

    for model_id in model_ids:
        logger.info(f"Running ASR model: {model_id} on {len(scheduled)} windows")

        # Unload previous models before loading new one
        if model_id.startswith("nemo-asr:"):
            # Only unload non-NeMo models
            global _faster_whisper_model
            _faster_whisper_model = None

        for window in scheduled:
            audio_path = os.path.join(windows_dir, f"{window['window_id']}.wav")
            if not os.path.isfile(audio_path):
                logger.warning(f"Window audio missing: {audio_path}")
                completed += 1
                continue

            start_time = time.time()
            result = transcribe_window(
                audio_path,
                model_id,
                language=language,
                allowed_languages=allowed_languages,
                forced_language=forced_language,
                transcription_mode=transcription_mode,
            )
            elapsed = time.time() - start_time
            result["transcription_mode"] = transcription_mode

            candidate = persist_candidate(session_id, window, model_id, result)
            candidate["elapsed_s"] = round(elapsed, 2)
            all_candidates.append(candidate)

            completed += 1
            if progress_callback:
                progress_callback(int(completed / max(1, total_work) * 100))

        # Mark stage as done if we have a stage for this model
        if stage_map and model_id in stage_map:
            stage = stage_map[model_id]
            stage.commit()

    # Write summary
    summary = {
        "session_id": session_id,
        "model_ids": model_ids,
        "window_count": len(scheduled),
        "candidate_count": len(all_candidates),
        "candidates_by_model": {},
    }
    for model_id in model_ids:
        model_cands = [c for c in all_candidates if c["model_id"] == model_id]
        summary["candidates_by_model"][model_id] = {
            "count": len(model_cands),
            "success": sum(1 for c in model_cands if c["confidence_features"].get("success")),
            "failed": sum(1 for c in model_cands if not c["confidence_features"].get("success")),
        }

    summary_path = sd / "candidates" / "asr_summary.json"
    existing_summary = safe_read_json(str(summary_path)) or {}
    merged_candidates_by_model = dict(existing_summary.get("candidates_by_model") or {})
    merged_candidates_by_model.update(summary["candidates_by_model"])

    merged_model_ids = []
    for model_id in list(existing_summary.get("model_ids") or []) + model_ids:
        if model_id not in merged_model_ids:
            merged_model_ids.append(model_id)

    merged_summary = {
        "session_id": session_id,
        "model_ids": merged_model_ids,
        "window_count": max(existing_summary.get("window_count", 0), len(scheduled)),
        "candidate_count": sum(item.get("count", 0) for item in merged_candidates_by_model.values()),
        "candidates_by_model": merged_candidates_by_model,
    }

    atomic_write_json(str(summary_path), merged_summary)

    return merged_summary
