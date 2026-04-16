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
from app.pipeline.witness_diagnostics import compute_candidate_flags
from app.storage.session_store import session_dir

logger = logging.getLogger(__name__)

# Model caches for sequential VRAM management
_faster_whisper_model = None
_faster_whisper_runtime_preferences: Dict[str, tuple[str, str]] = {}
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


def unload_faster_whisper():
    """Unload faster-whisper / ctranslate2 and aggressively reclaim CUDA state.

    ctranslate2 uses its own CUDA allocator which corrupts PyTorch's handle
    tracking.  We must fully destroy all ctranslate2 objects, force GC, then
    reset PyTorch's CUDA state before any PyTorch-based model (NeMo) can use
    the GPU.
    """
    global _faster_whisper_model
    if _faster_whisper_model is not None:
        logger.info("Unloading faster-whisper model and reclaiming CUDA state")
    _faster_whisper_model = None
    _faster_whisper_runtime_preferences.clear()
    # Double GC to ensure ctranslate2 C++ destructors run before PyTorch reclaims
    gc.collect()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # Reset memory stats so the allocator starts fresh
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


def unload_nemo_models():
    """Unload NeMo models and free VRAM."""
    if _nemo_model_cache:
        logger.info("Unloading %d NeMo model(s)", len(_nemo_model_cache))
    _nemo_model_cache.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def unload_all_models():
    """Free all cached models and VRAM."""
    unload_faster_whisper()
    unload_nemo_models()
    _hf_model_cache.clear()
    gc.collect()


def _resolve_gpu_runtime(cfg) -> dict:
    """Determine device + compute_type using explicit config, then hardware probes.

    Returns dict with: device, compute_type, method (how the decision was made),
    and diagnostic details for audit logging.
    """
    requested_device = cfg.worker.whisper_device  # "cuda" or "cpu"
    requested_compute = cfg.worker.whisper_compute_type  # "float16", "int8", etc.
    fallbacks = cfg.worker.whisper_compute_type_fallbacks  # ["int8_float16", "int8"]
    strict = cfg.worker.strict_cuda

    diag = {
        "requested_device": requested_device,
        "requested_compute_type": requested_compute,
        "strict_cuda": strict,
        "torch_cuda": False,
        "ctranslate2_cuda_devices": 0,
    }

    # Probe hardware
    try:
        import torch
        diag["torch_cuda"] = torch.cuda.is_available()
        if diag["torch_cuda"]:
            diag["gpu_name"] = torch.cuda.get_device_name(cfg.worker.cuda_device_index)
    except Exception:
        pass

    try:
        import ctranslate2
        diag["ctranslate2_cuda_devices"] = int(ctranslate2.get_cuda_device_count())
    except Exception:
        pass

    cuda_available = diag["torch_cuda"] or diag["ctranslate2_cuda_devices"] > 0

    if requested_device == "cuda":
        if cuda_available:
            diag["method"] = "config_cuda_confirmed"
            return {"device": "cuda", "compute_type": requested_compute, **diag}

        # CUDA requested but not available
        if strict:
            diag["method"] = "strict_cuda_failed"
            return {"device": "FAIL", "compute_type": requested_compute, **diag}

        logger.warning("WHISPER_DEVICE=cuda but no CUDA available — falling back to CPU")
        diag["method"] = "cuda_requested_but_unavailable_cpu_fallback"
        return {"device": "cpu", "compute_type": "int8", **diag}

    # CPU explicitly requested
    diag["method"] = "config_cpu_explicit"
    return {"device": "cpu", "compute_type": "int8", **diag}


def _transcribe_faster_whisper(audio_path: str, model_size: str,
                                language: str = None) -> Dict:
    """Transcribe using faster-whisper with explicit GPU strategy from config."""
    global _faster_whisper_model
    cfg = get_config()

    # Check for a sticky override (set after a CUDA failure with strict=false)
    override = _faster_whisper_runtime_preferences.get(model_size)
    if override:
        device, compute_type = override
        runtime_info = {"device": device, "compute_type": compute_type, "method": "sticky_override"}
    else:
        runtime_info = _resolve_gpu_runtime(cfg)
        device = runtime_info["device"]
        compute_type = runtime_info["compute_type"]

    if device == "FAIL":
        msg = (
            f"WHISPER_STRICT_CUDA=true but no CUDA runtime available. "
            f"torch_cuda={runtime_info.get('torch_cuda')}, "
            f"ctranslate2_cuda={runtime_info.get('ctranslate2_cuda_devices')}"
        )
        logger.error(msg)
        return _fallback_result(f"faster-whisper:{model_size}", msg, language)

    def _load_and_transcribe(dev: str, ct: str) -> Dict:
        global _faster_whisper_model
        from faster_whisper import WhisperModel

        needs_reload = (
            _faster_whisper_model is None
            or getattr(_faster_whisper_model, "_model_size", "") != model_size
            or getattr(_faster_whisper_model, "_device", "") != dev
        )
        if needs_reload:
            unload_all_models()
            logger.info(
                "Loading faster-whisper:%s on %s (compute_type=%s)",
                model_size, dev.upper(), ct,
            )
            _faster_whisper_model = WhisperModel(
                model_size, device=dev, compute_type=ct,
                device_index=cfg.worker.cuda_device_index if dev == "cuda" else 0,
            )
            _faster_whisper_model._model_size = model_size
            _faster_whisper_model._device = dev

        kwargs = {"language": language} if language and language != "auto" else {}
        kwargs["task"] = "transcribe"
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

        result = {
            "text": " ".join(full_text_parts),
            "segments": segments,
            "detection_info": {
                "detected_language": info.language,
                "language_probability": info.language_probability,
                "requested_language": language,
                "device": dev,
                "compute_type": ct,
            },
            "success": True,
            "segment_timestamp_unit": "seconds",
        }
        return result

    try:
        return _load_and_transcribe(device, compute_type)
    except ImportError as e:
        logger.warning("faster-whisper not installed; using degraded fallback")
        return _fallback_result(f"faster-whisper:{model_size}", f"provider_unavailable:{e}", language)
    except Exception as e:
        error_text = str(e).lower()
        is_cuda_error = any(tok in error_text for tok in ("cuda", "cublas", "cudnn", "ctranslate"))

        if device == "cuda" and is_cuda_error:
            # Try compute_type fallbacks before giving up on GPU
            for fb_ct in cfg.worker.whisper_compute_type_fallbacks:
                logger.warning(
                    "faster-whisper:%s CUDA failed with %s (%s); trying compute_type=%s",
                    model_size, compute_type, e, fb_ct,
                )
                try:
                    unload_all_models()
                    return _load_and_transcribe("cuda", fb_ct)
                except Exception:
                    continue

            # All GPU attempts exhausted — CPU fallback if not strict
            if not cfg.worker.strict_cuda:
                logger.warning(
                    "faster-whisper:%s all CUDA compute types failed; falling back to CPU (WHISPER_STRICT_CUDA=false)",
                    model_size,
                )
                try:
                    _faster_whisper_runtime_preferences[model_size] = ("cpu", "int8")
                    unload_all_models()
                    return _load_and_transcribe("cpu", "int8")
                except Exception as retry_error:
                    logger.error("faster-whisper CPU retry also failed: %s", retry_error)
                    return _fallback_result(f"faster-whisper:{model_size}", str(retry_error), language)
            else:
                logger.error(
                    "faster-whisper:%s CUDA failed and WHISPER_STRICT_CUDA=true — not falling back to CPU",
                    model_size,
                )

        logger.error("faster-whisper:%s transcription failed: %s", model_size, e)
        return _fallback_result(f"faster-whisper:{model_size}", str(e), language)


def _cuda_vram_free_mb() -> float:
    """Return free GPU memory in MB, or 0 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def _load_nemo_model(model_path: str, model_id: str, device: str = "cuda"):
    """Load (or retrieve cached) NeMo model, handling CUDA fallback to CPU.

    Strategy:
    1. Always restore_from with map_location='cpu' to avoid CUDA issues
       during deserialization (ctranslate2 corrupts PyTorch handle tracker).
    2. Only move to CUDA if enough free VRAM (model_size * 1.3 headroom).
    3. On CUDA OOM during .cuda(), keep the CPU model — never reload from disk.
    """
    import nemo.collections.asr as nemo_asr

    # Check cache — try requested device first, then fall back to any device
    cache_key = f"{model_path}:{device}"
    if cache_key in _nemo_model_cache:
        return _nemo_model_cache[cache_key], device
    # If we requested CUDA but have a CPU-cached version, use it
    if device == "cuda":
        cpu_key = f"{model_path}:cpu"
        if cpu_key in _nemo_model_cache:
            logger.info("Using cached CPU model for %s (skipping CUDA)", model_id)
            return _nemo_model_cache[cpu_key], "cpu"

    # Find .nemo file
    nemo_file = None
    for f in Path(model_path).iterdir():
        if f.suffix == ".nemo" and f.stat().st_size > 1_000_000:
            nemo_file = str(f)
            break
    if nemo_file is None:
        raise FileNotFoundError(f"no .nemo file found in {model_path}")

    # Always load on CPU first to avoid ctranslate2↔PyTorch CUDA handle conflict.
    # NeMo's restore_from internally instantiates sub-models which may try to use
    # CUDA. map_location='cpu' prevents this.
    logger.info("Loading NeMo model %s from %s (map_location=cpu)", model_id, nemo_file)
    try:
        model = nemo_asr.models.ASRModel.restore_from(
            nemo_file, map_location="cpu"
        )
    except Exception as e:
        logger.error("NeMo restore_from failed for %s: %s", model_id, e)
        raise

    # Optionally move to CUDA if requested and enough VRAM available
    actual_device = "cpu"
    if device == "cuda":
        import torch
        if torch.cuda.is_available():
            # Estimate model size from parameter count
            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            param_mb = param_bytes / (1024 * 1024)
            free_mb = _cuda_vram_free_mb()
            headroom = param_mb * 1.3  # 30% overhead for activations/buffers

            if free_mb > headroom:
                try:
                    logger.info(
                        "Moving NeMo %s to CUDA (params=%.0fMB, free=%.0fMB)",
                        model_id, param_mb, free_mb,
                    )
                    model = model.cuda()
                    actual_device = "cuda"
                except RuntimeError as cuda_err:
                    logger.warning(
                        "NeMo %s .cuda() failed (%s); keeping on CPU",
                        model_id, cuda_err,
                    )
                    # Model stays on CPU — do NOT reload from disk
                    model = model.cpu()
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            else:
                logger.info(
                    "Skipping CUDA for NeMo %s: params=%.0fMB needs ~%.0fMB but only %.0fMB free",
                    model_id, param_mb, headroom, free_mb,
                )

    final_key = f"{model_path}:{actual_device}"
    _nemo_model_cache[final_key] = model
    return model, actual_device


def _run_nemo_inference(model, tmp_path: str, model_id: str,
                        language: str, is_canary: bool) -> tuple:
    """Run NeMo transcription and extract text + segments."""
    if is_canary:
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
        result = model.transcribe([tmp_path], timestamps=True)

    # Extract text and segments
    if hasattr(result, '__iter__') and not isinstance(result, str):
        if hasattr(result[0], 'text'):
            text = result[0].text
            segments = []
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

    return text, segments


def _transcribe_nemo(audio_path: str, model_path: str,
                     model_id: str, language: str = None) -> Dict:
    """Transcribe using NeMo ASR (Parakeet or Canary).

    IMPORTANT: Parakeet is English-only. We do NOT fake multilingual capability.
    If language is non-English and model is Parakeet, we proceed but mark
    low confidence and provenance degradation.
    """
    try:
        from app.pipeline.ingest import normalize_audio_file
        import tempfile

        # Always normalize to mono 16kHz for NeMo
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            normalize_audio_file(audio_path, tmp_path)

            # Determine preferred device
            cfg = get_config()
            preferred_device = cfg.worker.whisper_device  # same GPU preference

            model, actual_device = _load_nemo_model(model_path, model_id, preferred_device)
            is_canary = "canary" in model_id.lower() or "MultiTask" in type(model).__name__

            try:
                text, segments = _run_nemo_inference(model, tmp_path, model_id, language, is_canary)
            except RuntimeError as e:
                error_text = str(e).lower()
                if "cuda" in error_text or "assert" in error_text or "allocat" in error_text:
                    logger.warning(
                        "NeMo %s inference CUDA error (%s); moving model to CPU and retrying",
                        model_id, e,
                    )
                    # Move existing model to CPU instead of reloading from disk
                    cache_key = f"{model_path}:{actual_device}"
                    _nemo_model_cache.pop(cache_key, None)
                    try:
                        model = model.cpu()
                    except Exception:
                        pass
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    cpu_key = f"{model_path}:cpu"
                    _nemo_model_cache[cpu_key] = model
                    actual_device = "cpu"
                    text, segments = _run_nemo_inference(model, tmp_path, model_id, language, is_canary)
                else:
                    raise

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
    model_key = str(model_id).replace(":", "_").replace("/", "_")
    candidate_id = f"cand_{window['window_id']}_{model_key}"

    detection_info = result.get("detection_info", {}) or {}
    requested_language = result.get("requested_language") or detection_info.get("requested_language")
    transcription_mode = result.get("transcription_mode")
    duration_s = max(0.0, (window["end_ms"] - window["start_ms"]) / 1000.0)

    witness = compute_candidate_flags(
        raw_text=result.get("text", ""),
        detected_language=detection_info.get("detected_language"),
        requested_language=requested_language,
        transcription_mode=transcription_mode,
        success=result.get("success", False),
        degraded=result.get("degraded", False),
        duration_s=duration_s,
        segments=result.get("segments", []),
    )

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
        "language_evidence": detection_info,
        "confidence_features": {
            "success": result.get("success", False),
            "error": result.get("error"),
            "degraded": result.get("degraded", False),
        },
        "candidate_flags": witness["candidate_flags"],
        "witness_audit": witness["diagnostics"],
        "decode_metadata": {
            "model_id": model_id,
            "requested_language": requested_language,
            "transcription_mode": transcription_mode,
            "segment_timestamp_unit": result.get("segment_timestamp_unit"),
        },
    }

    filename = f"{candidate_id}.json"
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
