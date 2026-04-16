"""
Witness diagnostics - Stage 6 addendum (canonical spec section 9.6).

Every raw ASR output must be audited locally with flags such as:
  - language_mismatch
  - script_mismatch
  - possible_translation
  - semantic_drift
  - statement_question_drift
  - repetition_anomaly
  - edge_truncation_suspected
  - empty_candidate
  - provider_degraded

These flags do not automatically discard a candidate — they affect downstream
trust scoring and reconciliation reasoning. This module is deliberately
deterministic and cheap so it can run on every candidate without a model.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional

# Script classification lookup (covers the main scripts we care about).
_SCRIPT_RANGES = [
    ("cyrillic", 0x0400, 0x04FF),
    ("cyrillic_supplement", 0x0500, 0x052F),
    ("latin", 0x0000, 0x024F),
    ("arabic", 0x0600, 0x06FF),
    ("han", 0x4E00, 0x9FFF),
    ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
    ("hangul", 0xAC00, 0xD7AF),
    ("greek", 0x0370, 0x03FF),
    ("hebrew", 0x0590, 0x05FF),
]

# Language families that default to a specific script.  When the detected or
# requested language expects one script but the text is rendered in another,
# we flag `script_mismatch` — a strong hint of translation or transliteration.
_LANGUAGE_SCRIPT_EXPECTATION = {
    "ru": "cyrillic",
    "uk": "cyrillic",
    "bg": "cyrillic",
    "sr": "cyrillic",
    "be": "cyrillic",
    "en": "latin",
    "fr": "latin",
    "de": "latin",
    "es": "latin",
    "it": "latin",
    "pt": "latin",
    "pl": "latin",
    "cs": "latin",
    "tr": "latin",
    "nl": "latin",
    "ro": "latin",
    "ja": "han",
    "zh": "han",
    "ko": "hangul",
    "ar": "arabic",
    "he": "hebrew",
    "el": "greek",
}

_QUESTION_MARKS = set("?¿؟？")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Subtitle / media pollution phrases that Whisper-family models hallucinate on
# silence, music, or training-set contamination.  Witness-level detection
# allows downstream stages (reconciliation, retrieval) to suppress this text
# before it becomes canonical truth.
_MEDIA_JUNK_PATTERNS = [
    r"субтитры\s+сделал",
    r"субтитры\s+подготовил",
    r"корректор\s+\w+",
    r"редактор\s+\w+",
    r"dimatorzok",
    r"продолжение\s+следует",
    r"подписывайтесь",
    r"спасибо\s+за\s+просмотр",
    r"thanks?\s+for\s+watching",
    r"subscribe\s+to\s+(my|our)\s+channel",
    r"like\s+and\s+subscribe",
    r"^\s*\[music\]\s*$",
    r"^\s*\(music\)\s*$",
    r"^\s*\[applause\]\s*$",
    r"sous[- ]titres?\s+réalisés?\s+par",
    r"merci\s+d(?:'|e\s+)avoir\s+regardé",
]
_MEDIA_JUNK_RE = re.compile("|".join(_MEDIA_JUNK_PATTERNS), re.IGNORECASE | re.UNICODE)


def looks_like_media_pollution(text: str) -> bool:
    """Return True if text is a known subtitle/media hallucination."""
    if not text or not text.strip():
        return False
    return bool(_MEDIA_JUNK_RE.search(text))


def classify_script(text: str) -> Optional[str]:
    """Return the dominant script name for the text, or None if too little evidence."""
    if not text:
        return None
    counts: Counter = Counter()
    total = 0
    for ch in text:
        if not ch.isalpha():
            continue
        total += 1
        code = ord(ch)
        bucket = None
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= code <= hi:
                bucket = "cyrillic" if name == "cyrillic_supplement" else name
                break
        if bucket is None:
            bucket = unicodedata.name(ch, "").split(" ", 1)[0].lower() or "other"
        counts[bucket] += 1
    if total < 3:
        return None
    dominant, _ = counts.most_common(1)[0]
    return dominant


def _repetition_ratio(text: str) -> float:
    tokens = _WORD_RE.findall((text or "").lower())
    if len(tokens) < 4:
        return 0.0
    counts = Counter(tokens)
    most_common_count = counts.most_common(1)[0][1]
    return round(most_common_count / len(tokens), 3)


def _looks_truncated_edge(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    # A final token with no closing punctuation AND a prefix starting lowercase
    # is a strong signal that a word was cut at window boundaries.
    first_word = stripped.split(" ", 1)[0]
    last_char = stripped[-1]
    starts_mid_word = (
        first_word
        and first_word[0].islower()
        and first_word.isalpha()
        and len(first_word) <= 2
    )
    ends_without_terminator = last_char not in ".!?…。！？»\")"
    return starts_mid_word and ends_without_terminator


def compute_candidate_flags(
    *,
    raw_text: str,
    detected_language: Optional[str],
    requested_language: Optional[str],
    transcription_mode: Optional[str],
    success: bool,
    degraded: bool,
    duration_s: Optional[float] = None,
    segments: Optional[List[Dict]] = None,
) -> Dict:
    """Produce the canonical witness-audit record for a single candidate.

    Returns a dict with `candidate_flags` (list) plus numeric supporting
    metrics so downstream reconciliation can reason about trust without
    re-analysing the text.
    """
    flags: List[str] = []
    details: Dict[str, object] = {}

    text = (raw_text or "").strip()

    if not success:
        flags.append("empty_candidate")
        details["failure"] = True
    elif not text:
        flags.append("empty_candidate")
    if degraded:
        flags.append("provider_degraded")

    script = classify_script(text) if text else None
    details["script"] = script

    expected_script = None
    if requested_language:
        expected_script = _LANGUAGE_SCRIPT_EXPECTATION.get(requested_language.lower())

    detected_script = None
    if detected_language:
        detected_script = _LANGUAGE_SCRIPT_EXPECTATION.get(detected_language.lower())

    if (
        requested_language
        and detected_language
        and requested_language.lower() != detected_language.lower()
    ):
        flags.append("language_mismatch")
        details["requested_language"] = requested_language
        details["detected_language"] = detected_language

    if expected_script and script and script != expected_script and script != "other":
        flags.append("script_mismatch")
        details["expected_script"] = expected_script
        details["observed_script"] = script

    # Possible translation: Cyrillic language expectation but Latin text output
    # (or vice versa) is the classic Whisper translation drift.
    if expected_script == "cyrillic" and script == "latin" and text:
        flags.append("possible_translation")
    elif detected_script == "cyrillic" and script == "latin" and text:
        flags.append("possible_translation")
    elif (
        transcription_mode == "verbatim_multilingual"
        and expected_script
        and script
        and expected_script != script
        and script != "other"
    ):
        flags.append("possible_translation")

    repetition = _repetition_ratio(text)
    details["repetition_ratio"] = repetition
    if repetition >= 0.45:
        flags.append("repetition_anomaly")

    if _looks_truncated_edge(text):
        flags.append("edge_truncation_suspected")

    if looks_like_media_pollution(text):
        flags.append("media_pollution_suspected")

    # Statement/question drift: very short windows that end with a question mark
    # when neighbour windows have none are worth auditing later.
    if text and text[-1] in _QUESTION_MARKS and len(text.split()) <= 3:
        flags.append("statement_question_drift")

    # semantic_drift is a weaker heuristic: when a candidate is wildly shorter
    # than the window duration would suggest, the ASR likely dropped content.
    if duration_s and text:
        expected_min_chars = max(1, int(duration_s * 2))
        if len(text) < expected_min_chars and duration_s > 5:
            flags.append("semantic_drift")
            details["observed_chars"] = len(text)
            details["expected_min_chars"] = expected_min_chars

    # Deduplicate while preserving order
    seen = set()
    unique_flags: List[str] = []
    for flag in flags:
        if flag in seen:
            continue
        seen.add(flag)
        unique_flags.append(flag)

    return {
        "candidate_flags": unique_flags,
        "diagnostics": details,
    }
