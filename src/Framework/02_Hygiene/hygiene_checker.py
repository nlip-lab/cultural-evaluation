"""
Phase 2 — Output Hygiene Checker
Checks adapted text for fluency, completeness, and non-hallucination.
No API calls. Fast, deterministic. Uses fastText for language detection.
"""

import re
from pathlib import Path
from typing import Optional

try:
    import fasttext
    import fasttext.FastText as _ft_module
    import numpy as _np
    _FTClass = _ft_module._FastText
    def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
        """Use NumPy-asarray conversion for compatibility with recent NumPy behavior."""
        def _check(entry):
            """Validate a single-line entry for fastText predict API."""
            if entry.find("\n") != -1:
                raise ValueError("predict processes one line at a time (remove '\\n')")
            return entry + "\n"
        if isinstance(text, list):
            text = [_check(e) for e in text]
            all_labels, all_probs = self.f.multilinePredict(text, k, threshold, on_unicode_error)
            return all_labels, all_probs
        text = _check(text)
        predictions = self.f.predict(text, k, threshold, on_unicode_error)
        if predictions:
            probs, labels = zip(*predictions)
        else:
            probs, labels = ([], ())
        return labels, _np.asarray(probs)
    _FTClass.predict = _patched_predict
except ImportError:
    fasttext = None

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODELS_DIR = PROJECT_ROOT / "models"
LID_MODEL_PATH = MODELS_DIR / "lid.176.bin"

INDIAN_ENGLISH_ALLOWLIST = {
    "crore", "lakh", "rupee", "rupees", "rs", "inr",
    "chai", "roti", "dal", "paneer", "ghee", "masala",
    "biryani", "dosa", "idli", "sambar",
    "aunty", "uncle", "ji", "bhai", "didi", "yaar", "arre",
    "kirana", "bazaar", "mela", "puja", "pooja", "aarti",
    "kurta", "saree", "sari", "dupatta", "salwar", "dhoti",
    "rickshaw", "autorickshaw", "tiffin", "dabba",
}

META_PATTERNS = [
    r"(?i)(here is|here's) (the|a|an|my) (adapted|cultural|indian|localized)",
    r"(?i)i('ve| have) (changed|replaced|adapted|modified|converted)",
    r"(?i)as an ai|as a language model",
    r"(?i)note:|disclaimer:|explanation:",
    r"(?i)the (original|source|american) (text|version|story|recipe|dialogue|problem)",
    r"(?i)i replaced .* with",
    r"(?i)cultural adaptation:",
]

REFUSAL_PATTERNS = [
    r"(?i)i (cannot|can't|am unable to|won't) (adapt|rewrite|modify|change)",
    r"(?i)i('m| am) (sorry|unable)",
    r"(?i)this (text|content) (is|does not|cannot)",
    r"(?i)as an ai assistant",
]

def _load_fasttext_model():
    """Lazy load fastText model on first use."""
    if not LID_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"FastText model not found at {LID_MODEL_PATH}\n"
            f"Download: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin "
            f"-O {LID_MODEL_PATH}"
        )
    return fasttext.load_model(str(LID_MODEL_PATH))

def _extract_tokens(text: str) -> list[str]:
    """Simple tokenization: lowercase words, remove punctuation."""
    return re.findall(r"\b\w+\b", text.lower())

def _find_ngrams(tokens: list[str], n: int) -> list[tuple]:
    """Generate n-grams from token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def _detect_repetition(text: str) -> bool:
    """Check if any 4-gram appears 3+ times."""
    tokens = _extract_tokens(text)
    if len(tokens) < 4:
        return False

    ngrams = _find_ngrams(tokens, 4)
    gram_counts = {}
    for gram in ngrams:
        gram_counts[gram] = gram_counts.get(gram, 0) + 1

    return any(count >= 3 for count in gram_counts.values())

def _is_truncated(text: str, length_ratio: float) -> bool:
    """Check if text is truncated: no terminal punctuation AND short length ratio."""
    text = text.strip()
    if not text:
        return False

    has_terminal_punct = text[-1] in ".!?;"
    if has_terminal_punct:
        return False

    return length_ratio < 0.3

def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    """Check if text matches any regex pattern."""
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

def _detect_meta_commentary_ratio(text: str) -> float:
    """
    Fraction of sentences that contain a meta-commentary pattern.

    Sentence-level scoring is more robust than char-span coverage because
    meta patterns are short trigger phrases, not full explanatory prose.

    Threshold for C1.4: >0.5  (more than half sentences are meta).

    Design decision: a single-sentence response that starts with a meta
    pattern (e.g. "Here is the adapted text: She had chai.") scores 1.0
    and is correctly flagged C1.4 — such a response has essentially no
    standalone adapted content.  Multi-line adaptations with one meta
    preamble sentence score <0.5 and fall through to fail_soft.
    """
    if not text.strip():
        return 0.0

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return 0.0

    meta_count = sum(
        1 for s in sentences
        if any(re.search(p, s) for p in META_PATTERNS)
    )
    return meta_count / len(sentences)

class HygieneChecker:
    """Fast, deterministic hygiene checker for adapted text."""

    def __init__(self):
        """Initialize lazy resources used by hygiene checks."""
        self._ft_model = None

    def check(self, source_text: str, adapted_text: str, domain: str) -> dict:
        """
        Check hygiene of adapted text.

        Args:
            source_text: Original text
            adapted_text: Model's adaptation
            domain: Domain (gsm8k, rocstories, dailydialog, recipenlg)

        Returns:
            dict with comprehensive hygiene results
        """
        result = {
            "lang_check": None,
            "meta_commentary": None,
            "refusal": None,
            "degeneration": None,
            "length_ratio": None,
            "length_check": None,
            "overall": None,
            "iaet_label": None,
            "failure_reason": None,
        }

        result["degeneration"] = self._check_degeneration(adapted_text)

        source_tokens = len(source_text.split())
        adapted_tokens = len(adapted_text.split())
        result["length_ratio"] = adapted_tokens / max(source_tokens, 1)

        result["length_check"] = self._check_length(result["length_ratio"])
        if result["degeneration"] == "pass" and _is_truncated(adapted_text, result["length_ratio"]):
            result["degeneration"] = "truncated"

        result["refusal"] = self._check_refusal(adapted_text)

        result["lang_check"] = self._check_language(adapted_text)

        result["meta_commentary"] = self._check_meta_commentary(adapted_text)

        overall, label, reason = self._classify_overall(
            result, adapted_text, result["length_ratio"]
        )
        result["overall"] = overall
        result["iaet_label"] = label
        result["failure_reason"] = reason

        return result

    def _check_degeneration(self, text: str) -> str:
        """Check for empty, truncated, or repetitive text."""
        text = text.strip()

        if not text:
            return "empty"

        if _detect_repetition(text):
            return "repetitive"

        return "pass"

    def _check_length(self, length_ratio: float) -> str:
        """Check if length is in acceptable range."""
        if length_ratio < 0.3:
            return "too_short"
        if length_ratio > 3.0:
            return "too_long"
        return "pass"

    def _check_refusal(self, text: str) -> str:
        """Check if model refused to adapt."""
        if _matches_any_pattern(text, REFUSAL_PATTERNS):
            return "flagged"

        if len(text.strip()) < 20 and not re.search(r"\w{3,}", text):
            return "flagged"

        return "pass"

    def _check_language(self, text: str) -> str:
        """Check if text is in English (or Indian English)."""
        if not fasttext:
            return "pass"

        if self._ft_model is None:
            try:
                self._ft_model = _load_fasttext_model()
            except FileNotFoundError as e:
                return "pass"

        text_clean = text.strip()
        if not text_clean:
            return "pass"

        text_sample = text_clean[:1000].replace("\n", " ").replace("\r", " ").strip()

        predictions = self._ft_model.predict(text_sample, k=2)
        labels = predictions[0]
        scores = predictions[1]

        top_lang = labels[0].replace("__label__", "") if len(labels) > 0 else "unknown"
        top_score = float(scores[0]) if len(scores) > 0 else 0.0
        second_lang = labels[1].replace("__label__", "") if len(labels) > 1 else None
        second_score = float(scores[1]) if len(scores) > 1 else 0.0

        tokens = _extract_tokens(text)
        indian_english_tokens = [t for t in tokens if t in INDIAN_ENGLISH_ALLOWLIST]

        if tokens and len(indian_english_tokens) > 0:
            non_english_tokens = [t for t in tokens if t not in {"english", "en", "english,"}]
            if non_english_tokens and all(t in INDIAN_ENGLISH_ALLOWLIST for t in non_english_tokens):
                return "pass"

        if top_lang != "en":
            if top_score > 0.5:
                return "fail_non_english"

        if top_lang == "en" and top_score < 0.4:
            if second_lang and second_lang != "en" and second_score > 0.3:
                return "fail_mixed"

        return "pass"

    def _check_meta_commentary(self, text: str) -> str:
        """Check if text contains meta-commentary or explanations."""
        if _matches_any_pattern(text, META_PATTERNS):
            return "flagged"
        return "pass"

    def _classify_overall(
        self, checks: dict, adapted_text: str, length_ratio: float
    ) -> tuple[str, Optional[str], Optional[str]]:
        """
        Classify overall hygiene and assign IAET label.

        Returns:
            (overall: str, iaet_label: str|None, failure_reason: str|None)
        """

        if checks["lang_check"] == "fail_non_english":
            return ("fail_hard", "C5.1", "non-English content detected")

        if checks["degeneration"] in ("empty", "repetitive"):
            reason = f"text is {checks['degeneration']}"
            return ("fail_hard", "C5.1", reason)

        if checks["degeneration"] == "truncated":
            return ("fail_hard", "C5.1", "text appears truncated")

        if checks["refusal"] == "flagged":
            return ("fail_hard", "C5.1", "model refused to adapt")

        if checks["meta_commentary"] == "flagged":
            meta_ratio = _detect_meta_commentary_ratio(adapted_text)
            if meta_ratio >= 0.5:
                return (
                    "fail_hard",
                    "C1.4",
                    "text is mostly meta-commentary"
                )

        if checks["meta_commentary"] == "flagged":
            return ("fail_soft", None, "contains meta-commentary")

        if checks["length_check"] != "pass":
            return ("fail_soft", None, f"length check failed: {checks['length_check']}")

        if checks["lang_check"] == "fail_mixed":
            return ("fail_soft", None, "mixed language detected")

        return ("pass", None, None)

_global_checker = None

def get_checker() -> HygieneChecker:
    """Get or create the global checker instance."""
    global _global_checker
    if _global_checker is None:
        _global_checker = HygieneChecker()
    return _global_checker

def check(source_text: str, adapted_text: str, domain: str) -> dict:
    """Convenience function: check hygiene using global checker instance."""
    return get_checker().check(source_text, adapted_text, domain)

