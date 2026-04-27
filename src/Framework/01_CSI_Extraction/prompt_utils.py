"""Prompt utilities for Phase 1 CSI extraction."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "source_csi_extraction.txt"
_TEMPLATE: str | None = None


def _load_template() -> str:
    """Load and cache the CSI extraction prompt template."""
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return _TEMPLATE


def format_csi_prompt(source_text: str, domain: str) -> str:
    """Fill the CSI extraction prompt template with source_text and domain."""
    return _load_template().replace("{source_text}", source_text).replace("{domain}", domain)
