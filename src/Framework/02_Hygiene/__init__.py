"""
Phase 2 — Output Hygiene Checker

Fast, deterministic checks for adapted text quality:
- Language detection (fastText with Indian English support)
- Meta-commentary flagging (regex-based)
- Refusal detection
- Degeneration checks (empty, truncated, repetitive)
- Length ratio analysis
- Overall classification (pass / fail_soft / fail_hard) with CRAFT labels

Usage:
    from hygiene_checker import check
    result = check(source_text, adapted_text, domain)
    print(result["overall"])
"""

from .hygiene_checker import HygieneChecker, check, get_checker

__all__ = ["HygieneChecker", "check", "get_checker"]
