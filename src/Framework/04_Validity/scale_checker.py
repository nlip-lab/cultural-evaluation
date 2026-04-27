"""
Phase 4.1 — Deterministic Scale / Unit / Currency Checker
==========================================================
No API calls. Pure regex + arithmetic.

Used by run_c2_judge.py to:
  - Override LLM scale_transfer scores when arithmetic errors are provable
  - Cross-check GSM8K adapted answers against source answers
"""

import re
from typing import Optional

_LB_TO_KG    = 0.4536
_OZ_TO_G     = 28.3495
_MILE_TO_KM  = 1.6093
_INCH_TO_CM  = 2.54
_FT_TO_M     = 0.3048
_CUP_TO_ML   = 236.588
_GALLON_TO_L = 3.7854
_TBSP_TO_ML  = 14.787
_TSP_TO_ML   = 4.929

_USD_INR_MIN = 50.0
_USD_INR_MAX = 120.0
_USD_INR_MID = 83.0

_TOLERANCE = 0.05

_UNIT_PAIRS = {
    "lb_to_kg":              (r"\blbs?\b|\bpounds?\b",
                              r"\bkgs?\b|\bkilograms?\b",
                              _LB_TO_KG),
    "oz_to_g":               (r"\boz\b|\bounces?\b",
                              r"\bg\b|\bgrams?\b",
                              _OZ_TO_G),
    "mile_to_km":            (r"\bmiles?\b",
                              r"\bkms?\b|\bkilometers?\b",
                              _MILE_TO_KM),
    "inch_to_cm":            (r"\binches?\b|\bin\b",
                              r"\bcms?\b|\bcentimeters?\b",
                              _INCH_TO_CM),
    "ft_to_m":               (r"\bfeet\b|\bft\b|\bfoot\b",
                              r"\bmeters?\b|\bm\b",
                              _FT_TO_M),
    "cup_to_ml":             (r"\bcups?\b",
                              r"\bml\b|\bmilliliters?\b",
                              _CUP_TO_ML),
    "gallon_to_liter":       (r"\bgallons?\b",
                              r"\bliters?\b|\bl\b",
                              _GALLON_TO_L),
    "tbsp_to_ml":            (r"\btbsp\b|\btablespoons?\b",
                              r"\bml\b",
                              _TBSP_TO_ML),
    "tsp_to_ml":             (r"\btsp\b|\bteaspoons?\b",
                              r"\bml\b",
                              _TSP_TO_ML),
    "fahrenheit_to_celsius": (r"\b°?F\b|\bFahrenheit\b",
                              r"\b°?C\b|\bCelsius\b",
                              lambda f: (f - 32) * 5 / 9),
}

_NUMBER_PATTERN = re.compile(
    r"""
    (?P<prefix>\$\s*|Rs\.?\s*|₹\s*|INR\s*)?
    (?P<value>
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
        | \d+(?:\.\d+)?
    )
    \s*
    (?P<unit>
        °F | °C |
        pounds?  | lbs?    | ounces?  | oz |
        miles?   | kilometers? | km   |
        cups?    | gallons?    | liters?  |
        tbsp     | tablespoons? | tsp | teaspoons? |
        inches?  | feet   | foot  | ft  |
        centimeters? | cms? |
        kilograms?   | kgs? |
        milliliters? | ml   |
        grams?       |
        rupees?  | Rs\.?  | INR | ₹ | USD |
        \bm\b | \bl\b | \bg\b
    )?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_ANSWER_PATTERNS = [
    re.compile(r"\u0023\u0023\u0023\u0023\s*([\d,]+\.?\d*)", re.IGNORECASE),
    re.compile(r"(?:answer|total|cost|price)\s*(?:is|:)\s*([\$₹Rs\.]*[\d,]+\.?\d*)", re.IGNORECASE),
    re.compile(r"([\$₹Rs\.]*[\d,]+\.?\d*)\s*\.?\s*$"),
]

class ScaleChecker:

    CONVERSIONS = {
        "lb_to_kg":              _LB_TO_KG,
        "oz_to_g":               _OZ_TO_G,
        "mile_to_km":            _MILE_TO_KM,
        "inch_to_cm":            _INCH_TO_CM,
        "ft_to_m":               _FT_TO_M,
        "fahrenheit_to_celsius": lambda f: (f - 32) * 5 / 9,
        "cup_to_ml":             _CUP_TO_ML,
        "gallon_to_liter":       _GALLON_TO_L,
        "tbsp_to_ml":            _TBSP_TO_ML,
        "tsp_to_ml":             _TSP_TO_ML,
        "usd_to_inr":            None,
    }

    def extract_numbers(self, text: str) -> list[dict]:
        """
        Extract all numbers with surrounding context.

        Returns a list of:
          {
            "value":    float,
            "unit":     str | None; unit suffix or currency prefix if no suffix
            "context":  str; ~40-char window around the match
            "position": int; char offset of match start
          }

        Currency prefixes ($, ₹, Rs., INR) are captured in 'unit' when no
        explicit unit suffix follows the number.
        """
        results = []
        for m in _NUMBER_PATTERN.finditer(text):
            raw = m.group("value")
            if not raw:
                continue
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                continue

            unit   = (m.group("unit") or "").strip()
            prefix = (m.group("prefix") or "").strip()

            if not unit and prefix:
                unit = prefix.rstrip()

            if value == 0 and not unit:
                continue

            start   = max(0, m.start() - 20)
            end     = min(len(text), m.end() + 20)
            context = text[start:end].replace("\n", " ")

            results.append({
                "value":    value,
                "unit":     unit or None,
                "context":  context,
                "position": m.start(),
            })
        return results

    def check_unit_conversion(
        self,
        source_span: str,
        adapted_span: str,
        source_value: float,
        adapted_value: float,
        unit_pair: str,
    ) -> dict:
        """
        Check if a unit conversion is numerically correct (5% tolerance).

        unit_pair: key from CONVERSIONS / _UNIT_PAIRS, e.g. "lb_to_kg",
                   "fahrenheit_to_celsius".

        Returns:
          {
            "conversion_correct": bool,
            "expected_value":     float,
            "actual_value":       float,
            "tolerance_pct":      float,
            "error":              null
                                  | "value_not_converted"   (value identical to source)
                                  | "label_only_change"      (value unchanged, unit label differs)
                                  | "incorrect_conversion"   (value changed but wrong)
                                  | "unknown_unit_pair"
          }
        """
        if unit_pair not in _UNIT_PAIRS:
            return {
                "conversion_correct": None,
                "expected_value":     None,
                "actual_value":       adapted_value,
                "tolerance_pct":      None,
                "error":              "unknown_unit_pair",
            }

        _, target_pat, factor = _UNIT_PAIRS[unit_pair]

        if callable(factor):
            expected = factor(source_value)
        else:
            expected = source_value * factor

        if abs(adapted_value - source_value) < 1e-9:
            target_unit_found = bool(re.search(target_pat, adapted_span, re.IGNORECASE))
            error = "label_only_change" if target_unit_found else "value_not_converted"
            return {
                "conversion_correct": False,
                "expected_value":     round(expected, 4),
                "actual_value":       adapted_value,
                "tolerance_pct":      round(abs(adapted_value - expected) / max(abs(expected), 1e-9), 4),
                "error":              error,
            }

        diff_pct = abs(adapted_value - expected) / max(abs(expected), 1e-9)

        if diff_pct <= _TOLERANCE:
            return {
                "conversion_correct": True,
                "expected_value":     round(expected, 4),
                "actual_value":       adapted_value,
                "tolerance_pct":      round(diff_pct, 4),
                "error":              None,
            }

        return {
            "conversion_correct": False,
            "expected_value":     round(expected, 4),
            "actual_value":       adapted_value,
            "tolerance_pct":      round(diff_pct, 4),
            "error":              "incorrect_conversion",
        }

    def check_currency_plausibility(
        self,
        source_value_usd: float,
        adapted_value_inr: float,
        item_description: str = "",
    ) -> dict:
        """
        Check USD→INR plausibility. Heuristic range: 1 USD ≈ 50–120 INR.

        item_description: the adapted text span (used to detect lingering $ symbol).

        Returns:
          {
            "plausible": bool,
            "ratio":     float | None,
            "error":     null
                         | "currency_symbol_retained"  ($ still in adapted text)
                         | "value_unchanged"            (INR value == USD value)
                         | "implausible_ratio"          (ratio outside [50, 120])
          }
        """
        if "$" in item_description:
            return {"plausible": False, "ratio": None, "error": "currency_symbol_retained"}

        if source_value_usd == 0:
            return {"plausible": True, "ratio": None, "error": None}

        if abs(adapted_value_inr - source_value_usd) < 0.01:
            return {"plausible": False, "ratio": 1.0, "error": "value_unchanged"}

        ratio = adapted_value_inr / source_value_usd
        if _USD_INR_MIN <= ratio <= _USD_INR_MAX:
            return {"plausible": True, "ratio": round(ratio, 2), "error": None}

        return {"plausible": False, "ratio": round(ratio, 2), "error": "implausible_ratio"}

    def check_gsm8k_answer(
        self,
        source_text: str,
        adapted_text: str,
        source_answer: Optional[float],
    ) -> dict:
        """
        For GSM8K items: extract the final numeric answer from the adapted text
        and verify:
          1. The answer is present (not missing).
          2. If no currency adaptation occurred, the numeric answer matches
             the source answer within 5% tolerance.
          3. Arithmetic internal consistency: the final answer is approximately
             reachable from numbers in the adapted text.

        Returns:
          {
            "answer_preserved":      bool,
            "answer_extracted":      float | None,
            "arithmetic_consistent": bool,
            "error_type":            None
                                     | "answer_missing"
                                     | "answer_changed"
                                     | "arithmetic_broken"
          }
        """
        adapted_answer = self._extract_final_answer(adapted_text)

        if adapted_answer is None:
            return {
                "answer_preserved":      False,
                "answer_extracted":      None,
                "arithmetic_consistent": False,
                "error_type":            "answer_missing",
            }

        if source_answer is not None:
            has_inr = bool(
                "₹" in adapted_text
                or re.search(r"\brupees?\b", adapted_text, re.IGNORECASE)
                or re.search(r"\bINR\b", adapted_text)
                or re.search(r"\bRs\.?\s", adapted_text, re.IGNORECASE)
            )
            if not has_inr:
                diff_pct = abs(adapted_answer - source_answer) / max(abs(source_answer), 1.0)
                if diff_pct > _TOLERANCE:
                    return {
                        "answer_preserved":      False,
                        "answer_extracted":      adapted_answer,
                        "arithmetic_consistent": False,
                        "error_type":            "answer_changed",
                    }

        consistent, broken = self._check_arithmetic_consistency(adapted_text, adapted_answer)

        return {
            "answer_preserved":      True,
            "answer_extracted":      adapted_answer,
            "arithmetic_consistent": consistent,
            "error_type":            "arithmetic_broken" if broken else None,
        }

    def _extract_final_answer(self, text: str) -> Optional[float]:
        """Try multiple patterns to extract the final numeric answer."""
        for pat in _ANSWER_PATTERNS:
            m = pat.search(text)
            if m:
                raw = (
                    m.group(1)
                    .replace(",", "")
                    .replace("$", "")
                    .replace("₹", "")
                    .replace("Rs.", "")
                    .replace("Rs", "")
                    .strip()
                )
                try:
                    return float(raw)
                except ValueError:
                    continue
        return None

    def _check_arithmetic_consistency(self, text: str, final_answer: float) -> tuple[bool, bool]:
        """
        Heuristic consistency check: can final_answer be derived from numbers in text?

        Strategy:
          1. Collect all non-answer positive numbers in the text (up to 8).
          2. Check all non-empty subsets for sum ≈ final_answer.
          3. Check all pairs for product ≈ final_answer.
          4. Check all pairs for difference ≈ final_answer.

        Returns (consistent: bool, broken: bool).
        'broken' is only True when we have ≥3 numbers and NO simple relationship
        works at all — a high-confidence signal of arithmetic corruption.
        We are conservative: if there are fewer than 3 candidate numbers we
        cannot reliably judge, so we return (True, False).
        """
        numbers = [
            n["value"]
            for n in self.extract_numbers(text)
            if n["value"] != final_answer and n["value"] > 0
        ]

        if len(numbers) < 3:
            return True, False

        numbers = numbers[:8]
        n = len(numbers)
        fa = final_answer

        for mask in range(1, 1 << n):
            total = sum(numbers[i] for i in range(n) if mask & (1 << i))
            if abs(total - fa) / max(abs(fa), 1.0) <= _TOLERANCE:
                return True, False

        for i in range(n):
            for j in range(i, n):
                prod = numbers[i] * numbers[j]
                if abs(prod - fa) / max(abs(fa), 1.0) <= _TOLERANCE:
                    return True, False
                diff = abs(numbers[i] - numbers[j])
                if abs(diff - fa) / max(abs(fa), 1.0) <= _TOLERANCE:
                    return True, False

        return False, True

