"""Format Pattern Analyzer — detects data format patterns for richer LLM evidence.

Runs before LLM generation to provide concrete, evidence-based context about
each column's value formats.
"""

import re
from collections import Counter
from typing import Any

import pandas as pd

from ..core.config import PLACEHOLDER_TOKENS


class FormatPatternAnalyzer:
    """Analyse column values to detect format patterns and anomalies."""

    # Known format patterns checked against every column sample.
    _BASE_PATTERNS: dict[str, str] = {
        "phone_intl": r"^\+\d{1,3}[-.\\s]?\d{1,14}$",
        "email":      r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$",
        "date_iso":   r"^\d{4}-\d{2}-\d{2}$",
        "currency":   r"^\$\d{1,3}(,\d{3})*(\.\d{2})?$",
        "url":        r"^https?://[^\s]+$",
        "uuid":       r"^[{]?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}[}]?$",
    }
    _SG_PATTERNS: dict[str, str] = {
        "phone_sg":      r"^[689]\d{7}$",
        "nric_sg":       r"^[STFGM]\d{7}[A-Z]$",
        "postal_code_sg": r"^\d{6}$",
    }

    def __init__(self, config) -> None:
        self.config = config
        self.patterns = dict(self._BASE_PATTERNS)
        if getattr(config, "enable_country_specific_patterns", False):
            self.patterns.update(self._SG_PATTERNS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_format_distribution(
        self,
        series: pd.Series,
        sample_size: int = 1000,
    ) -> dict[str, Any]:
        """Analyse format patterns in a column and return a structured report."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return {"status": "empty"}

        sample = non_null.sample(min(sample_size, len(non_null)), random_state=42)
        sample_str = sample.astype(str)

        known_patterns = self._detect_known_patterns(sample_str, len(sample))
        fingerprints = self._extract_fingerprints(sample_str)
        anomalies = self._find_anomalies(non_null, sample_str, fingerprints)
        coercibility = self._assess_coercibility(fingerprints)
        uniformity = self._calculate_uniformity(fingerprints)

        return {
            "total_values": len(non_null),
            "sample_size": len(sample),
            "known_patterns": known_patterns,
            "format_fingerprints": fingerprints,
            "anomalies": anomalies,
            "coercibility": coercibility,
            "uniformity_score": uniformity,
        }

    def generate_llm_context(
        self,
        column_name: str,
        series: pd.Series,
        analysis: dict | None = None,
    ) -> str:
        """Build a rich context string for the LLM evidence prompt.

        Pass a pre-computed *analysis* dict to avoid running the analysis twice.
        """
        if analysis is None:
            analysis = self.analyze_format_distribution(series)

        parts = [
            f"Column: {column_name}",
            f"Total values: {analysis['total_values']:,}",
        ]

        top_formats = analysis.get("format_fingerprints", {}).get("top_formats", [])
        meaningful = [f for f in top_formats[:5] if not self._is_length_variation_only(f["pattern"])]

        coercion = analysis.get("coercibility", {})
        non_coercible = coercion.get("non_coercible_formats", [])
        coercible_fmts = coercion.get("coercible_formats", [])
        all_length_only = (
            all(
                self._is_length_variation_only(f.get("pattern", ""))
                for f in (non_coercible + coercible_fmts)
            )
            if (non_coercible or coercible_fmts)
            else True
        )
        has_real_format_issue = (
            (coercion.get("is_coercible") or bool(non_coercible))
            and not all_length_only
        )

        # Suppress if a known pattern already explains most of the data.
        known_patterns = analysis.get("known_patterns", {})
        if sum(info["percentage"] for info in known_patterns.values()) >= 80:
            has_real_format_issue = False

        if has_real_format_issue and meaningful:
            parts.append("\nFormat Distribution:")
            for fmt in meaningful[:3]:
                parts.append(
                    f"  - {fmt['percentage']:.1f}%: {fmt['pattern']}"
                    f" (e.g., {fmt['examples'][0]})"
                )
            parts.append(f"\nFormat Uniformity: {analysis.get('uniformity_score', 0):.1%}")

        if known_patterns:
            parts.append("\nDetected Patterns:")
            for name, info in known_patterns.items():
                parts.append(f"  - {name}: {info['percentage']:.1f}%")

        anomalies = analysis.get("anomalies", {})
        if anomalies.get("total_anomaly_count", 0) > 0:
            parts.append(f"\nAnomalies Found: {anomalies['total_anomaly_count']}")
            if anomalies.get("placeholder_values"):
                parts.append(f"  - Placeholder values: {anomalies['placeholder_values'][:3]}")

        recommendation = coercion.get("recommendation", "")
        if (
            has_real_format_issue
            and recommendation
            and not all_length_only
            and recommendation != "No action needed - format already uniform"
        ):
            parts.append(f"\nFormat Analysis: {recommendation}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_known_patterns(
        self, sample_str: pd.Series, total: int
    ) -> dict[str, dict]:
        result = {}
        for name, regex in self.patterns.items():
            matches = sample_str.str.match(regex, case=False)
            count = int(matches.sum())
            if count > 0:
                result[name] = {
                    "count": count,
                    "percentage": round(count / total * 100, 1),
                    "examples": sample_str[matches].head(3).tolist(),
                }
        return result

    @staticmethod
    def _to_fingerprint(val: str) -> str:
        """Map each character to X (digit), a (alpha), or itself (other)."""
        return "".join(
            "X" if c.isdigit() else ("a" if c.isalpha() else c)
            for c in str(val)
        )

    def _extract_fingerprints(self, series: pd.Series) -> dict[str, Any]:
        fps = series.apply(self._to_fingerprint)
        counts = Counter(fps)
        top = []
        for pattern, count in counts.most_common(10):
            top.append({
                "pattern": pattern,
                "count": count,
                "percentage": round(count / len(series) * 100, 1),
                "examples": series[fps == pattern].head(3).tolist(),
            })
        return {
            "total_unique_formats": len(counts),
            "top_formats": top,
            "dominant_format": top[0] if top else None,
        }

    def _find_anomalies(
        self,
        series: pd.Series,
        series_str: pd.Series,
        format_info: dict,
    ) -> dict[str, Any]:
        anomalies: dict[str, list] = {
            "placeholder_values": [],
            "suspicious_values": [],
        }

        for val in series_str:
            if str(val).lower().strip() in PLACEHOLDER_TOKENS:
                anomalies["placeholder_values"].append(val)

        if pd.api.types.is_numeric_dtype(series):
            anomalies["suspicious_values"] = self._numeric_outliers(series)
        else:
            for val in series_str.head(100):
                if any(
                    re.match(p, str(val).lower())
                    for p in self.config.suspicious_string_patterns
                ):
                    anomalies["suspicious_values"].append(val)

        for key in ("placeholder_values", "suspicious_values"):
            anomalies[key] = list({str(v) for v in anomalies[key]})[:10]

        anomalies["total_anomaly_count"] = sum(
            len(anomalies[k]) for k in ("placeholder_values", "suspicious_values")
        )

        numeric_full = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric_full) > 0:
            anomalies["col_min"] = float(numeric_full.min())
            anomalies["col_max"] = float(numeric_full.max())

        return anomalies

    def _numeric_outliers(self, series: pd.Series) -> list:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) == 0:
            return []
        skewness = numeric.skew()
        mult = self.config.outlier_tail_multiplier
        if abs(skewness) > 1 or len(numeric) < 100:
            p99, p01 = numeric.quantile(0.999), numeric.quantile(0.001)
            iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
            upper, lower = p99 + mult * iqr, p01 - mult * iqr
            outliers = series[
                pd.to_numeric(series, errors="coerce").gt(upper)
                | pd.to_numeric(series, errors="coerce").lt(lower)
            ]
        else:
            mean, std = numeric.mean(), numeric.std()
            if std > 0:
                z = (pd.to_numeric(series, errors="coerce") - mean) / std
                outliers = series[z.abs() > self.config.outlier_z_score_threshold]
            else:
                outliers = series.iloc[0:0]
        return outliers.unique().tolist()[:10]

    def _assess_coercibility(self, format_info: dict) -> dict[str, Any]:
        """Determine whether multiple formats represent the same semantic content."""
        top = format_info["top_formats"]
        if not top or len(top) < 2:
            return {
                "is_coercible": False,
                "reason": "Single format already" if top else "No data",
            }

        dominant = top[0]
        dominant_pattern = dominant["pattern"]

        # Floating-point precision variation — not a real format issue.
        if all(re.sub(r"[X.]", "", f["pattern"]) == "" for f in top):
            return {"is_coercible": False, "reason": "Numeric precision variation only — not a format issue"}

        # Integer length variation — not a real format issue.
        if all(re.sub(r"X", "", f["pattern"]) == "" for f in top):
            return {"is_coercible": False, "reason": "Integer length variation only — not a format issue"}

        # Text structure variation — not a real format issue.
        if all(re.sub(r"[a\s.'\-\/]", "", f["pattern"]) == "" for f in top):
            return {"is_coercible": False, "reason": "Text length/word-count variation only — not a format issue"}

        coercible, non_coercible = [], []
        dom_stripped = re.sub(r"[^Xa]", "", dominant_pattern)

        for fmt in top[1:]:
            fmt_stripped = re.sub(r"[^Xa]", "", fmt["pattern"])
            # Skip pure integer length variation
            if re.sub(r"X", "", dom_stripped) == "" and re.sub(r"X", "", fmt_stripped) == "":
                continue
            if dom_stripped == fmt_stripped:
                coercible.append(fmt)
            else:
                non_coercible.append(fmt)

        if not coercible and dominant["percentage"] < 50:
            return {"is_coercible": False}

        return {
            "is_coercible": bool(coercible),
            "coercible_percentage": round(sum(f["percentage"] for f in coercible), 1),
            "coercible_formats": coercible,
            "non_coercible_formats": non_coercible,
            "target_format": dominant_pattern,
            "recommendation": self._coercion_recommendation(dominant, coercible, non_coercible),
        }

    @staticmethod
    def _coercion_recommendation(dominant: dict, coercible: list, non_coercible: list) -> str:
        if not coercible and not non_coercible:
            return "No action needed - format already uniform"
        parts = []
        if coercible:
            pct = sum(f["percentage"] for f in coercible)
            parts.append(f"Standardize {pct:.1f}% of values to {dominant['pattern']} format (coercible)")
        if non_coercible:
            pct = sum(f["percentage"] for f in non_coercible)
            parts.append(
                f"Clarify with data owner: {pct:.1f}% use incompatible format "
                f"{non_coercible[0]['pattern']}"
            )
        return " | ".join(parts)

    @staticmethod
    def _calculate_uniformity(format_info: dict) -> float:
        """Return 1.0 when all values share one format, 0.0 when every value differs."""
        top = format_info.get("top_formats", [])
        return round(top[0]["percentage"] / 100, 3) if top else 0.0

    @staticmethod
    def _is_length_variation_only(pattern: str) -> bool:
        """True when a fingerprint pattern only varies by text length, not structure."""
        stripped = re.sub(r"[\s.'\\-\\/@]", "", pattern)
        if not stripped:
            return False
        return all(c == "a" for c in stripped) or all(c == "X" for c in stripped)