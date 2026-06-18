"""Build human-readable column evidence for LLM prompts.

The functions here were moved from ColumnAnalyzer without changing their logic.
"""

import re

import pandas as pd

from ..core.config import PipelineConfig
from .format_pattern_analyzer import FormatPatternAnalyzer

def build_column_facts(
    config: PipelineConfig,
    format_analyzer: FormatPatternAnalyzer,
    col_name: str,
    raw_series: pd.Series,
    storage_type: str,
    permissible_values: list[str] | None = None,
    format_analysis: dict | None = None,
    intended_type: str | None = None,
) -> list[str]:
    """Build a list of natural-language fact sentences for the LLM prompt."""
    facts: list[str] = []
    non_null = raw_series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if len(non_null) > 10:
        if format_analysis is None:
            format_analysis = format_analyzer.analyze_format_distribution(raw_series)
        format_context = format_analyzer.generate_llm_context(col_name, raw_series, analysis=format_analysis)
        facts.append(format_context)
        # Extra evidence for stable identifier formats with alphabetic prefix/suffix variants.
        # Example: XXXXXX and aXXXXXX, such as 536365 and C536379.
        # This is universal: it does not hardcode InvoiceNo or cancellation.
        top_formats = (
            format_analysis
            .get("format_fingerprints", {})
            .get("top_formats", [])
        )

        if len(top_formats) >= 2:
            dominant = top_formats[0]
            dominant_pattern = dominant.get("pattern", "")

            if re.fullmatch(r"X{3,}", dominant_pattern):
                n_digits = len(dominant_pattern)
                variant_notes = []
                prefix_suffix_examples = []

                for fmt in top_formats[1:6]:
                    pattern = fmt.get("pattern", "")
                    pct = float(fmt.get("percentage", 0) or 0)
                    examples = [str(x) for x in fmt.get("examples", [])[:3]]

                    if pct < 0.5 or not examples:
                        continue

                    if re.fullmatch(rf"a+X{{{n_digits}}}", pattern):
                        n_letters = len(pattern) - n_digits
                        variant_notes.append(
                            f"{pct:.1f}% use {n_letters} letter(s) followed by {n_digits} digits "
                            f"(e.g. '{examples[0]}')"
                        )
                        prefix_suffix_examples.extend(examples)

                    elif re.fullmatch(rf"X{{{n_digits}}}a+", pattern):
                        n_letters = len(pattern) - n_digits
                        variant_notes.append(
                            f"{pct:.1f}% use {n_digits} digits followed by {n_letters} letter(s) "
                            f"(e.g. '{examples[0]}')"
                        )
                        prefix_suffix_examples.extend(examples)

                if variant_notes:
                    dom_examples = dominant.get("examples", [])
                    dom_example = f" (e.g. '{dom_examples[0]}')" if dom_examples else ""

                    facts.append(
                        f"Observed stable identifier formats: "
                        f"{dominant.get('percentage', 0):.1f}% use {n_digits} digits{dom_example}; "
                        + "; ".join(variant_notes)
                        + ". Alphabetic prefixes or suffixes may encode a status, category, subtype, "
                            "or business event, so their meaning should be confirmed before stripping or standardising them."
                    )
    n_total = len(raw_series)
    n_missing = int(raw_series.isna().sum())
    n_distinct = int(non_null.nunique())

    facts.append(f"Column {col_name} has {n_missing} NULL values out of {n_total} records.")
    facts.append(f"There are {n_distinct} distinct non-null values.")
    if n_distinct == 1 and len(non_null) > 1:
        facts.append(
            "All non-null values are identical — this column has zero analytical variance "
            "and may be a candidate for removal unless it serves a mandatory metadata purpose."
        )
    facts.append(f"The technical data type is {storage_type}.")
    if intended_type and intended_type != storage_type:
        facts.append(
            f"Column is stored as {storage_type} but the intended type is "
            f"{intended_type} — a type conversion is recommended."
        )

    if not non_null.empty:
        min_val: str | None = None
        max_val: str | None = None
        if "int" in storage_type.lower() or "float" in storage_type.lower():
            numeric_vals = pd.to_numeric(non_null, errors="coerce").dropna()
            if not numeric_vals.empty:
                min_val = (
                    str(int(numeric_vals.min()))
                    if storage_type == "int64"
                    else str(numeric_vals.min())
                )
                max_val = (
                    str(int(numeric_vals.max()))
                    if storage_type == "int64"
                    else str(numeric_vals.max())
                )
        else:
            min_val = str(non_null.min())
            max_val = str(non_null.max())
        if min_val is not None and max_val is not None:
            facts.append(f"The minimum value is '{min_val}' and the maximum value is '{max_val}'.")

        if storage_type in ("object", "string", "category"):
            samples = non_null.drop_duplicates().head(5).tolist()
            if samples:
                facts.append(
                    "Example observed values include "
                    + ", ".join(f"'{x}'" for x in samples) + "."
                )

        top_vals = non_null.value_counts().head(20).index.tolist()
        if top_vals:
            facts.append(
                "Most common non-NULL column values are "
                + ", ".join(f"'{x}'" for x in top_vals) + "."
            )

        _col_norm = col_name.lower().replace("_", "").replace(" ", "")
        _is_personal_name = any(
            token in _col_norm
            for token in ("firstname", "lastname", "forename", "givenname",
                          "fullname", "middlename", "surname")
        )

        if n_distinct == len(non_null):
            facts.append("All non-NULL values are unique.")
        else:
            repeated_record_count = len(non_null) - n_distinct

            if _is_personal_name:
                facts.append(
                    f"Not all non-NULL values are unique: {len(non_null)} non-null records "
                    f"contain {n_distinct} distinct values. Repeated names are expected for "
                    f"a personal name field and are not a data quality issue."
                )
            elif repeated_record_count <= max(3, int(len(non_null) * 0.01)):
                dup_vals = non_null[non_null.duplicated(keep=False)].unique().tolist()[:5]
                facts.append(
                    f"Near-unique column with {repeated_record_count} repeated non-null "
                    f"record(s); examples of repeated values: {dup_vals}."
                )
            else:
                facts.append(
                    f"Not all non-NULL values are unique: {len(non_null)} non-null records "
                    f"contain {n_distinct} distinct values."
                )

        lengths = non_null.str.len()
        unique_lengths = sorted(lengths.unique().tolist())
        if len(unique_lengths) == 1:
            facts.append(f"The values are always {unique_lengths[0]} characters long.")
        elif len(unique_lengths) <= 5:
            facts.append(f"Observed value lengths are {unique_lengths}.")

        numeric_parsed = pd.to_numeric(non_null, errors="coerce")
        numeric_ratio = numeric_parsed.notna().mean()

        if numeric_ratio == 1.0:
            facts.append("Every column value looks like a number.")
        elif numeric_ratio > 0.95:
            bad_values = non_null[numeric_parsed.isna()].drop_duplicates().tolist()[:5]
            bad_count = int(numeric_parsed.isna().sum())
            facts.append(
                f"Most values look numeric, but {bad_count} value(s) cannot be parsed as numbers: {bad_values}."
            )

    if permissible_values:
        facts.append(
            "Observed distinct low-cardinality values include "
            + ", ".join(f"'{x}'" for x in permissible_values[:10]) + ". "
            "These are observed values only, not an official approved reference list."
        )

    return facts

def extract_sample_values(series: pd.Series, config: PipelineConfig) -> list[str]:
    max_s = config.max_sample_values
    non_null = series.dropna()
    if len(non_null) == 0:
        return []

    # Random sample to avoid always picking values from the top of the file
    sample_size = min(len(non_null), max_s * 10)
    sampled = non_null.sample(sample_size, random_state=42)

    seen: set[str] = set()
    samples: list[str] = []
    for v in sampled:
        s = str(v).strip()
        if s and s not in seen:
            seen.add(s)
            samples.append(s)
        if len(samples) >= max_s * 3:
            break
    return samples
