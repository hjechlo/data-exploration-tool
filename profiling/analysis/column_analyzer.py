"""Coordinate per-column profiling and summary construction.

Detailed error detection, evidence construction, and MinHash sketch creation live
in focused sibling modules.
"""

import json
from pathlib import Path

import pandas as pd

from ..core.config import EMAIL_REGEX, PLACEHOLDER_TOKENS, PipelineConfig
from .column_errors import detect_column_errors
from .column_evidence import build_column_facts, extract_sample_values
from .format_pattern_analyzer import FormatPatternAnalyzer
from .minhash_sketches import (
    classify_column_for_similarity,
    compute_minhash,
    compute_minhash_shingle,
    should_compute_minhash,
)


class ColumnAnalyzer:
    """Build complete column summaries for one dataset."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.format_analyzer = FormatPatternAnalyzer(config)

    def get_storage_type(self, series: pd.Series, profile_type: str | None = None) -> str:
        """Infer the most accurate storage type from the actual dtype."""
        dtype = series.dtype

        if pd.api.types.is_integer_dtype(dtype):
            return str(dtype)
        if pd.api.types.is_float_dtype(dtype):
            return str(dtype)
        if pd.api.types.is_bool_dtype(dtype):
            return "bool"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return str(dtype)
        if isinstance(dtype, pd.CategoricalDtype):
            return "category"
        if pd.api.types.is_object_dtype(dtype):
            if series.isna().all():
                return "object"
            # Use full series as denominator so NaN is counted (not silently excluded)
            numeric_try = pd.to_numeric(series, errors="coerce")
            if numeric_try.notna().sum() / len(series) >= 0.95:
                whole = ((numeric_try.dropna() % 1) == 0).all()
                return "int64" if whole else "float64"
            non_null_mask = series.notna()
            if non_null_mask.any() and series[non_null_mask].map(lambda x: isinstance(x, str)).mean() >= 0.95:
                return "string"
            return "object"


        return str(dtype)

    def build_summary(
        self,
        profile_json_path: str | Path,
        raw_df: pd.DataFrame,
        df: pd.DataFrame,
        table_name: str,
    ) -> list[dict]:
        """
        Build the complete column-level summary list for one dataset.

        Each dict contains profile stats, error list, column facts, similarity
        kind, and (when applicable) MinHash sketches stored under private keys
        '_minhash' and '_minhash_shingle'.
        """
        with open(profile_json_path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)

        variables = profile_data.get("variables", {})
        rows: list[dict] = []
        max_perm = self.config.max_permissible_values
        max_s = self.config.max_sample_values

        for col_name, v in variables.items():
            profile_type = v.get("type", "Unknown")
            raw_series = raw_df[col_name] if col_name in raw_df.columns else df[col_name]
            storage_type = self.get_storage_type(raw_series, profile_type)
            # Derive intended type from actual data characteristics
            intended_type = storage_type  # default: actual = intended

            if storage_type in ("object", "string"):
                numeric_ratio = pd.to_numeric(raw_series, errors="coerce").notna().sum() / len(raw_series)

                if numeric_ratio >= 0.9:
                    intended_type = "float64"
                else:
                    # Check for boolean-like columns before datetime
                    common_bool_spellings = {
                        "true", "false", "yes", "no", "y", "n", "t", "f", "1", "0", "1.0", "0.0"
                    }
                    non_null_lower = raw_series.dropna().astype(str).str.strip().str.lower()
                    unique_bool_check = set(non_null_lower.unique())
                    if len(unique_bool_check) == 2 and unique_bool_check.issubset(common_bool_spellings):
                        intended_type = "bool"

                    else:
                        # Try parsing as datetime from the DATA itself, not the column name
                        date_ratio = pd.to_datetime(
                            raw_series, errors="coerce", format="mixed"
                        ).notna().sum() / len(raw_series)

                        if date_ratio >= 0.9:
                            intended_type = "datetime64[ns]"
            elif storage_type in ("float64", "int64", "Int64", "float32", "int32"):
                raw_vals = raw_series.dropna().astype(str).str.strip()
                sample_vals = raw_vals.str.replace(r"\.0+$", "", regex=True)

                if len(sample_vals) > 0:
                    digit_ratio = sample_vals.str.fullmatch(r"\d+", na=False).mean()
                    lengths = sample_vals.str.len()
                    dominant_len_pct = lengths.value_counts(normalize=True).iloc[0]
                    dominant_len = lengths.value_counts().index[0]
                    unique_ratio = sample_vals.nunique() / len(sample_vals)

                    fixed_width_code_like = (
                        digit_ratio >= 0.98
                        and dominant_len >= 4
                        and dominant_len_pct >= 0.95
                        and unique_ratio >= 0.5
                    )

                    if sample_vals.str.fullmatch(r"(19|20)\d{6}", na=False).mean() >= 0.8:
                        intended_type = "datetime64[ns]"

                    elif sample_vals.str.fullmatch(r"(19|20)\d{12}", na=False).mean() >= 0.8:
                        intended_type = "datetime64[ns]"

                    elif fixed_width_code_like:
                        intended_type = "string"

                    elif storage_type == "float64":
                        decimal_mask = raw_vals.str.fullmatch(r"\d+\.0+", na=False)
                        if decimal_mask.mean() >= 0.95:
                            intended_type = "int64"

            # Valid (non-placeholder, non-null) string values
            s = raw_series.dropna().astype(str).str.strip()
            s = s[(s != "") & (~s.str.lower().isin(PLACEHOLDER_TOKENS))]

            # Build invalid value set for sample filtering
            invalid_values: set[str] = set()
            if "email" in col_name.lower():
                invalid_values.update(x for x in s.unique() if not EMAIL_REGEX.match(x))
            if "datetime" in storage_type:
                parsed = pd.to_datetime(s, errors="coerce",format="mixed")
                invalid_values.update(s[parsed.isna()].unique().tolist())
            if pd.api.types.is_numeric_dtype(raw_series.dtype):
                converted = pd.to_numeric(s, errors="coerce")
                invalid_values.update(s[converted.isna()].unique().tolist())

            # Sample values
            raw_samples = extract_sample_values(raw_series, self.config)
            sample_values = [x for x in raw_samples if x not in invalid_values][:max_s]
            if storage_type == "int64":
                sample_values = [
                    str(int(float(x)))
                    if str(x).replace(".", "", 1).isdigit() and float(x).is_integer()
                    else x
                    for x in sample_values
                ]

            # Permissible values
            raw_non_null = raw_series.dropna().astype(str).str.strip()
            raw_non_null = raw_non_null[
                (raw_non_null != "") & (~raw_non_null.str.lower().isin(PLACEHOLDER_TOKENS))
            ]
            permissible_values: list[str] | None = None
            if storage_type in ("category", "string", "object") and raw_non_null.nunique() <= max_perm:
                permissible_values = sorted(raw_non_null.unique().tolist())
            elif storage_type == "int64" and raw_non_null.nunique() <= 5:
                permissible_values = sorted(raw_non_null.unique().tolist())

            # Missing stats
            s_as_str = raw_series.astype(str).str.strip()
            missing_mask = raw_series.isna() | s_as_str.str.lower().isin(PLACEHOLDER_TOKENS)
            profile_stats = {
                "missing_count": int(missing_mask.sum()),
                "missing_pct": float(missing_mask.mean()),
                "n_total": len(raw_series),
                # Use raw_non_null (already computed above, excludes nulls + placeholders)
                "n_distinct": int(raw_non_null.nunique()),
            }
            if pd.api.types.is_numeric_dtype(raw_series):
                non_null_vals = raw_series.dropna()
                if len(non_null_vals) > 0:
                    profile_stats["min"] = float(non_null_vals.min())
                    profile_stats["max"] = float(non_null_vals.max())

            similarity_kind = classify_column_for_similarity(raw_series, storage_type, col_name)

            #Analyze format patterns
            format_analysis = self.format_analyzer.analyze_format_distribution(raw_series)
            if similarity_kind in ("key_like", "categorical") and "int" in storage_type.lower():
                _anoms = format_analysis.get("anomalies", {})
                _anoms["suspicious_values"] = []
                #_anoms["outlier_values"] = []

            errors, flagged_idx = detect_column_errors(
                self.config, col_name, raw_series, storage_type,
                intended_type=intended_type,
                format_analysis=format_analysis,
            )
            column_facts = build_column_facts(
                self.config, self.format_analyzer, col_name, raw_series, storage_type, permissible_values,
                format_analysis=format_analysis,
                intended_type=intended_type,
            )
            # similarity_kind = classify_column_for_similarity(
            #     raw_series, storage_type, col_name
            # )

            # Exact MinHash
            minhash_sketch = (
                compute_minhash(raw_series, self.config)
                if should_compute_minhash(raw_series, storage_type, col_name, self.config)
                else None
            )

            # Shingled MinHash (non-ID string columns only)
            minhash_shingle_sketch = (
                compute_minhash_shingle(raw_series, self.config)
                if similarity_kind in {"categorical", "discrete_code"}
                else None
            )

            row: dict = {
                "table_name": table_name,
                "column_name": col_name,
                "data_type": storage_type,
                "intended_data_type": intended_type,
                "profile_type": profile_type,
                "sample_values": sample_values,
                "permissible_values": permissible_values,
                "profile": profile_stats,
                "errors": errors,
                "column_facts": column_facts,
                "similarity_kind": similarity_kind,
                "_format_analysis": format_analysis,
                "_flagged_indices": flagged_idx,
            }

            if minhash_sketch is not None:
                row["_minhash"] = minhash_sketch
            if minhash_shingle_sketch is not None:
                row["_minhash_shingle"] = minhash_shingle_sketch

            rows.append(row)

        return rows


def build_column_summaries(
    analyzer: ColumnAnalyzer,
    profile_results: dict,
) -> dict[str, list[dict]]:
    """Build column-level summaries for every profiled dataset."""
    column_summaries = {}
    for name, result in profile_results.items():
        print(f"  Analysing columns for {name}...")
        summary = analyzer.build_summary(
            profile_json_path=result["summary"]["json_report"],
            raw_df=result["raw_df"],
            df=result["df"],
            table_name=name,
        )
        column_summaries[name] = summary
        cols_with_minhash = sum(1 for row in summary if "_minhash" in row)
        print(
            f"    {len(summary)} columns, "
            f"{cols_with_minhash} with MinHash sketches"
        )
    return column_summaries