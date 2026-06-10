"""
DataPreprocessor — light, low-risk cleaning to keep data close to raw form.
"""

import pandas as pd

from .config import PipelineConfig


class DataPreprocessor:
    """Applies standardised column names, whitespace stripping, and dtype casting."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def basic_preclean(self, df: pd.DataFrame, normalize_case: bool = False) -> pd.DataFrame:
        """Standardise column names, remove exact duplicate names, strip whitespace."""
        df = df.copy()

        # Standardise column names
        df.columns = [
            str(col).strip().replace(" ", "_").replace("/", "_").replace("-", "_")
            for col in df.columns
        ]

        # Deduplicate column names
        seen: dict[str, int] = {}
        new_cols = []
        for col in df.columns:
            if col not in seen:
                seen[col] = 0
                new_cols.append(col)
            else:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
        df.columns = new_cols

        # Strip whitespace in string-like columns
        for col in df.select_dtypes(include=["object", "string"]).columns:
            try:
                df[col] = df[col].astype("string").str.strip()
            except Exception:
                pass

        if normalize_case:
            for col in df.select_dtypes(include=["object", "string"]).columns:
                try:
                    n = len(df[col].dropna())
                    if n > 0 and df[col].nunique(dropna=True) <= 50 and (df[col].nunique(dropna=True) / n) < 0.5:
                        df[col] = df[col].str.upper()
                except Exception:
                    pass

        return df

    def auto_cast_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast numeric-looking and low-cardinality columns to appropriate types."""
        df = df.copy()
        threshold = self.config.categorical_threshold

        for col in df.columns:
            series = df[col]

            if pd.api.types.is_datetime64_ns_dtype(series):
                continue

            if pd.api.types.is_numeric_dtype(series):
                try:
                    numeric = pd.to_numeric(series, errors="raise")
                    # Preserve integer semantics: avoid float64 artefacts on int columns
                    if pd.api.types.is_integer_dtype(series.dtype) or (
                        numeric.dropna() % 1 == 0
                    ).all():
                        df[col] = numeric.astype("Int64")   # nullable integer
                    else:
                        df[col] = numeric.astype("float64")
                except (ValueError, TypeError) as e:
                    bad_vals = (
                        series.dropna()
                        .loc[pd.to_numeric(series.dropna(), errors="coerce").isna()]
                        .unique()
                        .tolist()[:5]
                    )
                    print(f"    ⚠️  {col}: could not cast to numeric — bad values: {bad_vals} ({e})")
                continue

            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null = series.dropna()
                if len(non_null) == 0:
                    continue

                stripped = non_null.astype(str).str.strip()
                numeric_try = pd.to_numeric(stripped, errors="coerce")
                if numeric_try.notna().mean() >= 1.0:
                    try:
                        df[col] = pd.to_numeric(
                            series.astype(str).str.strip(), errors="raise"
                        ).astype("float64")
                    except (ValueError, TypeError) as e:
                        bad_vals = (
                            stripped.loc[pd.to_numeric(stripped, errors="coerce").isna()]
                            .unique()
                            .tolist()[:5]
                        )
                        print(f"    ⚠️  {col}: looked numeric but could not cast — bad values: {bad_vals} ({e})")
                    continue

                if non_null.nunique() <= threshold:
                    try:
                        df[col] = series.astype("string").str.strip().astype("category")
                    except Exception as e:
                        print(f"    ⚠️  {col}: could not cast to category — {e}")
                        # Leave as string — don't drop or modify data

        return df
    
    def _sanitize_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert out-of-range datetime columns to strings so ydata_profiling doesn't crash."""
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col].min().to_pydatetime()
                    df[col].max().to_pydatetime()
                except (OverflowError, ValueError):
                    print(f"    ⚠️  {col}: out-of-range timestamps, converting to string")
                    df[col] = df[col].astype(str)
        return df

    def infer_type_schema(self, df: pd.DataFrame) -> dict:
        """Build a type_schema dict for ydata_profiling."""
        schema = {}
        for col in df.columns:
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            lowered = set(non_null.astype(str).str.strip().str.lower().unique())
            if lowered.issubset({"yes", "no", "true", "false"}):
                schema[col] = "categorical"
            elif lowered.issubset({"0", "1"}):
                schema[col] = "categorical"
            elif "date" in col.lower() or "time" in col.lower():
                # Validate values are actually parseable and in range before marking as datetime
                try:
                    parsed = pd.to_datetime(non_null.astype(str), errors="coerce",format="mixed")
                    valid = parsed.dropna()
                    if len(valid) > 0:
                        valid.min().to_pydatetime()  # Triggers OverflowError if out of range
                        valid.max().to_pydatetime()
                        schema[col] = "datetime"
                except (OverflowError, ValueError):
                    pass  # Leave as string - values are out of Python's datetime range
        return schema

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps in order."""
        df = self.basic_preclean(df,normalize_case=True)
        df = self.auto_cast_dtypes(df)
        df = self._sanitize_date_columns(df) 
        return df
