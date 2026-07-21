"""DataPreprocessor — light, low-risk cleaning to keep data close to raw form."""

import json

import pandas as pd

from ..core.config import PipelineConfig


class DataPreprocessor:
    """Apply standardised column names, whitespace stripping, and dtype casting."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps in order."""
        df = self.basic_preclean(df, normalize_case=True)
        df = self.auto_cast_dtypes(df)
        df = self._sanitize_date_columns(df)
        return df

    def basic_preclean(self, df: pd.DataFrame, normalize_case: bool = False) -> pd.DataFrame:
        """Standardise column names, deduplicate them, and strip whitespace."""
        df = df.copy()
        df.columns = [
            str(col).strip().replace(" ", "_").replace("/", "_").replace("-", "_")
            for col in df.columns
        ]
        df.columns = self._deduplicate_columns(df.columns.tolist())
        df = self._flatten_nested_columns(df)
        df = self._strip_string_whitespace(df)
        if normalize_case:
            df = self._normalize_low_cardinality_case(df)
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
                df[col] = self._cast_numeric(col, series)
                continue

            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                result = self._cast_object(col, series, threshold)
                if result is not None:
                    df[col] = result

        return df

    def infer_type_schema(self, df: pd.DataFrame) -> dict:
        """Build a type_schema dict for ydata_profiling."""
        schema = {}
        for col in df.columns:
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            lowered = set(non_null.astype(str).str.strip().str.lower().unique())
            if lowered.issubset({"yes", "no", "true", "false", "0", "1"}):
                schema[col] = "categorical"
            elif "date" in col.lower() or "time" in col.lower():
                schema[col] = self._try_datetime_schema(non_null)
        return {k: v for k, v in schema.items() if v is not None}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_columns(cols: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        result = []
        for col in cols:
            if col not in seen:
                seen[col] = 0
                result.append(col)
            else:
                seen[col] += 1
                result.append(f"{col}_{seen[col]}")
        return result

    @staticmethod
    def _flatten_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Stringify dict/list cells so downstream nunique/regex calls don't crash."""
        for col in df.columns:
            if df[col].dtype != "object":
                continue
            sample = df[col].dropna()
            if len(sample) > 0 and sample.apply(lambda x: isinstance(x, (dict, list))).any():
                df[col] = df[col].apply(
                    lambda x: json.dumps(x, sort_keys=True, default=str)
                    if isinstance(x, (dict, list)) else x
                )
        return df

    @staticmethod
    def _strip_string_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["object", "string"]).columns:
            try:
                df[col] = df[col].astype("string").str.strip()
            except Exception:
                pass
        return df

    @staticmethod
    def _normalize_low_cardinality_case(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["object", "string"]).columns:
            try:
                non_null = df[col].dropna()
                n = len(non_null)
                if n > 0 and df[col].nunique(dropna=True) <= 50 and (
                    df[col].nunique(dropna=True) / n
                ) < 0.5:
                    df[col] = df[col].str.upper()
            except Exception:
                pass
        return df

    @staticmethod
    def _cast_numeric(col: str, series: pd.Series) -> pd.Series:
        try:
            numeric = pd.to_numeric(series, errors="raise")
            if pd.api.types.is_integer_dtype(series.dtype) or (
                numeric.dropna() % 1 == 0
            ).all():
                return numeric.astype("Int64")
            return numeric.astype("float64")
        except (ValueError, TypeError) as exc:
            bad = (
                series.dropna()
                .loc[pd.to_numeric(series.dropna(), errors="coerce").isna()]
                .unique()
                .tolist()[:5]
            )
            print(f"    ⚠️  {col}: could not cast to numeric — bad values: {bad} ({exc})")
            return series

    @staticmethod
    def _cast_object(col: str, series: pd.Series, threshold: int) -> pd.Series | None:
        non_null = series.dropna()
        if len(non_null) == 0:
            return None

        stripped = non_null.astype(str).str.strip()
        if pd.to_numeric(stripped, errors="coerce").notna().mean() >= 1.0:
            try:
                return pd.to_numeric(
                    series.astype(str).str.strip(), errors="raise"
                ).astype("float64")
            except (ValueError, TypeError) as exc:
                bad = (
                    stripped.loc[pd.to_numeric(stripped, errors="coerce").isna()]
                    .unique()
                    .tolist()[:5]
                )
                print(f"    ⚠️  {col}: looked numeric but could not cast — bad values: {bad} ({exc})")
                return None

        if non_null.nunique() <= threshold:
            try:
                return series.astype("string").str.strip().astype("category")
            except Exception as exc:
                print(f"    ⚠️  {col}: could not cast to category — {exc}")

        return None

    def _sanitize_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert out-of-range datetime columns to strings to prevent ydata crashes."""
        df = df.copy()
        for col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            try:
                df[col].min().to_pydatetime()
                df[col].max().to_pydatetime()
            except (OverflowError, ValueError):
                print(f"    ⚠️  {col}: out-of-range timestamps, converting to string")
                df[col] = df[col].astype(str)
        return df

    @staticmethod
    def _try_datetime_schema(non_null: pd.Series) -> str | None:
        try:
            parsed = pd.to_datetime(non_null.astype(str), errors="coerce", format="mixed")
            valid = parsed.dropna()
            if len(valid) > 0:
                valid.min().to_pydatetime()
                valid.max().to_pydatetime()
                return "datetime"
        except (OverflowError, ValueError):
            pass
        return None
