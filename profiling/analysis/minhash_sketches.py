"""MinHash sketch construction for individual columns.

This module prepares column sketches; cross-column relationship discovery remains
in minhash_analyzer.py.
"""

import json
import re

import pandas as pd
from datasketch import MinHash

from ..core.config import ID_NAME_HINTS, PLACEHOLDER_TOKENS, PipelineConfig

def classify_column_for_similarity(
    series: pd.Series, storage_type: str, col_name: str = ""
) -> str:
    """Classify a column into a similarity kind used to gate MinHash computation."""
    non_null = series.dropna()
    n = len(non_null)

    if n == 0:
        return "empty"

    # Nested JSON columns (dicts/lists) are unhashable — stringify for
    # uniqueness counting only; doesn't mutate the original series.
    if non_null.apply(lambda x: isinstance(x, (dict, list))).any():
        non_null = non_null.apply(
            lambda x: json.dumps(x, sort_keys=True, default=str)
            if isinstance(x, (dict, list)) else x
        )

    unique_count = non_null.nunique(dropna=True)
    uniqueness_ratio = unique_count / n
    storage_type_str = str(storage_type).lower()
    _camel_split = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", col_name)
    name_tokens = set(re.split(r"[^a-z0-9]+", _camel_split.lower()))

    if "datetime" in storage_type_str:
        return "datetime"

    if storage_type in ("string", "category", "object"):
        values = non_null.astype(str).str.strip()
        values = values[values != ""]
        if len(values) == 0:
            return "empty"

        avg_len = values.str.len().mean()
        digit_ratio = values.str.fullmatch(r"\d+").mean() if len(values) else 0.0
        unique_count = values.nunique()
        uniqueness_ratio = unique_count / len(values)

        if avg_len > 80:
            return "free_text"
        if name_tokens & ID_NAME_HINTS:
            return "key_like"
        if unique_count <= 20 or uniqueness_ratio < 0.2:
            return "categorical"
        short_digit_like = (
            digit_ratio >= 0.8 and values.str.len().quantile(0.95) <= 12
        )
        if short_digit_like and uniqueness_ratio < 0.98:
            return "discrete_code"
        if uniqueness_ratio < 0.98:
            return "discrete_code"
        return "other"

    if storage_type in ("float64", "float32"):
        return "discrete_code" if unique_count <= 30 else "continuous_numeric"

    if storage_type in ("int64", "int32", "Int64"):
        if name_tokens & ID_NAME_HINTS:
            return "key_like"
        if unique_count <= 30 or uniqueness_ratio < 0.2:
            return "categorical"
        if uniqueness_ratio < 0.98:
            return "discrete_code"
        return "other"

    return "other"

def should_compute_minhash(
    series: pd.Series,
    storage_type: str,
    col_name: str,
    config: PipelineConfig,
) -> bool:
    kind = classify_column_for_similarity(series, storage_type, col_name)
    storage_l = str(storage_type).lower()

    if kind == "key_like":
        return True

    string_like = (
        "string" in storage_l or "object" in storage_l or "category" in storage_l
    )
    if string_like:
        return kind in {"categorical", "discrete_code", "key_like"}

    # NEW: also sketch numeric columns if they have low cardinality
    # This catches float duplicate columns like FeedbackScore / FeedbackScore1
    if "float" in storage_l or "int" in storage_l:
        non_null = series.dropna()
        n_distinct = non_null.nunique()
        return 0 < n_distinct <= config.near_dupe_max_values

    return False

def normalize_value(val: str) -> str:
    s = str(val).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_values(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).map(normalize_value)
    values = values[(values != "") & (~values.str.lower().isin(PLACEHOLDER_TOKENS))]
    return values.unique().tolist()

def compute_minhash(series: pd.Series, config: PipelineConfig) -> MinHash:
    """Exact value-token MinHash sketch for join-path detection."""
    m = MinHash(num_perm=config.minhash_num_perm)
    for val in clean_values(series):
        m.update(val.encode("utf8"))
    return m

def compute_minhash_shingle(series: pd.Series, config: PipelineConfig) -> MinHash:
    """
    Character k-gram shingled MinHash sketch.

    Measures vocabulary-level textual similarity — useful for detecting
    that two columns store the same concept even when values have typos
    or formatting noise (e.g. suburb names across two datasets).
    Not applied to ID/key-like columns where partial string overlap
    is meaningless.
    """
    k = config.minhash_shingle_k
    m = MinHash(num_perm=config.minhash_num_perm)
    for val in clean_values(series):
        if len(val) < k:
            m.update(val.encode("utf8"))
        else:
            for shingle in {val[i: i + k] for i in range(len(val) - k + 1)}:
                m.update(shingle.encode("utf8"))
    return m