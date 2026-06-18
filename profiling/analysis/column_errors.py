"""Column-level data-quality detection functions.

These functions were moved out of ColumnAnalyzer without changing their logic.
"""

import copy
import re

import pandas as pd
from datasketch import MinHash, MinHashLSH

from ..core.config import EMAIL_REGEX, PLACEHOLDER_TOKENS, PipelineConfig
from ..core.utils import email_local, is_sequential_ordinal

def detect_numeric_anomalies(
    series: pd.Series,
    col_name: str,
    config: "PipelineConfig",
    is_date_like: bool = False,
) -> list[str]:
    """
    Detect statistical outliers in a numeric column using IQR-based fencing.

    Skips: date-like columns, ID/key/code columns, columns with fewer than
    10 non-null values, and columns with zero IQR (constant distributions).
    Uses config.outlier_tail_multiplier as the fence multiplier, tightened
    automatically for small datasets (< 100 rows).
    """
    errors: list[str] = []
    flagged_idx: set[int] = set()

    _is_id_like = any(
        hint in col_name.lower().replace("_", "").replace(" ", "")
        for hint in ("id", "key", "code", "no", "num", "sn", "seq")
    )
    if is_date_like or _is_id_like:
        return errors, flagged_idx

    numeric_vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric_vals) < 10:
        return errors, flagged_idx

    q25 = numeric_vals.quantile(0.25)
    q75 = numeric_vals.quantile(0.75)
    iqr = q75 - q25
    if iqr == 0:
        return errors, flagged_idx

    multiplier = config.outlier_tail_multiplier
    if len(numeric_vals) < 100:
        multiplier = min(multiplier, 1.5)

    upper_fence = q75 + multiplier * iqr
    lower_fence = q25 - multiplier * iqr
    outliers_high = numeric_vals[numeric_vals > upper_fence]
    outliers_low = numeric_vals[numeric_vals < lower_fence]

    if len(outliers_high) > 0:
        errors.append(f"{len(outliers_high)} value(s) exceed the statistically expected "
                       f"upper bound of {upper_fence:.4g} (Q3 + {multiplier}×IQR): "
                       f"{sorted(outliers_high.unique().tolist())[:5]}")
        flagged_idx.update(outliers_high.index)
    if len(outliers_low) > 0:
        errors.append(f"{len(outliers_low)} value(s) fall below the statistically expected "
                       f"lower bound of {lower_fence:.4g} (Q1 - {multiplier}×IQR): "
                       f"{sorted(outliers_low.unique().tolist())[:5]}")
        flagged_idx.update(outliers_low.index)
    return errors, flagged_idx

def detect_near_duplicate_values(series: pd.Series, config: PipelineConfig) -> list[list[str]]:
    cfg = config
    values = series.dropna().astype(str).str.strip().unique().tolist()
    values = [v for v in values if v and v.lower() not in PLACEHOLDER_TOKENS]

    if not (cfg.near_dupe_min_values <= len(values) <= cfg.near_dupe_max_values):
        return []

    # Auto-select k based on average value length:
    # short strings (e.g. state codes: 'nsw', 'vic') need k=2 to get
    # meaningful shingle overlap; longer strings work better with k=3
    avg_len = sum(len(v) for v in values) / len(values)
    k = 2 if avg_len <= 4 else cfg.near_dupe_k

    lsh = MinHashLSH(threshold=cfg.near_dupe_threshold, num_perm=cfg.near_dupe_num_perm)
    minhashes: dict[str, MinHash] = {}

    for val in values:
        shingles = (
            {val} if len(val) < k
            else {val[i: i + k] for i in range(len(val) - k + 1)}
        )
        m = MinHash(num_perm=cfg.near_dupe_num_perm)
        for sh in shingles:
            m.update(sh.encode("utf8"))
        try:
            lsh.insert(val, m)
            minhashes[val] = m
        except ValueError:
            pass

    groups: list[list[str]] = []
    seen: set[str] = set()

    for val, m in minhashes.items():
        if val in seen:
            continue
        neighbours = [n for n in lsh.query(m) if n != val]
        if neighbours:
            group = [val] + neighbours
            groups.append(group)
            seen.update(group)

    return groups

def detect_column_errors(
    config: PipelineConfig, col_name: str, series: pd.Series, storage_type: str, intended_type: str, is_foreign_key: bool = False, format_analysis: dict | None = None
) -> list[str]:
    """Return a list of human-readable error/quality strings for a column.

    Args:
        col_name: Column name
        series: Pandas Series with column data
        storage_type: Data type (string, datetime64[ns], int64, etc.)
        intended_type: Inferred semantic type (categorical, datetime, key_like, etc.)
        is_foreign_key: If True, skip duplicate errors (duplicates are expected in FKs)
        format_analysis: Optional dict with results from FormatPatternAnalyzer, used to detect formatting inconsistencies
    """
    errors: list[str] = []
    flagged_idx: set = set()

    s = series.astype(str).str.strip()
    null_mask = series.isna()
    placeholder_mask = s.str.lower().isin(PLACEHOLDER_TOKENS)
    missing_mask = null_mask | placeholder_mask

    null_count = int(null_mask.sum())
    placeholder_count = int((placeholder_mask & ~null_mask).sum())
    total = len(series)

    if null_count > 0:
        errors.append(f"{null_count} NULL values ({null_count/total*100:.1f}% of {total} records)")
        flagged_idx.update(series.index[null_mask])

    if placeholder_count > 0:
        ph_mask = placeholder_mask & ~null_mask
        examples = s[ph_mask].unique().tolist()[:5]
        errors.append(
            f"{placeholder_count} placeholder/missing-sentinel values "
            f"({placeholder_count/total*100:.1f}%) e.g. {examples}"
        )
        flagged_idx.update(series.index[ph_mask])

    non_missing = series[~missing_mask]
    non_missing_str = non_missing.astype(str).str.strip()

    if len(non_missing_str) == 0:
        return errors, flagged_idx
    numeric_storage_types = {"int64", "int32", "Int64", "float64", "float32"}

    if storage_type in numeric_storage_types:
        numeric_converted = pd.to_numeric(non_missing_str, errors="coerce")
        non_numeric_mask = numeric_converted.isna()

        if non_numeric_mask.any():
            bad_values = (
                non_missing_str[non_numeric_mask]
                .drop_duplicates()
                .tolist()[:5]
            )
            bad_count = int(non_numeric_mask.sum())

            errors.append(
                f"{bad_count} non-numeric value(s) "
                f"({bad_count / total * 100:.1f}% of {total} records) found in numeric-like column; "
                f"examples: {bad_values}"
            )
            flagged_idx.update(non_missing_str.index[non_numeric_mask])

    # Email validation — intentionally non-strict.
    # Only flag structural failures (missing @, spaces).
    # Plus-tags, new TLDs, and subdomains are all valid.
    if "email" in col_name.lower():
        bad_mask = non_missing_str.apply(lambda v: bool(v) and not EMAIL_REGEX.match(v))
        structurally_bad = non_missing_str[bad_mask].unique().tolist()
        if structurally_bad:
            errors.append(
                f"structurally malformed email values (missing @ or contains spaces): "
                f"{structurally_bad[:5]}"
            )
            flagged_idx.update(non_missing_str.index[bad_mask])

        # Case inconsistency — deduplication risk.
        # Use lowercase, never uppercase: uppercasing is destructive for unicode
        # (e.g. ß→SS, Turkish İ folds differently across systems/locales).
        case_mismatch_mask = non_missing_str != non_missing_str.str.lower()
        if case_mismatch_mask.any():
            errors.append(
                "email column contains mixed-case values — "
                "lowercase (not uppercase) before deduplication or uniqueness checks "
                "to avoid unicode case-folding issues"
            )
            flagged_idx.update(non_missing_str.index[case_mismatch_mask])

        fake_mask = non_missing_str.apply(lambda v: email_local(v) in PLACEHOLDER_TOKENS)
        fake_vals = non_missing_str[fake_mask].unique().tolist()
        if fake_vals:
            errors.append(
                f"probable placeholder email addresses detected "
                f"(local part matches known sentinel): {fake_vals[:5]}"
            )
            flagged_idx.update(non_missing_str.index[fake_mask])

        # Domain concentration — statistically anomalous uniformity signals
        # test/seeded data. Real customer datasets have high domain diversity.
        domains = non_missing_str.str.extract(r"@([^@\s]+)$", expand=False).str.lower()
        domain_counts = domains.value_counts(normalize=True)
        if len(domain_counts) > 1:
            top_domain, top_pct = domain_counts.index[0], domain_counts.iloc[0]
            if top_pct > 0.8:
                errors.append(
                    f"email domain unusually concentrated: '{top_domain}' appears in "
                    f"{top_pct:.0%} of values — may indicate test or seeded data"
                )

        # Plus-tag prevalence — informational, not an error.
        plus_count = int(non_missing_str.str.contains(r"\+", regex=True).sum())
        plus_pct = plus_count / len(non_missing_str) * 100
        if plus_pct >= 5:
            errors.append(
                f"{plus_count} values ({plus_pct:.1f}%) use plus-tag subaddressing "
                f"(e.g. user+tag@domain.com) — valid RFC addresses; "
                f"do not strip the +tag when deduplicating without business confirmation"
            )

    # Identifier duplicates
    name = col_name.lower()
    name_tokens = set(re.split(r"[^a-z0-9]+", name))
    uniqueness_ratio = (
        non_missing_str.nunique() / len(non_missing_str)
        if len(non_missing_str) else 0.0
    )
    is_identifier_like = (
        name == "id"
        or name.endswith("_id")
        or bool(name_tokens & {"id", "key", "identifier", "uuid", "ssn", "soc_sec"})
    )

    # Only flag duplicates for likely PRIMARY keys, not foreign keys
    _is_rank_col = is_sequential_ordinal(series) and not any(
        h in col_name.lower() for h in {"id", "key", "code", "number", "no", "num"}
    )

    likely_primary_key = (
        is_identifier_like
        and uniqueness_ratio >= 0.98
        and non_missing_str.nunique() > 20
        and not is_foreign_key
        and not _is_rank_col
    )

    if likely_primary_key:
        dup_mask = non_missing_str.duplicated(keep=False)
        dupes = non_missing_str[non_missing_str.duplicated()].unique().tolist()
        if dupes:
            errors.append(f"candidate primary key contains duplicate values {dupes[:5]}")
            flagged_idx.update(non_missing_str.index[dup_mask])
    # Decimal artefact
    decimal_mask = non_missing_str.str.fullmatch(r"\d+\.0+", na=False)
    if decimal_mask.mean() >= 0.5:
        # Only flag if ALL values are X.0 form — if mixed with real decimals like 0.15,
        # this is a genuine float column and 0.0 is not an artefact
        non_integer_floats = non_missing_str.str.fullmatch(r"\d+\.\d*[1-9]\d*", na=False)
        if non_integer_floats.sum() == 0:  # no real decimal values → artefact likely
            examples = non_missing_str[decimal_mask].unique().tolist()[:5]
            errors.append(f"raw source values include decimal suffix artefacts {examples}")
            flagged_idx.update(non_missing_str.index[decimal_mask])

    # Numeric code inconsistency
    digit_mask = non_missing_str.str.fullmatch(r"\d+(\.0+)?", na=False)
    if digit_mask.mean() >= 0.8:
        cleaned_digits = non_missing_str.str.replace(r"\.0+$", "", regex=True)
        lengths = cleaned_digits.str.len()
        most_common_length_pct = lengths.value_counts(normalize=True).iloc[0]
        is_fixed_format_code = most_common_length_pct >= 0.9

        known_patterns = (format_analysis or {}).get("known_patterns", {})
        has_known_format = bool(known_patterns)

        if is_fixed_format_code and lengths.nunique() > 1 and has_known_format:
            dominant_length = lengths.value_counts().index[0]
            outlier_mask = lengths != dominant_length
            outlier_examples = cleaned_digits[outlier_mask].unique().tolist()[:5]
            numeric_vals = pd.to_numeric(cleaned_digits, errors="coerce").dropna()
            max_fits_dominant = len(numeric_vals) == 0 or numeric_vals.max() < (10 ** dominant_length)
            all_shorter = all(len(str(v)) < dominant_length for v in outlier_examples)
            if all_shorter and max_fits_dominant:
                errors.append(
                    f"code-like values have inconsistent length — dominant length is "
                    f"{dominant_length} digits, but found: {outlier_examples}"
                )
                errors.append("numeric code values may have lost leading zeros")
                flagged_idx.update(cleaned_digits.index[outlier_mask])

    # Sentinel/placeholder value detection for numeric columns
    if storage_type in ("int64", "Int64", "float64"):
        numeric_vals = pd.to_numeric(non_missing_str, errors="coerce").dropna()
        if len(numeric_vals) > 0:
            # Always check for large round-number sentinels
            always_check = [99999, 9999, 999999]
            _signed_sentinel_exclusions = {
                "qty", "quantity", "amount", "balance", "value",
                "price", "total", "revenue", "sales", "profit",
                "margin", "adjustment", "delta", "change", "diff",
            }
            col_l = col_name.lower()
            _allow_negatives = any(
                token in col_l for token in _signed_sentinel_exclusions
            )
            signed_check = [] if _allow_negatives else [-1, -9999]

            sentinel_patterns = always_check + signed_check
            sentinel_mask = numeric_vals.isin(sentinel_patterns)
            found_sentinels = [v for v in sentinel_patterns if (numeric_vals == v).any()]
            if found_sentinels:
                errors.append(
                    f"{found_sentinels} "
                    f"— confirm if these represent missing or invalid data"
                )
                flagged_idx.update(numeric_vals.index[sentinel_mask])

    # Date validation — detect from data, not column name
    is_date_storage = storage_type == "datetime64[ns]"
    numeric_looking = pd.to_numeric(non_missing_str, errors="coerce").notna().mean() >= 0.9
    yyyymmdd_ratio = (
        non_missing_str.str.replace(r"\.0+$", "", regex=True)
        .str.fullmatch(r"(19|20)\d{6}")
        .mean()
    ) if numeric_looking else 0.0

    # Add ISO date string detection:
    iso_date_ratio = non_missing_str.str.fullmatch(
        r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?", na=False
    ).mean()

    is_date_like = is_date_storage or yyyymmdd_ratio >= 0.3 or iso_date_ratio >= 0.8

    if is_date_like:
        compact_mask = non_missing_str.str.fullmatch(r"\d{8}(\.0+)?", na=False)
        if compact_mask.mean() >= 0.3:
            examples = non_missing_str[compact_mask].unique().tolist()[:5]
            errors.append(
                f"date-like values stored in non-standard numeric/string format {examples}"
            )
            flagged_idx.update(non_missing_str.index[compact_mask])
        cleaned = non_missing_str.str.replace(r"\.0+$", "", regex=True)
        parsed_compact = pd.to_datetime(cleaned, errors="coerce", format="%Y%m%d")
        parsed_general = pd.to_datetime(non_missing_str, errors="coerce", format="mixed")

        bad_date_mask = ~(parsed_compact.notna() | parsed_general.notna())
        bad_date_count = int(bad_date_mask.sum())
        bad_date_examples = (
            non_missing_str[bad_date_mask]
            .drop_duplicates()
            .tolist()[:5]
        )

        if bad_date_count > 0:
            errors.append(
                f"{bad_date_count} unparseable date-like values; "
                f"examples: {bad_date_examples}"
            )
            flagged_idx.update(non_missing_str.index[bad_date_mask])

    # Age range check — flag implausibly low values for service accounts
    if "age" in col_name.lower() and storage_type in ("int64", "Int64", "float64"):
        numeric_vals = pd.to_numeric(non_missing_str, errors="coerce").dropna()
        suspicious_low = numeric_vals[numeric_vals < 13]
        if len(suspicious_low) > 0:
            errors.append(
                f"{len(suspicious_low)} age value(s) below 13 — may be unrealistic "
                f"for a service account: {suspicious_low.unique().tolist()[:5]}"
            )
            flagged_idx.update(suspicious_low.index)

    # Statistical outlier detection for numeric columns
    if storage_type in ("int64", "Int64", "float64"):
        anomaly_errors, anomaly_idx = detect_numeric_anomalies(series, col_name, config, is_date_like)
        errors.extend(anomaly_errors)
        flagged_idx.update(anomaly_idx)

    # Contaminated categorical column detection
    n_distinct = non_missing_str.nunique()
    if (uniqueness_ratio < 0.05
        and n_distinct > 50 
        and storage_type in ("string", "object", "category")
        and intended_type not in ("datetime64[ns]", "date", "datetime")):
        errors.append(
            f"column appears categorical (low uniqueness: {uniqueness_ratio:.1%}) "
            f"but has {n_distinct} distinct values — possible contamination or encoding issues"
        )

    # Boolean inconsistency check
    non_missing_lower = non_missing_str.str.lower().str.strip()
    unique_vals = set(non_missing_lower.unique())
    if len(unique_vals) == 2 and unique_vals != {"true", "false"}:
        common_bool_spellings = {
            "true", "false", "yes", "no", "y", "n", "t", "f", "1", "0", "1.0", "0.0"
        }
        if unique_vals.issubset(common_bool_spellings):
            mixed = non_missing_str.unique().tolist()[:5]
            errors.append(
                f"boolean column contains non-standard representations: {mixed} "
                f"— standardize to 'True'/'False' before converting to bool dtype"
            )
            flagged_idx.update(non_missing_str.index)

    # Near-duplicate strings (skip for dates and numeric columns)
    is_mostly_numeric = pd.to_numeric(non_missing_str, errors="coerce").notna().mean() > 0.7
    avg_length = non_missing_str.str.len().mean()
    is_compound_values = avg_length > 30

    if (storage_type in ("string", "category", "object")
            and not is_date_like
            and not is_mostly_numeric
            and not is_compound_values):
        groups = detect_near_duplicate_values(non_missing, config)
        if groups:
            errors.append(
                f"near-duplicate values suggesting variants or typos: {groups[:3]}"
            )
            flat = {v for g in groups for v in g}
            flagged_idx.update(non_missing_str.index[non_missing_str.isin(flat)])

    return errors, flagged_idx

def remove_fk_errors_from_results( 
    data_dictionary: list[dict], 
    identified_fks: set[str]
) -> list[dict]:
    """
    Remove false positive errors from FK columns.
    This runs AFTER FK detection to clean up any errors that were
    flagged before we knew the column was a foreign key.
    """
    data_dictionary = copy.deepcopy(data_dictionary)
    for row in data_dictionary:
        col_name = row.get("column_name", "")
        if col_name in identified_fks:
            # Remove duplicate-related errors
            errors = row.get("errors", [])
            cleaned_errors = []

            for error in errors:
                if "duplicate values" in error.lower():
                    continue  # Expected for FKs
                cleaned_errors.append(error)

            row["errors"] = cleaned_errors

            # Mark as FK in description
            if row.get("description") and "foreign key" not in row["description"].lower():
                row["description"] = f"Foreign key column. {row['description']}"

    return data_dictionary