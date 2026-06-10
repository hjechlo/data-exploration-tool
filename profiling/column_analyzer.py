"""
ColumnAnalyzer — per-column profiling, error detection, and MinHash sketching.
"""

import json
import re
import copy
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH
from .format_pattern_analyzer import FormatPatternAnalyzer
from .json_utils import is_sequential_ordinal

from .config import EMAIL_REGEX, ID_NAME_HINTS, PLACEHOLDER_TOKENS, PipelineConfig


class ColumnAnalyzer:
    """
    Computes the full column summary for every column in a dataset.

    Responsibilities
    ----------------
    - Determine storage type
    - Detect data quality errors (missing values, date issues, duplicates, etc.)
    - Detect near-duplicate string values within a column (shingled MinHash + LSH)
    - Build human-readable column_facts for LLM evidence
    - Compute exact MinHash and shingled MinHash sketches for cross-column comparison
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.format_analyzer = FormatPatternAnalyzer(config)

    # ------------------------------------------------------------------
    # Type inference
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Similarity classification
    # ------------------------------------------------------------------

    def classify_column_for_similarity(
        self, series: pd.Series, storage_type: str, col_name: str = ""
    ) -> str:
        """Classify a column into a similarity kind used to gate MinHash computation."""
        non_null = series.dropna()
        n = len(non_null)

        if n == 0:
            return "empty"

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
        self, series: pd.Series, storage_type: str, col_name: str = ""
    ) -> bool:
        kind = self.classify_column_for_similarity(series, storage_type, col_name)
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
            return 0 < n_distinct <= self.config.near_dupe_max_values

        return False

    # ------------------------------------------------------------------
    # MinHash sketches
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(val: str) -> str:
        s = str(val).strip().lower()
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _clean_values(self, series: pd.Series) -> list[str]:
        values = series.dropna().astype(str).map(self._normalize)
        values = values[(values != "") & (~values.str.lower().isin(PLACEHOLDER_TOKENS))]
        return values.unique().tolist()

    def compute_minhash(self, series: pd.Series) -> MinHash:
        """Exact value-token MinHash sketch for join-path detection."""
        m = MinHash(num_perm=self.config.minhash_num_perm)
        for val in self._clean_values(series):
            m.update(val.encode("utf8"))
        return m

    def compute_minhash_shingle(self, series: pd.Series) -> MinHash:
        """
        Character k-gram shingled MinHash sketch.

        Measures vocabulary-level textual similarity — useful for detecting
        that two columns store the same concept even when values have typos
        or formatting noise (e.g. suburb names across two datasets).
        Not applied to ID/key-like columns where partial string overlap
        is meaningless.
        """
        k = self.config.minhash_shingle_k
        m = MinHash(num_perm=self.config.minhash_num_perm)
        for val in self._clean_values(series):
            if len(val) < k:
                m.update(val.encode("utf8"))
            else:
                for shingle in {val[i: i + k] for i in range(len(val) - k + 1)}:
                    m.update(shingle.encode("utf8"))
        return m

    # ------------------------------------------------------------------
    # Error detection
    # ------------------------------------------------------------------

    def detect_near_duplicate_values(self, series: pd.Series) -> list[list[str]]:
        cfg = self.config
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
        self, col_name: str, series: pd.Series, storage_type: str, is_foreign_key: bool = False, format_analysis: dict | None = None
    ) -> list[str]:
        """Return a list of human-readable error/quality strings for a column.
        
        Args:
            col_name: Column name
            series: Pandas Series with column data
            storage_type: Data type (string, datetime64[ns], int64, etc.)
            is_foreign_key: If True, skip duplicate errors (duplicates are expected in FKs)
        """
        errors: list[str] = []

        s = series.astype(str).str.strip()
        null_mask = series.isna()
        placeholder_mask = s.str.lower().isin(PLACEHOLDER_TOKENS)
        missing_mask = null_mask | placeholder_mask

        null_count = int(null_mask.sum())
        placeholder_count = int((placeholder_mask & ~null_mask).sum())
        total = len(series)

        if null_count > 0:
            errors.append(f"{null_count} NULL values ({null_count/total*100:.1f}% of {total} records)")

        if placeholder_count > 0:
            examples = s[placeholder_mask & ~null_mask].unique().tolist()[:5]
            errors.append(
                f"{placeholder_count} placeholder/missing-sentinel values "
                f"({placeholder_count/total*100:.1f}%) e.g. {examples}"
            )

        non_missing = series[~missing_mask]
        non_missing_str = non_missing.astype(str).str.strip()

        if len(non_missing_str) == 0:
            return errors
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
        
        # Email validation
        if "email" in col_name.lower():
            bad = [v for v in non_missing_str.unique() if v and not EMAIL_REGEX.match(v)]
            if bad:
                errors.append(f"malformed email values {bad[:5]}")
        
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
            dupes = non_missing_str[non_missing_str.duplicated()].unique().tolist()
            if dupes:
                errors.append(f"candidate primary key contains duplicate values {dupes[:5]}")
        # Decimal artefact
        decimal_mask = non_missing_str.str.fullmatch(r"\d+\.0+", na=False)
        if decimal_mask.mean() >= 0.5:
            # Only flag if ALL values are X.0 form — if mixed with real decimals like 0.15,
            # this is a genuine float column and 0.0 is not an artefact
            non_integer_floats = non_missing_str.str.fullmatch(r"\d+\.\d*[1-9]\d*", na=False)
            if non_integer_floats.sum() == 0:  # no real decimal values → artefact likely
                examples = non_missing_str[decimal_mask].unique().tolist()[:5]
                errors.append(f"raw source values include decimal suffix artefacts {examples}")
        
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
                found_sentinels = [v for v in sentinel_patterns if (numeric_vals == v).any()]
                if found_sentinels:
                    errors.append(
                        f"possible sentinel/placeholder value(s) detected: {found_sentinels} "
                        f"— confirm if these represent missing or invalid data"
                    )

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

        # Age range check — flag implausibly low values for service accounts
        if "age" in col_name.lower() and storage_type in ("int64", "Int64", "float64"):
            numeric_vals = pd.to_numeric(non_missing_str, errors="coerce").dropna()
            suspicious_low = numeric_vals[numeric_vals < 13]
            if len(suspicious_low) > 0:
                errors.append(
                    f"{len(suspicious_low)} age value(s) below 13 — may be unrealistic "
                    f"for a service account: {suspicious_low.unique().tolist()[:5]}"
                )

        # Contaminated categorical column detection
        n_distinct = non_missing_str.nunique()
        if (uniqueness_ratio < 0.05
            and n_distinct > 50   # ← was 20
            and storage_type in ("string", "object", "category")):
            errors.append(
                f"column appears categorical (low uniqueness: {uniqueness_ratio:.1%}) "
                f"but has {n_distinct} distinct values — possible contamination or encoding issues"
            )

        # Boolean inconsistency check
        bool_true = {"true", "yes", "1"}
        bool_false = {"false", "no", "0"}
        bool_all = bool_true | bool_false
        non_missing_lower = non_missing_str.str.lower()
        if non_missing_lower.isin(bool_all).mean() >= 0.9:
            unique_vals = set(non_missing_lower.unique())
            if unique_vals & bool_true and unique_vals & bool_false:
                mixed = non_missing_str[~non_missing_lower.isin({"true", "false"})].unique().tolist()[:5]
                if mixed:
                    errors.append(
                        f"boolean column contains non-standard representations: {mixed} "
                        f"— standardize to 'True'/'False' before converting to bool dtype"
                    )

        # Near-duplicate strings (skip for dates and numeric columns)
        is_mostly_numeric = pd.to_numeric(non_missing_str, errors="coerce").notna().mean() > 0.7
        avg_length = non_missing_str.str.len().mean()
        is_compound_values = avg_length > 30

        if (storage_type in ("string", "category", "object")
                and not is_date_like
                and not is_mostly_numeric
                and not is_compound_values):
            groups = self.detect_near_duplicate_values(non_missing)
            if groups:
                errors.append(
                    f"near-duplicate values suggesting variants or typos: {groups[:3]}"
                )

        return errors
    
    def remove_fk_errors_from_results(
        self, 
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

    # ------------------------------------------------------------------
    # Column facts (LLM evidence)
    # ------------------------------------------------------------------

    def build_column_facts(
        self,
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
                format_analysis = self.format_analyzer.analyze_format_distribution(raw_series)
            format_context = self.format_analyzer.generate_llm_context(col_name, raw_series, analysis=format_analysis)
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

    # ------------------------------------------------------------------
    # Sample value extraction
    # ------------------------------------------------------------------

    def extract_sample_values(self, series: pd.Series) -> list[str]:
        max_s = self.config.max_sample_values
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

    # ------------------------------------------------------------------
    # Full column summary
    # ------------------------------------------------------------------

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
                    bool_vals = {"true", "false", "yes", "no", "1", "0"}
                    non_null_lower = raw_series.dropna().astype(str).str.strip().str.lower()
                    if non_null_lower.isin(bool_vals).mean() >= 0.9:
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
            raw_samples = self.extract_sample_values(raw_series)
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

            similarity_kind = self.classify_column_for_similarity(raw_series, storage_type, col_name)
            
            #Analyze format patterns
            format_analysis = self.format_analyzer.analyze_format_distribution(raw_series)
            if similarity_kind in ("key_like", "categorical") and "int" in storage_type.lower():
                _anoms = format_analysis.get("anomalies", {})
                _anoms["suspicious_values"] = []
                _anoms["outlier_values"] = []

            errors = self.detect_column_errors(
                col_name, raw_series, storage_type,
                format_analysis=format_analysis
            )
            column_facts = self.build_column_facts(
                col_name, raw_series, storage_type, permissible_values,
                format_analysis=format_analysis,
                intended_type=intended_type,
            )
            similarity_kind = self.classify_column_for_similarity(
                raw_series, storage_type, col_name
            )

            # Exact MinHash
            minhash_sketch = (
                self.compute_minhash(raw_series)
                if self.should_compute_minhash(raw_series, storage_type, col_name)
                else None
            )

            # Shingled MinHash (non-ID string columns only)
            minhash_shingle_sketch = (
                self.compute_minhash_shingle(raw_series)
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
            }

            if minhash_sketch is not None:
                row["_minhash"] = minhash_sketch
            if minhash_shingle_sketch is not None:
                row["_minhash_shingle"] = minhash_shingle_sketch

            rows.append(row)

        return rows
    
    def generate_column_validation_rules(
        self,
        column_summary: list[dict],
        table_name: str,
    ) -> list[dict]:
        """
        Generate per-column validation rules deterministically from profiled data.
        Each rule includes check_params so it can be executed against real records.
        """
        rules: list[dict] = []
        rule_id = 1

        for row in column_summary:
            col = row["column_name"]
            intended = row.get("intended_data_type", row["data_type"])
            storage = row["data_type"]
            permissible = row.get("permissible_values")
            role = row.get("relationship_role", "")
            is_identifier = role in ("primary_key", "foreign_key", "join_key")

            # 1. Primary key integrity.
            # FK referential rules are generated later in pipeline.py using cross-table metadata.
            if role == "primary_key":
                rules.append({
                    "rule_id": rule_id,
                    "table": table_name,
                    "column": col,
                    "type": "primary_key",
                    "rule": f"{table_name}.{col} must be non-null and unique",
                    "check_params": {},
                })
                rule_id += 1

            # 1. Type mismatch
            if intended != storage:
                rules.append({
                    "rule_id": rule_id,
                    "table": table_name,
                    "column": col,
                    "type": "type",
                    "rule": f"value must be stored as {intended}, not {storage}",
                    "check_params": {"intended_type": intended, "storage_type": storage},
                })
                rule_id += 1
            
            # 1B. Strict compact YYYYMMDD date validation.
            errors_text = " | ".join(row.get("errors", []))

            compact_yyyymmdd_evidence = (
                "date-like values stored in non-standard numeric/string format" in errors_text
            )

            if intended.startswith("datetime") and compact_yyyymmdd_evidence:
                date_format = "YYYYMMDD"

                rules.append({
                    "rule_id": rule_id,
                    "table": table_name,
                    "column": col,
                    "type": "valid_yyyymmdd_date",
                    "rule": (
                        f"{col} must be a valid real calendar date in {date_format} format "
                        f"after removing any .0 suffix; month must be 01–12 and day must exist "
                        f"for that month"
                    ),
                    "check_params": {
                        "format": date_format,
                        "strip_decimal_suffix": True,
                    },
                })
                rule_id += 1
            
            # 1C. Numeric-like columns containing non-numeric raw values.
            _format_analysis_1c = row.get("_format_analysis", {}) or {}
            _top_formats_1c = (
                _format_analysis_1c
                .get("format_fingerprints", {})
                .get("top_formats", [])
            )
            _has_alpha_prefix_variant = any(
                re.fullmatch(r"a+X+", fmt.get("pattern", ""))
                for fmt in _top_formats_1c
            )

            if (
                "non-numeric value(s)" in errors_text
                and "numeric-like column" in errors_text
                and not _has_alpha_prefix_variant
            ):
                intended_l = str(intended).lower()
                is_integer_target = "int" in intended_l

                rule_type = "integer_parseable" if is_integer_target else "numeric_parseable"

                rules.append({
                    "rule_id": rule_id,
                    "table": table_name,
                    "column": col,
                    "type": rule_type,
                    "rule": (
                        f"{col} must contain values that can be parsed as "
                        f"{'integers' if is_integer_target else 'numbers'}"
                    ),
                    "check_params": {
                        "allow_decimal": not is_integer_target,
                    },
                })
                rule_id += 1
            
            if "sentinel/placeholder value(s) detected" in errors_text:
                rules.append({
                    "rule_id": rule_id,
                    "table": table_name,
                    "column": col,
                    "type": "sentinel_check",
                    "rule": (
                        f"{col} should not contain sentinel/placeholder values "
                        f"such as 99999, 9999, or 999999 — investigate and replace "
                        f"with NULL or a valid value"
                    ),
                    "check_params": {
                        "sentinel_values": [99999, 9999, 999999, -1, -9999],
                    },
                })
                rule_id += 1

            # 2. Enumeration — only truly categorical (already cast to category by preprocessor)
            if permissible and storage == "category" and not is_identifier:
                rules.append({
                    "rule_id": rule_id,
                    "table": table_name,
                    "column": col,
                    "type": "enumeration",
                    "rule": f"value must be one of: {permissible}",
                    "check_params": {"values": permissible},
                })
                rule_id += 1

            # 3. Format — only for string/object/category structural formats.
            storage_l = str(storage).lower()
            is_string_like = (
                "string" in storage_l
                or "object" in storage_l
                or "category" in storage_l
            )

            if is_string_like:
                col_l = col.lower()
                format_analysis = row.get("_format_analysis", {})
                known_patterns = format_analysis.get("known_patterns", {}) or {}

                # 3A. Datetime-like strings
                # This is allowed because it is a real structural rule:
                # YYYY-MM-DD HH:MM:SS, not a natural-language text fingerprint.
                if intended.startswith("datetime"):
                    rules.append({
                        "rule_id": rule_id,
                        "table": table_name,
                        "column": col,
                        "type": "format",
                        "rule": f"{col} must follow a valid datetime format such as YYYY-MM-DD HH:MM:SS",
                        "check_params": {
                            "regex": r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?$",
                            "pattern_name": "datetime_string",
                        },
                    })
                    rule_id += 1

                # 3B. Email
                # Only create this rule if the column name says it is email.
                elif "email" in col_l:
                    rules.append({
                        "rule_id": rule_id,
                        "table": table_name,
                        "column": col,
                        "type": "format",
                        "rule": f"{col} must follow a valid email address structure, e.g. name@example.com",
                        "check_params": {
                            # More permissive than ASCII-only; supports internationalized local parts.
                            "regex": r"^[^\s@]+@[^\s@]+\.[^\s@]+$",
                            "pattern_name": "email",
                        },
                    })
                    rule_id += 1


                # 3D. Postal / ZIP code
                # Only generate this for postal-code-like columns.
                # Do not use known_patterns blindly because numeric fields like Milliseconds
                # can accidentally match postal-code-like regexes.
                elif any(token in col_l for token in ("postal", "postcode", "zipcode", "zip")):
                    rules.append({
                        "rule_id": rule_id,
                        "table": table_name,
                        "column": col,
                        "type": "format",
                        "rule": f"{col} should follow a valid postal/ZIP code structure for the relevant country",
                        "check_params": {
                            # Generic: letters/digits/spaces/hyphens, avoids country-specific hardcoding.
                            "regex": r"^[A-Za-z0-9][A-Za-z0-9\s-]{2,15}$",
                            "pattern_name": "postal_code_generic",
                        },
                    })
                    rule_id += 1

                # 3E. URL
                elif any(token in col_l for token in ("url", "website", "link")):
                    rules.append({
                        "rule_id": rule_id,
                        "table": table_name,
                        "column": col,
                        "type": "format",
                        "rule": f"{col} must follow a valid URL structure, e.g. https://example.com",
                        "check_params": {
                            "regex": r"^https?://[^\s]+$",
                            "pattern_name": "url",
                        },
                    })
                    rule_id += 1

                # 3F. Stable alphanumeric code
                # Only for code/id/reference-like columns, not names/titles/descriptions.
                elif any(token in col_l for token in ("code", "ref", "reference", "number", "no")):
                    top_formats = (
                        format_analysis
                        .get("format_fingerprints", {})
                        .get("top_formats", [])
                    )

                    stable_code_patterns = []
                    readable_formats = []
                    cumulative_pct = 0.0

                    for fmt in top_formats[:6]:
                        pattern = fmt.get("pattern", "")
                        pct = float(fmt.get("percentage", 0) or 0)

                        if not pattern:
                            continue

                        # Only allow true code-like formats with both letters and digits.
                        # This blocks pure text fingerprints like aaaaa aaaaaaaa.
                        if not ("a" in pattern and "X" in pattern):
                            continue

                        # Also block free-text-looking formats with too many spaces/words.
                        if str(pattern).count(" ") > 1:
                            continue

                        if pct >= 1.0 or cumulative_pct < 95.0:
                            stable_code_patterns.append(pattern)
                            cumulative_pct += pct

                            if re.fullmatch(r"a+X+a+", pattern):
                                n_prefix = len(re.match(r"^a+", pattern).group(0))
                                n_digits = len(re.search(r"X+", pattern).group(0))
                                n_suffix = len(re.search(r"a+$", pattern).group(0))

                                if n_prefix == 1 and n_suffix == 1:
                                    readable_formats.append(
                                        f"one alphabetic prefix, followed by {n_digits} digits, "
                                        f"ending with one alphabetic suffix"
                                    )
                                else:
                                    readable_formats.append(
                                        f"{n_prefix} alphabetic prefix letter(s), followed by "
                                        f"{n_digits} digit(s), ending with "
                                        f"{n_suffix} alphabetic suffix letter(s)"
                                    )

                            elif re.fullmatch(r"a+X+", pattern):
                                n_letters = len(re.match(r"^a+", pattern).group(0))
                                n_digits = len(re.search(r"X+$", pattern).group(0))
                                readable_formats.append(
                                    f"{n_letters} letter(s) followed by {n_digits} digit(s)"
                                )

                            elif re.fullmatch(r"X+a+", pattern):
                                n_digits = len(re.match(r"^X+", pattern).group(0))
                                n_letters = len(re.search(r"a+$", pattern).group(0))
                                readable_formats.append(
                                    f"{n_digits} digit(s) followed by {n_letters} letter(s)"
                                )

                            else:
                                readable_formats.append(pattern)

                        if cumulative_pct >= 99.0:
                            break

                    if stable_code_patterns and cumulative_pct >= 95.0:
                        regex_parts = []
                        for pattern in stable_code_patterns:
                            regex = ""
                            for ch in pattern:
                                if ch == "X":
                                    regex += r"\d"
                                elif ch == "a":
                                    regex += r"[A-Za-z]"
                                else:
                                    regex += re.escape(ch)
                            regex_parts.append(regex)

                        combined_regex = r"^(?:" + "|".join(regex_parts) + r")$"

                        if len(readable_formats) == 1:
                            rule_text = (
                                f"{col} must match the observed stable code format: "
                                f"{readable_formats[0]}."
                            )
                        else:
                            rule_text = (
                                f"{col} must match one of the observed stable code formats: "
                                + " OR ".join(readable_formats)
                                + "."
                            )

                        rules.append({
                            "rule_id": rule_id,
                            "table": table_name,
                            "column": col,
                            "type": "format",
                            "rule": rule_text,
                            "check_params": {
                                "regex": combined_regex,
                                "pattern_name": "stable_code_format",
                            },
                        })
                        rule_id += 1

            # 7. Binary flag constraint — int columns where observed range is exactly {0, 1}
            if "int" in str(storage).lower() and not is_identifier:
                col_min = row.get("profile", {}).get("min")
                col_max = row.get("profile", {}).get("max")
                if col_min is not None and col_max is not None and float(col_min) == 0.0 and float(col_max) == 1.0:
                    rules.append({
                        "rule_id": rule_id,
                        "table": table_name,
                        "column": col,
                        "type": "enumeration",
                        "rule": f"{col} must be 0 or 1",
                        "check_params": {"values": [0, 1]},
                    })
                    rule_id += 1

        return rules

    def detect_cross_column_relationships(
        self,
        df: pd.DataFrame,
        column_summary: list[dict],
    ) -> list[dict]:
        """
        Run record-wise cross-column checks and return findings with violation counts.

        Checks:
        - Date ordering: col_a <= col_b for all date/time column pairs
        - Numeric sum: col_c ≈ col_a + col_b for numeric column triplets
        - Null consistency: if col_a is non-null, col_b should also be non-null
        """

        findings = []

        date_cols = [
            r["column_name"] for r in column_summary
            if (
                "date" in r["column_name"].lower()
                or r.get("intended_data_type", "").startswith("datetime")
            )
            and r["column_name"] in df.columns
        ]

        numeric_cols = [
            r["column_name"] for r in column_summary
            if r.get("data_type") in ("int64", "float64")
            and r.get("relationship_role") not in ("primary_key", "foreign_key", "join_key")
            and r["column_name"] in df.columns
        ]

        # 1. Date ordering
        def _is_ordered_pair(a: str, b: str) -> bool:
            a_l, b_l = a.lower(), b.lower()
            a_is_start = any(h in a_l for h in self.config.date_ordering_start_hints)
            b_is_end   = any(h in b_l for h in self.config.date_ordering_end_hints)
            b_is_start = any(h in b_l for h in self.config.date_ordering_start_hints)
            a_is_end   = any(h in a_l for h in self.config.date_ordering_end_hints)
            return (a_is_start and b_is_end) or (b_is_start and a_is_end)
        
        for i, col_a in enumerate(date_cols):
            for col_b in date_cols[i + 1:]:
                try:
                    a_parsed = pd.to_datetime(df[col_a], errors="coerce")
                    b_parsed = pd.to_datetime(df[col_b], errors="coerce")
                    both_valid = a_parsed.notna() & b_parsed.notna()
                    n_both = int(both_valid.sum())
                    if n_both == 0:
                        continue
                    violations_ab = int((a_parsed[both_valid] > b_parsed[both_valid]).sum())
                    violations_ba = int((b_parsed[both_valid] > a_parsed[both_valid]).sum())
                    if violations_ab / n_both <= 0.05:
                        findings.append({
                            "type": "date_ordering",
                            "columns": [col_a, col_b],
                            "rule": f"{col_a} must be on or before {col_b}",
                            "n_checked": n_both,
                            "n_violations": violations_ab,
                            "violation_rate": round(violations_ab / n_both, 3),
                            "check_params": {"col_a": col_a, "col_b": col_b},
                            "sample_violations": df.loc[
                                both_valid & (a_parsed > b_parsed), [col_a, col_b]
                            ].head(3).to_dict("records"),
                        })
                    elif violations_ba / n_both <= 0.05:
                        findings.append({
                            "type": "date_ordering",
                            "columns": [col_b, col_a],
                            "rule": f"{col_b} must be on or before {col_a}",
                            "n_checked": n_both,
                            "n_violations": violations_ba,
                            "violation_rate": round(violations_ba / n_both, 3),
                            "check_params": {"col_a": col_b, "col_b": col_a},
                            "sample_violations": df.loc[
                                both_valid & (b_parsed > a_parsed), [col_a, col_b]
                            ].head(3).to_dict("records"),
                        })
                except Exception:
                    continue

        # 2. Numeric sum
        if len(numeric_cols) >= 3:
            for k, col_c in enumerate(numeric_cols):
                for i, col_a in enumerate(numeric_cols):
                    if i == k:
                        continue
                    for col_b in numeric_cols[i + 1:]:
                        if col_b == col_c:
                            continue
                        try:
                            mask = (
                                df[col_a].notna()
                                & df[col_b].notna()
                                & df[col_c].notna()
                            )
                            n_checked = int(mask.sum())
                            if n_checked < 10:
                                continue
                            expected = df.loc[mask, col_a] + df.loc[mask, col_b]
                            actual = df.loc[mask, col_c]
                            mean_abs = actual.abs().mean()
                            tolerance = mean_abs * 0.01 if mean_abs > 0 else 0.01
                            violations = int((abs(expected - actual) > tolerance).sum())
                            if violations / n_checked <= 0.02:
                                findings.append({
                                    "type": "numeric_sum",
                                    "columns": [col_a, col_b, col_c],
                                    "rule": f"{col_c} must equal {col_a} + {col_b}",
                                    "n_checked": n_checked,
                                    "n_violations": violations,
                                    "violation_rate": round(violations / n_checked, 3),
                                    "check_params": {"col_a": col_a, "col_b": col_b, "col_c": col_c},
                                })
                        except Exception:
                            continue

        # 3. Date columns must not be in the future
        today = pd.Timestamp.now().normalize()
        for col_date in date_cols:
            try:
                parsed = pd.to_datetime(df[col_date], errors="coerce")
                valid = parsed.notna()
                n_valid = int(valid.sum())
                if n_valid == 0:
                    continue
                future_mask = valid & (parsed > today)
                n_future = int(future_mask.sum())
                # Only emit rule if no violations (or very few) — it should hold universally
                if n_future / n_valid <= 0.02:
                    findings.append({
                        "type": "date_not_future",
                        "columns": [col_date],
                        "rule": f"{col_date} must not be a future date",
                        "n_checked": n_valid,
                        "n_violations": n_future,
                        "violation_rate": round(n_future / n_valid, 3),
                        "check_params": {"col_a": col_date},
                    })
            except Exception:
                continue

        # 4. NRIC birth year consistent with Age column (universally applicable — derived from data)
        nric_cols = [
            r["column_name"] for r in column_summary
            if r["column_name"] in df.columns
            and r.get("_format_analysis", {}).get("known_patterns", {}).get("nric_sg", {}).get("percentage", 0) >= 80
        ]
        age_cols = [
            r["column_name"] for r in column_summary
            if r["column_name"] in df.columns
            and "age" in r["column_name"].lower()
            and r.get("data_type") in ("int64", "float64", "string", "object")
        ]
        current_year = pd.Timestamp.now().year
        dob_cols = []
        for r in column_summary:
            col_name = r["column_name"]
            if col_name not in df.columns:
                continue
            if not r.get("intended_data_type", "").startswith("datetime"):
                continue
            # A DOB column is a date column where all values fall in a plausible birth year range.
            # This is fully data-driven — no column name keywords used.
            parsed = pd.to_datetime(df[col_name], errors="coerce")
            valid_years = parsed.dropna().dt.year
            if len(valid_years) == 0:
                continue
            if valid_years.min() >= 1900 and valid_years.max() <= (current_year - 10):
                dob_cols.append(col_name)
        current_year = pd.Timestamp.now().year

        def _extract_nric_birth_year(nric_series: pd.Series, current_year: int) -> pd.Series:
            two_digit = nric_series.str[1:3].apply(pd.to_numeric, errors="coerce")
            prefix = nric_series.str[0]
            century = pd.Series(1900, index=nric_series.index)
            century[prefix == "T"] = 2000
            century[(prefix != "T") & (two_digit <= (current_year % 100))] = 2000
            return century + two_digit

        for nric_col in nric_cols:
            nric_str = df[nric_col].astype(str).str.strip().str.upper()
            valid_nric = nric_str.str.match(r'^[STFG]\d{7}[A-Z]$', na=False)

            # DOB path — preferred, exact year match
            for dob_col in dob_cols:
                try:
                    dob_parsed = pd.to_datetime(df[dob_col], errors="coerce")
                    both_valid = valid_nric & dob_parsed.notna()
                    n_checked = int(both_valid.sum())
                    if n_checked < 5:
                        continue
                    nric_year = _extract_nric_birth_year(nric_str[both_valid], current_year)
                    dob_year = dob_parsed[both_valid].dt.year
                    violations = int((nric_year != dob_year).sum())
                    if violations / n_checked <= 0.02:
                        findings.append({
                            "type": "nric_dob_consistency",
                            "columns": [nric_col, dob_col],
                            "rule": f"Birth year encoded in {nric_col} must match year in {dob_col}",
                            "n_checked": n_checked,
                            "n_violations": violations,
                            "violation_rate": round(violations / n_checked, 3),
                            "check_params": {
                                "col_a": nric_col,
                                "col_b": dob_col,
                                "check_mode": "dob",
                                "current_year": current_year,
                            },
                        })
                except Exception:
                    continue

            # Age path — fallback when no DOB column present, ±2 year tolerance
            if not dob_cols:
                for age_col in age_cols:
                    try:
                        age_vals = pd.to_numeric(df[age_col], errors="coerce")
                        both_valid = valid_nric & age_vals.notna()
                        n_checked = int(both_valid.sum())
                        if n_checked < 5:
                            continue
                        nric_year = _extract_nric_birth_year(nric_str[both_valid], current_year)
                        expected_age = current_year - nric_year
                        actual_age = age_vals[both_valid]
                        violations = int((abs(expected_age - actual_age) > 2).sum())
                        if violations / n_checked <= 0.05:
                            findings.append({
                                "type": "nric_age_consistency",
                                "columns": [nric_col, age_col],
                                "rule": f"{age_col} must be consistent with birth year encoded in {nric_col} (±2 years)",
                                "n_checked": n_checked,
                                "n_violations": violations,
                                "violation_rate": round(violations / n_checked, 3),
                                "check_params": {
                                    "col_a": nric_col,
                                    "col_b": age_col,
                                    "check_mode": "age",
                                    "current_year": current_year,
                                },
                            })
                    except Exception:
                        continue


        return findings
    
    def run_validation_checks(
        self,
        df: pd.DataFrame,
        validation_rules: list[dict],
    ) -> dict:
        """
        Apply validation rules against actual records.

        Returns:
        - per_rule: summary per validation rule
        - violation_records: clean record-centric rows for Word rendering
        """
        per_rule_results = []
        record_failed_rules: dict[int, list[dict]] = {}

        def _find_record_identifier_column() -> str | None:
            """
            Pick a readable record identifier for failed-validation tables.

            Universal logic:
            - Prefer identifier-like names.
            - Prefer high uniqueness.
            - Prefer compact values over long free text.
            This is display-only and does not classify PK/FK relationships.
            """
            def _norm(x: str) -> str:
                return "".join(ch for ch in str(x).lower() if ch.isalnum())

            def _name_score(col: str) -> int:
                n = _norm(col)

                if n == "id":
                    return 100
                if n.endswith("id") or "identifier" in n or n.endswith("key"):
                    return 90
                if n.endswith("no") or n.endswith("number") or "serial" in n:
                    return 70
                if n in {"sn", "sno", "seq", "sequence", "rowno", "recordno"}:
                    return 60

                return 0

            candidates = []

            for c in df.columns:
                non_null = df[c].dropna()
                if len(non_null) == 0:
                    continue

                uniqueness = non_null.nunique(dropna=True) / len(non_null)
                score = _name_score(c)

                if score == 0 and uniqueness < 1.0:
                    continue

                avg_len = non_null.astype(str).str.len().mean()

                candidates.append({
                    "column": c,
                    "name_score": score,
                    "uniqueness": uniqueness,
                    "avg_len": avg_len,
                })

            if not candidates:
                return None

            candidates.sort(
                key=lambda x: (
                    x["name_score"],
                    x["uniqueness"],
                    -x["avg_len"],
                ),
                reverse=True,
            )

            return candidates[0]["column"]

        record_id_col = _find_record_identifier_column()

        for rule in validation_rules:
            rule_id = rule.get("rule_id")
            col = rule.get("column", "")
            rule_type = rule.get("type")
            check_params = rule.get("check_params", {})

            try:
                failing_mask = self._apply_rule_check(df, rule_type, col, check_params)
                if failing_mask is None:
                    continue

                failing_mask = failing_mask.fillna(False)
                failing_indices = df.index[failing_mask].tolist()
                n_violations = len(failing_indices)

                label = (
                    f"#{rule_id}: {rule.get('rule', '')}"
                    if rule_id is not None
                    else rule.get("rule", "")
                )

                for idx in failing_indices:
                    failed_value = "—"
                    failed_col = col

                    if col in df.columns:
                        failed_value = df.at[idx, col]
                    else:
                        related_cols = check_params.get("related_cols", [])
                        first_related = next((c for c in related_cols if c in df.columns), None)
                        if first_related:
                            failed_col = first_related
                            failed_value = df.at[idx, first_related]

                    record_failed_rules.setdefault(idx, []).append({
                        "rule_id": rule_id,
                        "label": label,
                        "column": failed_col,
                        "failed_value": failed_value,
                        "rule_type": rule_type,
                    })

                result = {
                    "rule_id": rule_id,
                    "column": col,
                    "rule": rule.get("rule"),
                    "type": rule_type,
                    "n_records_checked": len(df),
                    "n_violations": n_violations,
                    "violation_rate": round(n_violations / len(df), 3) if len(df) > 0 else 0,
                    "passed": n_violations == 0,
                }

                if n_violations > 0 and col in df.columns:
                    result["sample_violations"] = (
                        df.loc[failing_indices[:5], [col]].to_dict("records")
                    )

                per_rule_results.append(result)

            except Exception as e:
                per_rule_results.append({
                    "rule_id": rule_id,
                    "column": col,
                    "rule": rule.get("rule"),
                    "type": rule_type,
                    "error": str(e),
                    "n_violations": None,
                    "passed": None,
                })

        violation_records = []

        for idx, failures in record_failed_rules.items():
            failed_cols = list(dict.fromkeys(
                f["column"] for f in failures if f.get("column")
            ))

            failed_values = [
                str(f.get("failed_value", "—"))
                for f in failures
            ]

            record_identifier = "—"
            if record_id_col and record_id_col in df.columns:
                record_identifier = df.at[idx, record_id_col]

            violation_records.append({
                "Row": str(idx),
                "Record Identifier": str(record_identifier),
                "Failed Column": ", ".join(failed_cols) if failed_cols else "—",
                "Failed Value": " | ".join(failed_values) if failed_values else "—",
                "Validation Rules Failed": " | ".join(f["label"] for f in failures),
            })

        return {
            "per_rule": per_rule_results,
            "violation_records": violation_records,
            "total_failing_records": len(record_failed_rules),
        }

    def _apply_rule_check(
        self,
        df: pd.DataFrame,
        rule_type: str,
        col: str,
        check_params: dict,
    ) -> pd.Series | None:
        """Returns a boolean mask of rows that FAIL the rule (True = failing)."""

        if rule_type == "format":
            pattern = check_params.get("regex")
            length = check_params.get("expected_length")
            pattern_name = check_params.get("pattern_name")

            if col not in df.columns:
                return None

            col_str = df[col].astype(str).str.strip()
            non_missing = df[col].notna() & (col_str != "")

            if pattern_name == "datetime_string":
                structural_ok = col_str.str.match(pattern, na=False, case=False) if pattern else True

                def _is_real_datetime(x: str) -> bool:
                    try:
                        pd.to_datetime(x, errors="raise")
                        return True
                    except Exception:
                        return False

                calendar_ok = col_str.map(_is_real_datetime)
                return non_missing & ~(structural_ok & calendar_ok)

            if pattern:
                valid = col_str.str.match(pattern, na=False, case=False)
                return non_missing & ~valid

            if length:
                return non_missing & (col_str.str.len() != length)

        elif rule_type == "enumeration":
            values = check_params.get("values", [])
            if values and col in df.columns:
                return df[col].notna() & ~df[col].isin(values)
            
        elif rule_type == "numeric_sum":
            col_a = check_params.get("col_a")
            col_b = check_params.get("col_b")
            col_c = check_params.get("col_c")  # col_c = col_a + col_b
            if not all(c and c in df.columns for c in [col_a, col_b, col_c]):
                return None
            mask = df[col_a].notna() & df[col_b].notna() & df[col_c].notna()
            expected = df.loc[mask, col_a] + df.loc[mask, col_b]
            actual = df.loc[mask, col_c]
            mean_abs = actual.abs().mean()
            tolerance = mean_abs * 0.01 if mean_abs > 0 else 0.01
            failing = pd.Series(False, index=df.index)
            failing.loc[mask[mask].index] = (abs(expected - actual) > tolerance).values
            return failing

        elif rule_type == "date_ordering":
            col_a = check_params.get("col_a")
            col_b = check_params.get("col_b")
            if col_a and col_b and col_a in df.columns and col_b in df.columns:
                a_parsed = pd.to_datetime(df[col_a], errors="coerce")
                b_parsed = pd.to_datetime(df[col_b], errors="coerce")
                both_valid = a_parsed.notna() & b_parsed.notna()
                return both_valid & (a_parsed > b_parsed)

        elif rule_type == "null_consistency":
            col_a = check_params.get("col_a", col)
            col_b = check_params.get("col_b")
            if col_b and col_a in df.columns and col_b in df.columns:
                return df[col_a].notna() & df[col_b].isna()
            
        elif rule_type == "valid_yyyymmdd_date":
            if col not in df.columns:
                return None

            raw = df[col].astype(str).str.strip()
            cleaned = raw.str.replace(r"\.0+$", "", regex=True)

            non_missing = df[col].notna() & (cleaned != "")
            shape_ok = cleaned.str.fullmatch(r"\d{8}", na=False)

            def _is_valid_yyyymmdd(x: str) -> bool:
                from datetime import datetime
                try:
                    datetime.strptime(x, "%Y%m%d")
                    return True
                except ValueError:
                    return False

            valid_calendar = cleaned.where(shape_ok, "").map(
                lambda x: _is_valid_yyyymmdd(x) if x else False
            )

            return non_missing & (~shape_ok | ~valid_calendar)
        
        elif rule_type in {"numeric_parseable", "integer_parseable"}:
            if col not in df.columns:
                return None

            raw = df[col].astype(str).str.strip()
            non_missing = df[col].notna() & (raw != "")

            def _is_number(x: str) -> bool:
                try:
                    float(str(x).strip())
                    return True
                except ValueError:
                    return False

            def _is_integer(x: str) -> bool:
                try:
                    v = float(str(x).strip())
                    return v.is_integer()
                except ValueError:
                    return False

            if rule_type == "integer_parseable":
                valid = raw.map(_is_integer)
            else:
                valid = raw.map(_is_number)

            return non_missing & ~valid
        
        elif rule_type == "date_not_future":
            col_a = check_params.get("col_a", col)
            if col_a not in df.columns:
                return None
            parsed = pd.to_datetime(df[col_a], errors="coerce")
            today = pd.Timestamp.now().normalize()
            return parsed.notna() & (parsed > today)

        elif rule_type in ("nric_age_consistency", "nric_dob_consistency"):
            nric_col = check_params.get("col_a")
            other_col = check_params.get("col_b")
            mode = check_params.get("check_mode")
            current_year = check_params.get("current_year", pd.Timestamp.now().year)
            if not nric_col or not other_col or nric_col not in df.columns or other_col not in df.columns:
                return None
            nric_str = df[nric_col].astype(str).str.strip().str.upper()
            valid_nric = nric_str.str.match(r'^[STFG]\d{7}[A-Z]$', na=False)
            two_digit = nric_str.str[1:3].apply(pd.to_numeric, errors="coerce")
            prefix = nric_str.str[0]
            century = pd.Series(1900, index=df.index, dtype=int)
            century[prefix == "T"] = 2000
            century[(prefix != "T") & (two_digit <= (current_year % 100))] = 2000
            nric_year = century + two_digit
            failing = pd.Series(False, index=df.index)
            if mode == "dob":
                dob_parsed = pd.to_datetime(df[other_col], errors="coerce")
                both_valid = valid_nric & dob_parsed.notna()
                failing.loc[both_valid[both_valid].index] = (
                    nric_year[both_valid] != dob_parsed[both_valid].dt.year
                ).values
            else:
                age_vals = pd.to_numeric(df[other_col], errors="coerce")
                both_valid = valid_nric & age_vals.notna()
                expected_age = current_year - nric_year[both_valid]
                failing.loc[both_valid[both_valid].index] = (
                    abs(expected_age - age_vals[both_valid]) > 2
                ).values
            return failing

        elif rule_type == "phone_validity":
            col_a = check_params.get("col_a", col)
            country_code = check_params.get("country_code")
            valid_first = check_params.get("valid_first_digits", [str(d) for d in range(1, 10)])
            dominant_length = check_params.get("dominant_length")
            if col_a not in df.columns or not dominant_length:
                return None
            raw = df[col_a].astype(str).str.strip()
            if country_code:
                stripped = raw.str.replace(rf'^\+?{re.escape(country_code)}[\s\-]?', '', regex=True)
            else:
                stripped = raw
            digit_only = stripped.str.fullmatch(r'\d+', na=False)
            failing = pd.Series(False, index=df.index)
            local_nums = stripped[digit_only]
            invalid = (
                ~local_nums.str.len().isin([dominant_length]) |
                ~local_nums.str[0].isin(valid_first)
            )
            failing.loc[digit_only[digit_only].index] = invalid.values
            # Non-digit local numbers after stripping also fail
            failing.loc[df[col_a].notna() & ~digit_only] = True
            return failing
        
        elif rule_type == "referential_cross_table":
            col_a = check_params.get("col_a", col)
            pk_values = set(check_params.get("pk_values", []))
            if col_a not in df.columns or not pk_values:
                return None
            col_str = (
                df[col_a]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0+$", "", regex=True)
            )

            pk_values = {
                str(v).strip().replace(".0", "")
                for v in pk_values
                if str(v).strip() != ""
            }

            return df[col_a].notna() & ~col_str.isin(pk_values)

        elif rule_type == "sentinel_check":
            sentinel_values = check_params.get("sentinel_values", [])
            if col not in df.columns or not sentinel_values:
                return None
            numeric = pd.to_numeric(df[col], errors="coerce")
            return df[col].notna() & numeric.isin(sentinel_values)

        elif rule_type == "not_null":
            if col in df.columns:
                return df[col].isna()

        elif rule_type == "range":
            col_min = check_params.get("min")
            col_max = check_params.get("max")
            if col not in df.columns:
                return None
            numeric = pd.to_numeric(df[col], errors="coerce")
            failing = pd.Series(False, index=df.index)
            if col_min is not None:
                failing = failing | (df[col].notna() & (numeric < col_min))
            if col_max is not None:
                failing = failing | (df[col].notna() & (numeric > col_max))
            return failing

        elif rule_type in ("type", "referential"):
            return None 
    
    
    
