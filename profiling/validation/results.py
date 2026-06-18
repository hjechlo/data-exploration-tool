"""Validation result building and the legacy Python rule executor.

Functions were moved from the former column analyzer without changing their
validation behavior.
"""

import json
import re

import pandas as pd

from ..core.config import ID_NAME_HINTS, PipelineConfig

def run_validation_checks(
    config: PipelineConfig,
    df: pd.DataFrame,
    validation_rules: list[dict],
    record_id_col: str | None = None,
    use_llm_indices: bool = False,
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

            # Exact match on known ID hints
            if n in {h.replace("_", "") for h in ID_NAME_HINTS}:
                return 100

            # Ends with or contains an ID hint token
            if any(n.endswith(h.replace("_", "")) or h.replace("_", "") in n
                   for h in ID_NAME_HINTS):
                return 90

            # Ends with "no" or "number" but is NOT a descriptive/contact field
            descriptive = (
                config.relationship_descriptive_terms
                | config.relationship_descriptive_prefixes
            )
            descriptive_norm = {d.replace("_", "") for d in descriptive}
            if n.endswith("no") or n.endswith("number") or "serial" in n:
                if any(d in n for d in descriptive_norm):
                    return 10
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

        best = candidates[0]
        return best["column"] if best["name_score"] > 0 else None

    if record_id_col is None or record_id_col not in df.columns:
        record_id_col = _find_record_identifier_column()

    for rule in validation_rules:
        rule_id = rule.get("rule_id")
        col = rule.get("column") or (rule.get("columns") or [""])[0]
        rule_type = rule.get("type")
        check_params = rule.get("check_params", {})

        try:
            if use_llm_indices:
                raw_indices = rule.get(
                    "failing_record_indices",
                    [],
                )

                if not isinstance(raw_indices, list):
                    raise ValueError(
                        "failing_record_indices must be a list"
                    )

                failing_indices = sorted({
                    int(index)
                    for index in raw_indices
                })

                invalid_indices = [
                    index
                    for index in failing_indices
                    if index not in df.index
                ]

                if invalid_indices:
                    raise ValueError(
                        f"Invalid failing row indices: "
                        f"{invalid_indices[:10]}"
                    )

            else:
                failing_mask = _apply_rule_check(
                    df,
                    rule_type,
                    col,
                    check_params,
                )

                if failing_mask is None:
                    continue

                failing_mask = failing_mask.fillna(False)
                failing_indices = (
                    df.index[failing_mask].tolist()
                )

            n_violations = len(failing_indices)

            label = (
                f"#{rule_id}: {rule.get('rule', '')}"
                if rule_id is not None
                else rule.get("rule", "")
            )

            for idx in failing_indices:
                failed_value = "—"
                failed_col = col

                if check_params.get("col_b") in df.columns and check_params.get("col_b") != col:
                    failed_col = check_params["col_b"]
                    failed_value = df.at[idx, failed_col]
                elif col in df.columns:
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
        seen_pairs = set()
        failed_cols = []
        failed_values = []
        for f in failures:
            col_name = f.get("column") or "—"
            val = str(f.get("failed_value", "—"))
            pair = (col_name, val)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                failed_cols.append(col_name)
                failed_values.append(val)

        record_identifier = "—"
        if record_id_col and record_id_col in df.columns:
            record_identifier = df.at[idx, record_id_col]

        violation_records.append({
            "Row": str(idx+1),
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
            # Coerce to string for comparison to handle int/string type mismatches
            str_values = [str(v) for v in values]
            col_str = df[col].astype(str).str.strip()
            return df[col].notna() & ~col_str.isin(str_values)

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
        cutoff_str = check_params.get("cutoff_date")
        if cutoff_str:
            cutoff = pd.to_datetime(cutoff_str, errors="coerce")
            if pd.isna(cutoff):
                cutoff = pd.Timestamp.now().normalize()
        else:
            cutoff = pd.Timestamp.now().normalize()
        return parsed.notna() & (parsed > cutoff)

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

    elif rule_type == "custom":
        logic = check_params.get("logic")
        if not logic:
            return None

        # Pattern: inverted parseability check — LLM wrote the PASS condition
        # ("not pd.isna(...)") instead of the FAIL condition ("pd.isna(...)")
        _inverted_match = re.match(
            r"^\s*not\s+pd\.isna\(pd\.to_(?:datetime|numeric)\(row\['([^']+)'\].*\)\s*\)\s*$",
            logic.strip()
        )
        if _inverted_match:
            target_col = _inverted_match.group(1)
            if target_col in df.columns:
                if "to_datetime" in logic:
                    parsed = pd.to_datetime(df[target_col], errors="coerce")
                else:
                    parsed = pd.to_numeric(df[target_col], errors="coerce")
                return df[target_col].notna() & parsed.isna()

        # Redirect known bad LLM patterns to correct native handlers
        # rather than accumulating guards — keeps this branch clean

        # Pattern: combined 'free'/sentinel check on numeric column
        # (LLM merges two rules into one custom expression that misfires on floats)
        if col in df.columns and ("'free'" in logic or '"free"' in logic) and "99999" in logic:
            col_str = df[col].astype(str).str.strip().str.lower()
            sentinel_hit = pd.to_numeric(df[col], errors="coerce") == 99999
            free_hit = col_str == "free"
            return df[col].notna() & (free_hit | sentinel_hit)

        # Pattern: numeric parseability or range check on already-numeric column
        # (isdigit/isnumeric/isinstance break on Int64/float64 — extract range and apply natively)
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if any(kw in logic for kw in ("isdigit", "isnumeric", "is_integer", "isinstance")):
                min_match = re.search(r'>=\s*(\d+)', logic)
                max_match = re.search(r'<=\s*(\d+)', logic)
                col_min = int(min_match.group(1)) if min_match else None
                col_max = int(max_match.group(1)) if max_match else None
                if col_min is None and col_max is None:
                    return pd.Series(False, index=df.index)

                numeric = pd.to_numeric(df[col], errors="coerce")
                failing = pd.Series(False, index=df.index)
                if col_min is not None:
                    failing = failing | (df[col].notna() & (numeric < col_min))
                if col_max is not None:
                    failing = failing | (df[col].notna() & (numeric > col_max))
                return failing

        _cols = re.findall(r"row\['([^']+)'\]", logic)
        if len(_cols) == 2 and all(c in df.columns for c in _cols):
            if any(op in logic for op in (">=", "<=", " > ", " < ")):
                    _a = pd.to_datetime(df[_cols[0]], errors="coerce")
                    _b = pd.to_datetime(df[_cols[1]], errors="coerce")
                    _both = _a.notna() & _b.notna()
                    if ">=" in logic:
                        return _both & (_a < _b)
                    elif "<=" in logic:
                        return _both & (_a > _b)
                    elif " > " in logic:
                        return _both & (_a <= _b)
                    elif " < " in logic:
                        return _both & (_a >= _b)
        # Pattern: identity checks that always return True — skip silently
        if " is not pd.NaT" in logic or " is pd.NaT" in logic or \
        (" is not None" in logic and "pd.to_datetime" in logic) or \
        ("pd.to_numeric" in logic and " is not " in logic):
            print(f"    [custom rule] skipping unsafe identity check: {logic}")
            return None

        # General eval — df is available for cross-row checks
        try:
            referenced_cols = [
                c for c in re.findall(r"row\['([^']+)'\]", logic)
                if c in df.columns
            ]

            def _safe_eval(row):
                # If any referenced column is null, this row cannot
                # meaningfully satisfy a cross-column rule — skip it.
                if referenced_cols and any(pd.isna(row[c]) for c in referenced_cols):
                    return False
                try:
                    return bool(eval(logic, {"row": row, "pd": pd, "re": re, "df": df}))
                except Exception:
                    return False

            failing = df.apply(_safe_eval, axis=1)
            return failing
        except Exception as e:
            print(f"    [custom rule] eval failed for '{logic}': {e}")
            return None

    elif rule_type in ("type", "referential"):
        return None 
