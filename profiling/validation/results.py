"""Validation result building and the legacy Python rule executor.

Functions were moved from the former column analyzer without changing their
validation behavior.
"""

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
            # Tokenize on underscores/camelCase before normalizing, so we match whole tokens
            tokens = set(re.split(r"[^a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", "_", col).lower()))

            hint_tokens = {h.replace("_", "") for h in ID_NAME_HINTS}

            # Exact match on known ID hints (whole token, not substring)
            if tokens & hint_tokens:
                return 100

            # Ends with or contains an ID hint as a distinct token, not a substring
            if any(n.endswith(h.replace("_", "")) for h in ID_NAME_HINTS):
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

            if score == 0 and uniqueness < 0.9:
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

        # Prefer a genuine name-matched identifier. If none exists, fall back
        # to the most identifier-like column by uniqueness, as long as it's
        # reasonably unique (avoids picking a low-cardinality categorical).
        if best["name_score"] > 0:
            return best["column"]

        fallback_candidates = [c for c in candidates if c["uniqueness"] >= 0.9]
        if fallback_candidates:
            fallback_candidates.sort(
                key=lambda x: (x["uniqueness"], -x["avg_len"]),
                reverse=True,
            )
            return fallback_candidates[0]["column"]

        return None

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
            structural_ok = col_str.str.match(pattern, na=False) if pattern else True

            def _is_real_datetime(x: str) -> bool:
                try:
                    pd.to_datetime(x, errors="raise")
                    return True
                except Exception:
                    return False

            calendar_ok = col_str.map(_is_real_datetime)
            return non_missing & ~(structural_ok & calendar_ok)

        if pattern:
            valid = col_str.str.match(pattern, na=False)
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
            a_parsed = pd.to_datetime(df[col_a], errors="coerce",format="mixed")
            b_parsed = pd.to_datetime(df[col_b], errors="coerce",format="mixed")
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
    
    elif rule_type == "datetime_parseable":
        if col not in df.columns:
            return None
        raw = df[col].astype(str).str.strip()
        non_missing = df[col].notna() & (raw != "")
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        return non_missing & parsed.isna()

    elif rule_type == "date_not_future":
        col_a = check_params.get("col_a", col)
        if col_a not in df.columns:
            return None
        parsed = pd.to_datetime(df[col_a], errors="coerce",format="mixed")
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
        dominant_length = check_params.get("dominant_length")

        if col_a not in df.columns or not dominant_length:
            return None

        try:
            dominant_length = int(dominant_length)
        except (TypeError, ValueError):
            return None

        raw = df[col_a].astype(str).str.strip()
        non_missing = df[col_a].notna()

        if country_code:
            country_code = str(country_code)

            # Require the exact international format:
            # +<country code><space><local number>
            canonical_regex = (
                rf"^\+{re.escape(country_code)} "
                rf"\d{{{dominant_length}}}$"
            )
        else:
            # If no country code is provided, require only the local number.
            canonical_regex = rf"^\d{{{dominant_length}}}$"

        valid_format = raw.str.fullmatch(
            canonical_regex,
            na=False,
        )

        # Missing values are handled by the separate not_null rule.
        return non_missing & ~valid_format

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
        if col not in df.columns:
            return None

        sentinel_values = check_params.get("sentinel_values")

        if not isinstance(sentinel_values, list) or not sentinel_values:
            return None

        non_missing = df[col].notna()

        raw = df[col].astype(str).str.strip()
        sentinel_strings = {
            str(value).strip()
            for value in sentinel_values
        }

        string_hit = raw.isin(sentinel_strings)

        numeric = pd.to_numeric(df[col], errors="coerce")
        numeric_sentinels = pd.to_numeric(
            pd.Series(sentinel_values),
            errors="coerce",
        ).dropna()

        numeric_hit = numeric.isin(numeric_sentinels.tolist())

        return non_missing & (string_hit | numeric_hit)

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
    
    elif rule_type == "uniqueness":
        if col not in df.columns:
            return None

        non_missing = df[col].notna()

        return (
            non_missing
            & df[col].duplicated(keep=False)
        )

    elif rule_type == "custom":
        logic = check_params.get("logic")
        if not logic:
            return None

        # Pattern: inverted parseability check — LLM wrote the PASS condition
        # ("not pd.isna(...)") instead of the FAIL condition ("pd.isna(...)")
        _inverted_match = re.match(
            r"^\s*not\s+pd\.isna\(pd\.to_(?:datetime|numeric)\(row\['([^']+)'\][^)]*\)\s*\)\s*$",
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
            
        # Pattern: straight parseability check — pd.isna(pd.to_datetime/numeric(...))
        # Exclude nulls so they are flagged only by not_null.
        _parse_match = re.match(
            r"^\s*pd\.isna\(pd\.to_(datetime|numeric)\(row\['([^']+)'\].*\)\s*\)\s*$",
            logic.strip()
        )
        if _parse_match:
            target_col = _parse_match.group(2)
            if target_col in df.columns:
                if _parse_match.group(1) == "datetime":
                    parsed = pd.to_datetime(df[target_col], errors="coerce", format="mixed")
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
        if (
            len(_cols) == 2
            and all(c in df.columns for c in _cols)
            and not any(pd.api.types.is_numeric_dtype(df[c]) for c in _cols)
            and any(op in logic for op in (">=", "<=", " > ", " < "))
        ):
            _a = pd.to_datetime(df[_cols[0]], errors="coerce", format="mixed")
            _b = pd.to_datetime(df[_cols[1]], errors="coerce", format="mixed")
            if _a.notna().any() and _b.notna().any():
                _both = _a.notna() & _b.notna()
                if ">=" in logic:
                    return _both & (_a >= _b)
                elif "<=" in logic:
                    return _both & (_a <= _b)
                elif " > " in logic:
                    return _both & (_a > _b)
                elif " < " in logic:
                    return _both & (_a < _b)
                    
        # Pattern: row-wise column equality/inequality check
        _eq_match = re.match(
            r"^\s*row\['([^']+)'\]\s*(!=|==)\s*row\['([^']+)'\]\s*$",
            logic.strip()
        )
        if _eq_match:
            col_a, op, col_b = _eq_match.group(1), _eq_match.group(2), _eq_match.group(3)
            if col_a in df.columns and col_b in df.columns:
                if op == "!=":
                    return df[col_a] != df[col_b]
                else:
                    return ~(df[col_a] == df[col_b])
                
        # Pattern: identity checks that always return True — skip silently
        if " is not pd.NaT" in logic or " is pd.NaT" in logic or \
        (" is not None" in logic and "pd.to_datetime" in logic) or \
        ("pd.to_numeric" in logic and " is not " in logic):
            print(f"    [custom rule] skipping unsafe identity check: {logic}")
            return None
        
        # Row-wise expressions — evaluate per row deterministically.
        if "row[" in logic:
            clean_logic = logic.strip()

            # Strip leading import statements (handles both semicolon and newline forms)
            clean_logic = re.sub(
                r"^import\s+\w+(?:\s+as\s+\w+)?\s*[;\n]?\s*",
                "",
                clean_logic,
                flags=re.MULTILINE,
            ).strip()

            # Fix common LLM mistakes
            clean_logic = re.sub(r"\.strip(?!\s*\()", ".strip()", clean_logic)
            clean_logic = re.sub(r"\.lower(?!\s*\()", ".lower()", clean_logic)
            clean_logic = re.sub(r"\.upper(?!\s*\()", ".upper()", clean_logic)
            _err_count = [0]
            def _row_eval(row, _logic=clean_logic):
                try:
                    result = eval(_logic, {
                        "__builtins__": {},
                        "pd": pd,
                        "re": re,
                        "str": str,
                        "int": int,
                        "float": float,
                        "bool": bool,
                        "abs": abs,
                        "len": len,
                        "any": any,
                        "all": all,
                        "row": row,
                        "df": df,
                    })
                    if result is pd.NA:
                        return False
                    return bool(result)
                except SyntaxError:
                    try:
                        local_ns = {
                            "row": row, "df": df, "pd": pd, "re": re,
                            "str": str, "int": int, "float": float,
                            "bool": bool, "abs": abs, "len": len,
                            "any": any, "all": all,
                        }
                        lines = [l.strip() for l in _logic.split(";") if l.strip()]
                        for line in lines[:-1]:
                            exec(line, {"__builtins__": {}}, local_ns)
                        result = eval(lines[-1], {"__builtins__": {}}, local_ns)
                        if result is pd.NA:
                            return False
                        return bool(result)
                    except Exception:
                        _err_count[0] += 1
                        return False
                except Exception:
                    _err_count[0] += 1
                    return False
            _mask = df.apply(_row_eval, axis=1)
            if _err_count[0] > 0:
                print(
                    f"    [custom rule] logic failed on {_err_count[0]} row(s) — "
                    f"deferring entire rule to LLM: {logic[:80]}"
                )
                return None
            return _mask

        # General eval — vectorised df['col'] expression (fast, whole dataframe at once)
        try:
            mask = eval(
                logic,
                {
                    "__builtins__": {},
                    "pd": pd,
                    "re": re,
                    "df": df,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "abs": abs,
                },
            )
        except Exception as exc:
            print(
                f"    [custom rule] vectorised expression failed — "
                f"deferring to LLM: {logic[:80]} ({exc})"
            )
            return None

        # After
        if not isinstance(mask, pd.Series):
            if isinstance(mask, (bool, int, float)):
                mask = pd.Series(bool(mask), index=df.index)
            else:
                raise TypeError(
                    "Custom validation expression must return a pandas Series, "
                    f"but returned {type(mask).__name__}: {logic!r}"
                )

        mask = mask.reindex(df.index, fill_value=False)
        return mask.fillna(False).astype(bool) 
