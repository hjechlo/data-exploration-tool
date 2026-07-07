"""Validation rule generation.

Functions were moved from the former LLM generator and pipeline without
adding rule logic.
"""

import json
import re

from ..core.config import BOOLEAN_SPELLINGS, PipelineConfig
from ..llm.utils import clean_output
from .prompts import GENERATE_VALIDATION_RULES_PROMPT


def generate_validation_rules(
    config: PipelineConfig,
    llm_generator,
    table_name: str,
    column_summary: list[dict],
    df,  # pandas DataFrame — the actual data
    join_hints: dict[str, list[str]] | None = None,
    n_sample: int = 300,
    sibling_evidence: dict | None = None,
    dataset_description: str = "",
) -> list[dict]:
    """
    Ask the LLM to generate validation rules for one dataset table.

    Failure identification is performed separately by
    identify_validation_failures().
    """
    import pandas as pd

    cfg = config
    cache_dir = cfg.output_dir / f"{table_name}_llm_chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "validation_rules.json"

    if cfg.llm_resume and cached.exists():
        print(f"  [{table_name}] Validation rules: reusing cached result.")
        with open(cached, encoding="utf-8") as f:
            rules = json.load(f)
        for rule in rules:
            rule.pop("failing_record_indices", None)
        return rules

    # Build column evidence (same structure as dictionary evidence but lighter)
    evidence = []
    for row in column_summary:
        evidence.append(
            {
                "column_name": row["column_name"],
                "data_type": row["data_type"],
                "intended_data_type": row.get(
                    "intended_data_type",
                    row["data_type"],
                ),
                "sample_values": row["sample_values"],
                "observed_distinct_values": row.get("permissible_values"),
                "missing_pct": row.get("profile", {}).get("missing_pct", 0),
                "n_distinct": row.get("profile", {}).get("n_distinct"),
                "n_total": row.get("profile", {}).get("n_total"),
                "min": row.get("profile", {}).get("min"),
                "max": row.get("profile", {}).get("max"),
                "upper_fence": row.get("profile", {}).get("upper_fence"),
                "lower_fence": row.get("profile", {}).get("lower_fence"),
                "errors": row.get("errors", []),
                "relationship_role": row.get("relationship_role", ""),
                "column_facts": row.get("column_facts", []),
                "description": row.get("description", ""),
                "recommended_actions": row.get("recommended_actions", []),
            }
        )

    # Stratified sample: dirty rows first, then fill with random rows.
    # Ensures the LLM sees actual violations, not just the clean top of the file.
    error_cols = [row["column_name"] for row in column_summary if row.get("errors")]
    if error_cols:
        flagged_indices: set = set()
        for row in column_summary:
            flagged_indices.update(row.get("_flagged_indices") or set())

        if flagged_indices:
            dirty_mask = df.index.isin(flagged_indices)
            dirty_rows = df[dirty_mask]
            clean_rows = df[~dirty_mask]
            n_dirty = min(len(dirty_rows), n_sample // 2)
            n_clean = min(len(clean_rows), n_sample - n_dirty)
            sample_df = pd.concat([
                dirty_rows.head(n_dirty),
                clean_rows.sample(min(n_clean, len(clean_rows)), random_state=42),
            ]).head(n_sample)
        else:
            sample_df = df.head(n_sample)
    else:
        sample_df = df.head(n_sample)
    sample_records = json.loads(
        sample_df.astype(str).to_json(orient="records", force_ascii=False)
    )

    evidence_json = json.dumps(
        evidence,
        indent=2,
        ensure_ascii=False,
    )

    sample_records_json = json.dumps(
        sample_records,
        indent=2,
        ensure_ascii=False,
    )

    join_hints_json = json.dumps(
        join_hints or {},
        indent=2,
        ensure_ascii=False,
    )

    sibling_evidence_json = json.dumps(
        {k: v for k, v in (sibling_evidence or {}).items() if k != table_name},
        indent=2,
        ensure_ascii=False,
    )


    prompt = (
        GENERATE_VALIDATION_RULES_PROMPT.replace("{table_name}", table_name)
        .replace("{evidence_json}", evidence_json)
        .replace("{sample_records_json}", sample_records_json)
        .replace("{join_hints_json}", join_hints_json)
        .replace("{sibling_evidence_json}", sibling_evidence_json)
        .replace("{dataset_description}", dataset_description)
        .replace("{n_sample}", str(n_sample))
    )

    for attempt in range(1, cfg.llm_max_retries + 1):
        print(
            f"  [{table_name}] Validation rules, attempt {attempt}/{cfg.llm_max_retries}"
        )
        raw = llm_generator.call(prompt)
        try:
            rules = clean_output(raw)
            if not isinstance(rules, list):
                raise ValueError("Expected JSON array")
            # Stamp table name and rule_id on every rule
            for i, rule in enumerate(rules):
                rule.setdefault("rule_id", i + 1)
                rule.setdefault("table", table_name)
                rule.setdefault("columns", [rule.get("column", "")])
                rule.pop("failing_record_indices", None)

            # ----------------------------------------------------------------
            # Post-generation deterministic overrides
            # ----------------------------------------------------------------

            similarity_by_col = {
                row["column_name"]: row.get("similarity_kind", "")
                for row in column_summary
            }
            col_summary_by_name = {
                row["column_name"]: row for row in column_summary
            }
            col_intended_type = {
                row["column_name"]: row.get("intended_data_type", row["data_type"])
                for row in column_summary
            }
            numeric_storage = {"int64", "int32", "Int64", "float64", "float32"}

            # Pass 1 — Remove range rules on key-like, categorical, and
            # discrete_code columns (identifiers and enumerations have no
            # meaningful continuous domain bound).
            rules = [
                rule for rule in rules
                if not (
                    rule.get("type") == "range"
                    and similarity_by_col.get(rule.get("column", ""), "") == "key_like"
                )
            ]

            # Pass 2 — Remove range rules with implausible bounds.
            # A real domain ceiling (age ≤ 120, rating ≤ 5) produces a bound
            # at most a few multiples of the IQR fence. An invented round number
            # (5000, 50000) is far beyond it. Threshold lives in config.
            fence_multiplier = getattr(cfg, "range_fence_suspicion_multiplier", 3.0)

            filtered: list[dict] = []
            for rule in rules:
                if rule.get("type") != "range":
                    filtered.append(rule)
                    continue
                col = rule.get("column", "")
                col_profile = col_summary_by_name.get(col, {}).get("profile", {})
                llm_max = rule.get("check_params", {}).get("max")
                llm_min = rule.get("check_params", {}).get("min")
                upper_fence = col_profile.get("upper_fence")
                lower_fence = col_profile.get("lower_fence")

                suspicious = False
                if llm_max is not None and upper_fence is not None and upper_fence > 0:
                    suspicious = suspicious or (llm_max > upper_fence * fence_multiplier)
                if llm_min is not None and lower_fence is not None and lower_fence < 0:
                    suspicious = suspicious or (llm_min < lower_fence * fence_multiplier)

                if not suspicious:
                    filtered.append(rule)
            rules = filtered

            # Pass 3 — Replace range rules on code-like columns with format rules.
            # Trigger: numeric storage + intended_data_type == "string" (the
            # profiler already flagged this as a fixed-width code, not a
            # continuous measurement). Format regex derived from the dominant
            # all-digit pattern in _format_analysis — no column-name matching.
            filtered = []
            for rule in rules:
                col = rule.get("column", "")
                is_code_like_range = (
                    rule.get("type") == "range"
                    and col_intended_type.get(col) == "string"
                    and col_summary_by_name.get(col, {}).get("data_type") in numeric_storage
                )
                if not is_code_like_range:
                    filtered.append(rule)
                    continue

                col_row = col_summary_by_name.get(col, {})
                top_formats = (
                    col_row.get("_format_analysis", {})
                    .get("format_fingerprints", {})
                    .get("top_formats", [])
                )
                # Dominant pattern must be all-digit (X+) with >= 50% coverage
                dominant_len = next(
                    (
                        len(fmt["pattern"])
                        for fmt in top_formats
                        if re.fullmatch(r"X+", fmt.get("pattern", ""))
                        and float(fmt.get("percentage", 0)) >= 50
                    ),
                    None,
                )
                if dominant_len:
                    rule["type"] = "format"
                    rule["check_params"] = {"regex": rf"^\d{{{dominant_len}}}$"}
                    rule["rule"] = f"{col} must be a {dominant_len}-digit code"
                    rule["rationale"] = (
                        f"Dominant observed format is {dominant_len} digits "
                        f"(derived from the data distribution). Values of different "
                        f"lengths are likely truncated, padded, or sentinel values."
                    )
                    filtered.append(rule)
                # else: no dominant digit pattern — drop the rule rather than
                # keeping an invented bound
            rules = filtered

            # Pass 4 — Normalise custom digit-length checks into format rules
            # so the format-rule supplement and dedup recognise them.
            for rule in rules:
                if rule.get("type") != "custom":
                    continue
                logic = (rule.get("check_params") or {}).get("logic") or ""
                m = re.search(
                    r"len\(str\(row\['([^']+)'\]\)(?:\.strip\(\))?\)\s*!=\s*(\d+)",
                    logic,
                )
                if m and m.group(1) == rule.get("column"):
                    n = m.group(2)
                    rule["type"] = "format"
                    rule["check_params"] = {"regex": rf"^\d{{{n}}}$"}
                    rule["rule"] = f"{rule['column']} must be a {n}-digit code"
            
            # Pass 5 — Convert custom uniqueness expressions to named type,
            # normalise phone rules, normalise Active boolean rules.
            for rule in rules:
                if rule.get("type") != "custom":
                    continue
                logic = (rule.get("check_params") or {}).get("logic") or ""
                # 5a: df[df['col']==row['col']].shape[0]>1 → uniqueness
                m = re.match(
                    r"df\[df\['([^']+)'\]\s*==\s*row\['([^']+)'\]\]\.shape\[0\]\s*>\s*1",
                    logic.strip()
                )
                if m and m.group(1) == m.group(2) == rule.get("column"):
                    rule["type"] = "uniqueness"
                    rule["check_params"] = {}
                    continue
                # 5b: custom phone regex → phone_validity
                if (
                    rule.get("category") == "per_column"
                    and "re." in logic
                    and any(kw in (rule.get("column") or "").lower()
                            for kw in ("contact", "phone", "tel", "mobile"))
                ):
                    existing = rule.get("check_params") or {}
                    rule["type"] = "phone_validity"
                    rule["check_params"] = {
                        "country_code": existing.get("country_code", "65"),
                        "dominant_length": 8,
                        "valid_first_digits": ["3", "6", "8", "9"],
                    }
                    continue

                # 5d: token-shadowing repair — when one string literal in a
                _logic5d = (rule.get("check_params") or {}).get("logic") or ""
                if "row['" in _logic5d and " in " in _logic5d:
                    _lits = set(re.findall(r"'([^']{2,})'", _logic5d))
                    _lits = {l for l in _lits if not l.startswith("row[")}
                    _shadowed_all = {
                        a for a in _lits
                        if any(a != b and a in b for b in _lits)
                    }
                    # Direct rewrites only for tokens in an `in` membership test
                    _shadowed = {
                        a for a in _shadowed_all
                        if re.search(rf"'{re.escape(a)}'\s+in\s+", _logic5d)
                    }
                    
                    _new_logic = _logic5d
                    _strip_case = re.compile(r'\.(?:upper|lower)\(\)$')
                    for _tok in _shadowed:
                        _pat = re.escape(_tok)
                        _new_logic = re.sub(
                            rf"'{_pat}'\s+in\s+(str\(row\['[^']+'\]\)(?:\.(?:upper|lower)\(\))?|row\['[^']+'\](?:\.(?:upper|lower)\(\))?|[A-Za-z_]\w*)",
                            lambda m, p=_pat: f"bool(re.search(r'(?<![a-z0-9]){p}(?![a-z0-9])', {_strip_case.sub('', m.group(1))}.lower()))",
                            _new_logic,
                        )
                    # Also handle: any(v in X for v in ['bin', ...]) where a
                    # list element is shadowed by another literal in the logic.
                    def _rewrite_any(m):
                        var, operand, listbody = m.group(1), m.group(2), m.group(3)
                        toks = re.findall(r"'([^']+)'", listbody)
                        if not any(t in _shadowed_all for t in toks):
                            return m.group(0)
                        alt = "|".join(re.escape(t) for t in toks)
                        clean = _strip_case.sub("", operand)
                        return (
                            f"bool(re.search(r'(?<![a-z0-9])(?:{alt})(?![a-z0-9])', "
                            f"{clean}.lower()))"
                        )
                    _new_logic = re.sub(
                        r"any\(\s*(\w+)\s+in\s+(str\(row\['[^']+'\]\)(?:\.(?:upper|lower)\(\))?)\s+for\s+\1\s+in\s+\[([^\]]*)\]\s*\)",
                        _rewrite_any,
                        _new_logic,
                    )
                    if _new_logic != _logic5d:
                        rule["check_params"]["logic"] = _new_logic
                # 5e: casing-normalised enumeration in custom logic defeats
                # casing validation (title()/upper()/lower() before not-in).
                # Convert to strict enumeration using the listed values.
                _logic5e = (rule.get("check_params") or {}).get("logic") or ""

                m5e = re.fullmatch(
                    r"(?:pd\.notna\(row\['[^']+'\]\)\s+and\s+)?"
                    r"str\(row\['([^']+)'\]\)(?:\.strip\(\))?\."
                    r"(?:title|upper|lower)\(\)"
                    r"\s+not\s+in\s+\[([^\]]*)\]",
                    _logic5e.strip(),
                )

                if m5e and m5e.group(1) == rule.get("column"):
                    col_name = rule.get("column")
                    col_row = col_summary_by_name.get(col_name, {})

                    mappings: dict[str, str] = {}

                    for action in col_row.get("recommended_actions", []):
                        for source, target in re.findall(
                            r"'([^']+)'\s*→\s*'([^']+)'",
                            action,
                        ):
                            mappings[source.strip()] = target.strip()

                    observed_values = [
                        str(value).strip()
                        for value in (col_row.get("permissible_values") or [])
                    ]

                    if mappings:
                        canonical_values: list[str] = []

                        for observed in observed_values:
                            canonical = mappings.get(observed, observed)

                            if canonical not in canonical_values:
                                canonical_values.append(canonical)

                        if canonical_values:
                            rule["type"] = "enumeration"
                            rule["check_params"] = {
                                "values": canonical_values
                            }
                            rule["rule"] = (
                                f"{col_name} must use the canonical values: "
                                f"{', '.join(canonical_values)}"
                            )

            # Pass 6 — Drop custom per-column rules redundant with an
            # enumeration on the same column (e.g. macOs custom rule
            # when enumeration already covers that column).
            enum_cols = {
                r.get("column") for r in rules if r.get("type") == "enumeration"
            }
            rules = [
                r for r in rules
                if not (
                    r.get("type") == "custom"
                    and r.get("category") == "per_column"
                    and r.get("column") in enum_cols
                    and "row['" in ((r.get("check_params") or {}).get("logic") or "")
                    and "==" in ((r.get("check_params") or {}).get("logic") or "")
                )
            ]

            '''# Pass 6b — Collapse case-duplicate enumeration values, keeping
            # the casing most frequent in the data (canonical form).
            for rule in rules:
                if rule.get("type") != "enumeration":
                    continue
                _col = rule.get("column")
                _vals = (rule.get("check_params") or {}).get("values") or []
                if _col not in df.columns or not _vals:
                    continue
                _counts = df[_col].dropna().astype(str).str.strip().value_counts()
                _by_fold: dict[str, list[str]] = {}
                for v in _vals:
                    _by_fold.setdefault(str(v).strip().lower(), []).append(str(v))
                rule["check_params"]["values"] = [
                    _group[0] if len(_group) == 1
                    else max(_group, key=lambda g: int(_counts.get(g, 0)))
                    for _group in _by_fold.values()
                ]'''
            
            # Pass 6c — Canonicalise boolean-intent enumerations regardless
            # of the rule type the LLM chose.
            for rule in rules:
                col_name = rule.get("column")

                if (
                    rule.get("category") == "per_column"
                    and rule.get("type") in ("enumeration", "custom")
                    and col_name in df.columns
                    and col_intended_type.get(col_name) == "bool"
                ):
                    counts = (
                        df[col_name]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .value_counts()
                    )

                    truthy = [
                        value
                        for value in counts.index
                        if BOOLEAN_SPELLINGS.get(str(value).strip().lower()) is True
                    ]

                    falsy = [
                        value
                        for value in counts.index
                        if BOOLEAN_SPELLINGS.get(str(value).strip().lower()) is False
                    ]

                    if truthy and falsy:
                        canonical_true = max(
                            truthy,
                            key=lambda value: int(counts.get(value, 0)),
                        )
                        canonical_false = max(
                            falsy,
                            key=lambda value: int(counts.get(value, 0)),
                        )

                        rule["type"] = "enumeration"
                        rule["check_params"] = {
                            "values": [
                                str(canonical_true),
                                str(canonical_false),
                            ]
                        }
                        rule["rule"] = (
                            f"{col_name} must use the dominant boolean representations "
                            f"{canonical_true} or {canonical_false}"
                        )

            '''# Pass 7 — Drop equality/difference rules between columns
            # that are always equal in the sample (near-duplicate noise).
            rules = [
                r for r in rules
                if not (
                    r.get("type") == "custom"
                    and any(op in ((r.get("check_params") or {}).get("logic") or "")
                            for op in ("!=", "abs("))
                    and len(re.findall(
                        r"row\['([^']+)'\]",
                        (r.get("check_params") or {}).get("logic") or ""
                    )) == 2
                    and all(
                        c in df.columns
                        for c in re.findall(
                            r"row\['([^']+)'\]",
                            (r.get("check_params") or {}).get("logic") or ""
                        )
                    )
                    and df[
                        re.findall(r"row\['([^']+)'\]",
                            (r.get("check_params") or {}).get("logic") or "")[0]
                    ].equals(df[
                        re.findall(r"row\['([^']+)'\]",
                            (r.get("check_params") or {}).get("logic") or "")[1]
                    ])
                )
            ]'''
                    
            # Re-stamp rule_ids after filtering
            for i, rule in enumerate(rules):
                rule["rule_id"] = i + 1
                rule.setdefault("table", table_name)
                rule.setdefault("columns", [rule.get("column", "")])
                rule.setdefault("check_params", {})

                '''if not rule.get("column"):
                    raise ValueError(
                        f"Rule #{rule['rule_id']} is missing its column."
                    )'''

                if not rule.get("type"):
                    raise ValueError(
                        f"Rule #{rule['rule_id']} is missing its type."
                    )

            with open(cached, "w", encoding="utf-8") as f:
                json.dump(rules, f, indent=2, ensure_ascii=False)
            print(f"  [{table_name}] Validation rules: success ({len(rules)} rules).")
            return rules
        except Exception as e:
            print(f"  [{table_name}] Validation rules attempt {attempt} failed: {e}")

    return []


def generate_rules_for_tables(
    config: PipelineConfig,
    llm_generator,
    column_summaries: dict,
    minhash_results: dict,
    profile_results: dict | None = None,
    dataset_descriptions: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """
    Generate validation rules for all tables via the LLM.

    Failure identification is performed later by
    validate_tables().
    Cross-table referential integrity rules are supplemented from
    detected foreign-key relationships.
    """
    all_rules = {}

    for table_name, table_summary in column_summaries.items():
        print(f"  Generating validation rules for {table_name}...")

        df = profile_results[table_name]["df"] if profile_results else None

        if df is not None:
            # Build join hints from MinHash relationships — all typed
            # relationships, not just FK
            join_hints = {row["column_name"]: [] for row in table_summary}
            for jp in minhash_results.get("join_paths", []):
                rel_type = jp.get("relationship_type", "")

                if rel_type == "foreign_key":
                    fk_table = jp.get("foreign_key_table")
                    fk_col = jp.get("foreign_key_column")
                    pk_table = jp.get("primary_key_table")
                    pk_col = jp.get("primary_key_column")
                    if fk_table == table_name and fk_col in join_hints:
                        join_hints[fk_col].append(f"FK → {pk_table}.{pk_col}")
                    if pk_table == table_name and pk_col in join_hints:
                        join_hints[pk_col].append(f"PK ← {fk_table}.{fk_col}")

                elif rel_type == "one_to_one_key":
                    col_a, col_b = jp.get("col_a"), jp.get("col_b")
                    t_a, t_b = jp.get("table_a"), jp.get("table_b")
                    if t_a == table_name and col_a in join_hints:
                        join_hints[col_a].append(f"one-to-one key with {t_b}.{col_b}")
                    if t_b == table_name and col_b in join_hints:
                        join_hints[col_b].append(f"one-to-one key with {t_a}.{col_a}")

                elif rel_type == "shared_value_domain":
                    col_a, col_b = jp.get("col_a"), jp.get("col_b")
                    t_a, t_b = jp.get("table_a"), jp.get("table_b")
                    if t_a == table_name and col_a in join_hints:
                        join_hints[col_a].append(
                            f"shared value domain with {t_b}.{col_b} — consistency check candidate"
                        )
                    if t_b == table_name and col_b in join_hints:
                        join_hints[col_b].append(
                            f"shared value domain with {t_a}.{col_a} — consistency check candidate"
                        )
            
            sibling_evidence = {
                tname: [
                    {
                        "column_name": row["column_name"],
                        "data_type": row["data_type"],
                        "sample_values": row["sample_values"],
                    }
                    for row in tsummary
                ]
                for tname, tsummary in column_summaries.items()
            }

            rules = generate_validation_rules(
                config=config,
                llm_generator=llm_generator,
                table_name=table_name,
                column_summary=table_summary,
                df=df,
                join_hints=join_hints,
                n_sample=config.llm_validation_sample_size,
                sibling_evidence=sibling_evidence,
                dataset_description=(dataset_descriptions or {}).get(table_name, ""),
            )
        else:
            rules = []

        all_rules[table_name] = rules
        print(f"  [{table_name}] Validation rules: {len(rules)} rules.")

    # Cross-table referential integrity — appended from FK join paths
    all_dfs = {tn: profile_results[tn]["df"] for tn in profile_results or {}}
    for jp in minhash_results.get("join_paths", []):
        if jp.get("relationship_type") != "foreign_key":
            continue
        fk_table = jp.get("foreign_key_table")
        fk_col = jp.get("foreign_key_column")
        pk_table = jp.get("primary_key_table")
        pk_col = jp.get("primary_key_column")
        if not all([fk_table, fk_col, pk_table, pk_col]):
            continue
        if fk_table not in all_rules or pk_table not in all_dfs:
            continue
        # Skip if LLM already generated a referential rule for this column
        already_has_referential = any(
            r.get("type") in ("referential", "referential_cross_table")
            and r.get("column") == fk_col
            for r in all_rules[fk_table]
        )
        if already_has_referential:
            continue
        rule_id = len(all_rules[fk_table]) + 1
        all_rules[fk_table].append(
            {
                "rule_id": rule_id,
                "table": fk_table,
                "column": fk_col,
                "columns": [fk_col],
                "category": "cross_table",
                "type": "referential_cross_table",
                "rule": f"{fk_table}.{fk_col} must exist in {pk_table}.{pk_col}",
                "rationale": "Foreign key relationship detected by MinHash analysis.",
                "check_params": {
                    "col_a": fk_col,
                    "pk_table": pk_table,
                    "pk_col": pk_col,
                },
            }
        )

    # Referential integrity in both directions for confirmed one-to-one keys.
    for jp in minhash_results.get("join_paths", []):
        if jp.get("relationship_type") != "one_to_one_key":
            continue

        table_a = jp.get("table_a")
        col_a = jp.get("col_a")
        table_b = jp.get("table_b")
        col_b = jp.get("col_b")

        if not all([table_a, col_a, table_b, col_b]):
            continue

        directions = [
            (table_a, col_a, table_b, col_b),
            (table_b, col_b, table_a, col_a),
        ]

        for child_table, child_col, parent_table, parent_col in directions:
            if child_table not in all_rules or parent_table not in all_dfs:
                continue

            already_exists = any(
                rule.get("type") in (
                    "referential",
                    "referential_cross_table",
                )
                and rule.get("column") == child_col
                and (
                    rule.get("check_params", {}).get("pk_table")
                    or rule.get("check_params", {}).get("ref_table")
                ) == parent_table
                and (
                    rule.get("check_params", {}).get("pk_col")
                    or rule.get("check_params", {}).get("ref_column")
                ) == parent_col
                for rule in all_rules[child_table]
            )

            if already_exists:
                continue

            all_rules[child_table].append(
                {
                    "rule_id": len(all_rules[child_table]) + 1,
                    "table": child_table,
                    "column": child_col,
                    "columns": [child_col],
                    "category": "cross_table",
                    "type": "referential_cross_table",
                    "rule": (
                        f"Every {child_col} in {child_table} must exist in "
                        f"{parent_table}.{parent_col}"
                    ),
                    "rationale": (
                        "Confirmed one-to-one key relationship; unmatched values "
                        "break the relationship."
                    ),
                    "check_params": {
                        "col_a": child_col,
                        "pk_table": parent_table,
                        "pk_col": parent_col,
                    },
                }
            )

    # Uniqueness — appended for columns participating in one-to-one key
    # join paths (a duplicate on either side breaks the relationship).
    for jp in minhash_results.get("join_paths", []):
        if jp.get("relationship_type") != "one_to_one_key":
            continue
        for tname, cname in (
            (jp.get("table_a"), jp.get("col_a")),
            (jp.get("table_b"), jp.get("col_b")),
        ):
            if not tname or not cname or tname not in all_rules:
                continue
            if any(
                r.get("type") == "uniqueness" and r.get("column") == cname
                for r in all_rules[tname]
            ):
                continue
            all_rules[tname].append({
                "rule_id": len(all_rules[tname]) + 1,
                "table": tname,
                "column": cname,
                "columns": [cname],
                "category": "per_column",
                "type": "uniqueness",
                "rule": f"{cname} must be unique within {tname} (one-to-one join key)",
                "rationale": "One-to-one key relationship detected by MinHash analysis; duplicate values break the join.",
                "check_params": {},
            })
    
    # # Uniqueness — appended for near-unique columns participating in any
    # # detected cross-table relationship. A dirty candidate key (e.g. a
    # # duplicated account number) cannot be classified as one_to_one_key
    # # by MinHash precisely because its duplicates break the join, so it
    # # is caught here via its observed uniqueness ratio instead.
    # _min_ratio = getattr(config, "uniqueness_rule_min_ratio", 0.9)
    # for jp in minhash_results.get("join_paths", []):
    #     _pairs = (
    #         (jp.get("table_a") or jp.get("foreign_key_table"),
    #          jp.get("col_a") or jp.get("foreign_key_column")),
    #         (jp.get("table_b") or jp.get("primary_key_table"),
    #          jp.get("col_b") or jp.get("primary_key_column")),
    #     )
    #     for tname, cname in _pairs:
    #         if not tname or not cname or tname not in all_rules:
    #             continue
    #         if any(
    #             r.get("type") == "uniqueness" and r.get("column") == cname
    #             for r in all_rules[tname]
    #         ):
    #             continue
    #         _tdf = all_dfs.get(tname)
    #         if _tdf is None or cname not in _tdf.columns:
    #             continue
    #         _non_null = _tdf[cname].dropna()
    #         if len(_non_null) == 0:
    #             continue
    #         _ratio = _non_null.nunique(dropna=True) / len(_non_null)
    #         if _ratio < _min_ratio:
    #             continue
    #         all_rules[tname].append({
    #             "rule_id": len(all_rules[tname]) + 1,
    #             "table": tname,
    #             "column": cname,
    #             "columns": [cname],
    #             "category": "per_column",
    #             "type": "uniqueness",
    #             "rule": f"{cname} must be unique within {tname}",
    #             "rationale": (
    #                 f"Column participates in a detected cross-table relationship "
    #                 f"and is {_ratio:.1%} distinct; treated as a candidate key, "
    #                 f"duplicates flagged for review."
    #             ),
    #             "check_params": {},
    #         })

    # # Uniqueness — also inject for columns used as sibling_data_col in
    # # cross_table_semantic rules, which MinHash may not have linked directly.
    # for tname, trules in all_rules.items():
    #     _tdf = all_dfs.get(tname)
    #     if _tdf is None:
    #         continue
    #     for rule in trules:
    #         if rule.get("type") != "cross_table_semantic":
    #             continue
    #         cname = rule.get("column")
    #         if not cname or cname not in _tdf.columns:
    #             continue
    #         if any(
    #             r.get("type") == "uniqueness" and r.get("column") == cname
    #             for r in all_rules[tname]
    #         ):
    #             continue
    #         _non_null = _tdf[cname].dropna()
    #         if len(_non_null) == 0:
    #             continue
    #         _ratio = _non_null.nunique(dropna=True) / len(_non_null)
    #         if _ratio < _min_ratio or _ratio >= 1.0:
    #             continue
    #         all_rules[tname].append({
    #             "rule_id": len(all_rules[tname]) + 1,
    #             "table": tname,
    #             "column": cname,
    #             "columns": [cname],
    #             "category": "per_column",
    #             "type": "uniqueness",
    #             "rule": f"{cname} must be unique within {tname}",
    #             "rationale": (
    #                 f"Column is referenced in a cross-table semantic rule "
    #                 f"and is {_ratio:.1%} distinct; treated as a candidate key, "
    #                 f"duplicates flagged for review."
    #             ),
    #             "check_params": {},
    #         })

    for table_name, table_summary in column_summaries.items():
        for col_row in table_summary:
            col = col_row["column_name"]
            if not (
                col_row.get("intended_data_type") == "string"
                and col_row.get("data_type") in {"int64", "int32", "Int64", "float64", "float32"}
            ):
                continue

            # Skip if a format rule already exists for this column
            if any(
                r.get("type") == "format" and r.get("column") == col
                for r in all_rules.get(table_name, [])
            ):
                continue

            top_formats = (
                col_row.get("_format_analysis", {})
                .get("format_fingerprints", {})
                .get("top_formats", [])
            )
            dominant_len = next(
                (
                    len(fmt["pattern"])
                    for fmt in top_formats
                    if re.fullmatch(r"X+", fmt.get("pattern", ""))
                    and float(fmt.get("percentage", 0)) >= 50
                ),
                None,
            )
            if not dominant_len:
                continue

            rule_id = len(all_rules[table_name]) + 1
            all_rules[table_name].append({
                "rule_id": rule_id,
                "table": table_name,
                "column": col,
                "columns": [col],
                "category": "per_column",
                "type": "format",
                "rule": f"{col} must be a {dominant_len}-digit code",
                "rationale": (
                    f"Dominant observed format is {dominant_len} digits "
                    f"(derived from data distribution). Values of different "
                    f"lengths are likely truncated, padded, or sentinel values."
                ),
                "check_params": {"regex": rf"^\d{{{dominant_len}}}$"},
            })

    # # Consistency supplement — near-duplicate column pairs within a table
    # # (detected by MinHash) must agree row-by-row; divergent rows are the
    # # violation. Pairs that are exactly equal produce no rule (no dirt).
    # for dc in minhash_results.get("duplicate_columns", []):
    #     tname = dc.get("table_a")
    #     if not tname or dc.get("table_b") != tname or tname not in all_rules:
    #         continue
    #     col_a, col_b = dc.get("col_a"), dc.get("col_b")
    #     _tdf = all_dfs.get(tname)
    #     if (
    #         not col_a or not col_b
    #         or _tdf is None
    #         or col_a not in _tdf.columns
    #         or col_b not in _tdf.columns
    #         or _tdf[col_a].equals(_tdf[col_b])
    #     ):
    #         continue
    #     if any(
    #         r.get("type") == "custom"
    #         and set(re.findall(r"row\['([^']+)'\]",
    #                 (r.get("check_params") or {}).get("logic") or ""))
    #             == {col_a, col_b}
    #         for r in all_rules[tname]
    #     ):
    #         continue
    #     all_rules[tname].append({
    #         "rule_id": len(all_rules[tname]) + 1,
    #         "table": tname,
    #         "column": col_a,
    #         "columns": [col_a, col_b],
    #         "category": "cross_column",
    #         "type": "custom",
    #         "rule": f"{col_a} and {col_b} must contain identical values",
    #         "rationale": (
    #             f"MinHash detected {col_a} and {col_b} as near-duplicate "
    #             f"columns; rows where they diverge indicate an inconsistency."
    #         ),
    #         "check_params": {
    #             "logic": f"str(row['{col_a}']) != str(row['{col_b}'])"
    #         },
    #     })

    # Sentinel supplement
    _sentinel_pattern = re.compile(r"\[([^\]]+)\] is a statistical outlier.*sentinel-coded")
    for table_name, table_summary in column_summaries.items():
        for col_row in table_summary:
            col = col_row["column_name"]
            sentinel_values = []
            for error in col_row.get("errors", []):
                m = _sentinel_pattern.match(error)
                if m:
                    raw_val = m.group(1).strip()
                    try:
                        numeric_val = float(raw_val)
                        sentinel_values.append(numeric_val)
                        if numeric_val == int(numeric_val):
                            sentinel_values.append(str(int(numeric_val)))
                    except ValueError:
                        sentinel_values.append(raw_val)
            if not sentinel_values:
                continue
            if any(
                r.get("type") == "sentinel_check" and r.get("column") == col
                for r in all_rules.get(table_name, [])
            ):
                continue
            all_rules[table_name].append({
                "rule_id": len(all_rules[table_name]) + 1,
                "table": table_name,
                "column": col,
                "columns": [col],
                "category": "per_column",
                "type": "sentinel_check",
                "rule": f"{col} must not contain sentinel placeholder values",
                "rationale": (
                    f"Profiler detected {sentinel_values[:3]} as statistically outlying "
                    f"sentinel-coded value(s) in {col}."
                ),
                "check_params": {"sentinel_values": sentinel_values},
            })
    return all_rules


