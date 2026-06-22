"""Validation rule generation.

Functions were moved from the former LLM generator and pipeline without
adding rule logic.
"""

import json
import re

from ..core.config import PipelineConfig
from ..llm.utils import clean_output
from .prompts import GENERATE_VALIDATION_RULES_PROMPT


def generate_validation_rules(
    config: PipelineConfig,
    llm_generator,
    table_name: str,
    column_summary: list[dict],
    df,  # pandas DataFrame — the actual data
    join_hints: dict[str, list[str]] | None = None,
    n_sample: int = 100,
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
                "intended_data_type": row.get("intended_data_type", row["data_type"]),
                "sample_values": row["sample_values"],
                "observed_distinct_values": row.get("permissible_values"),
                "missing_pct": row.get("profile", {}).get("missing_pct", 0),
                "min": row.get("profile", {}).get("min"),
                "max": row.get("profile", {}).get("max"),
                "upper_fence": row.get("profile", {}).get("upper_fence"),
                "lower_fence": row.get("profile", {}).get("lower_fence"),
                "errors": row.get("errors", []),
                "relationship_role": row.get("relationship_role", ""),
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

    prompt = (
        GENERATE_VALIDATION_RULES_PROMPT.replace("{table_name}", table_name)
        .replace("{evidence_json}", evidence_json)
        .replace(
            "{sample_records_json}",
            sample_records_json,
        )
        .replace(
            "{join_hints_json}",
            join_hints_json,
        )
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
                    and similarity_by_col.get(rule.get("column", ""), "")
                        in ("key_like", "categorical", "discrete_code")
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

            # Re-stamp rule_ids after filtering
            for i, rule in enumerate(rules):
                rule["rule_id"] = i + 1

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
            rules = generate_validation_rules(
                config=config,
                llm_generator=llm_generator,
                table_name=table_name,
                column_summary=table_summary,
                df=df,
                join_hints=join_hints,
                n_sample=config.llm_validation_sample_size,
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

    return all_rules

