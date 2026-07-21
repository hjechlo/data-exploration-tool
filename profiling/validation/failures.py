"""
Deterministically identify records that fail generated validation rules.

The LLM generates rule definitions. Pandas applies those rules to the full
dataframe, and Python builds the record-level validation results.
"""
import json
import re
import pandas as pd

from ..core.config import PipelineConfig
from ..llm.utils import clean_output
from .prompts import APPLY_VALIDATION_RULES_PROMPT
from .results import run_validation_checks, _apply_rule_check


def identify_validation_failures(
    config: PipelineConfig,
    llm_generator,
    table_name: str,
    validation_rules: list[dict],
    df,
    batch_size: int | None = None,
) -> dict[int, list[int]]:
    cfg = config
    batch_size = batch_size or cfg.llm_validation_batch_size

    cache_dir = cfg.output_dir / f"{table_name}_llm_chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "validation_llm_failures.json"

    if cfg.llm_resume and cached.exists():
        print(f"  [{table_name}] LLM validation failures: reusing cached result.")
        with open(cached, encoding="utf-8") as f:
            stored = json.load(f)
        return {int(k): [int(i) for i in v] for k, v in stored.items()}

    if not df.index.equals(pd.RangeIndex(start=0, stop=len(df), step=1)):
        raise ValueError(
            "Validation requires a zero-based RangeIndex. "
            "Reset the dataframe index before running validation."
        )

    # ── Deterministic pass ───────────────────────────────────────────────
    # Run every rule that _apply_rule_check can handle in Python.
    # Only rules that return None fall through to the LLM.
    deterministic_results: dict[int, list[int]] = {}
    llm_rules: list[dict] = []

    for rule in validation_rules:
        rule_id = int(rule["rule_id"])
        col = rule.get("column") or (rule.get("columns") or [""])[0]
        mask = _apply_rule_check(df, rule.get("type"), col, rule.get("check_params", {}))
        if mask is not None:
            deterministic_results[rule_id] = sorted(df.index[mask.fillna(False)].tolist())
        else:
            llm_rules.append(rule)

    if not llm_rules:
        with open(cached, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in deterministic_results.items()}, f, indent=2)
        return deterministic_results

    # ── LLM pass (synchronous) ───────────────────────────────────────────
    # Only cross_table_semantic rules and genuinely unevaluable custom
    # expressions reach here. Batching is kept for large dataframes but
    # runs sequentially — determinism matters more than speed here.
    required_columns: set[str] = set()
    for rule in llm_rules:
        for col in rule.get("columns") or []:
            if col in df.columns:
                required_columns.add(col)
        col = rule.get("column")
        if col in df.columns:
            required_columns.add(col)
        params = rule.get("check_params") or {}
        for key in ("col_a", "col_b", "col_c", "join_col"):
            col = params.get(key)
            if col in df.columns:
                required_columns.add(col)
        for col in re.findall(r"row\['([^']+)'\]", params.get("logic") or ""):
            if col in df.columns:
                required_columns.add(col)

    selected_columns = sorted(required_columns)
    serializable_rules = []
    for rule in llm_rules:
        copied = {k: v for k, v in rule.items() if k != "failing_record_indices"}
        params = dict(copied.get("check_params") or {})
        if isinstance(params.get("pk_values"), set):
            params["pk_values"] = sorted(str(v) for v in params["pk_values"])
        copied["check_params"] = params
        serializable_rules.append(copied)

    rules_json = json.dumps(serializable_rules, indent=2, ensure_ascii=False, default=str)
    expected_rule_ids = {int(r["rule_id"]) for r in llm_rules if r.get("rule_id") is not None}
    failures: dict[int, list[int]] = {rule_id: [] for rule_id in expected_rule_ids}

    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"  [{table_name}] Dispatching {total_batches} LLM validation batch(es) sequentially...")

    for batch_num, start in enumerate(range(0, len(df), batch_size), start=1):
        batch_df = df.iloc[start:start + batch_size][selected_columns].copy()
        batch_df.insert(0, "_row_index", batch_df.index.astype(int))
        allowed_indices = set(batch_df.index.astype(int))

        records_json = json.dumps(
            json.loads(batch_df.to_json(orient="records", force_ascii=False, date_format="iso")),
            indent=2, ensure_ascii=False, default=str,
        )
        prompt = (
            APPLY_VALIDATION_RULES_PROMPT
            .replace("{rules_json}", rules_json)
            .replace("{records_json}", records_json)
        )

        last_error = None
        for attempt in range(1, cfg.llm_max_retries + 1):
            print(f"  [{table_name}] batch {batch_num}/{total_batches}, attempt {attempt}/{cfg.llm_max_retries}")
            raw = llm_generator.call(prompt)
            raw_path = cache_dir / f"validation_batch_{batch_num}_attempt_{attempt}_raw.txt"
            raw_path.write_text(raw, encoding="utf-8")
            try:
                rows = clean_output(raw)
                returned_ids: set[int] = set()
                for row in rows:
                    rule_id = int(row["rule_id"])
                    returned_ids.add(rule_id)
                    if rule_id not in failures:
                        raise ValueError(f"Unknown rule_id returned: {rule_id}")
                    indices = row.get("failing_record_indices", [])
                    if not isinstance(indices, list):
                        raise ValueError("failing_record_indices must be a JSON list.")
                    for idx in indices:
                        idx = int(idx)
                        if idx not in allowed_indices:
                            raise ValueError(f"Returned row index {idx} was not in this batch.")
                        if idx not in failures[rule_id]:
                            failures[rule_id].append(idx)
                if returned_ids != expected_rule_ids:
                    raise ValueError(
                        f"Rule ID mismatch. Expected {sorted(expected_rule_ids)}, "
                        f"got {sorted(returned_ids)}."
                    )
                last_error = None
                break
            except Exception as e:
                last_error = e
                print(f"  [{table_name}] batch {batch_num} failed: {e}")

        if last_error is not None:
            raise ValueError(
                f"{table_name} validation batch {batch_num} failed after "
                f"{cfg.llm_max_retries} attempts. Last error: {last_error}"
            )

    result = {**deterministic_results, **{k: sorted(v) for k, v in failures.items()}}
    with open(cached, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in result.items()}, f, indent=2, ensure_ascii=False)
    return result


def validate_tables(
    config: PipelineConfig,
    llm_generator,
    validation_rules: dict[str, list[dict]],
    profile_results: dict,
) -> dict[str, dict]:
    """Ask the LLM to apply generated rules to every record."""
    all_check_results = {}

    all_dfs = {
        table_name: result["df"] for table_name, result in profile_results.items()
    }

    for table_name, rules in validation_rules.items():
        if table_name not in profile_results:
            continue

        df = profile_results[table_name]["df"]

        if not df.index.equals(
            pd.RangeIndex(start=0, stop=len(df), step=1)
        ):
            df = df.reset_index(drop=True)
            profile_results[table_name]["df"] = df
            all_dfs[table_name] = df

        for rule in rules:
            params = rule.setdefault("check_params", {})

            if rule.get("type") in ("referential", "referential_cross_table"):
                rule["type"] = "referential_cross_table"
                params.setdefault("col_a", rule.get("column"))
                if "ref_table" in params:
                    params.setdefault("pk_table", params.pop("ref_table"))
                if "ref_column" in params:
                    params.setdefault("pk_col", params.pop("ref_column"))

            if rule.get("type") == "referential_cross_table":
                parent_table = params.get("pk_table")
                parent_column = params.get("pk_col")
                if (
                    parent_table
                    and parent_column
                    and parent_table in all_dfs
                    and parent_column in all_dfs[parent_table].columns
                ):
                    params["pk_values"] = (
                        all_dfs[parent_table][parent_column]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .str.replace(r"\.0+$", "", regex=True)
                        .drop_duplicates()
                        .tolist()
                    )
            
            if rule.get("type") == "cross_table_semantic":
                sibling_table = params.get("sibling_table")
                sibling_columns = params.get("sibling_join_col"), params.get("sibling_data_col")
                current_join_col = params.get("join_col")

                if (
                    sibling_table
                    and sibling_table in all_dfs
                    and all(c and c in all_dfs[sibling_table].columns for c in sibling_columns)
                    and current_join_col
                    and sibling_columns[0] != sibling_columns[1]
                ):
                    sibling_df = all_dfs[sibling_table]
                    join_col, data_col = sibling_columns
                    _sib = sibling_df[[join_col, data_col]].dropna()
                    _dup_keys = _sib[join_col][_sib[join_col].duplicated(keep=False)]
                    if not _dup_keys.empty:
                        print(
                            f"    [cross_table_semantic] excluding "
                            f"{_dup_keys.nunique()} duplicated join key(s) in "
                            f"{sibling_table}.{join_col} from sibling_lookup."
                        )
                    params["sibling_lookup"] = (
                        _sib[~_sib[join_col].isin(_dup_keys)]
                        .set_index(join_col)[data_col]
                        .astype(str)
                        .to_dict()
                    )

        print(f"  Identifying failed records for {table_name}...")

        failures_by_rule = identify_validation_failures(
            config=config,
            llm_generator=llm_generator,
            table_name=table_name,
            validation_rules=rules,
            df=df,
            batch_size=config.llm_validation_batch_size,
        )

        for rule in rules:
            rule_id = rule.get("rule_id")
            rule["failing_record_indices"] = (
                failures_by_rule.get(int(rule_id), [])
                if rule_id is not None
                else []
            )

        results = run_validation_checks(
            config=config,
            df=df,
            validation_rules=rules,
            use_llm_indices=True,
        )

        all_check_results[table_name] = results

        n_failing = sum(
            1 for result in results["per_rule"]
            if (result.get("n_violations") or 0) > 0
        )

        print(
            f"  [{table_name}] "
            f"{len(results['per_rule'])} rules checked, "
            f"{n_failing} with violations. "
            f"{results['total_failing_records']} unique failing records."
        )

    return all_check_results