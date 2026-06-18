"""LLM-based identification of records that fail generated rules.

Functions were moved from the former LLM generator and pipeline without
adding failure-detection logic.
"""

import json

import pandas as pd

from ..core.config import PipelineConfig
from ..llm.utils import clean_output
from .prompts import APPLY_VALIDATION_RULES_PROMPT
from .results import run_validation_checks


def identify_validation_failures(
    config: PipelineConfig,
    llm_generator,
    table_name: str,
    validation_rules: list[dict],
    df,
    batch_size: int | None = None,
) -> dict[int, list[int]]:
    """Ask the LLM to apply generated rules to every dataframe row."""
    import pandas as pd

    cfg = config
    batch_size = batch_size or cfg.llm_validation_batch_size

    cache_dir = cfg.output_dir / f"{table_name}_llm_chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "validation_llm_failures.json"

    if cfg.llm_resume and cached.exists():
        print(f"  [{table_name}] LLM validation failures: reusing cached result.")
        with open(cached, encoding="utf-8") as f:
            stored = json.load(f)
        return {
            int(rule_id): [int(index) for index in indices]
            for rule_id, indices in stored.items()
        }

    if not df.index.equals(pd.RangeIndex(start=0, stop=len(df), step=1)):
        raise ValueError(
            "Validation requires a zero-based RangeIndex. "
            "Reset the dataframe index before running validation."
        )

    required_columns: set[str] = set()
    for rule in validation_rules:
        for column in rule.get("columns") or []:
            if column in df.columns:
                required_columns.add(column)
        column = rule.get("column")
        if column in df.columns:
            required_columns.add(column)
        params = rule.get("check_params") or {}
        for key in ("col_a", "col_b", "col_c"):
            column = params.get(key)
            if column in df.columns:
                required_columns.add(column)

    selected_columns = sorted(required_columns)

    serializable_rules = []
    for rule in validation_rules:
        copied = {
            key: value for key, value in rule.items() if key != "failing_record_indices"
        }
        params = dict(copied.get("check_params") or {})
        if isinstance(params.get("pk_values"), set):
            params["pk_values"] = sorted(str(v) for v in params["pk_values"])
        copied["check_params"] = params
        serializable_rules.append(copied)

    expected_rule_ids = {
        int(rule["rule_id"])
        for rule in validation_rules
        if rule.get("rule_id") is not None
    }
    failures: dict[int, set[int]] = {rule_id: set() for rule_id in expected_rule_ids}

    for batch_number, start in enumerate(range(0, len(df), batch_size), start=1):
        stop = min(start + batch_size, len(df))
        batch_df = df.iloc[start:stop][selected_columns].copy()
        batch_df.insert(0, "_row_index", batch_df.index.astype(int))

        records = json.loads(
            batch_df.to_json(
                orient="records",
                force_ascii=False,
                date_format="iso",
            )
        )

        rules_json = json.dumps(
            serializable_rules,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        records_json = json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        prompt = APPLY_VALIDATION_RULES_PROMPT.replace(
            "{rules_json}", rules_json
        ).replace("{records_json}", records_json)

        allowed_indices = set(int(index) for index in batch_df.index)
        last_error = None

        for attempt in range(1, cfg.llm_max_retries + 1):
            print(
                f"  [{table_name}] LLM validation batch {batch_number}, "
                f"attempt {attempt}/{cfg.llm_max_retries}"
            )
            raw = llm_generator.call(prompt)
            raw_path = (
                cache_dir / f"validation_batch_{batch_number}_attempt_{attempt}_raw.txt"
            )
            raw_path.write_text(raw, encoding="utf-8")

            try:
                rows = clean_output(raw)
                returned_rule_ids: set[int] = set()

                for row in rows:
                    rule_id = int(row["rule_id"])
                    returned_rule_ids.add(rule_id)

                    if rule_id not in failures:
                        raise ValueError(f"Unknown rule_id returned: {rule_id}")

                    indices = row.get("failing_record_indices", [])
                    if not isinstance(indices, list):
                        raise ValueError("failing_record_indices must be a JSON list.")

                    for index in indices:
                        index = int(index)
                        if index not in allowed_indices:
                            raise ValueError(
                                f"Returned row index {index} was not in this batch."
                            )
                        failures[rule_id].add(index)

                if returned_rule_ids != expected_rule_ids:
                    raise ValueError(
                        "The LLM did not return every rule_id. "
                        f"Expected {sorted(expected_rule_ids)}, "
                        f"returned {sorted(returned_rule_ids)}."
                    )

                last_error = None
                break

            except Exception as error:
                last_error = error
                print(
                    f"  [{table_name}] Validation batch {batch_number} "
                    f"failed: {error}"
                )

        if last_error is not None:
            raise ValueError(
                f"{table_name} validation batch {batch_number} failed after "
                f"{cfg.llm_max_retries} attempts. Last error: {last_error}"
            )

    result = {rule_id: sorted(indices) for rule_id, indices in failures.items()}

    with open(cached, "w", encoding="utf-8") as f:
        json.dump(
            {str(rule_id): indices for rule_id, indices in result.items()},
            f,
            indent=2,
            ensure_ascii=False,
        )

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
            pd.RangeIndex(
                start=0,
                stop=len(df),
                step=1,
            )
        ):
            df = df.reset_index(drop=True)
            profile_results[table_name]["df"] = df
            all_dfs[table_name] = df

        # Normalize cross-table rule schemas and provide the
        # parent-table values needed by the LLM.
        for rule in rules:
            params = rule.setdefault(
                "check_params",
                {},
            )

            if rule.get("type") == "referential":
                rule["type"] = "referential_cross_table"

                params.setdefault(
                    "col_a",
                    rule.get("column"),
                )
                params.setdefault(
                    "pk_table",
                    params.pop(
                        "ref_table",
                        None,
                    ),
                )
                params.setdefault(
                    "pk_col",
                    params.pop(
                        "ref_column",
                        None,
                    ),
                )

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
                        .str.replace(
                            r"\.0+$",
                            "",
                            regex=True,
                        )
                        .drop_duplicates()
                        .tolist()
                    )

        print("  Asking the LLM to identify " f"failed records for {table_name}...")

        failures_by_rule = identify_validation_failures(
            config=config,
            llm_generator=llm_generator,
            table_name=table_name,
            validation_rules=rules,
            df=df,
            batch_size=(config.llm_validation_batch_size),
        )

        for rule in rules:
            rule_id = rule.get("rule_id")

            rule["failing_record_indices"] = (
                failures_by_rule.get(
                    int(rule_id),
                    [],
                )
                if rule_id is not None
                else []
            )

        # Python validates the indices and builds the
        # report payload. It does not execute the rule.
        results = run_validation_checks(
            config=config,
            df=df,
            validation_rules=rules,
            use_llm_indices=True,
        )

        all_check_results[table_name] = results

        n_failing = sum(
            1
            for result in results["per_rule"]
            if result.get(
                "n_violations",
                0,
            )
            > 0
        )

        print(
            f"  [{table_name}] "
            f"{len(results['per_rule'])} "
            f"rules checked by LLM, "
            f"{n_failing} with violations. "
            f"{results['total_failing_records']} "
            f"unique failing records."
        )

    return all_check_results
