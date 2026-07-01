"""
Deterministically identify records that fail generated validation rules.

The LLM generates rule definitions. Pandas applies those rules to the full
dataframe, and Python builds the record-level validation results.
"""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from ..core.config import PipelineConfig
from ..llm.utils import clean_output
from .prompts import APPLY_VALIDATION_RULES_PROMPT
from .results import run_validation_checks

# Guards the shared `failures` dict when batches write results concurrently.
_failures_lock = threading.Lock()

def identify_validation_failures(
    config: PipelineConfig,
    llm_generator,
    table_name: str,
    validation_rules: list[dict],
    df,
    batch_size: int | None = None,
) -> dict[int, list[int]]:
    """Ask the LLM to apply generated rules to every dataframe row.
 
    Batches are dispatched concurrently up to ``config.llm_validation_concurrency``
    (default: 5) using a thread pool so that ``llm_generator.call()`` does not need
    to be async. Results are merged and written to the same cache file as before,
    so ``llm_resume=True`` still works unchanged.
    """
    cfg = config
    batch_size = batch_size or cfg.llm_validation_batch_size
    max_concurrency: int = getattr(cfg, "llm_validation_concurrency", 5)
 
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
 
    # ------------------------------------------------------------------ #
    # Build the column subset and serialisable rule list once, up front.  #
    # ------------------------------------------------------------------ #
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
 
    rules_json = json.dumps(serializable_rules, indent=2, ensure_ascii=False, default=str)
 
    expected_rule_ids = {
        int(rule["rule_id"])
        for rule in validation_rules
        if rule.get("rule_id") is not None
    }
    failures: dict[int, set[int]] = {rule_id: set() for rule_id in expected_rule_ids}
 
    # ------------------------------------------------------------------ #
    # Pre-build every (batch_number, prompt, allowed_indices) triple so   #
    # the async loop only needs to call the LLM.                          #
    # ------------------------------------------------------------------ #
    batches: list[tuple[int, str, set[int]]] = []
    for batch_number, start in enumerate(range(0, len(df), batch_size), start=1):
        stop = min(start + batch_size, len(df))
        batch_df = df.iloc[start:stop][selected_columns].copy()
        batch_df.insert(0, "_row_index", batch_df.index.astype(int))
 
        records_json = json.dumps(
            json.loads(batch_df.to_json(orient="records", force_ascii=False, date_format="iso")),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        prompt = APPLY_VALIDATION_RULES_PROMPT.replace(
            "{rules_json}", rules_json
        ).replace("{records_json}", records_json)
 
        allowed_indices = {int(i) for i in batch_df.index}
        batches.append((batch_number, prompt, allowed_indices))
 
    total_batches = len(batches)
    print(
        f"  [{table_name}] Dispatching {total_batches} validation batches "
        #f"(concurrency={max_concurrency})..."
    )
 
    # ------------------------------------------------------------------ #
    # Async worker: retries for one batch, writes raw file, merges hits.  #
    # ------------------------------------------------------------------ #
    async def _run_batch(
        executor: ThreadPoolExecutor,
        loop: asyncio.AbstractEventLoop,
        batch_number: int,
        prompt: str,
        allowed_indices: set[int],
    ) -> None:
        last_error: Exception | None = None
 
        for attempt in range(1, cfg.llm_max_retries + 1):
            print(
                f"  [{table_name}] batch {batch_number}/{total_batches}, "
                f"attempt {attempt}/{cfg.llm_max_retries}"
            )
            # Run the blocking LLM call in the thread pool so the event
            # loop stays free for other concurrent batches.
            raw: str = await loop.run_in_executor(executor, llm_generator.call, prompt)
 
            raw_path = cache_dir / f"validation_batch_{batch_number}_attempt_{attempt}_raw.txt"
            raw_path.write_text(raw, encoding="utf-8")
 
            try:
                rows = clean_output(raw)
                returned_rule_ids: set[int] = set()
                batch_hits: dict[int, set[int]] = {}
 
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
                        batch_hits.setdefault(rule_id, set()).add(index)
 
                if returned_rule_ids != expected_rule_ids:
                    raise ValueError(
                        "The LLM did not return every rule_id. "
                        f"Expected {sorted(expected_rule_ids)}, "
                        f"returned {sorted(returned_rule_ids)}."
                    )
 
                # Merge under lock — other batches may be writing simultaneously.
                with _failures_lock:
                    for rule_id, indices in batch_hits.items():
                        failures[rule_id].update(indices)
 
                last_error = None
                break
 
            except Exception as error:
                last_error = error
                print(f"  [{table_name}] batch {batch_number} failed: {error}")
 
        if last_error is not None:
            raise ValueError(
                f"{table_name} validation batch {batch_number} failed after "
                f"{cfg.llm_max_retries} attempts. Last error: {last_error}"
            )
 
    # ------------------------------------------------------------------ #
    # Run all batches concurrently, bounded by the semaphore.             #
    # ------------------------------------------------------------------ #
    async def _run_all() -> None:
        loop = asyncio.get_running_loop()
        semaphore = asyncio.Semaphore(max_concurrency)
 
        async def _guarded(batch_number, prompt, allowed_indices):
            async with semaphore:
                await _run_batch(executor, loop, batch_number, prompt, allowed_indices)
 
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            await asyncio.gather(*[
                _guarded(bn, pr, ai) for bn, pr, ai in batches
            ])
 
    asyncio.run(_run_all())
 
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
            pd.RangeIndex(start=0, stop=len(df), step=1)
        ):
            df = df.reset_index(drop=True)
            profile_results[table_name]["df"] = df
            all_dfs[table_name] = df

        for rule in rules:
            params = rule.setdefault("check_params", {})

            if rule.get("type") == "referential":
                rule["type"] = "referential_cross_table"
                params.setdefault("col_a", rule.get("column"))
                params.setdefault("pk_table", params.pop("ref_table", None))
                params.setdefault("pk_col", params.pop("ref_column", None))

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
                    params["sibling_lookup"] = (
                        sibling_df[[join_col, data_col]]
                        .dropna()
                        .drop_duplicates(subset=[join_col])
                        .set_index(join_col)[data_col]
                        .astype(str)
                        .to_dict()
                    )

        print(f"  Asking the LLM to identify failed records for {table_name}...")

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
            if result.get("n_violations") or 0 > 0
        )

        print(
            f"  [{table_name}] "
            f"{len(results['per_rule'])} rules checked by LLM, "
            f"{n_failing} with violations. "
            f"{results['total_failing_records']} unique failing records."
        )

    return all_check_results