"""Column-level LLM data-dictionary generation and merging."""

import json
import os
import re
from pathlib import Path

from ..core.config import PipelineConfig
from .evidence import prepare_dictionary_evidence
from .prompts import DATA_DICTIONARY_SYSTEM_PROMPT, DATA_DICTIONARY_USER_PROMPT
from .semantic_ordering import group_by_logic
from .summaries import generate_dataset_summary
from .utils import clean_actions, clean_output, semantic_chunks, validate_llm_rows


class LLMDictionaryGenerator:
    """Generate and merge column-level dictionary entries through an LLM."""

    def __init__(self, llm_client, config: PipelineConfig):
        self.engine = llm_client
        self.config = config

    def call(self, prompt: str, system_prompt: str | None = None) -> str:
        """Call the primary Azure deployment (used for summaries and interpretations)."""
        endpoint = self.config.llm_endpoint or os.environ.get("ENDPOINT_KIMI", "")
        deployment = self.config.llm_model or os.environ.get("DEPLOYMENT_KIMI", "")
        is_native = self.config.llm_is_native_azure
        return self.engine.generate_response(
            endpoint=endpoint,
            deployment_name=deployment,
            system_prompt=system_prompt or "",
            user_payload=prompt,
            is_native_azure=is_native,
        )

    def generate(
        self,
        column_summary: list[dict],
        table_name: str,
        join_hints: dict[str, list[str]] | None = None,
        dataset_description: str = "",
    ) -> list[dict]:
        """
        Run the full LLM generation for one table with chunking, caching,
        and retry logic.
        """
        cfg = self.config
        evidence = prepare_dictionary_evidence(column_summary, table_name, join_hints)
        chunk_dir = cfg.output_dir / f"{table_name}_llm_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        all_rows: list[dict] = []
        chunk_size = cfg.llm_chunk_size
        if chunk_size > 1:
            reordered, sim_matrix, original_order = group_by_logic(evidence)
            evidence_chunks = semantic_chunks(
                reordered, sim_matrix, original_order, max_chunk_size=chunk_size
            )
        else:
            evidence_chunks = [[col] for col in evidence]
        for i, chunk in enumerate(evidence_chunks, start=1):
            expected_names = [r["column_name"] for r in chunk]
            cached = chunk_dir / f"chunk_{i}_validated.json"
            if cfg.llm_resume and cached.exists():
                print(f"  [{table_name}] Chunk {i}: reusing cached result.")
                with open(cached, encoding="utf-8") as f:
                    rows = json.load(f)
                all_rows.extend(validate_llm_rows(rows, expected_names))
                continue
            prompt = DATA_DICTIONARY_USER_PROMPT.replace(
                "{column_evidence_json}",
                json.dumps(chunk, indent=2, ensure_ascii=False),
            )
            if dataset_description:
                prompt = f"## Dataset Background\n{dataset_description}\n\n" + prompt
            last_error = None
            for attempt in range(1, cfg.llm_max_retries + 1):
                print(
                    f"  [{table_name}] Chunk {i}, attempt {attempt}/{cfg.llm_max_retries}"
                )
                raw = self.call(prompt, system_prompt=DATA_DICTIONARY_SYSTEM_PROMPT)
                raw_path = chunk_dir / f"chunk_{i}_attempt_{attempt}_raw.txt"
                raw_path.write_text(raw, encoding="utf-8")
                try:
                    rows = clean_output(raw)
                    rows = validate_llm_rows(rows, expected_names)
                    with open(cached, "w", encoding="utf-8") as f:
                        json.dump(rows, f, indent=2, ensure_ascii=False)
                    all_rows.extend(rows)
                    print(f"  [{table_name}] Chunk {i}: success.")
                    break
                except Exception as e:
                    last_error = e
                    snippet = raw[:300].replace("\n", " ")
                    print(f"  [{table_name}] Chunk {i} failed: {e}")
                    print(f"  Raw response preview: {snippet!r}")
            else:
                if len(chunk) > 1:
                    print(
                        f"  [{table_name}] Chunk {i}: all retries failed. Falling back to one column at a time..."
                    )
                    chunk_rows = self._generate_one_by_one(
                        chunk,
                        table_name,
                        chunk_dir,
                        chunk_index=i,
                        dataset_description=dataset_description,
                    )
                    all_rows.extend(chunk_rows)
                else:
                    raise ValueError(
                        f"{table_name} chunk {i} failed after {cfg.llm_max_retries} attempts. Last error: {last_error}"
                    )
        return all_rows

    def _generate_one_by_one(
        self,
        chunk: list[dict],
        table_name: str,
        chunk_dir: Path,
        chunk_index: int,
        dataset_description: str = "",
    ) -> list[dict]:
        """
        Process each column in a chunk individually when the full chunk fails.

        Called automatically as a fallback when all retries on a multi-column
        chunk are exhausted — typically caused by reasoning models consuming
        their token budget in the <thinking> block before producing JSON.
        """
        cfg = self.config
        results = []
        for col_evidence in chunk:
            col_name = col_evidence["column_name"]
            col_cached = chunk_dir / f"chunk_{chunk_index}_{col_name}_validated.json"
            if cfg.llm_resume and col_cached.exists():
                print(f"    [{table_name}] {col_name}: reusing cached result.")
                with open(col_cached, encoding="utf-8") as f:
                    row = json.load(f)
                results.append(row)
                continue
            single_prompt = DATA_DICTIONARY_USER_PROMPT.replace(
                "{column_evidence_json}",
                json.dumps([col_evidence], indent=2, ensure_ascii=False),
            )
            if dataset_description:
                single_prompt = (
                    f"## Dataset Background\n{dataset_description}\n\n" + single_prompt
                )
            last_error = None
            for attempt in range(1, cfg.llm_max_retries + 1):
                print(
                    f"    [{table_name}] {col_name}, attempt {attempt}/{cfg.llm_max_retries}"
                )
                raw = self.call(
                    single_prompt, system_prompt=DATA_DICTIONARY_SYSTEM_PROMPT
                )
                raw_path = (
                    chunk_dir
                    / f"chunk_{chunk_index}_{col_name}_attempt_{attempt}_raw.txt"
                )
                raw_path.write_text(raw, encoding="utf-8")
                try:
                    rows = clean_output(raw)
                    rows = validate_llm_rows(rows, [col_name])
                    with open(col_cached, "w", encoding="utf-8") as f:
                        json.dump(rows[0], f, indent=2, ensure_ascii=False)
                    results.append(rows[0])
                    print(f"    [{table_name}] {col_name}: success.")
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    print(f"    [{table_name}] {col_name} failed: {e}")
            else:
                raise ValueError(
                    f"{table_name}.{col_name} failed after {cfg.llm_max_retries} attempts. Last error: {last_error}"
                )
        return results

    def merge(self, column_summary: list[dict], llm_rows: list[dict]) -> list[dict]:
        """Merge LLM-generated fields into the column summary."""
        llm_map = {r["column_name"]: r for r in llm_rows}
        merged = []
        for row in column_summary:
            llm = llm_map.get(row["column_name"], {})
            _role = row.get("relationship_role", "")
            _actions = list(llm.get("recommended_actions", []))
            if _role == "foreign_key":
                _has_ref = any(
                    (
                        kw in a.lower()
                        for a in _actions
                        for kw in (
                            "referential",
                            "integrity",
                            "must exist in",
                            "foreign key",
                        )
                    )
                )
                if not _has_ref:
                    _parent = None
                    for _hint in row.get("join_hints", []):
                        _m = re.search(
                            "references primary key candidate ([\\w.]+)", _hint
                        )
                        if _m:
                            _parent = _m.group(1)
                            break
                    if _parent:
                        _actions.insert(
                            0,
                            f"[VALIDATE] Enforce referential integrity against {_parent} — all values must reference a valid record in the parent table.",
                        )
            merged.append(
                {
                    "column_name": row["column_name"],
                    "data_type": row["data_type"],
                    "intended_data_type": row.get(
                        "intended_data_type", row["data_type"]
                    ),
                    "sample_values": row["sample_values"],
                    "permissible_values": row.get("permissible_values"),
                    "profile": row.get("profile", {}),
                    "errors": row["errors"],
                    "description": llm.get("description", ""),
                    "recommended_actions": clean_actions(_actions),
                }
            )
        return merged


def generate_dictionaries(
    generator: LLMDictionaryGenerator,
    column_summaries: dict,
    minhash_results: dict,
    dataset_descriptions: dict[str, str] | None = None,
    join_hints: dict | None = None,
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Generate and merge data dictionaries and dataset summaries for all tables."""
    from ..analysis.relationship_reporting import build_join_hints

    if join_hints is None:
        join_hints = build_join_hints(minhash_results)
    all_dictionaries = {}
    dataset_summaries = {}
    descriptions = dataset_descriptions or {}
    for table_name, table_summary in column_summaries.items():
        desc = descriptions.get(table_name, "")
        print(f"\n  Generating dictionary for {table_name}...")
        llm_rows = generator.generate(
            column_summary=table_summary,
            table_name=table_name,
            join_hints=join_hints,
            dataset_description=desc,
        )
        all_dictionaries[table_name] = generator.merge(table_summary, llm_rows)
        print(f"  Generating dataset summary for {table_name}...")
        dataset_summaries[table_name] = generate_dataset_summary(
            call_llm=generator.call,
            config=generator.config,
            table_name=table_name,
            column_summary=table_summary,
            join_hints=join_hints,
            dataset_description=desc,
        )
    return (all_dictionaries, dataset_summaries)
    
    