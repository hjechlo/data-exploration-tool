"""
LLMDictionaryGenerator — calls the LLM to generate column descriptions and
recommended actions, with retry logic, chunk caching, and output validation.
"""

import json
import re
import os
from pathlib import Path as _Path

from .config import PipelineConfig
from .llm_utils import clean_output, validate_llm_rows, clean_actions, chunks, strip_thinking, semantic_chunks
from .prompts import (
    DATA_DICTIONARY_SYSTEM_PROMPT,
    DATA_DICTIONARY_USER_PROMPT,
    REPORT_SUMMARY_PROMPT,
    DATASET_SUMMARY_PROMPT,
    JOIN_PATH_INTERPRETATION_PROMPT,
    GENERATE_VALIDATION_RULES_PROMPT,
)


class LLMDictionaryGenerator:
    """
    Generates data dictionary descriptions and recommended actions via an LLM.
 
    Parameters
    ----------
    llm_client : AzureLLMEngine
    Engine instance configured with Azure OpenAI credentials.
    config     : PipelineConfig
    """

    _st_model = None

    @classmethod
    def _get_st_model(cls):
        """Lazy-load SentenceTransformer and cache at class level."""
        if cls._st_model is None:
            from sentence_transformers import SentenceTransformer
            cls._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._st_model
 
    def __init__(self, llm_client, config: PipelineConfig):
        self.engine = llm_client   
        self.config = config
 
    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------
 
    def _call_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        """Call the primary Azure deployment (used for summaries and interpretations)."""
        endpoint = (
            self.config.llm_endpoint
            or os.environ.get("ENDPOINT_KIMI", "")
        )
        deployment = (
            self.config.llm_model
            or os.environ.get("DEPLOYMENT_KIMI", "")
        )
        is_native = self.config.llm_is_native_azure

        return self.engine.generate_response(
            endpoint=endpoint,
            deployment_name=deployment,
            system_prompt=system_prompt or "",
            user_payload=prompt,
            is_native_azure=is_native,
        )
    
    def _call_chunk_llm(self, prompt: str, system_prompt: str | None = None) -> str:
        """Call the chunk-specific deployment (per-column JSON generation).

        Falls back to the primary model/endpoint if chunk-specific values are not set.
        """
        endpoint = (
            self.config.llm_chunk_endpoint
            or self.config.llm_endpoint
            or os.environ.get("ENDPOINT_KIMI", "")
        )
        deployment = (
            self.config.llm_chunk_model
            or self.config.llm_model
            or os.environ.get("DEPLOYMENT_KIMI", "")
        )
        is_native = self.config.llm_chunk_is_native_azure

        return self.engine.generate_response(
            endpoint=endpoint,
            deployment_name=deployment,
            system_prompt=system_prompt or "",
            user_payload=prompt,
            is_native_azure=is_native,
        )

    
 
    # ------------------------------------------------------------------
    # Evidence preparation
    # ------------------------------------------------------------------
 
    def prepare_evidence(
        self,
        column_summary: list[dict],
        table_name: str,
        join_hints: dict[str, list[str]] | None = None,
    ) -> list[dict]:
        """Build the list of evidence dicts sent to the LLM."""
        evidence = []
        for row in column_summary:
            key = f"{table_name}.{row['column_name']}"
            profile = row.get("profile") or {}
            n_total = int(profile.get("n_total") or 0)
            missing_count = int(profile.get("missing_count") or 0)
            n_non_missing = max(n_total - missing_count, 1)

            n_distinct = profile.get("n_distinct")
            if n_distinct is None and row.get("permissible_values") is not None:
                n_distinct = len(row.get("permissible_values") or [])

            n_distinct = int(n_distinct or 0)
            unique_ratio = n_distinct / n_non_missing if n_non_missing else 1.0

            # Data-driven category guard:
            # If a string column has a small repeated value set, treat its values as labels,
            # not structural formats. This prevents things like '<=50K' and '>50K'
            # from being interpreted as incompatible fingerprints.
            suppress_abstract_format_for_category = (
                str(row.get("data_type", "")).lower() in {"string", "object", "category"}
                and row.get("permissible_values") is not None
                and n_distinct >= 2
                and unique_ratio <= 0.5
            )

            raw_column_facts = [
                f for f in row.get("column_facts", [])
                if not f.startswith("Most common non-NULL column values are")
            ]

            if suppress_abstract_format_for_category:
                column_facts = [
                    f for f in raw_column_facts
                    if not (
                        isinstance(f, str)
                        and f.startswith(f"Column: {row['column_name']}")
                        and (
                            "Format Distribution:" in f
                            or "Format Analysis:" in f
                            or "Format Uniformity:" in f
                        )
                    )
                ]
            else:
                column_facts = raw_column_facts
            entry = {
                "table_name": table_name,
                "column_name": row["column_name"],
                "data_type": row["data_type"],
                "profile_type": row.get("profile_type"),
                "sample_values": row["sample_values"],
                "observed_distinct_values": row.get("permissible_values"),
                "column_facts": [
                    f for f in row.get("column_facts", [])
                    if not f.startswith("Most common non-NULL column values are")
                ],
                "errors": row["errors"],
            }
            if "_format_analysis" in row:
                format_analysis = row["_format_analysis"]

                def is_length_variation_only(pattern: str) -> bool:
                    """
                    Suppress abstract fingerprint patterns that only describe text length,
                    word count, or simple numeric length.

                    Examples suppressed:
                    - aaaa
                    - aaa aaaa
                    - aaa a/a aaaaaaaa
                    - XXXXXX

                    These are internal fingerprint codes and should not be shown to the LLM.
                    """
                    stripped = re.sub(r"[\s.'\-\/_(),:&]+", "", str(pattern))

                    if not stripped:
                        return False
                    return (
                        all(c == 'a' for c in stripped) or
                        all(c == 'X' for c in stripped)
                    )

                # Filter out letter-only/digit-only patterns from top_formats
                raw_fingerprints = format_analysis.get("format_fingerprints", {})
                raw_top_formats = raw_fingerprints.get("top_formats", [])
                meaningful_formats = [
                    f for f in raw_top_formats
                    if not is_length_variation_only(f.get("pattern", ""))
                ]

                # Only include coercibility if it has real structural differences
                raw_coercibility = format_analysis.get("coercibility", {})
                non_coercible = raw_coercibility.get("non_coercible_formats", [])
                coercible_fmts = raw_coercibility.get("coercible_formats", [])
                all_flagged_length_variation = all(
                    is_length_variation_only(f.get("pattern", ""))
                    for f in (non_coercible + coercible_fmts)
                ) if (non_coercible or coercible_fmts) else True

                # Replace with — suppress it when _assess_coercibility flagged it as precision-only:
                is_numeric_precision_only = (
                    raw_coercibility.get("reason") == "Numeric precision variation only — not a format issue"
                )

                # AFTER: only include fields that have meaningful content
                fa_entry = {}

                uniformity = None if is_numeric_precision_only else format_analysis.get("uniformity_score", 0)
                if uniformity and not row.get("permissible_values"):
                    fa_entry["uniformity_score"] = uniformity

                has_format_issue = (
                    raw_coercibility.get('is_coercible')
                    or bool(raw_coercibility.get('non_coercible_formats'))
                ) and not all_flagged_length_variation

                if (
                    meaningful_formats
                    and has_format_issue
                    and not suppress_abstract_format_for_category
                ):
                    fa_entry["format_fingerprints"] = {"top_formats": meaningful_formats}

                if (
                    not all_flagged_length_variation
                    and raw_coercibility
                    and not suppress_abstract_format_for_category
                ):
                    fa_entry["coercibility"] = raw_coercibility

                # Only include anomalies if there are actual anomalies
                raw_anomalies = format_analysis.get("anomalies", {})
                if raw_anomalies.get("total_anomaly_count", 0) > 0:
                    fa_entry["anomalies"] = {
                        k: v for k, v in raw_anomalies.items()
                        if v and k != "total_anomaly_count"
                    }

                if fa_entry:
                    entry["format_analysis"] = fa_entry
            if join_hints:
                entry["join_hints"] = join_hints.get(key, [])
            relationship_role = row.get("relationship_role", "")
            if relationship_role:
                entry["relationship_role"] = relationship_role
            evidence.append(entry)
        
        return evidence
 
    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------
 
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
        evidence = self.prepare_evidence(column_summary, table_name, join_hints)
 
        chunk_dir = cfg.output_dir / f"{table_name}_llm_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
 
        all_rows: list[dict] = []
        chunk_size = cfg.llm_chunk_size
 
        # Reorder by semantic similarity before chunking so related
        # columns are processed together rather than by file sequence.
        # Only meaningful when chunk_size > 1.
        if chunk_size > 1:
            reordered, sim_matrix, original_order = self._group_by_logic(evidence)
            evidence_chunks =semantic_chunks(
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
                json.dumps(chunk, indent=2, ensure_ascii=False)
            )
            if dataset_description:
                prompt = (
                    f"## Dataset Background\n{dataset_description}\n\n"
                    + prompt
                )
            last_error = None
 
            for attempt in range(1, cfg.llm_max_retries + 1):
                print(f"  [{table_name}] Chunk {i}, attempt {attempt}/{cfg.llm_max_retries}")
                raw = self._call_chunk_llm(prompt, system_prompt=DATA_DICTIONARY_SYSTEM_PROMPT)
 
                # Always save raw output — essential for debugging failures
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
                # All retries exhausted — fall back to one column at a time
                if len(chunk) > 1:
                    print(
                        f"  [{table_name}] Chunk {i}: all retries failed. "
                        f"Falling back to one column at a time..."
                    )
                    chunk_rows = self._generate_one_by_one(
                        chunk, table_name, chunk_dir, chunk_index=i,
                        dataset_description=dataset_description,
                    )
                    all_rows.extend(chunk_rows)
                else:
                    raise ValueError(
                        f"{table_name} chunk {i} failed after {cfg.llm_max_retries} attempts. "
                        f"Last error: {last_error}"
                    )
 
        return all_rows
 
    # ------------------------------------------------------------------
    # Single-column fallback
    # ------------------------------------------------------------------
 
    def _generate_one_by_one(
        self,
        chunk: list[dict],
        table_name: str,
        chunk_dir: _Path,
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
                json.dumps([col_evidence], indent=2, ensure_ascii=False)
            )
            if dataset_description:
                single_prompt = (
                    f"## Dataset Background\n{dataset_description}\n\n"
                    + single_prompt
                )
            last_error = None
 
            for attempt in range(1, cfg.llm_max_retries + 1):
                print(
                    f"    [{table_name}] {col_name}, "
                    f"attempt {attempt}/{cfg.llm_max_retries}"
                )
                raw = self._call_chunk_llm(single_prompt, system_prompt=DATA_DICTIONARY_SYSTEM_PROMPT,)
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
                    f"{table_name}.{col_name} failed after {cfg.llm_max_retries} attempts. "
                    f"Last error: {last_error}"
                )
 
        return results 
 
    # ------------------------------------------------------------------
    # Dataset-level summary
    # ------------------------------------------------------------------
 
    def generate_report_summary(
        self,
        dataset_summaries: dict[str, str],
        all_dictionaries: dict[str, list[dict]],
        minhash_results: dict,
        output_dir,
    ) -> str:
        """Generate a report-level executive summary across all datasets."""
        cfg = self.config
        cached = _Path(output_dir) / "report_summary.txt"
 
        if cfg.llm_resume and cached.exists():
            print("  Report summary: reusing cached result.")
            return cached.read_text(encoding="utf-8")
 
        # Build compact evidence for the report-level prompt
        evidence = {
            "datasets": {
                name: {
                    "summary": dataset_summaries.get(name, ""),
                    "row_count": next(
                        (c["profile"]["n_total"] for c in cols
                         if c.get("profile", {}).get("n_total") is not None),
                        None,
                    ),
                    "total_columns": len(cols),
                }
                for name, cols in all_dictionaries.items()
            },
            "candidate_linkage_paths": [
                {
                    "table_a": jp.get("table_a"),
                    "col_a": jp.get("col_a"),
                    "table_b": jp.get("table_b"),
                    "col_b": jp.get("col_b"),
                    "method": jp.get("method"),
                    "exact_resemblance": jp.get("resemblance"),
                    "shingle_resemblance": jp.get("resemblance_shingle"),
                }
                for jp in minhash_results.get("candidate_join_paths", [])
            ],
            "classified_relationships": [
                {
                    "table_a": jp.get("table_a"),
                    "col_a": jp.get("col_a"),
                    "table_b": jp.get("table_b"),
                    "col_b": jp.get("col_b"),
                    "relationship_type": jp.get("relationship_type"),
                    "relationship_interpretation": jp.get("relationship_interpretation"),
                }
                for jp in minhash_results.get("join_paths", [])
                if jp.get("relationship_type") in {
                    "foreign_key",
                    "one_to_one_key",
                    "shared_join_key",
                    "shared_value_domain",
                }
            ],
        }
 
        prompt = REPORT_SUMMARY_PROMPT.format(
            evidence_json=json.dumps(evidence, indent=2, ensure_ascii=False)
        )
 
        for attempt in range(1, cfg.llm_max_retries + 1):
            print(f"  Report summary, attempt {attempt}/{cfg.llm_max_retries}")
            raw = self._call_llm(prompt)
            text = strip_thinking(raw)
            text = text.strip()
            if text:
                cached.write_text(text, encoding="utf-8")
                print("  Report summary: success.")
                return text
 
        return "Executive summary not available."
 
 
    def generate_dataset_summary(
        self,
        table_name: str,
        column_summary: list[dict],
        join_hints: dict[str, list[str]] | None = None,
        dataset_description: str = "",
    ) -> str:
        """Generate a high-level natural-language overview of the dataset."""
        cfg = self.config
        cache_dir = cfg.output_dir / f"{table_name}_llm_chunks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / "dataset_summary.txt"

        if cfg.llm_resume and cached.exists():
            print(f"  [{table_name}] Dataset summary: reusing cached result.")
            return cached.read_text(encoding="utf-8")

        # Infer total row count from any column that has partial missingness.
        # Falls back to None if all columns are fully populated (missing_pct == 0).
        row_count: int | None = next(
            (row["profile"]["n_total"] for row in column_summary
             if row.get("profile", {}).get("n_total") is not None),
            None,
        )

        # Lean evidence — structure only, no quality signals.
        # Errors, column_facts, and format_analysis are deliberately excluded
        # so the LLM describes what the data IS, not what is wrong with it.
        evidence = {
            "table_name": table_name,
            "row_count": row_count,
            "column_count": len(column_summary),
            "columns": [
                {
                    "column_name": row["column_name"],
                    "data_type": row.get("intended_data_type", row["data_type"]),
                    "sample_values": row["sample_values"][:3],
                    "observed_distinct_values": row.get("permissible_values"),
                }
                for row in column_summary
            ],
        }

        prompt = DATASET_SUMMARY_PROMPT.format(
            table_name=table_name,
            column_evidence_json=json.dumps(evidence, indent=2, ensure_ascii=False),
        )
        if dataset_description:
            prompt = f"## Dataset Background\n{dataset_description}\n\n" + prompt

        for attempt in range(1, cfg.llm_max_retries + 1):
            print(f"  [{table_name}] Dataset summary, attempt {attempt}/{cfg.llm_max_retries}")
            raw = self._call_llm(prompt)
            text = strip_thinking(raw).strip()
            if text:
                cached.write_text(text, encoding="utf-8")
                print(f"  [{table_name}] Dataset summary: success.")
                return text

        return f"Summary not available for {table_name}."
 
 
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
                    kw in a.lower()
                    for a in _actions
                    for kw in ("referential", "integrity", "must exist in", "foreign key")
                )
                if not _has_ref:
                    _parent = None
                    for _hint in row.get("join_hints", []):
                        _m = re.search(r"references primary key candidate ([\w.]+)", _hint)
                        if _m:
                            _parent = _m.group(1)
                            break
                    if _parent:
                        _actions.insert(
                            0,
                            f"[VALIDATE] Enforce referential integrity against {_parent} "
                            f"— all values must reference a valid record in the parent table."
                        )

            merged.append({
                "column_name": row["column_name"],
                "data_type": row["data_type"],
                "intended_data_type": row.get("intended_data_type", row["data_type"]),
                "sample_values": row["sample_values"],
                "permissible_values": row.get("permissible_values"),
                "profile": row.get("profile", {}),
                "errors": row["errors"],
                "description": llm.get("description", ""),
                "recommended_actions": clean_actions(_actions),
            })
        return merged
    
    # ------------------------------------------------------------------
    # Join table interpretation
    # ------------------------------------------------------------------
    def generate_join_interpretation(
        self,
        join_paths: list[dict],
        join_threshold: float,
        shingle_join_threshold: float,
        output_dir,
    ) -> str:
        cfg = self.config
        cached = _Path(output_dir) / "join_interpretation.txt"

        if cfg.llm_resume and cached.exists():
            print("  Join interpretation: reusing cached result.")
            return cached.read_text(encoding="utf-8")

        '''reportable_relationship_types = {
            "foreign_key",
            "one_to_one_key",
            "shared_value_domain",
            "shared_join_key",
        }'''

        clean_join_paths = join_paths

        compact_join_paths = []
        for jp in clean_join_paths:
            rel_type = jp.get("relationship_type")

            if rel_type == "foreign_key":
                fk_table = jp.get("foreign_key_table")
                fk_col = jp.get("foreign_key_column")
                pk_table = jp.get("primary_key_table")
                pk_col = jp.get("primary_key_column")

                compact_join_paths.append({
                    "relationship_type": "foreign_key",
                    "foreign_key": f"{fk_table}.{fk_col}",
                    "primary_key": f"{pk_table}.{pk_col}",
                    "direction": jp.get("primary_direction"),
                    "exact_resemblance": jp.get("resemblance"),
                    "shingle_resemblance": jp.get("resemblance_shingle"),
                    "directional_containment": max(
                        jp.get("coverage_a", 0),
                        jp.get("coverage_b", 0),
                    ),
                    "interpretation": jp.get("relationship_interpretation"),
                })

            elif rel_type == "shared_value_domain":
                compact_join_paths.append({
                    "relationship_type": "shared_value_domain",
                    "column_a": f"{jp.get('table_a')}.{jp.get('col_a')}",
                    "column_b": f"{jp.get('table_b')}.{jp.get('col_b')}",
                    "exact_resemblance": jp.get("resemblance"),
                    "shingle_resemblance": jp.get("resemblance_shingle"),
                    "coverage_a": jp.get("coverage_a"),
                    "coverage_b": jp.get("coverage_b"),
                    "interpretation": jp.get("relationship_interpretation"),
                })

            elif rel_type in {"one_to_one_key", "shared_join_key"}:
                compact_join_paths.append({
                    "relationship_type": rel_type,
                    "column_a": f"{jp.get('table_a')}.{jp.get('col_a')}",
                    "column_b": f"{jp.get('table_b')}.{jp.get('col_b')}",
                    "exact_resemblance": jp.get("resemblance"),
                    "shingle_resemblance": jp.get("resemblance_shingle"),
                    "coverage_a": jp.get("coverage_a"),
                    "coverage_b": jp.get("coverage_b"),
                    "interpretation": jp.get("relationship_interpretation"),
                })
            else:
                compact_join_paths.append({
                    "relationship_type": "candidate_linkage_path",
                    "column_a": f"{jp.get('table_a')}.{jp.get('col_a')}",
                    "column_b": f"{jp.get('table_b')}.{jp.get('col_b')}",
                    "method": jp.get("method"),
                    "exact_resemblance": jp.get("resemblance"),
                    "shingle_resemblance": jp.get("resemblance_shingle"),
                    "interpretation": (
                        "Exploratory MinHash/shingle candidate. Review as a possible "
                        "exact linkage, fuzzy linkage, or blocking field depending on domain context."
                    ),
                })

        evidence = {
            "exact_resemblance_threshold": join_threshold,
            "shingle_resemblance_threshold": shingle_join_threshold,
            "note": (
                "For foreign keys, directional_containment is the main signal. "
                "Exact resemblance can be low when the FK uses only a subset of the parent PK values. "
                "Shingle resemblance is mainly for fuzzy text domains, not numeric IDs."
            ),
            "join_paths": compact_join_paths,
        }

        prompt = JOIN_PATH_INTERPRETATION_PROMPT.format(
            evidence_json=json.dumps(evidence, indent=2, ensure_ascii=False)
        )

        for attempt in range(1, cfg.llm_max_retries + 1):
            print(f"  Join interpretation, attempt {attempt}/{cfg.llm_max_retries}")
            raw = self._call_llm(prompt)
            text = strip_thinking(raw)
            text = text.strip()
            if text:
                cached.write_text(text, encoding="utf-8")
                print("  Join interpretation: success.")
                return text

        return ""
    
    # ------------------------------------------------------------------
    # Validation rule generation
    # ------------------------------------------------------------------
    
    def generate_validation_rules(
        self,
        table_name: str,
        column_summary: list[dict],
        df,  # pandas DataFrame — the actual data
        join_hints: dict[str, list[str]] | None = None,
        n_sample: int = 100,
    ) -> list[dict]:
        """
        Ask the LLM to generate validation rules AND identify failing records.
        Replaces the hardcoded generate_column_validation_rules() in column_analyzer.
        """
        import pandas as pd

        cfg = self.config
        cache_dir = cfg.output_dir / f"{table_name}_llm_chunks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / "validation_rules.json"

        if cfg.llm_resume and cached.exists():
            print(f"  [{table_name}] Validation rules: reusing cached result.")
            with open(cached, encoding="utf-8") as f:
                return json.load(f)

        # Build column evidence (same structure as prepare_evidence but lighter)
        evidence = []
        for row in column_summary:
            evidence.append({
                "column_name": row["column_name"],
                "data_type": row["data_type"],
                "intended_data_type": row.get("intended_data_type", row["data_type"]),
                "sample_values": row["sample_values"],
                "observed_distinct_values": row.get("permissible_values"),
                "missing_pct": row.get("profile", {}).get("missing_pct", 0),
                "errors": row.get("errors", []),
                "relationship_role": row.get("relationship_role", ""),
            })

        # Stratified sample: dirty rows first, then fill with random rows.
        # Ensures the LLM sees actual violations, not just the clean top of the file.
        error_cols = [row["column_name"] for row in column_summary if row.get("errors")]
        if error_cols:
            valid_cols = [c for c in error_cols if c in df.columns]
            dirty_mask = df[valid_cols].isnull().any(axis=1) if valid_cols else pd.Series(False, index=df.index)
            dirty_rows = df[dirty_mask]
            clean_rows = df[~dirty_mask]
            n_dirty = min(len(dirty_rows), n_sample // 2)
            n_clean = min(len(clean_rows), n_sample - n_dirty)
            sample_df = pd.concat([
                dirty_rows.head(n_dirty),
                clean_rows.sample(min(n_clean, len(clean_rows)), random_state=42)
            ]).head(n_sample)
        else:
            sample_df = df.head(n_sample)
        sample_records = json.loads(
            sample_df.astype(str).to_json(orient="records", force_ascii=False)
        )

        prompt = GENERATE_VALIDATION_RULES_PROMPT.format(
            table_name=table_name,
            evidence_json=json.dumps(evidence, indent=2, ensure_ascii=False),
            sample_records_json=json.dumps(sample_records, indent=2, ensure_ascii=False),
            join_hints_json=json.dumps(join_hints or {}, indent=2, ensure_ascii=False),
            n_sample=n_sample,
        )

        for attempt in range(1, cfg.llm_max_retries + 1):
            print(f"  [{table_name}] Validation rules, attempt {attempt}/{cfg.llm_max_retries}")
            raw = self._call_llm(prompt)
            try:
                rules = clean_output(raw)
                if not isinstance(rules, list):
                    raise ValueError("Expected JSON array")
                # Stamp table name on every rule
                for i, rule in enumerate(rules):
                    rule.setdefault("rule_id", i + 1)
                    rule.setdefault("table", table_name)
                    rule.setdefault("columns", [rule.get("column", "")])
                with open(cached, "w", encoding="utf-8") as f:
                    json.dump(rules, f, indent=2, ensure_ascii=False)
                print(f"  [{table_name}] Validation rules: success ({len(rules)} rules).")
                return rules
            except Exception as e:
                print(f"  [{table_name}] Validation rules attempt {attempt} failed: {e}")

        return []
    
    
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _group_by_logic(evidence: list[dict]) -> list[dict]:
        """
        Reorder columns by semantic similarity using sentence-transformers
        + greedy nearest-neighbour traversal.

        Instead of hard clustering (which produces variable-size groups),
        this builds a single ordered sequence where each column is followed
        by its most semantically similar unvisited neighbour. _chunks() then
        splits this sequence into equal-sized chunks, guaranteeing that
        similar columns end up in the same chunk without any group being
        too large or too small.
        """
        if len(evidence) < 2:
            return evidence

        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            print("    [semantic grouping] sentence-transformers not available — skipping.")
            n = len(evidence)
            return evidence, [[1.0] * n for _ in range(n)], list(range(n))
 
        def build_sentence(col: dict) -> str:
            """
            Describe the column in natural language so the embedding model
            can capture its semantic meaning.
 
            Example:
              "date of birth: datetime column with sample values 19801231, 19750101"
              "suburb: string column with sample values sydney, melbourne, brisbane"
            """
            col_name  = col.get("column_name", "").replace("_", " ")
            dtype     = col.get("data_type", "")
            intended  = col.get("intended_data_type", "")
            samples   = ", ".join(str(v) for v in col.get("sample_values", [])[:3])
            type_desc = intended if intended and intended != dtype else dtype

            facts = col.get("column_facts", [])
            is_unique = any("unique" in f.lower() and "all" in f.lower() for f in facts)

            perm = col.get("permissible_values")
            n_distinct = len(perm) if perm else None

            if is_unique:
                role = " unique identifier"
            elif n_distinct is not None and n_distinct <= 5:
                role = f" binary flag with {n_distinct} values"
            elif n_distinct is not None:
                role = f" categorical with {n_distinct} values"
            else:
                role = ""

            return f"{col_name}: {type_desc}{role} column with sample values {samples}"

        n         = len(evidence)
        sentences = [build_sentence(col) for col in evidence]

        model = LLMDictionaryGenerator._get_st_model()
        embeddings = model.encode(sentences, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)

        # Greedy nearest-neighbour traversal
        # Start from column 0, always move to the most similar unvisited column
        visited = [False] * n
        order   = []
        current = 0

        for _ in range(n):
            visited[current] = True
            order.append(current)
            # Find most similar unvisited neighbour
            best_sim  = -1
            best_next = -1
            for j in range(n):
                if not visited[j] and sim_matrix[current][j] > best_sim:
                    best_sim  = sim_matrix[current][j]
                    best_next = j
            current = best_next if best_next != -1 else 0

        reordered = [evidence[i] for i in order]

        # Enforce adjacency: unit-measure columns must sit immediately after
        # their value column so both land in the same LLM chunk.
        _unit_suffix = re.compile(r'^(.+?)(UnitMeasureCode|Unit|Units)$', re.IGNORECASE)
        _col_names = [c["column_name"] for c in reordered]
        for _col in list(_col_names):
            _m = _unit_suffix.match(_col)
            if not _m:
                continue
            _base = _m.group(1).rstrip("_")
            _val_col = next(
                (c for c in _col_names if c.lower() == _base.lower() and c != _col),
                None,
            )
            if _val_col is None:
                continue
            _unit_idx  = next(i for i, c in enumerate(reordered) if c["column_name"] == _col)
            _value_idx = next(i for i, c in enumerate(reordered) if c["column_name"] == _val_col)
            if _unit_idx != _value_idx + 1:
                _entry = reordered.pop(_unit_idx)
                _value_idx = next(i for i, c in enumerate(reordered) if c["column_name"] == _val_col)
                reordered.insert(_value_idx + 1, _entry)
                _col_names = [c["column_name"] for c in reordered]

        print(f"    [semantic ordering] {n} columns reordered: "
            f"{[c['column_name'] for c in reordered]}")

        return reordered, sim_matrix, order