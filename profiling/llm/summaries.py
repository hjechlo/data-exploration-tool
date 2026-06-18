"""Dataset-, report-, and relationship-level narrative generation."""

import json
from pathlib import Path

from ..core.config import PipelineConfig
from .prompts import (
    DATASET_SUMMARY_PROMPT,
    JOIN_PATH_INTERPRETATION_PROMPT,
    REPORT_SUMMARY_PROMPT,
)
from .utils import strip_thinking


def generate_report_summary(
    call_llm,
    config: PipelineConfig,
    dataset_summaries: dict[str, str],
    all_dictionaries: dict[str, list[dict]],
    minhash_results: dict,
    output_dir,
) -> str:
    """Generate a report-level executive summary across all datasets."""
    cfg = config
    cached = Path(output_dir) / "report_summary.txt"
    if cfg.llm_resume and cached.exists():
        print("  Report summary: reusing cached result.")
        return cached.read_text(encoding="utf-8")
    evidence = {
        "datasets": {
            name: {
                "summary": dataset_summaries.get(name, ""),
                "row_count": next(
                    (
                        c["profile"]["n_total"]
                        for c in cols
                        if c.get("profile", {}).get("n_total") is not None
                    ),
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
            if jp.get("relationship_type")
            in {
                "foreign_key",
                "one_to_one_key",
                #"shared_join_key",
                "shared_value_domain",
            }
        ],
    }
    prompt = REPORT_SUMMARY_PROMPT.format(
        evidence_json=json.dumps(evidence, indent=2, ensure_ascii=False)
    )
    for attempt in range(1, cfg.llm_max_retries + 1):
        print(f"  Report summary, attempt {attempt}/{cfg.llm_max_retries}")
        raw = call_llm(prompt)
        text = strip_thinking(raw)
        text = text.strip()
        if text:
            cached.write_text(text, encoding="utf-8")
            print("  Report summary: success.")
            return text
    return "Executive summary not available."


def generate_dataset_summary(
    call_llm,
    config: PipelineConfig,
    table_name: str,
    column_summary: list[dict],
    join_hints: dict[str, list[str]] | None = None,
    dataset_description: str = "",
) -> str:
    """Generate a high-level natural-language overview of the dataset."""
    cfg = config
    cache_dir = cfg.output_dir / f"{table_name}_llm_chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "dataset_summary.txt"
    if cfg.llm_resume and cached.exists():
        print(f"  [{table_name}] Dataset summary: reusing cached result.")
        return cached.read_text(encoding="utf-8")
    row_count: int | None = next(
        (
            row["profile"]["n_total"]
            for row in column_summary
            if row.get("profile", {}).get("n_total") is not None
        ),
        None,
    )
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
        print(
            f"  [{table_name}] Dataset summary, attempt {attempt}/{cfg.llm_max_retries}"
        )
        raw = call_llm(prompt)
        text = strip_thinking(raw).strip()
        if text:
            cached.write_text(text, encoding="utf-8")
            print(f"  [{table_name}] Dataset summary: success.")
            return text
    return f"Summary not available for {table_name}."


def generate_join_interpretation(
    call_llm,
    config: PipelineConfig,
    join_paths: list[dict],
    join_threshold: float,
    shingle_join_threshold: float,
    output_dir,
) -> str:
    """Generate a narrative interpretation of detected relationship paths."""
    cfg = config
    cached = Path(output_dir) / "join_interpretation.txt"
    if cfg.llm_resume and cached.exists():
        print("  Join interpretation: reusing cached result.")
        return cached.read_text(encoding="utf-8")
    clean_join_paths = join_paths
    compact_join_paths = []
    for jp in clean_join_paths:
        rel_type = jp.get("relationship_type")
        if rel_type == "foreign_key":
            fk_table = jp.get("foreign_key_table")
            fk_col = jp.get("foreign_key_column")
            pk_table = jp.get("primary_key_table")
            pk_col = jp.get("primary_key_column")
            compact_join_paths.append(
                {
                    "relationship_type": "foreign_key",
                    "foreign_key": f"{fk_table}.{fk_col}",
                    "primary_key": f"{pk_table}.{pk_col}",
                    "direction": jp.get("primary_direction"),
                    "exact_resemblance": jp.get("resemblance"),
                    "shingle_resemblance": jp.get("resemblance_shingle"),
                    "directional_containment": max(
                        jp.get("coverage_a", 0), jp.get("coverage_b", 0)
                    ),
                    "interpretation": jp.get("relationship_interpretation"),
                }
            )
        elif rel_type == "shared_value_domain":
            compact_join_paths.append(
                {
                    "relationship_type": "shared_value_domain",
                    "column_a": f"{jp.get('table_a')}.{jp.get('col_a')}",
                    "column_b": f"{jp.get('table_b')}.{jp.get('col_b')}",
                    "exact_resemblance": jp.get("resemblance"),
                    "shingle_resemblance": jp.get("resemblance_shingle"),
                    "coverage_a": jp.get("coverage_a"),
                    "coverage_b": jp.get("coverage_b"),
                    "interpretation": jp.get("relationship_interpretation"),
                }
            )
        elif rel_type in {"one_to_one_key", 
                          #"shared_join_key"
                          }:
            compact_join_paths.append(
                {
                    "relationship_type": rel_type,
                    "column_a": f"{jp.get('table_a')}.{jp.get('col_a')}",
                    "column_b": f"{jp.get('table_b')}.{jp.get('col_b')}",
                    "exact_resemblance": jp.get("resemblance"),
                    "shingle_resemblance": jp.get("resemblance_shingle"),
                    "coverage_a": jp.get("coverage_a"),
                    "coverage_b": jp.get("coverage_b"),
                    "interpretation": jp.get("relationship_interpretation"),
                }
            )
        else:
            compact_join_paths.append(
                {
                    "relationship_type": "candidate_linkage_path",
                    "column_a": f"{jp.get('table_a')}.{jp.get('col_a')}",
                    "column_b": f"{jp.get('table_b')}.{jp.get('col_b')}",
                    "method": jp.get("method"),
                    "exact_resemblance": jp.get("resemblance"),
                    "shingle_resemblance": jp.get("resemblance_shingle"),
                    "interpretation": "Exploratory MinHash/shingle candidate. Review as a possible exact linkage, fuzzy linkage, or blocking field depending on domain context.",
                }
            )
    evidence = {
        "exact_resemblance_threshold": join_threshold,
        "shingle_resemblance_threshold": shingle_join_threshold,
        "note": "For foreign keys, directional_containment is the main signal. Exact resemblance can be low when the FK uses only a subset of the parent PK values. Shingle resemblance is mainly for fuzzy text domains, not numeric IDs.",
        "join_paths": compact_join_paths,
    }
    prompt = JOIN_PATH_INTERPRETATION_PROMPT.format(
        evidence_json=json.dumps(evidence, indent=2, ensure_ascii=False)
    )
    for attempt in range(1, cfg.llm_max_retries + 1):
        print(f"  Join interpretation, attempt {attempt}/{cfg.llm_max_retries}")
        raw = call_llm(prompt)
        text = strip_thinking(raw)
        text = text.strip()
        if text:
            cached.write_text(text, encoding="utf-8")
            print("  Join interpretation: success.")
            return text
    return ""
