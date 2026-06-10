"""
Exporters — writes pipeline outputs to CSV, JSON, and Word document.
 
Word document is generated via a Node.js script (generate_word_report.js)
using the 'docx' npm package.  This class writes the data payload JSON and
then invokes the script.
"""
 
import json
import subprocess
from datetime import datetime
from pathlib import Path
 
import pandas as pd
 
from .config import PipelineConfig
from .json_utils import json_default, clean_for_json
 
 
class DataDictionaryExporter:
    """Exports a completed data dictionary to CSV, JSON, and optionally Word."""
 
    def __init__(self, config: PipelineConfig):
        self.config = config
 
    # ------------------------------------------------------------------
    # CSV / JSON
    # ------------------------------------------------------------------
 
    def to_json(self, data_dictionary: list[dict], table_name: str) -> Path:
        path = self.config.output_dir / f"{table_name}_data_dictionary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                clean_for_json(data_dictionary),
                f,
                indent=2,
                ensure_ascii=False,
                default=json_default,
                allow_nan=False,
            )
        return path
 
    def to_csv(self, data_dictionary: list[dict], table_name: str) -> Path:
        flat = []
        for row in data_dictionary:
            flat.append({
                "column_name": row["column_name"],
                "data_type": row["data_type"],
                "intended_data_type": row.get("intended_data_type", row["data_type"]),
                "sample_values": " | ".join(map(str, row["sample_values"]))
                    if row["sample_values"] else "",
                "missing_count": row["profile"].get("missing_count"),
                "missing_pct": row["profile"].get("missing_pct"),
                "errors": " | ".join(row["errors"]) if row["errors"] else "",
                "description": row["description"],
                "recommended_actions": " | ".join(row["recommended_actions"])
                    if row["recommended_actions"] else "",
            })
        path = self.config.output_dir / f"{table_name}_data_dictionary.csv"
        pd.DataFrame(flat).to_csv(path, index=False, encoding="utf-8")
        return path
 
    def export_table(self, data_dictionary: list[dict], table_name: str) -> dict[str, Path]:
        """Export both formats for one table and return paths."""
        return {
            "json": self.to_json(data_dictionary, table_name),
            "csv": self.to_csv(data_dictionary, table_name),
        }
 
    # ------------------------------------------------------------------
    # Word document
    # ------------------------------------------------------------------
 
    def _parse_relationships(self, join_paths: list[dict]) -> dict:
        """
        Parse join paths to extract relationship information for the Word report.
        """
        foreign_keys = []
        lookup_tables = []
        one_to_one_keys = []
        shared_join_keys = []
        shared_value_domains = []
        primary_keys = {}

        for jp in join_paths:
            rel_type = jp.get("relationship_type")

            if rel_type == "foreign_key":
                pk_table = jp.get("primary_key_table")
                pk_col = jp.get("primary_key_column")
                fk_table = jp.get("foreign_key_table")
                fk_col = jp.get("foreign_key_column")

                if pk_table and pk_col and fk_table and fk_col:
                    coverage_a = jp.get("coverage_a", 0)
                    coverage_b = jp.get("coverage_b", 0)
                    referential_integrity = max(coverage_a, coverage_b) * 100

                    fk_info = {
                        "foreign_key": f"{fk_table}.{fk_col}",
                        "references": f"{pk_table}.{pk_col}",
                        "interpretation": jp.get("relationship_interpretation", ""),
                        "referential_integrity": round(referential_integrity, 1),
                        "quality_score": jp.get("quality_score", 0),
                        "quality_grade": jp.get("quality_grade", "N/A"),
                        "coverage_a": coverage_a,
                        "coverage_b": coverage_b,
                    }
                    foreign_keys.append(fk_info)

                    pk_key = f"{pk_table}.{pk_col}"
                    if pk_key not in primary_keys:
                        primary_keys[pk_key] = []
                    primary_keys[pk_key].append(f"{fk_table}.{fk_col}")

            elif rel_type == "one_to_one_key":
                one_to_one_keys.append({
                    "key_a": jp.get("key_a", f"{jp.get('table_a')}.{jp.get('col_a')}"),
                    "key_b": jp.get("key_b", f"{jp.get('table_b')}.{jp.get('col_b')}"),
                    "interpretation": jp.get("relationship_interpretation", ""),
                    "coverage_a": jp.get("coverage_a", 0),
                    "coverage_b": jp.get("coverage_b", 0),
                    "quality_score": jp.get("quality_score", 0),
                    "quality_grade": jp.get("quality_grade", "N/A"),
                })

            elif rel_type == "shared_value_domain":
                shared_value_domains.append({
                    "column_a": f"{jp.get('table_a')}.{jp.get('col_a')}",
                    "column_b": f"{jp.get('table_b')}.{jp.get('col_b')}",
                    "interpretation": jp.get("relationship_interpretation", ""),
                    "coverage_a": jp.get("coverage_a", 0),
                    "coverage_b": jp.get("coverage_b", 0),
                    "quality_score": jp.get("quality_score", 0),
                    "quality_grade": jp.get("quality_grade", "N/A"),
                })

            elif rel_type == "shared_join_key":
                shared_join_keys.append({
                    "column_a": f"{jp.get('table_a')}.{jp.get('col_a')}",
                    "column_b": f"{jp.get('table_b')}.{jp.get('col_b')}",
                    "interpretation": jp.get("relationship_interpretation", ""),
                    "coverage_a": jp.get("coverage_a", 0),
                    "coverage_b": jp.get("coverage_b", 0),
                    "quality_score": jp.get("quality_score", 0),
                    "quality_grade": jp.get("quality_grade", "N/A"),
                })

            elif rel_type == "many_to_many_or_lookup":
                lookup_tables.append({
                    "column_a": f"{jp.get('table_a')}.{jp.get('col_a')}",
                    "column_b": f"{jp.get('table_b')}.{jp.get('col_b')}",
                    "interpretation": jp.get("relationship_interpretation", ""),
                    "jaccard": jp.get("jaccard", jp.get("resemblance", 0)),
                    "coverage_a": jp.get("coverage_a", 0),
                    "coverage_b": jp.get("coverage_b", 0),
                })

        return {
            "foreign_keys": foreign_keys,
            "lookup_tables": lookup_tables,
            "one_to_one_keys": one_to_one_keys,
            "shared_join_keys": shared_join_keys,
            "shared_value_domains": shared_value_domains,
            "primary_keys": primary_keys,
            "total_relationships": (
                len(foreign_keys)
                + len(lookup_tables)
                + len(one_to_one_keys)
                + len(shared_join_keys)
                + len(shared_value_domains)
            ),
        }
 
    def to_word(
        self,
        all_data_dictionaries: dict[str, list[dict]],
        minhash_results: dict,
        js_script_path: str | Path = "generate_word_report.js",
        report_title: str = "",
        dataset_summaries: dict[str, str] | None = None,
        report_summary: str = "",
        join_interpretation="",
        validation_rules: dict[str, list[dict]] | None = None,
        validation_check_results: dict[str, list[dict]] | None = None,
    ) -> Path:
        """
        Generate a Word document by:
        1. Writing a payload JSON with all data dictionaries + MinHash findings + CSV paths.
        2. Invoking the Node.js docx generation script.
        """
        clean_join_paths = [
            {k: v for k, v in jp.items() if not k.startswith("_")}
            for jp in minhash_results.get("join_paths", [])
        ]
        clean_dupes = [
            {k: v for k, v in dc.items() if not k.startswith("_")}
            for dc in minhash_results.get("duplicate_columns", [])
        ]

        # Only typed paths feed the Key Identification / relationship sections.
        _typed_for_relationships = [
            jp for jp in minhash_results.get("join_paths", [])
            if jp.get("relationship_type") in {
                "foreign_key", "one_to_one_key",
                "shared_value_domain", "shared_join_key",
            }
        ]
        relationships = self._parse_relationships(_typed_for_relationships)
 
 
        csv_paths = {
            name: str((self.config.output_dir / f"{name}_data_dictionary.csv").resolve())
            for name in all_data_dictionaries
        }
 
        # Derive title from dataset names if not provided
        if not report_title:
            names = " & ".join(all_data_dictionaries.keys())
            report_title = f"Data Dictionary — {names}"
 
        # Build per-dataset overview stats
        dataset_overviews = {}
        for name, rows in all_data_dictionaries.items():
            total_cols = len(rows)
            cols_with_errors = sum(1 for r in rows if r.get("errors"))
            missing_cols = sum(
                1 for r in rows
                if r.get("profile", {}).get("missing_count", 0) > 0
            )
            type_counts: dict[str, int] = {}
            for r in rows:
                t = r.get("data_type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
 
            dataset_overviews[name] = {
                "total_columns": total_cols,
                "columns_with_errors": cols_with_errors,
                "columns_with_missing": missing_cols,
                "type_breakdown": type_counts,
                "summary": (dataset_summaries or {}).get(name, ""),
            }

        # Unique non-null columns are useful evidence, but not automatically primary keys.
        unique_non_null_columns = []
        likely_primary_keys = []

        def _norm_name(x: str) -> str:
            return "".join(ch for ch in str(x).lower() if ch.isalnum())

        for _ck_table, _ck_rows in all_data_dictionaries.items():
            table_norm = _norm_name(_ck_table)
            table_singular = table_norm[:-1] if table_norm.endswith("s") else table_norm

            for _ck_row in _ck_rows:
                _p = _ck_row.get("profile", {})
                _missing  = _p.get("missing_count", -1)
                _distinct = _p.get("n_distinct")
                _total    = _p.get("n_total")

                if (
                    _missing == 0
                    and _distinct is not None
                    and _total is not None
                    and _distinct > 1
                    and _distinct == _total
                ):
                    entry = {
                        "table":      _ck_table,
                        "column":     _ck_row["column_name"],
                        "data_type":  _ck_row["data_type"],
                        "n_distinct": _distinct,
                        "n_total":    _total,
                    }

                    unique_non_null_columns.append(entry)

                    col_norm = _norm_name(_ck_row["column_name"])
                    is_id_named = (
                        col_norm == "id"
                        or col_norm.endswith("id")
                        or col_norm.endswith("key")
                    )
                    table_matches = table_singular and table_singular in col_norm

                    if is_id_named and (table_matches or col_norm in {"id", f"{table_singular}id"}):
                        likely_primary_keys.append(entry)

        clean_candidate_join_paths = [
            {k: v for k, v in jp.items() if not k.startswith("_")}
            for jp in minhash_results.get("candidate_join_paths", minhash_results.get("join_paths", []))
        ]

        payload = {
            "report_title": report_title,
            "report_subtitle": "Automated Profiling & Quality Analysis",
            "report_summary": report_summary,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "output_dir": str(self.config.output_dir.resolve()),
            "csv_paths": csv_paths,
            "dataset_overviews": dataset_overviews,
            "tables": {
                name: [
                    {k: v for k, v in row.items() if not k.startswith("_")}
                    for row in rows
                ]
                for name, rows in all_data_dictionaries.items()
            },
            "join_paths": clean_join_paths,
            "duplicate_columns": clean_dupes,
            "cross_table_duplicates": [
                dc for dc in clean_dupes if dc["table_a"] != dc["table_b"]
            ],
            "join_interpretation": join_interpretation,
            "join_threshold": self.config.join_threshold,
            "shingle_join_threshold": self.config.shingle_join_threshold,
            "validation_rules": validation_rules or {},
            "validation_check_results": validation_check_results or {},
            "relationships": relationships,
            "unique_non_null_columns": unique_non_null_columns,
            "likely_primary_keys": likely_primary_keys,
            "candidate_join_paths": clean_candidate_join_paths,
            # Backwards-compatible alias for older report JS.
            "candidate_keys": unique_non_null_columns,
        }
 
        payload_path = self.config.output_dir / "word_report_payload.json"
        payload = clean_for_json(payload)

        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(
                payload,
                f,
                indent=2,
                ensure_ascii=False,
                default=json_default,
                allow_nan=False,
            )
 
        out_path = self.config.output_dir / "data_dictionary_report.docx"
 
        result = subprocess.run(
            ["node","--max-old-space-size=8192", str(js_script_path), str(payload_path), str(out_path)],
            capture_output=True,
            text=True,
        )
 
        if result.returncode != 0:
            raise RuntimeError(
                f"Word report generation failed:\n{result.stderr}"
            )
 
        print(f"Word report saved to: {out_path}")
        return out_path