"""
DataDictionaryPipeline — top-level orchestrator.

Wires together all components and exposes a single run() method
that produces data dictionaries, MinHash findings, and all output files.
"""

from datetime import datetime
from pathlib import Path
from dataclasses import replace as _dc_replace
import json as _json
import numpy as _np
import datetime as _dt
import pandas as _pd
import re

from .column_analyzer import ColumnAnalyzer
from .config import PipelineConfig
from .exporters import DataDictionaryExporter
from .llm_generator import LLMDictionaryGenerator
from .loader import DataLoader
from .minhash_analyzer_enhanced import MinHashAnalyzer
from .profiler import DataProfiler
from .preprocessor import DataPreprocessor
from .json_utils import json_default

class DataDictionaryPipeline:
    """
    End-to-end pipeline from raw files to data dictionary outputs.
 
    Usage
    -----
    >>> config = PipelineConfig(data_dir="data/raw", output_dir="profile_outputs")
    >>> pipeline = DataDictionaryPipeline(config, llm_client=h2o_client)
    >>> results = pipeline.run(dataset_paths=[path_a, path_b])
    """
 
    def __init__(self, config: PipelineConfig, llm_client=None):
        self.config = config
        self._base_output_dir = config.output_dir
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor(config)
        self.profiler = DataProfiler(config, self.preprocessor)
        self.column_analyzer = ColumnAnalyzer(config)
        self.minhash_analyzer = MinHashAnalyzer(config)
        self.llm_generator = (
            LLMDictionaryGenerator(llm_client, config) if llm_client else None
        )
        self.exporter = DataDictionaryExporter(config)
 
    # ------------------------------------------------------------------
    # Run folder
    # ------------------------------------------------------------------
 
    def _create_run_folder(self, dataset_paths: list[Path]) -> Path:
        """
        Create and return a named run subfolder inside config.output_dir.

        Format: <dataset1>_<dataset2>_YYYY-MM-DD_HH-MM
        Example: febrl4a_febrl4b_2026-04-17_14-30
        """
        # Sanitise stems — replace spaces and special chars with underscores
        dataset_names = "_".join(
            re.sub(r"[^\w]+", "_", p.stem).strip("_")
            for p in dataset_paths
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{dataset_names}_{timestamp}"
        run_dir = self._base_output_dir / folder_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Replace config with a new instance pointing at the run folder
        self.config = _dc_replace(self.config, output_dir=run_dir)

        # Propagate the new config to every component that holds a reference
        # to the old one — without this they keep writing to profile_outputs/
        self.preprocessor.config = self.config
        self.profiler.config = self.config
        self.column_analyzer.config = self.config
        self.minhash_analyzer.config = self.config
        self.exporter.config = self.config
        if self.llm_generator:
            self.llm_generator.config = self.config

        print(f"Run folder: {run_dir}")
        return run_dir
 
    # ------------------------------------------------------------------
    # Step helpers (callable individually for notebook exploration)
    # ------------------------------------------------------------------
 
    def step_profile(self, dataset_paths: list[Path]) -> dict:
        """Load, preprocess, and profile all datasets."""
        profile_results = {}
        for path in dataset_paths:
            name = path.stem
            raw_df = self.loader.load(path)
            df, report, summary = self.profiler.profile(raw_df, name)
            profile_results[name] = {
                "raw_df": raw_df,
                "df": df,
                "report": report,
                "summary": summary,
                "path": path,
            }
            print(f"  Profiled {name}: {df.shape[0]} rows × {df.shape[1]} cols")
        return profile_results
 
    def step_column_summaries(self, profile_results: dict) -> dict[str, list[dict]]:
        """Build column-level summaries with error detection and MinHash sketches."""
        column_summaries = {}
        for name, result in profile_results.items():
            print(f"  Analysing columns for {name}...")
            summary = self.column_analyzer.build_summary(
                profile_json_path=result["summary"]["json_report"],
                raw_df=result["raw_df"],
                df=result["df"],
                table_name=name,
            )
            column_summaries[name] = summary
            cols_with_minhash = sum(1 for r in summary if "_minhash" in r)
            print(f"    {len(summary)} columns, {cols_with_minhash} with MinHash sketches")
        return column_summaries
 
    def step_minhash(self, column_summaries: dict,profile_results: dict) -> dict:
        """Run MinHash analysis across all columns."""
        all_summaries = [
            row
            for rows in column_summaries.values()
            for row in rows
        ]
        for table_name, rows in column_summaries.items():
            for row in rows:
                all_summaries.append({**row, "table_name": table_name})
        
        dataframes = {name: result["df"] for name, result in profile_results.items()}
        results = self.minhash_analyzer.find_joinable_columns(all_summaries, dataframes=dataframes)
        self.minhash_analyzer.report(results)
        self.minhash_analyzer.report_cross_table_duplicates(results["duplicate_columns"])
        return results
    
    def _annotate_relationship_roles(
        self,
        column_summaries: dict,
        minhash_results: dict,
    ) -> dict:
        """
        Annotate each column summary row with its relationship_role and join_hints.
        Called before LLM generation so the LLM evidence includes PK/FK context.
        """
        join_hints_text = self.minhash_analyzer.build_join_hints(minhash_results)
        relationship_roles: dict[str, str] = {}
        for jp in minhash_results.get("join_paths", []):
            rel_type = jp.get("relationship_type", "")
            if rel_type == "foreign_key":
                pk_key = f"{jp['primary_key_table']}.{jp['primary_key_column']}"
                fk_key = f"{jp['foreign_key_table']}.{jp['foreign_key_column']}"
                relationship_roles[pk_key] = "primary_key"
                relationship_roles[fk_key] = "foreign_key"
            elif rel_type == "one_to_one_key":
                relationship_roles[jp.get("key_a", "")] = "join_key"
                relationship_roles[jp.get("key_b", "")] = "join_key"
            elif rel_type == "shared_join_key":
                key_a = f"{jp['table_a']}.{jp['col_a']}"
                key_b = f"{jp['table_b']}.{jp['col_b']}"
                relationship_roles[key_a] = "join_key"
                relationship_roles[key_b] = "join_key"

        annotated: dict[str, list[dict]] = {}
        for table_name, table_summary in column_summaries.items():
            annotated_rows = []
            for row in table_summary:
                col_key = f"{table_name}.{row['column_name']}"
                hints = join_hints_text.get(col_key, [])
                new_row = {
                    **row,
                    "relationship_role": relationship_roles.get(col_key, ""),
                    "join_hints": hints,
                }
                # Surface duplicate-column hint into errors[] so it is
                # visible in validation check results and the Word report.
                if any("likely duplicate column" in h for h in hints):
                    new_row["errors"] = list(new_row.get("errors", [])) + [
                        "column appears to be a near-duplicate of another column "
                        "in this table — confirm if both are needed or if one should be removed"
                    ]
                annotated_rows.append(new_row)
            annotated[table_name] = annotated_rows
        return annotated
 
    def step_generate_dictionaries(
        self,
        column_summaries: dict,
        minhash_results: dict,
        dataset_descriptions: dict[str, str] | None = None,
        join_hints: dict | None = None,
    ) -> tuple[dict[str, list[dict]], dict[str, str]]:
        """Generate LLM descriptions and merge into final data dictionaries.
        
        Parameters
        ----------
        dataset_descriptions : dict[str, str], optional
            Per-table background descriptions (e.g. source agency, domain,
            coverage). Keyed by table name. Passed into LLM prompts to
            provide context beyond column names and sample values.

        Returns
        -------
        all_dictionaries  : {table_name: [column_dict, ...]}
        dataset_summaries : {table_name: summary_string}
        """
        if self.llm_generator is None:
            raise RuntimeError(
                "LLM client was not provided to DataDictionaryPipeline."
            )
        if join_hints is None:
            join_hints = self.minhash_analyzer.build_join_hints(minhash_results)
        all_dictionaries = {}
        dataset_summaries = {}
        descriptions = dataset_descriptions or {}
 
        for table_name, table_summary in column_summaries.items():
            desc = descriptions.get(table_name, "")
            print(f"\n  Generating dictionary for {table_name}...")
            llm_rows = self.llm_generator.generate(
                column_summary=table_summary,
                table_name=table_name,
                join_hints=join_hints,
                dataset_description=desc,
            )
            merged = self.llm_generator.merge(table_summary, llm_rows)
            all_dictionaries[table_name] = merged
 
            print(f"  Generating dataset summary for {table_name}...")
            summary = self.llm_generator.generate_dataset_summary(
                table_name=table_name,
                column_summary=table_summary,
                join_hints=join_hints,
                dataset_description=desc,
            )
            dataset_summaries[table_name] = summary
 
        return all_dictionaries, dataset_summaries
    
    def step_validation_rules(
        self,
        column_summaries: dict,
        minhash_results: dict,
        profile_results: dict | None = None,
    ) -> dict[str, list[dict]]:
        """
        Generate validation rules for all tables via LLM.
        The LLM generates rules AND identifies failing records in the sample.
        Cross-table referential integrity rules are appended from FK join paths.
        """
        all_rules = {}

        for table_name, table_summary in column_summaries.items():
            print(f"  Generating validation rules for {table_name}...")

            df = profile_results[table_name]["df"] if profile_results else None

            if self.llm_generator and df is not None:
                # Build join hints from MinHash FK relationships
                join_hints = {row["column_name"]: [] for row in table_summary}
                for jp in minhash_results.get("join_paths", []):
                    fk_table = jp.get("foreign_key_table")
                    fk_col = jp.get("foreign_key_column")
                    pk_table = jp.get("primary_key_table")
                    pk_col = jp.get("primary_key_column")
                    if fk_table == table_name and fk_col in join_hints:
                        join_hints[fk_col].append(f"FK → {pk_table}.{pk_col}")
                    if pk_table == table_name and pk_col in join_hints:
                        join_hints[pk_col].append(f"PK ← {fk_table}.{fk_col}")
                rules = self.llm_generator.generate_validation_rules(
                    table_name=table_name,
                    column_summary=table_summary,
                    df=df,
                    join_hints=join_hints,
                    n_sample=self.config.llm_validation_sample_size,
                )
            else:
                # Fallback: no LLM — empty rules
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
            all_rules[fk_table].append({
                "rule_id": rule_id,
                "table": fk_table,
                "column": fk_col,
                "columns": [fk_col],
                "category": "cross_table",
                "type": "referential",
                "rule": f"{fk_table}.{fk_col} must exist in {pk_table}.{pk_col}",
                "rationale": "Foreign key relationship detected by MinHash analysis.",
                "check_params": {
                    "col_a": fk_col,
                    "pk_table": pk_table,
                    "pk_col": pk_col,
                },
                "failing_record_indices": [],
            })

        return all_rules
    
    def step_run_validation_checks(
        self,
        validation_rules: dict[str, list[dict]],
        profile_results: dict,
    ) -> dict[str, dict]:
        """Run validation rules against actual records."""
        all_check_results = {}
        all_dfs = {tn: profile_results[tn]["df"] for tn in profile_results}

        for table_name, rules in validation_rules.items():
            if table_name not in profile_results:
                continue
            # Inject pk_values at runtime for referential cross-table rules
            for rule in rules:
                if rule.get("type") == "referential_cross_table":
                    cp = rule.get("check_params", {})
                    if "pk_values" not in cp:
                        pt = cp.get("pk_table")
                        pc = cp.get("pk_col")
                        if pt and pc and pt in all_dfs and pc in all_dfs[pt].columns:
                            cp["pk_values"] = set(
                                all_dfs[pt][pc]
                                .dropna()
                                .astype(str)
                                .str.strip()
                                .str.replace(r"\.0+$", "", regex=True)
                            )
            df = profile_results[table_name]["df"]
            print(f"  Running validation checks for {table_name}...")
            results = self.column_analyzer.run_validation_checks(df, rules)
            all_check_results[table_name] = results

            n_failing = sum(
                1 for r in results["per_rule"]
                if r.get("n_violations") and r["n_violations"] > 0
            )
            print(
                f"  [{table_name}] {len(results['per_rule'])} rules checked, "
                f"{n_failing} with violations. "
                f"{results['total_failing_records']} unique failing records."
            )

        return all_check_results
    
    def step_cleanup_fk_errors(
        self,
        column_summaries: dict,
        minhash_results: dict,
    ) -> dict[str, list[dict]]:
        """
        Remove false positive errors from FK columns after relationship detection.
        
        This step runs AFTER MinHash analysis identifies FK relationships,
        and BEFORE LLM generation, so the LLM gets clean data.
        """
        # Extract FK columns by table from identified relationships
        fk_columns_by_table = {}
        for join in minhash_results.get("join_paths", []):
            if join.get("relationship_type") == "foreign_key":
                fk_table = join.get("foreign_key_table")
                fk_col = join.get("foreign_key_column")
                if fk_table and fk_col:
                    if fk_table not in fk_columns_by_table:
                        fk_columns_by_table[fk_table] = set()
                    fk_columns_by_table[fk_table].add(fk_col)
        
        if not fk_columns_by_table:
            print("  No FK columns detected - skipping cleanup")
            return column_summaries
        
        # Clean up false positive errors for each table
        cleaned_summaries = {}
        for table_name, summary in column_summaries.items():
            identified_fks = fk_columns_by_table.get(table_name, set())
            if identified_fks:
                print(f"  Cleaning {table_name}: {', '.join(identified_fks)}")
                cleaned = self.column_analyzer.remove_fk_errors_from_results(
                    summary, identified_fks
                )
                cleaned_summaries[table_name] = cleaned
            else:
                cleaned_summaries[table_name] = summary
        
        return cleaned_summaries
    
 
    def step_export(
        self,
        all_dictionaries: dict[str, list[dict]],
        minhash_results: dict,
        generate_word: bool = True,
        word_script: str = "generate_word_report.js",
        report_title: str = "",
        dataset_summaries: dict[str, str] | None = None,
        report_summary: str = "",
        join_interpretation="",
        validation_rules: dict[str, list[dict]] | None = None,
        validation_check_results: dict[str, list[dict]] | None = None,
    ) -> dict:
        """Export all outputs (CSV, JSON, and optionally Word)."""
        output_paths: dict = {}

        for table_name, dictionary in all_dictionaries.items():
            paths = self.exporter.export_table(dictionary, table_name)
            output_paths[table_name] = paths
            print(f"  {table_name}: CSV → {paths['csv']}, JSON → {paths['json']}")

            if validation_rules and table_name in validation_rules:
                rules_path = self.config.output_dir / f"{table_name}_validation_rules.json"
                with open(rules_path, "w", encoding="utf-8") as f:
                    _json.dump(
                        validation_rules[table_name],
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=json_default,
                    )
                print(f"  {table_name}: Validation rules → {rules_path}")

        if generate_word:
            word_path = self.exporter.to_word(
                all_dictionaries, minhash_results, word_script,
                report_title=report_title,
                dataset_summaries=dataset_summaries or {},
                report_summary=report_summary,
                join_interpretation=join_interpretation,
                validation_rules=validation_rules or {},
                validation_check_results=validation_check_results or {},
            )
            output_paths["_word_report"] = word_path

        return output_paths
 
    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------
 
    def run(
        self,
        dataset_paths: list[Path],
        generate_word: bool = True,
        word_script: str = "generate_word_report.js",
        report_title: str = "",
        dataset_descriptions: dict[str, str] | None = None,
        join_hints: dict | None = None,
    ) -> dict:
        """
        Execute the full pipeline end-to-end.
 
        Returns a dict with all intermediate and final results:
        {
            "profile_results":    {...},
            "column_summaries":   {...},
            "minhash_results":    {...},
            "all_dictionaries":   {...},
            "output_paths":       {...},
        }
        """
        self._create_run_folder(dataset_paths)
 
        print("\n── Step 1: Profiling ───────────────────────────────")
        profile_results = self.step_profile(dataset_paths)
 
        print("\n── Step 2: Column analysis ─────────────────────────")
        column_summaries = self.step_column_summaries(profile_results)
 
        print("\n── Step 3: MinHash analysis ────────────────────────")
        minhash_results = self.step_minhash(column_summaries, profile_results)


        print("\n── Step 3.5: Cleaning FK false positive errors ─────")
        column_summaries = self.step_cleanup_fk_errors(column_summaries, minhash_results)
        # Annotate relationship roles + join hints before LLM generation
        # so the LLM evidence includes PK/FK context for each column.
        column_summaries = self._annotate_relationship_roles(column_summaries, minhash_results)

 
        print("\n── Step 4: LLM dictionary generation ───────────────")
        all_dictionaries, dataset_summaries = self.step_generate_dictionaries(
            column_summaries, minhash_results,
            dataset_descriptions=dataset_descriptions,
            join_hints=join_hints
        )
 
        print("\n  Generating report-level executive summary...")
        report_summary = self.llm_generator.generate_report_summary(
            dataset_summaries=dataset_summaries,
            all_dictionaries=all_dictionaries,
            minhash_results=minhash_results,
            output_dir=self.config.output_dir,
        ) if self.llm_generator else ""

        print("\n  Generating join path interpretation...")
        join_paths_for_interpretation = (
            minhash_results.get("candidate_join_paths", [])
            + minhash_results.get("join_paths", [])
        )

        join_interpretation = self.llm_generator.generate_join_interpretation(
            join_paths=join_paths_for_interpretation,
            join_threshold=self.config.join_threshold,
            shingle_join_threshold=self.config.shingle_join_threshold,
            output_dir=self.config.output_dir,
        ) if self.llm_generator and join_paths_for_interpretation else ""
 
        print("\n── Step 5: Validation rules ────────────────────────")
        validation_rules = self.step_validation_rules(
            column_summaries, minhash_results, profile_results
        )

        print("\n── Step 5b: Validation checks (record-wise) ────────")
        validation_check_results = self.step_run_validation_checks(
            validation_rules, profile_results
        )
        # Export check results
        for table_name, check_results in validation_check_results.items():
            check_path = self.config.output_dir / f"{table_name}_validation_check_results.json"
            with open(check_path, "w", encoding="utf-8") as f:
                _json.dump(
                    check_results,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=json_default,
                )
            print(
                f"  {table_name}: {check_results['total_failing_records']} "
                f"failing records → {check_path}"
            )

        print("\n── Step 6: Export ──────────────────────────────────")
        output_paths = self.step_export(
            all_dictionaries, minhash_results, generate_word, word_script,
            report_title=report_title,
            dataset_summaries=dataset_summaries,
            report_summary=report_summary,
            join_interpretation=join_interpretation,
            validation_rules=validation_rules,
            validation_check_results=validation_check_results,
        )
 
        print("\n✓ Pipeline complete.")
        return {
            "profile_results": profile_results,
            "column_summaries": column_summaries,
            "minhash_results": minhash_results,
            "all_dictionaries": all_dictionaries,
            "output_paths": output_paths,
        }