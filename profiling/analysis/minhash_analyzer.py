"""Discover joinable and duplicate columns using MinHash and coverage.

Relationship interpretation is delegated to RelationshipClassifier; human-readable
output and join hints live in relationship_reporting.py.
"""

import pandas as pd
from datasketch import MinHashLSH

from ..core.config import PipelineConfig
from .relationship_classifier import RelationshipClassifier
from .relationship_reporting import report_cross_table_duplicates, report_relationships

def _normalize_val_set(series: "pd.Series") -> set:
    """Normalise a column to a set of strings, stripping .0 float artefacts."""
    s = series.dropna().astype(str).str.strip()
    # Remove trailing .0 so int-stored-as-float matches integer column
    s = s.str.replace(r'\.0+$', '', regex=True)
    return set(s[s != ''])


class MinHashAnalyzer:
    """Find candidate joins and duplicate columns across datasets."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.classifier = RelationshipClassifier(config)

    @staticmethod
    def _are_compatible(kind_a: str, kind_b: str) -> bool:
        allowed = {"categorical", "discrete_code", "key_like"}
        return kind_a in allowed and kind_b in allowed

    @classmethod
    def _should_report_join(cls, jp: dict) -> bool:
        """
        Final report filter. This prevents raw numeric overlap/debug pairs
        from leaking into the Word report.
        """
        rel_type = jp.get("relationship_type")

        if rel_type in {"foreign_key", "one_to_one_key", "shared_value_domain"}:
            return True

        # # Keep old shared_join_key only if both sides are non-numeric and clearly named alike.
        # if rel_type == "shared_join_key":
        #     type_a = str(jp.get("type_a", "")).lower()
        #     type_b = str(jp.get("type_b", "")).lower()
        #     numeric = ("int", "float", "double", "decimal", "number")
        #     if any(t in type_a for t in numeric) or any(t in type_b for t in numeric):
        #         return False
        #     return True

        # Hide low-confidence / unclassified MinHash-only pairs from final report.
        return False

    def find_joins_by_coverage_with_data(
        self,
        column_summary: list[dict],
        dataframes: dict[str, pd.DataFrame],
    ) -> list[dict]:
        """
        Coverage-based join detection using actual values.

        Universal strategy:
        1. Compare plausible name pairs, not every numeric pair.
        2. Treat descriptive copied attributes as shared value domains.
        3. Require strong name evidence before FK classification.
        4. Allow self-FKs only for hierarchy-like column names.
        """
        cfg = self.config
        joins: list[dict] = []

        excluded_kinds = {"datetime", "continuous_numeric", "free_text", "empty"}

        candidate_rows = [
            row for row in column_summary
            if row.get("similarity_kind", "other") not in excluded_kinds
        ]

        for i, row_a in enumerate(candidate_rows):
            for row_b in candidate_rows[i + 1:]:
                table_a = row_a["table_name"]
                table_b = row_b["table_name"]
                col_a = row_a["column_name"]
                col_b = row_b["column_name"]

                if table_a == table_b and col_a == col_b:
                    continue

                # Same-table comparisons are dangerous.
                # Only allow plausible self-referencing hierarchy FKs.
                if table_a == table_b and not self.classifier._looks_like_hierarchy_fk(table_a, col_a, table_b, col_b):
                    continue

                # Universal name gate: prevents accidental numeric overlap.
                if not self.classifier._candidate_name_pair(table_a, col_a, table_b, col_b):
                    continue

                df_a = dataframes.get(table_a)
                df_b = dataframes.get(table_b)

                if df_a is None or df_b is None:
                    continue

                if col_a not in df_a.columns or col_b not in df_b.columns:
                    continue

                vals_a = _normalize_val_set(df_a[col_a])
                vals_b = _normalize_val_set(df_b[col_b])

                if not vals_a or not vals_b:
                    continue

                # Allow small lookup domains, but avoid single-value coincidences.
                if min(len(vals_a), len(vals_b)) < 2:
                    continue

                overlap = vals_a & vals_b
                if not overlap:
                    continue

                coverage_a = len(overlap) / len(vals_a)
                coverage_b = len(overlap) / len(vals_b)
                jaccard = len(overlap) / len(vals_a | vals_b)

                row_count_a = len(df_a)
                row_count_b = len(df_b)

                uniqueness_a = len(vals_a) / row_count_a if row_count_a > 0 else 0
                uniqueness_b = len(vals_b) / row_count_b if row_count_b > 0 else 0

                name_score_a_pk = self.classifier._relationship_name_score(
                    table_a, col_a, table_b, col_b
                )
                name_score_b_pk = self.classifier._relationship_name_score(
                    table_b, col_b, table_a, col_a
                )

                relationship = self.classifier.identify_pk_fk_relationship(
                    table_a=table_a,
                    col_a=col_a,
                    table_b=table_b,
                    col_b=col_b,
                    coverage_a=coverage_a,
                    coverage_b=coverage_b,
                    cardinality_a=len(vals_a),
                    cardinality_b=len(vals_b),
                    uniqueness_a=uniqueness_a,
                    uniqueness_b=uniqueness_b,
                    name_score_a_pk=name_score_a_pk,
                    name_score_b_pk=name_score_b_pk,
                    series_a=df_a[col_a],
                    series_b=df_b[col_b],
                )

                rel_type = relationship.get("relationship_type")

                is_fk_or_one_to_one = rel_type in {"foreign_key", "one_to_one_key"}

                is_shared_value_domain = (
                    rel_type == "shared_value_domain"
                    and max(coverage_a, coverage_b) >= cfg.coverage_join_threshold
                )

                # Do not report ambiguous numeric/shared-id overlaps.
                if not (is_fk_or_one_to_one or is_shared_value_domain):
                    continue

                quality = self.classifier.compute_join_quality_score(
                    coverage_a=coverage_a,
                    coverage_b=coverage_b,
                    cardinality_a=len(vals_a),
                    cardinality_b=len(vals_b),
                    overlap_count=len(overlap),
                    uniqueness_a=uniqueness_a,
                    uniqueness_b=uniqueness_b,
                )

                joins.append({
                    "table_a": table_a,
                    "col_a": col_a,
                    "table_b": table_b,
                    "col_b": col_b,

                    "coverage_a": round(coverage_a, 4),
                    "coverage_b": round(coverage_b, 4),
                    "jaccard": round(jaccard, 4),
                    "resemblance": round(jaccard, 4),
                    "resemblance_shingle": None,
                    "overlap_count": len(overlap),

                    "type_a": row_a["data_type"],
                    "type_b": row_b["data_type"],
                    "kind_a": row_a.get("similarity_kind"),
                    "kind_b": row_b.get("similarity_kind"),

                    "quality_score": quality["score"],
                    "quality_grade": quality["grade"],
                    "score_components": quality,
                    "uniqueness_a": round(uniqueness_a, 4),
                    "uniqueness_b": round(uniqueness_b, 4),
                    "name_score_a_pk": round(name_score_a_pk, 4),
                    "name_score_b_pk": round(name_score_b_pk, 4),

                    "method": "coverage",
                    "primary_direction": relationship.get("direction"),
                    "relationship_type": rel_type,
                    "primary_key_table": relationship.get("primary_key_table"),
                    "primary_key_column": relationship.get("primary_key_column"),
                    "foreign_key_table": relationship.get("foreign_key_table"),
                    "foreign_key_column": relationship.get("foreign_key_column"),
                    "key_a": relationship.get("key_a"),
                    "key_b": relationship.get("key_b"),
                    "relationship_interpretation": relationship.get("interpretation"),
                })

        def sort_key(x: dict):
            if x.get("relationship_type") == "foreign_key":
                return (0, -max(x.get("coverage_a", 0), x.get("coverage_b", 0)))
            if x.get("relationship_type") == "one_to_one_key":
                return (1, -max(x.get("coverage_a", 0), x.get("coverage_b", 0)))
            if x.get("relationship_type") == "shared_value_domain":
                return (2, -max(x.get("coverage_a", 0), x.get("coverage_b", 0)))
            return (3, -x.get("quality_score", 0))

        joins.sort(key=sort_key)
        return joins

    def find_joinable_columns(
        self, 
        column_summary: list[dict],
        dataframes: dict[str, pd.DataFrame] | None = None
    ) -> dict:
        """
        Identify pairs of columns that are candidates for joins or deduplication.

        Strategy
        --------
        1. Use MinHash LSH for high-similarity pairs (Jaccard-based)
        2. Use coverage-based detection for foreign key joins (asymmetric)
        3. Merge results and deduplicate
        4. Route to join_paths or duplicate_columns based on similarity score

        Returns
        -------
        {
          "join_paths":       [pair_record, ...],
          "duplicate_columns": [pair_record, ...],
        }
        """
        cfg = self.config

        # === PART 1: MinHash-based detection (original logic) ===

        candidates = [r for r in column_summary if r.get("_minhash") is not None]
        key_to_row = {
            f"{r['table_name']}.{r['column_name']}": r for r in candidates
        }

        # ── Exact LSH ──────────────────────────────────────────────────
        lsh_exact = MinHashLSH(threshold=cfg.join_threshold, num_perm=cfg.minhash_num_perm)
        for key, row in key_to_row.items():
            lsh_exact.insert(key, row["_minhash"])

        # ── Shingle LSH ─────────────────────────────────────────────────
        shingle_candidates = {
            k: r for k, r in key_to_row.items() if r.get("_minhash_shingle") is not None
        }
        lsh_shingle = MinHashLSH(
            threshold=cfg.shingle_join_threshold, num_perm=cfg.minhash_num_perm
        )
        for key, row in shingle_candidates.items():
            lsh_shingle.insert(key, row["_minhash_shingle"])

        # ── Union candidate pairs ───────────────────────────────────────
        seen_pairs: set[tuple[str, str]] = set()
        candidate_pairs: list[tuple[str, str]] = []

        def _collect(key: str, neighbours: list[str]) -> None:
            for other_key in neighbours:
                if other_key == key:
                    continue
                pair = tuple(sorted([key, other_key]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    candidate_pairs.append(pair)

        for key, row in key_to_row.items():
            _collect(key, lsh_exact.query(row["_minhash"]))

        for key, row in shingle_candidates.items():
            _collect(key, lsh_shingle.query(row["_minhash_shingle"]))

        # ── Score and route MinHash results ─────────────────────────────
        join_paths_minhash: list[dict] = []
        duplicate_columns: list[dict] = []

        for key_a, key_b in candidate_pairs:
            row = key_to_row.get(key_a)
            other_row = key_to_row.get(key_b)
            if not row or not other_row:
                continue

            if not self._are_compatible(
                row.get("similarity_kind", "other"),
                other_row.get("similarity_kind", "other"),
            ):
                continue

            same_table = row["table_name"] == other_row["table_name"]

            # Safety gate: do not allow numeric MinHash join candidates.
            # Numeric overlap is often accidental, e.g. GenreId ↔ PlaylistId.
            # Still allow same-table numeric pairs to be checked as possible duplicates.
            type_a = str(row.get("data_type", "")).lower()
            type_b = str(other_row.get("data_type", "")).lower()
            numeric_markers = ("int", "float", "double", "decimal", "number")

            any_numeric = (
                any(t in type_a for t in numeric_markers)
                or any(t in type_b for t in numeric_markers)
            )

            both_are_named_ids = (
                row.get("similarity_kind") == "key_like"
                and other_row.get("similarity_kind") == "key_like"
            )

            if any_numeric and not both_are_named_ids and not same_table:
                continue

            resemblance = round(row["_minhash"].jaccard(other_row["_minhash"]), 4)

            resemblance_shingle: float | None = None
            if row.get("_minhash_shingle") and other_row.get("_minhash_shingle"):
                resemblance_shingle = round(
                    row["_minhash_shingle"].jaccard(other_row["_minhash_shingle"]), 4
                )

            exact_passes = resemblance >= cfg.join_threshold
            shingle_passes = (resemblance_shingle or 0) >= cfg.shingle_join_threshold

            if not exact_passes and not shingle_passes:
                continue

            pair_record = {
                "table_a": row["table_name"],
                "col_a": row["column_name"],
                "table_b": other_row["table_name"],
                "col_b": other_row["column_name"],
                "resemblance": resemblance,
                "resemblance_shingle": resemblance_shingle,
                "type_a": row["data_type"],
                "type_b": other_row["data_type"],
                "kind_a": row.get("similarity_kind"),
                "kind_b": other_row.get("similarity_kind"),
                "method": "minhash",
            }

            # Only treat as duplicate if it is within the same table.
            # Cross-table identical columns are often valid join keys, not duplicate columns.
            # Skip duplicate flag if both columns have low cardinality
            # e.g. binary Yes/No columns naturally share the same value set
            # and will always show Jaccard=1.0 without being actual duplicates
            low_cardinality_a = row.get("permissible_values") is not None
            low_cardinality_b = other_row.get("permissible_values") is not None
            both_low_cardinality = low_cardinality_a and low_cardinality_b

            if same_table and resemblance >= cfg.duplicate_threshold and not both_low_cardinality:
                duplicate_columns.append(pair_record)

            elif (
                same_table
                and resemblance >= cfg.near_identical_threshold
                and resemblance < cfg.duplicate_threshold
                and not both_low_cardinality
            ):
                near_identical_record = {
                    **pair_record,
                    "relationship_type": "near_identical",
                    "note": (
                        f"Columns are {resemblance:.1%} similar — likely duplicates that have "
                        f"diverged. Confirm if both are needed or if one has been incorrectly modified."
                    ),
                }
                duplicate_columns.append(near_identical_record)

            if not same_table:
                if self.classifier._candidate_name_pair(
                    row["table_name"],
                    row["column_name"],
                    other_row["table_name"],
                    other_row["column_name"],
                ):
                    join_paths_minhash.append(pair_record)

        # === PART 2: Coverage-based detection ===

        join_paths_coverage = []

        if dataframes is not None:
            coverage_joins = self.find_joins_by_coverage_with_data(
                column_summary,
                dataframes
            )

            # Preserve all PK/FK metadata from coverage detection.
            # Do not rebuild a smaller dictionary here.
            for cj in coverage_joins:
                coverage_record = dict(cj)
                coverage_record["resemblance"] = coverage_record.get("jaccard", 0)
                coverage_record.setdefault("resemblance_shingle", None)
                coverage_record["method"] = "coverage"
                join_paths_coverage.append(coverage_record)

        # === PART 3: Merge and deduplicate ===

        # Put coverage first because coverage records contain PK/FK metadata.
        all_joins = join_paths_coverage + join_paths_minhash

        join_by_pair: dict[tuple[str, str], dict] = {}

        for jp in all_joins:
            pair_key = tuple(sorted([
                f"{jp['table_a']}.{jp['col_a']}",
                f"{jp['table_b']}.{jp['col_b']}",
            ]))

            if pair_key not in join_by_pair:
                join_by_pair[pair_key] = jp
                continue

            existing = join_by_pair[pair_key]

            # Prefer coverage records because they contain relationship_type,
            # primary_key_table, foreign_key_table, etc.
            if jp.get("method") == "coverage" and existing.get("method") != "coverage":
                merged = {**existing, **jp}
                join_by_pair[pair_key] = merged
            else:
                # Otherwise keep the existing record, but fill missing fields.
                for k, v in jp.items():
                    if k not in existing or existing[k] is None:
                        existing[k] = v

        join_paths_final = list(join_by_pair.values())

        def sort_key(x):
            if x.get("relationship_type") == "foreign_key":
                return (0, -x.get("quality_score", 0), -max(x.get("coverage_a", 0), x.get("coverage_b", 0)))
            if x.get("relationship_type") == "one_to_one_key":
                return (1, -x.get("quality_score", 0), -max(x.get("coverage_a", 0), x.get("coverage_b", 0)))
            if x.get("method") == "coverage":
                return (2, -x.get("quality_score", 0), -max(x.get("coverage_a", 0), x.get("coverage_b", 0)))
            return (3, -x.get("resemblance", 0), 0)

        join_paths_final.sort(key=sort_key)
        # Fallback: classify any MinHash-only pairs that coverage detection missed.
        if dataframes is not None:
            for jp in join_paths_final:
                if jp.get("relationship_type") is not None:
                    continue
                if jp.get("table_a") == jp.get("table_b"):
                    continue
                ta, ca = jp["table_a"], jp["col_a"]
                tb, cb = jp["table_b"], jp["col_b"]
                df_a = dataframes.get(ta)
                df_b = dataframes.get(tb)
                if df_a is None or df_b is None:
                    continue
                if ca not in df_a.columns or cb not in df_b.columns:
                    continue
                vals_a = _normalize_val_set(df_a[ca])
                vals_b = _normalize_val_set(df_b[cb])
                if not vals_a or not vals_b:
                    continue
                overlap = vals_a & vals_b
                cov_a = len(overlap) / len(vals_a)
                cov_b = len(overlap) / len(vals_b)
                uniq_a = len(vals_a) / len(df_a)
                uniq_b = len(vals_b) / len(df_b)
                rel = self.classifier.identify_pk_fk_relationship(
                    table_a=ta,
                    col_a=ca,
                    table_b=tb,
                    col_b=cb,
                    coverage_a=cov_a,
                    coverage_b=cov_b,
                    cardinality_a=len(vals_a),
                    cardinality_b=len(vals_b),
                    uniqueness_a=uniq_a,
                    uniqueness_b=uniq_b,
                    series_a=df_a[ca],
                    series_b=df_b[cb],
                )
                jp.update(rel)
                jp["coverage_a"] = round(cov_a, 4)
                jp["coverage_b"] = round(cov_b, 4)
                jp["uniqueness_a"] = round(uniq_a, 4)
                jp["uniqueness_b"] = round(uniq_b, 4)
        # Keep ALL discovered candidates for broad MinHash/shingle join discovery.
        candidate_join_paths = [
            jp for jp in join_paths_final
            if jp.get("relationship_type") not in {
                "shared_value_domain",
                "unreported_ordinal_domain",
            }
        ]

        # Keep only high-confidence classified relationships for PK/FK/shared-domain sections.
        classified_relationship_paths = [
            jp for jp in join_paths_final
            if self._should_report_join(jp)
        ]

        duplicate_columns.sort(key=lambda x: -x["resemblance"])

        return {
            "candidate_join_paths": candidate_join_paths,
            "join_paths": classified_relationship_paths,
            "duplicate_columns": duplicate_columns,
        }

    def drop_duplicate_columns(
        self,
        df: pd.DataFrame,
        duplicate_columns: list[dict],
        table_name: str,
        keep: str = "first",
        dry_run: bool = True,
    ) -> pd.DataFrame:
        """Drop same-table duplicate columns (cross-table pairs are skipped)."""
        to_drop: set[str] = set()

        for dc in duplicate_columns:
            if dc["table_a"] != table_name or dc["table_b"] != table_name:
                continue
            drop_col = dc["col_b"] if keep == "first" else dc["col_a"]
            keep_col = dc["col_a"] if keep == "first" else dc["col_b"]
            if drop_col in df.columns and drop_col not in to_drop:
                to_drop.add(drop_col)
                prefix = "DRY RUN - " if dry_run else ""
                print(
                    f"  [{prefix}DROP] '{drop_col}' "
                    f"(duplicate of '{keep_col}', resemblance={dc['resemblance']})"
                )

        if dry_run:
            print(f"\n  Dry run complete. {len(to_drop)} column(s) would be dropped.")
            print("  Set dry_run=False to apply.")
            return df

        df = df.drop(columns=list(to_drop))
        print(f"\n  {len(to_drop)} duplicate column(s) dropped.")
        return df


def analyze_relationships(
    analyzer: MinHashAnalyzer,
    column_summaries: dict,
    profile_results: dict,
) -> dict:
    """Run cross-column relationship analysis for all datasets."""
    all_summaries = []
    for table_name, rows in column_summaries.items():
        for row in rows:
            all_summaries.append({
                **row,
                "table_name": row.get("table_name", table_name),
            })

    dataframes = {
        name: result["df"]
        for name, result in profile_results.items()
    }

    results = analyzer.find_joinable_columns(
        all_summaries,
        dataframes=dataframes,
    )
    report_relationships(results)
    report_cross_table_duplicates(results["duplicate_columns"])
    return results

