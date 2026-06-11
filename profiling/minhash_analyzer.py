"""
MinHashAnalyzer — cross-column join-path detection and duplicate identification.

Uses dual LSH indexes:
  - Exact MinHash LSH  : finds columns sharing the same actual values
  - Shingled MinHash LSH: finds columns sharing character-level vocabulary
                          (useful when values are corrupted, e.g. suburb typos)

ENHANCED: Now includes coverage-based join detection for foreign key relationships
where one table is much larger than the other (MinHash's Jaccard doesn't work well here).

Results flow into join_hints which the LLM uses as evidence.
"""

from collections import defaultdict

import pandas as pd
import numpy as np
from datasketch import MinHashLSH
from .utils import is_sequential_ordinal

from .config import PipelineConfig

def _normalize_val_set(series: "pd.Series") -> set:
    """Normalise a column to a set of strings, stripping .0 float artefacts."""
    s = series.dropna().astype(str).str.strip()
    # Remove trailing .0 so int-stored-as-float matches integer column
    s = s.str.replace(r'\.0+$', '', regex=True)
    return set(s[s != ''])


class MinHashAnalyzer:
    """Finds joinable columns and near-duplicate columns across all datasets."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Compatibility gate
    # ------------------------------------------------------------------

    @staticmethod
    def _are_compatible(kind_a: str, kind_b: str) -> bool:
        allowed = {"categorical", "discrete_code", "key_like"}
        return kind_a in allowed and kind_b in allowed
    
    @staticmethod
    def _norm_name(name: str) -> str:
        """Lowercase alphanumeric-only name used for loose schema matching."""
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    @classmethod
    def _table_stem(cls, table: str) -> str:
        """
        Crude singular table stem for schema-name scoring.
        Examples:
        Customers -> customer, OrderDetails -> orderdetail
        """
        t = cls._norm_name(table)
        return t[:-1] if t.endswith("s") and len(t) > 3 else t

    @classmethod
    def _stem_id_name(cls, name: str) -> str:
        """
        Return semantic stem of an ID-like name.
        Examples:
        CustomerId -> customer
        customer_id -> customer
        customer_key -> customer
        """
        n = cls._norm_name(name)
        for suffix in ("identifier", "id", "key"):
            if n.endswith(suffix) and len(n) > len(suffix):
                return n[: -len(suffix)]
        return n

    @classmethod
    def _is_id_like_name(cls, col: str) -> bool:
        """
        Generic ID-like name detector.
        Does not assume a specific dataset.
        """
        n = cls._norm_name(col)
        if (
            n == "id"
            or n.endswith("id")
            or n.endswith("key")
            or "identifier" in n
            or n in {
                "reportsto", "manager", "managerid",
                "parent", "parentid",
                "supervisor", "supervisorid",
                "owner", "ownerid",
            }
        ):
            return True
        # Token-level check: split on separators so 'ID_No' → {'id','no'}
        # and 'Account_No' → {'account','no'} — catches compound names where
        # 'id' is a component token rather than a suffix.
        import re as _re
        tokens = set(_re.split(r'[^a-z0-9]+', col.lower()))
        return bool(tokens & {"id", "key", "identifier", "uuid", "ssn"})
    
    def _cfg_set(self, name: str, default: set[str]) -> set[str]:
        """Read configurable relationship heuristic terms from PipelineConfig."""
        value = getattr(self.config, name, default)
        return set(value or default)

    def _is_descriptive_name(self, col: str) -> bool:
        """
        Descriptive attributes can overlap across tables, but should not become PK/FK relationships.
        The term list is configurable in PipelineConfig.
        """
        n = self._norm_name(col)

        prefixes = self._cfg_set("relationship_descriptive_prefixes", set())
        descriptive_terms = self._cfg_set("relationship_descriptive_terms", set())

        for p in prefixes:
            if n.startswith(p) and len(n) > len(p):
                n = n[len(p):]
                break

        return n in descriptive_terms

    def _looks_like_shared_value_domain(self, col_a: str, col_b: str) -> bool:
        """
        True for descriptive copied domains like BillingCity <-> City.
        These are useful consistency checks, not safe PK/FK joins.
        """
        a = self._norm_name(col_a)
        b = self._norm_name(col_b)

        prefixes = self._cfg_set("relationship_descriptive_prefixes", set())
        descriptive_terms = self._cfg_set("relationship_descriptive_terms", set())

        for p in prefixes:
            if a.startswith(p) and len(a) > len(p):
                a = a[len(p):]
            if b.startswith(p) and len(b) > len(p):
                b = b[len(p):]

        return a == b and a in descriptive_terms

    def _looks_like_hierarchy_fk(
        self,
        table_a: str,
        col_a: str,
        table_b: str,
        col_b: str,
    ) -> bool:
        """
        Generic self-reference detector:
        e.g. parent_id, manager_id, reports_to, supervisor_id.
        Terms are configurable in PipelineConfig.
        """
        if self._table_stem(table_a) != self._table_stem(table_b):
            return False

        a = self._norm_name(col_a)
        b = self._norm_name(col_b)

        hierarchy_terms = self._cfg_set("relationship_hierarchy_terms", set())

        return any(term in a for term in hierarchy_terms) or any(term in b for term in hierarchy_terms)

    def _relationship_name_score(
        self,
        pk_table: str,
        pk_col: str,
        fk_table: str,
        fk_col: str,
    ) -> float:
        """
        Score whether fk_col is semantically plausible as a FK to pk_table.pk_col.
        Uses generic configurable heuristics, not dataset-specific table names.
        """
        pk_col_n = self._norm_name(pk_col)
        fk_col_n = self._norm_name(fk_col)
        pk_table_n = self._table_stem(pk_table)

        pk_stem = self._stem_id_name(pk_col)
        fk_stem = self._stem_id_name(fk_col)

        # Exact same identifier name, e.g. CustomerId in both tables.
        if pk_col_n == fk_col_n and self._is_id_like_name(pk_col):
            return 1.0

        # Classic table-name FK:
        # Parent.ParentId <- Child.ParentId
        if self._is_id_like_name(pk_col) and fk_stem in {pk_table_n, pk_stem}:
            return 0.95

        # FK column includes parent table concept:
        # CreatedByUserId -> User.UserId
        if self._is_id_like_name(fk_col) and pk_table_n and pk_table_n in fk_stem:
            return 0.90

        # Same-table hierarchy:
        # ReportsTo -> EmployeeId, ParentId -> CategoryId, etc.
        if self._looks_like_hierarchy_fk(pk_table, pk_col, fk_table, fk_col):
            return 0.95

        # Configurable assignment relationship:
        # e.g. support_rep_id / owner_id / assignee_id may reference a people-like table.
        assignment_parent_terms = self._cfg_set("relationship_assignment_parent_terms", set())
        assignment_fk_terms = self._cfg_set("relationship_assignment_fk_terms", set())

        if pk_table_n in assignment_parent_terms and any(term in fk_col_n for term in assignment_fk_terms):
            return 0.85

        return 0.0

    def _candidate_name_pair(
        self,
        table_a: str,
        col_a: str,
        table_b: str,
        col_b: str,
    ) -> bool:
        """
        Cheap schema-name gate before reporting MinHash candidate paths.

        Keep pairs that are name-plausible:
        - same column name across different tables
        - generic id-vs-specific-id pairs, e.g. id ↔ soc_sec_id
        - shared descriptive domains, e.g. BillingCity ↔ City
        - FK-style names supported by relationship_name_score()

        This prevents accidental numeric overlaps like AlbumId ↔ InvoiceId.
        """
        if table_a == table_b and col_a == col_b:
            return False

        a = self._norm_name(col_a)
        b = self._norm_name(col_b)

        # Same column name across different tables:
        # Account_No ↔ Account_No, CustomerId ↔ CustomerId, TrackId ↔ TrackId.
        if table_a != table_b and a == b:
            return True

        # Generic record-linkage style:
        # id ↔ soc_sec_id, id ↔ customer_id, id ↔ person_identifier.
        # This keeps FEBRL4 soc_sec_id ↔ id without hardcoding FEBRL.
        if table_a != table_b:
            if (a == "id" and self._is_id_like_name(col_b)) or (
                b == "id" and self._is_id_like_name(col_a)
            ):
                return True

        # Descriptive domains such as Customer.City ↔ Invoice.BillingCity.
        if self._looks_like_shared_value_domain(col_a, col_b):
            return True

        score_a_pk = self._relationship_name_score(table_a, col_a, table_b, col_b)
        score_b_pk = self._relationship_name_score(table_b, col_b, table_a, col_a)

        return max(score_a_pk, score_b_pk) >= 0.65

    @classmethod
    def _should_report_join(cls, jp: dict) -> bool:
        """
        Final report filter. This prevents raw numeric overlap/debug pairs
        from leaking into the Word report.
        """
        rel_type = jp.get("relationship_type")

        if rel_type in {"foreign_key", "one_to_one_key", "shared_value_domain"}:
            return True

        # Keep old shared_join_key only if both sides are non-numeric and clearly named alike.
        if rel_type == "shared_join_key":
            type_a = str(jp.get("type_a", "")).lower()
            type_b = str(jp.get("type_b", "")).lower()
            numeric = ("int", "float", "double", "decimal", "number")
            if any(t in type_a for t in numeric) or any(t in type_b for t in numeric):
                return False
            return True

        # Hide low-confidence / unclassified MinHash-only pairs from final report.
        return False

    # ------------------------------------------------------------------
    # Coverage-based join detection
    # ------------------------------------------------------------------
    
    def compute_join_quality_score(
        self,
        coverage_a: float,
        coverage_b: float,
        cardinality_a: int,
        cardinality_b: int,
        overlap_count: int,
        uniqueness_a: float,
        uniqueness_b: float,
    ) -> dict:
        """
        Compute a single join quality score (0-1) combining all signals.
        
        Score interpretation:
            0.9-1.0: Excellent join (definitely FK→PK)
            0.7-0.9: Good join (likely FK→PK)
            0.5-0.7: Possible join (needs review)
            0.0-0.5: Poor join (likely false positive)
        """
        
        # Component 1: Coverage strength
        coverage_strength = max(coverage_a, coverage_b)
        
        # Component 2: Directional asymmetry
        asymmetry = abs(coverage_a - coverage_b)
        
        # Component 3: Uniqueness
        uniqueness_strength = max(uniqueness_a, uniqueness_b)
        
        # Component 4: Size asymmetry
        smaller = min(cardinality_a, cardinality_b)
        larger = max(cardinality_a, cardinality_b)
        size_asymmetry = 1 - (smaller / larger) if larger > 0 else 0
        
        # Component 5: Overlap significance (sigmoid)
        overlap_significance = 1 / (1 + np.exp(-0.005 * (overlap_count - 1000)))
        
        # Weighted composite score
        score = (
            0.25 * coverage_strength +
            0.25 * asymmetry +
            0.20 * uniqueness_strength +
            0.15 * size_asymmetry +
            0.15 * overlap_significance
        )

        cfg = self.config
        if min(coverage_a, coverage_b) >= cfg.coverage_join_threshold:
            score = max(score, cfg.join_quality_threshold)

        if (
            max(coverage_a, coverage_b) >= 0.99
            and min(coverage_a, coverage_b) >= 0.70
        ):
            score = max(score, cfg.join_quality_threshold)

        # Grade the score
        if score >= 0.9:
            grade = "Excellent"
        elif score >= 0.7:
            grade = "Good"
        elif score >= 0.5:
            grade = "Possible"
        else:
            grade = "Poor"

        return {
            'score': round(score, 4),
            'grade': grade,
            'coverage_strength': round(coverage_strength, 4),
            'asymmetry': round(asymmetry, 4),
            'uniqueness_strength': round(uniqueness_strength, 4),
            'size_asymmetry': round(size_asymmetry, 4),
            'overlap_significance': round(overlap_significance, 4),
        }
    
    def identify_pk_fk_relationship(
        self,
        table_a: str,
        col_a: str,
        table_b: str,
        col_b: str,
        coverage_a: float,
        coverage_b: float,
        cardinality_a: int,
        cardinality_b: int,
        uniqueness_a: float,
        uniqueness_b: float,
        name_score_a_pk: float | None = None,
        name_score_b_pk: float | None = None,
        series_a: pd.Series | None = None,
        series_b: pd.Series | None = None,
    ) -> dict:
        """
        Classify an overlapping pair.

        coverage_a = % of A distinct values found in B.
        coverage_b = % of B distinct values found in A.

        If A is PK and B is FK, then B values should exist in A,
        so coverage_b should be high.
        """
        cfg = self.config

        pk_uniqueness_threshold = cfg.pk_uniqueness_threshold
        fk_coverage_threshold = cfg.fk_coverage_threshold
        one_to_one_coverage_threshold = cfg.one_to_one_coverage_threshold

        if name_score_a_pk is None:
            name_score_a_pk = self._relationship_name_score(table_a, col_a, table_b, col_b)

        if name_score_b_pk is None:
            name_score_b_pk = self._relationship_name_score(table_b, col_b, table_a, col_a)

        a_unique = uniqueness_a >= pk_uniqueness_threshold
        b_unique = uniqueness_b >= pk_uniqueness_threshold

        b_values_exist_in_a = coverage_b >= fk_coverage_threshold
        a_values_exist_in_b = coverage_a >= fk_coverage_threshold

        # IMPORTANT: descriptive copied attributes must be handled BEFORE FK logic.
        # Example: Customer.Address <-> Invoice.BillingAddress.
        if self._looks_like_shared_value_domain(col_a, col_b):
            return {
                "relationship_type": "shared_value_domain",
                "interpretation": (
                    f"{table_a}.{col_a} and {table_b}.{col_b} share a descriptive value domain. "
                    f"This can support consistency checks, but should not be treated as a primary/foreign key relationship."
                ),
            }
        
        _a_is_sequential_ordinal = series_a is not None and is_sequential_ordinal(series_a)
        _b_is_sequential_ordinal = series_b is not None and is_sequential_ordinal(series_b)

        if (
            table_a != table_b
            and _a_is_sequential_ordinal
            and _b_is_sequential_ordinal
            and not self._is_id_like_name(col_a)
            and not self._is_id_like_name(col_b)
        ):
            return {
                "relationship_type": "unreported_ordinal_domain",
                "interpretation": (
                    f"{table_a}.{col_a} and {table_b}.{col_b} both look like "
                    f"sequential ordinal/rank fields. They may overlap strongly, "
                    f"but should not be treated as stable join keys."
                ),
            }

        # One-to-one shared key, e.g. same business identifier in two aligned tables.
        _same_col_name = self._norm_name(col_a) == self._norm_name(col_b)
        _id_name_evidence = (
            self._is_id_like_name(col_a)
            and self._is_id_like_name(col_b)
            and max(name_score_a_pk, name_score_b_pk) >= 0.75
        )
        
        if (
            a_unique
            and b_unique
            and coverage_a >= one_to_one_coverage_threshold
            and coverage_b >= one_to_one_coverage_threshold
            and (_same_col_name or _id_name_evidence)
        ):
            min_coverage = min(coverage_a, coverage_b)
            if min_coverage < 0.99:
                weaker_side = f"{table_a}.{col_a}" if coverage_a < coverage_b else f"{table_b}.{col_b}"
                gap_pct = round((1 - min_coverage) * 100, 1)
                coverage_note = (
                    f" Note: {gap_pct}% of values on the weaker side ({weaker_side}) have no match — "
                    f"this is likely caused by a duplicate or unmatched record on that side; "
                    f"investigate before using as a join key."
                )
            else:
                coverage_note = ""

            return {
                "relationship_type": "one_to_one_key",
                "key_a": f"{table_a}.{col_a}",
                "key_b": f"{table_b}.{col_b}",
                "direction": f"{table_a}.{col_a} ↔ {table_b}.{col_b}",
                "interpretation": (
                    f"{table_a}.{col_a} and {table_b}.{col_b} are both highly unique "
                    f"and strongly overlapping, suggesting a one-to-one shared key.{coverage_note}"
                ),
            }

        # A is PK, B is FK.
        if (
            a_unique
            and b_values_exist_in_a
            and name_score_a_pk >= 0.75
            and self._is_id_like_name(col_a)
        ):
            return {
                "relationship_type": "foreign_key",
                "primary_key_table": table_a,
                "primary_key_column": col_a,
                "foreign_key_table": table_b,
                "foreign_key_column": col_b,
                "direction": f"{table_b}.{col_b} → {table_a}.{col_a}",
                "interpretation": f"{table_b}.{col_b} likely references {table_a}.{col_a}.",
            }

        # B is PK, A is FK.
        if (
            b_unique
            and a_values_exist_in_b
            and name_score_b_pk >= 0.75
            and self._is_id_like_name(col_b)
        ):
            return {
                "relationship_type": "foreign_key",
                "primary_key_table": table_b,
                "primary_key_column": col_b,
                "foreign_key_table": table_a,
                "foreign_key_column": col_a,
                "direction": f"{table_a}.{col_a} → {table_b}.{col_b}",
                "interpretation": f"{table_a}.{col_a} likely references {table_b}.{col_b}.",
            }

        # Do NOT report numeric shared domains as join paths by default.
        # Example: two child tables both contain TrackId; useful internally, but not a direct join.
        if (
            self._is_id_like_name(col_a)
            and self._is_id_like_name(col_b)
            and (coverage_a >= fk_coverage_threshold or coverage_b >= fk_coverage_threshold)
        ):
            return {
                "relationship_type": "unreported_shared_id_domain",
                "interpretation": (
                    f"{table_a}.{col_a} and {table_b}.{col_b} share an identifier value domain, "
                    f"but neither side is clearly the parent key."
                ),
            }

        return {
            "relationship_type": "many_to_many_or_lookup",
            "interpretation": (
                f"{table_a}.{col_a} and {table_b}.{col_b} share values, "
                f"but the relationship is ambiguous and should be reviewed manually."
            ),
        }

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
                if table_a == table_b and not self._looks_like_hierarchy_fk(table_a, col_a, table_b, col_b):
                    continue

                # Universal name gate: prevents accidental numeric overlap.
                if not self._candidate_name_pair(table_a, col_a, table_b, col_b):
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

                name_score_a_pk = self._relationship_name_score(
                    table_a, col_a, table_b, col_b
                )
                name_score_b_pk = self._relationship_name_score(
                    table_b, col_b, table_a, col_a
                )

                relationship = self.identify_pk_fk_relationship(
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

                quality = self.compute_join_quality_score(
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
    
    

    # ------------------------------------------------------------------
    # Core analysis (MODIFIED to use both methods)
    # ------------------------------------------------------------------

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
                if self._candidate_name_pair(
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
                rel = self.identify_pk_fk_relationship(
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

    # ------------------------------------------------------------------
    # Join hints for LLM evidence (UPDATED)
    # ------------------------------------------------------------------

    def build_join_hints(self, minhash_results: dict) -> dict[str, list[str]]:
        """Convert MinHash findings into human-readable hints for each column key."""
        hints: dict[str, list[str]] = defaultdict(list)

        for jp in minhash_results["join_paths"]:
            key_a = f"{jp['table_a']}.{jp['col_a']}"
            key_b = f"{jp['table_b']}.{jp['col_b']}"
            rel_type = jp.get("relationship_type")

            if rel_type == "foreign_key":
                pk_table = jp.get("primary_key_table")
                pk_col  = jp.get("primary_key_column")
                fk_table = jp.get("foreign_key_table")
                fk_col  = jp.get("foreign_key_column")
                pk_key  = f"{pk_table}.{pk_col}"
                fk_key  = f"{fk_table}.{fk_col}"
                coverage = max(jp.get("coverage_a", 0), jp.get("coverage_b", 0))
                hints[fk_key].append(
                    f"foreign key candidate: {fk_key} references primary key candidate "
                    f"{pk_key} with {coverage:.1%} referential integrity"
                )
                hints[pk_key].append(
                    f"primary key candidate: {pk_key} is referenced by foreign key "
                    f"{fk_key} with {coverage:.1%} referential integrity"
                )

            elif rel_type == "one_to_one_key":
                cov_a = jp.get("coverage_a", 0)
                cov_b = jp.get("coverage_b", 0)
                hints[key_a].append(
                    f"one-to-one shared key with '{key_b}' "
                    f"(coverage {cov_a:.1%} / {cov_b:.1%})"
                )
                hints[key_b].append(
                    f"one-to-one shared key with '{key_a}' "
                    f"(coverage {cov_a:.1%} / {cov_b:.1%})"
                )
            
            elif rel_type == "shared_value_domain":
                cov_a = jp.get("coverage_a", 0)
                cov_b = jp.get("coverage_b", 0)

                hints[key_a].append(
                    f"shared value domain with '{key_b}' "
                    f"(coverage {cov_a:.1%} / {cov_b:.1%}); "
                    f"not a primary/foreign key relationship"
                )
                hints[key_b].append(
                    f"shared value domain with '{key_a}' "
                    f"(coverage {cov_a:.1%} / {cov_b:.1%}); "
                    f"not a primary/foreign key relationship"
                )

            elif rel_type == "shared_join_key":
                cov_a = jp.get("coverage_a", 0)
                cov_b = jp.get("coverage_b", 0)
                hints[key_a].append(
                    f"shared join key with '{key_b}' "
                    f"(coverage {cov_a:.1%} / {cov_b:.1%})"
                )
                hints[key_b].append(
                    f"shared join key with '{key_a}' "
                    f"(coverage {cov_a:.1%} / {cov_b:.1%})"
                )

            else:
                # MinHash-only (unclassified) — symmetric, no FK framing
                type_a = str(jp.get("type_a", "")).lower()
                type_b = str(jp.get("type_b", "")).lower()
                numeric_markers = ("int", "float", "double", "decimal", "number")

                if any(t in type_a for t in numeric_markers) or any(t in type_b for t in numeric_markers):
                    continue
                shingle_note = (
                    f", shingle resemblance={jp['resemblance_shingle']} (fuzzy/typo-tolerant)"
                    if jp.get("resemblance_shingle") is not None else ""
                )
                hint_a = (
                    f"high value overlap with '{key_b}' "
                    f"(exact resemblance={jp['resemblance']}{shingle_note}) — possible join path"
                )
                hint_b = (
                    f"high value overlap with '{key_a}' "
                    f"(exact resemblance={jp['resemblance']}{shingle_note}) — possible join path"
                )
                hints[key_a].append(hint_a)
                hints[key_b].append(hint_b)

        for dc in minhash_results["duplicate_columns"]:
            key_a = f"{dc['table_a']}.{dc['col_a']}"
            key_b = f"{dc['table_b']}.{dc['col_b']}"
            hint = f"(resemblance={dc['resemblance']}) — likely duplicate column"
            hints[key_a].append(f"near-identical values to '{key_b}' {hint}")
            hints[key_b].append(f"near-identical values to '{key_a}' {hint}")

        return dict(hints)

    # ------------------------------------------------------------------
    # Reporting (UPDATED)
    # ------------------------------------------------------------------

    def report(self, minhash_results: dict) -> None:
        join_paths = minhash_results["join_paths"]
        duplicate_columns = minhash_results["duplicate_columns"]

        print()
        print("=" * 60)
        print(f"POSSIBLE JOIN PATHS ({len(join_paths)} found)")
        print("=" * 60)
        if join_paths:
            for jp in join_paths:
                if jp.get("method") == "coverage":
                    # Coverage-based join
                    print(
                        f"  {jp['table_a']}.{jp['col_a']} <--> "
                        f"{jp['table_b']}.{jp['col_b']}"
                        f"  | coverage: {jp['coverage_a']:.1%} → {jp['coverage_b']:.1%}"
                        f"  | jaccard: {jp['resemblance']}"
                        f"  | overlap: {jp['overlap_count']} values"
                        f"  | method: coverage-based ✓"
                    )
                else:
                    # MinHash-based join
                    shingle_str = (
                        f" | shingle resemblance: {jp['resemblance_shingle']}"
                        if jp.get("resemblance_shingle") is not None else ""
                    )
                    print(
                        f"  {jp['table_a']}.{jp['col_a']} <--> "
                        f"{jp['table_b']}.{jp['col_b']}"
                        f"  | exact resemblance: {jp['resemblance']}"
                        f"{shingle_str}"
                        f"  | types: {jp['type_a']} / {jp['type_b']}"
                        f"  | method: minhash"
                    )
        else:
            print("  None found.")

        print()
        print("=" * 60)
        print(f"LIKELY DUPLICATE COLUMNS ({len(duplicate_columns)} found)")
        print("=" * 60)
        if duplicate_columns:
            for dc in duplicate_columns:
                print(
                    f"  {dc['table_a']}.{dc['col_a']} <--> "
                    f"{dc['table_b']}.{dc['col_b']}"
                    f"  | resemblance: {dc['resemblance']}"
                    f"  | types: {dc['type_a']} / {dc['type_b']}"
                )
        else:
            print("  None found.")

    def report_cross_table_duplicates(self, duplicate_columns: list[dict]) -> None:
        cross = [dc for dc in duplicate_columns if dc["table_a"] != dc["table_b"]]
        print()
        print("=" * 60)
        print(f"CROSS-TABLE DUPLICATE COLUMNS ({len(cross)} found — review manually)")
        print("=" * 60)
        if cross:
            for dc in cross:
                print(
                    f"  {dc['table_a']}.{dc['col_a']} <--> "
                    f"{dc['table_b']}.{dc['col_b']}"
                    f"  | resemblance: {dc['resemblance']}"
                )
        else:
            print("  None found.")

    # ------------------------------------------------------------------
    # Duplicate column dropping
    # ------------------------------------------------------------------

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
