"""Classify candidate column relationships using schema and coverage evidence.

The logic was moved from MinHashAnalyzer without changing its decisions.
"""

import numpy as np
import pandas as pd

from ..core.config import PipelineConfig
from ..core.utils import is_sequential_ordinal


class RelationshipClassifier:
    """Interpret candidate pairs as keys, foreign keys, or shared domains."""

    def __init__(self, config: PipelineConfig):
        self.config = config

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
