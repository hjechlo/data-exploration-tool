"""Human-readable relationship hints, reporting, and annotations.

These functions format relationship-analysis results but do not discover or classify
relationships.
"""

from collections import defaultdict

from .column_errors import remove_fk_errors_from_results

def build_join_hints(minhash_results: dict) -> dict[str, list[str]]:
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

        # elif rel_type == "shared_join_key":
        #     cov_a = jp.get("coverage_a", 0)
        #     cov_b = jp.get("coverage_b", 0)
        #     hints[key_a].append(
        #         f"shared join key with '{key_b}' "
        #         f"(coverage {cov_a:.1%} / {cov_b:.1%})"
        #     )
        #     hints[key_b].append(
        #         f"shared join key with '{key_a}' "
        #         f"(coverage {cov_a:.1%} / {cov_b:.1%})"
        #     )

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

def report_relationships(minhash_results: dict) -> None:
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

def report_cross_table_duplicates(duplicate_columns: list[dict]) -> None:
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

def annotate_relationship_roles(
    column_summaries: dict,
    minhash_results: dict,
) -> dict:
    """
    Annotate each column summary row with its relationship_role and join_hints.
    Called before LLM generation so the LLM evidence includes PK/FK context.
    """
    join_hints_text = build_join_hints(minhash_results)
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
        # elif rel_type == "shared_join_key":
        #     key_a = f"{jp['table_a']}.{jp['col_a']}"
        #     key_b = f"{jp['table_b']}.{jp['col_b']}"
        #     relationship_roles[key_a] = "join_key"
        #     relationship_roles[key_b] = "join_key"

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

def cleanup_fk_errors(
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
            cleaned = remove_fk_errors_from_results(
                summary, identified_fks
            )
            cleaned_summaries[table_name] = cleaned
        else:
            cleaned_summaries[table_name] = summary

    return cleaned_summaries
