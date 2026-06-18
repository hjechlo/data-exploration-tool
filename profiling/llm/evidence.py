"""Build evidence payloads for column-level dictionary generation."""

import re


def prepare_dictionary_evidence(
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
        suppress_abstract_format_for_category = (
            str(row.get("data_type", "")).lower() in {"string", "object", "category"}
            and row.get("permissible_values") is not None
            and (n_distinct >= 2)
            and (unique_ratio <= 0.5)
        )
        entry = {
            "table_name": table_name,
            "column_name": row["column_name"],
            "data_type": row["data_type"],
            "profile_type": row.get("profile_type"),
            "sample_values": row["sample_values"],
            "observed_distinct_values": row.get("permissible_values"),
            "column_facts": [
                f
                for f in row.get("column_facts", [])
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
                stripped = re.sub("[\\s.'\\-\\/_(),:&]+", "", str(pattern))
                if not stripped:
                    return False
                return all((c == "a" for c in stripped)) or all(
                    (c == "X" for c in stripped)
                )

            raw_fingerprints = format_analysis.get("format_fingerprints", {})
            raw_top_formats = raw_fingerprints.get("top_formats", [])
            meaningful_formats = [
                f
                for f in raw_top_formats
                if not is_length_variation_only(f.get("pattern", ""))
            ]
            raw_coercibility = format_analysis.get("coercibility", {})
            non_coercible = raw_coercibility.get("non_coercible_formats", [])
            coercible_fmts = raw_coercibility.get("coercible_formats", [])
            all_flagged_length_variation = (
                all(
                    (
                        is_length_variation_only(f.get("pattern", ""))
                        for f in non_coercible + coercible_fmts
                    )
                )
                if non_coercible or coercible_fmts
                else True
            )
            is_numeric_precision_only = (
                raw_coercibility.get("reason")
                == "Numeric precision variation only — not a format issue"
            )
            fa_entry = {}
            uniformity = (
                None
                if is_numeric_precision_only
                else format_analysis.get("uniformity_score", 0)
            )
            if uniformity and (not row.get("permissible_values")):
                fa_entry["uniformity_score"] = uniformity
            has_format_issue = (
                raw_coercibility.get("is_coercible")
                or bool(raw_coercibility.get("non_coercible_formats"))
            ) and (not all_flagged_length_variation)
            if (
                meaningful_formats
                and has_format_issue
                and (not suppress_abstract_format_for_category)
            ):
                fa_entry["format_fingerprints"] = {"top_formats": meaningful_formats}
            if (
                not all_flagged_length_variation
                and raw_coercibility
                and (not suppress_abstract_format_for_category)
            ):
                fa_entry["coercibility"] = raw_coercibility
            raw_anomalies = format_analysis.get("anomalies", {})
            if raw_anomalies.get("total_anomaly_count", 0) > 0:
                fa_entry["anomalies"] = {
                    k: v
                    for k, v in raw_anomalies.items()
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
