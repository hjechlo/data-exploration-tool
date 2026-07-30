from ..llm.utils import clean_output
from .rules import generate_rules_for_tables
from .failures import validate_tables
from .graph_state import PipelineState
from .results import _apply_rule_check
from langgraph.types import Send
from langgraph.graph import END
from ..validation.prompts import (
    INSPECT_RULES_PROMPT_WITH_HISTORY,
    REVISE_RULE_PROMPT_WITH_CONTEXT,
    ASSESS_RULES_PROMPT,
)
import json

MAX_REVISIONS = 2
MAX_REGENERATIONS = 1


def make_nodes(config, llm_generator, profile_results, all_dictionaries,
               minhash_results, dataset_summaries):
    """Factory that closes over non-serialisable dependencies."""

    def node_generate_rules(state: PipelineState) -> dict:
        rules = generate_rules_for_tables(
            config=config,
            llm_generator=llm_generator,
            column_summaries=all_dictionaries,
            minhash_results=minhash_results,
            profile_results=profile_results,
            dataset_descriptions=dataset_summaries,
        )
        rules_path = config.output_dir / "rules_pass_0.json"
        rules_path.write_text(
            json.dumps(rules, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )
        print(f"  [generate_rules] Initial rules saved to rules_pass_0.json")

        return {
            "validation_rules":     rules,
            "revision_count":       0,
            "regeneration_count":   0,
            "tables_to_regenerate": [],
            "revision_history":     [],
        }

    def node_assess_rules(state: PipelineState) -> dict:
        rules = state["validation_rules"]

        # Build column evidence summary per table
        column_evidence = {}
        for table_name, col_rows in all_dictionaries.items():
            column_evidence[table_name] = [
                {
                    "column":   row.get("column_name"),
                    "dtype":    row.get("dtype"),
                    "has_errors": bool(row.get("errors")),
                    "n_distinct": row.get("n_distinct"),
                    "n_missing":  row.get("n_missing"),
                }
                for row in col_rows
            ]

        # Build join path summary
        join_paths = []
        for result in minhash_results.get("candidate_pairs", []):
            join_paths.append({
                "table_a":  result.get("table_a"),
                "col_a":    result.get("col_a"),
                "table_b":  result.get("table_b"),
                "col_b":    result.get("col_b"),
                "score":    result.get("exact_resemblance"),
            })

        # Rule count summary per table for display
        for table_name, table_rules in rules.items():
            n_cols = len(all_dictionaries.get(table_name, []))
            print(f"  [assess_rules] {table_name}: {len(table_rules)} rules "
                  f"for {n_cols} columns "
                  f"({len(table_rules)/n_cols:.1f} rules/col)")

        prompt = ASSESS_RULES_PROMPT.replace(
            "{rules_json}", json.dumps(rules, indent=2, ensure_ascii=False, default=str)
        ).replace(
            "{column_evidence_json}", json.dumps(column_evidence, indent=2)
        ).replace(
            "{join_paths_json}", json.dumps(join_paths, indent=2)
        )

        raw = llm_generator.call(prompt)

        try:
            verdicts = clean_output(raw)
            print(f"  [assess_rules] LLM assessment:")
            for v in verdicts:
                print(f"    {v.get('table')} → {v.get('verdict', '').upper()} | "
                      f"{v.get('reason', '')[:80]}")
                if v.get("problem_columns"):
                    print(f"    Problem columns: {v.get('problem_columns')}")
        except Exception as e:
            print(f"  [assess_rules] Could not parse LLM response: {e} — proceeding")
            verdicts = []

        tables_to_regenerate = [
            v["table"] for v in verdicts
            if v.get("verdict") == "regenerate"
            and v.get("table") in rules
        ]

        if tables_to_regenerate:
            print(f"  [assess_rules] Tables flagged for regeneration: {tables_to_regenerate}")
        else:
            print(f"  [assess_rules] All tables cleared — proceeding to validation.")

        return {"tables_to_regenerate": tables_to_regenerate}

    def node_regenerate_rules(state: PipelineState) -> dict:
        tables = state.get("tables_to_regenerate", [])
        regeneration_num = state["regeneration_count"] + 1

        print(f"\n  [regenerate_rules] Regenerating rules for: {tables} "
              f"(regeneration {regeneration_num}/{MAX_REGENERATIONS})")

        # Only regenerate the flagged tables — keep rules for other tables
        partial_column_summaries = {
            t: all_dictionaries[t] for t in tables if t in all_dictionaries
        }
        partial_profile_results = {
            t: profile_results[t] for t in tables if t in profile_results
        }
        partial_dataset_summaries = {
            t: dataset_summaries[t] for t in tables if t in dataset_summaries
        }

        new_rules = generate_rules_for_tables(
            config=config,
            llm_generator=llm_generator,
            column_summaries=partial_column_summaries,
            minhash_results=minhash_results,
            profile_results=partial_profile_results,
            dataset_descriptions=partial_dataset_summaries,
        )

        # Merge new rules into existing rules
        merged_rules = dict(state["validation_rules"])
        for table_name, table_rules in new_rules.items():
            merged_rules[table_name] = table_rules
            print(f"  [regenerate_rules] {table_name}: regenerated "
                  f"{len(table_rules)} rules")

        rules_path = config.output_dir / f"rules_regenerated_{regeneration_num}.json"
        rules_path.write_text(
            json.dumps(merged_rules, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )
        print(f"  [regenerate_rules] Saved to rules_regenerated_{regeneration_num}.json")

        return {
            "validation_rules":     merged_rules,
            "regeneration_count":   regeneration_num,
            "tables_to_regenerate": [],
            "validation_check_results": {},
            "inspection_notes":     [],
        }

    def fan_out_validation(state: PipelineState) -> list[Send]:
        return [
            Send("validate_table", {
                "table_name": table_name,
                "rules":      state["validation_rules"],
            })
            for table_name in state["validation_rules"]
        ]

    def node_validate_table(state: dict) -> dict:
        single_table_rules = {
            state["table_name"]: state["rules"][state["table_name"]]
        }
        result = validate_tables(
            config=config,
            llm_generator=llm_generator,
            validation_rules=single_table_rules,
            profile_results=profile_results,
        )
        return {"validation_check_results": result}

    def node_inspect_rules(state: PipelineState) -> dict:
        suspicious = []
        revision_count = state.get("revision_count", 0)
        revision_history = state.get("revision_history", [])

        previously_revised = {}
        already_kept = set()
        for entry in revision_history:
            key = (entry["table"], entry["rule_id"])
            previously_revised.setdefault(key, []).append(entry)
            if entry.get("verdict") == "keep":
                already_kept.add(key)

        for table_name, result in state["validation_check_results"].items():
            total_rows = len(profile_results[table_name]["df"])

            columns_with_errors = {
                row["column_name"]
                for row in all_dictionaries.get(table_name, [])
                if row.get("errors")
            }

            rule_id_to_samples: dict[int, list[dict]] = {}
            for record in result.get("violation_records", []):
                rules_failed_str = record.get("Validation Rules Failed", "")
                failed_cols = [c.strip() for c in record.get("Failed Column", "").split(",")]
                failed_vals = [v.strip() for v in record.get("Failed Value", "").split("|")]
                for segment in rules_failed_str.split("|"):
                    segment = segment.strip()
                    if segment.startswith("#"):
                        try:
                            rid = int(segment.split(":")[0].lstrip("#").strip())
                        except ValueError:
                            continue
                        if rid not in rule_id_to_samples:
                            rule_id_to_samples[rid] = []
                        if len(rule_id_to_samples[rid]) < 5:
                            for fc, fv in zip(failed_cols, failed_vals):
                                rule_id_to_samples[rid].append({
                                    "record_id": record.get("Record Identifier", "—"),
                                    "column": fc,
                                    "failed_value": fv,
                                })
                            break

            for rule_result in result.get("per_rule", []):
                n = rule_result.get("n_violations") or 0
                rate = n / total_rows if total_rows else 0
                col = rule_result.get("column", "")
                rule_id = rule_result.get("rule_id")

                is_total_failure = rate == 1.0
                has_error = bool(rule_result.get("error"))
                is_high_rate = rate >= 0.8 and n > 0

                if is_total_failure or has_error or is_high_rate:
                    if (table_name, rule_id) in already_kept:
                        print(f"    Skipping Rule #{rule_id} ({table_name}) "
                              f"— already kept in a prior pass")
                        continue
                    prior = previously_revised.get((table_name, rule_id), [])
                    suspicious.append({
                        "table":                table_name,
                        "rule_id":              rule_id,
                        "rule":                 rule_result["rule"],
                        "type":                 rule_result["type"],
                        "failure_rate":         round(rate, 3),
                        "n_violations":         n,
                        "column":               col,
                        "error":                rule_result.get("error", None),
                        "sample_failing_values": rule_id_to_samples.get(rule_id, []),
                        "prior_revisions":      prior,
                    })

        print(f"\n  [inspect_rules] Revision count: {revision_count}")

        if not suspicious:
            print(f"  [inspect_rules] No suspicious rules found — exiting loop.")
            return {"inspection_notes": [], "revision_history": []}

        print(f"  [inspect_rules] {len(suspicious)} suspicious rule(s) flagged:")
        for s in suspicious:
            print(f"    Table: {s['table']} | Rule #{s['rule_id']} | "
                  f"Failure rate: {s['failure_rate']} | Type: {s['type']}")
            print(f"    Rule: {s['rule'][:80]}")
            if s.get("error"):
                print(f"    Error: {s['error'][:80]}")
            if s.get("prior_revisions"):
                print(f"    Previously revised {len(s['prior_revisions'])} time(s)")

        prompt = INSPECT_RULES_PROMPT_WITH_HISTORY.replace(
            "{suspicious_json}", json.dumps(suspicious, indent=2)
        ).replace(
            "{revision_history_json}", json.dumps(revision_history, indent=2)
        )

        raw = llm_generator.call(prompt)

        verdicts = []
        try:
            verdicts = clean_output(raw)
            print(f"  [inspect_rules] LLM verdicts:")
            for v in verdicts:
                print(f"    Rule #{v.get('rule_id')} ({v.get('table')}) → "
                      f"{v.get('verdict', '').upper()} | "
                      f"{v.get('reason', '')[:80]}")
        except Exception:
            print(f"  [inspect_rules] Could not parse LLM response for display.")

        # Record KEEP verdicts into history so they are skipped on future passes
        kept_entries = [
            {
                "pass":     revision_count + 1,
                "table":    v["table"],
                "rule_id":  int(v["rule_id"]),
                "verdict":  "keep",
                "reason":   v.get("reason", ""),
                "reverted": False,
            }
            for v in verdicts
            if v.get("verdict") == "keep"
        ]

        # Check if any table needs full regeneration
        tables_to_regenerate = list({
            v["table"] for v in verdicts
            if v.get("verdict") == "regenerate"
            and v.get("table") in state["validation_rules"]
        })

        if tables_to_regenerate:
            print(f"  [inspect_rules] Tables flagged for regeneration "
                  f"by inspect: {tables_to_regenerate}")

        return {
            "inspection_notes":     [raw],
            "revision_history":     kept_entries,
            "tables_to_regenerate": tables_to_regenerate,
        }

    def node_revise_rules(state: PipelineState) -> dict:
        notes_raw = state["inspection_notes"][0] if state["inspection_notes"] else "[]"
        try:
            verdicts = clean_output(notes_raw)
        except Exception:
            return {
                "revision_count":   state["revision_count"] + 1,
                "inspection_notes": [],
            }

        to_revise = [v for v in verdicts if v.get("verdict") == "revise"]
        new_history_entries = []

        print(f"\n  [revise_rules] {len(to_revise)} rule(s) marked for revision.")

        if not to_revise:
            print(f"  [revise_rules] Nothing to revise — exiting loop.")
            return {
                "revision_count":   state["revision_count"] + 1,
                "inspection_notes": [],
            }

        rule_sample_lookup: dict[tuple, list] = {}
        for table_name, result in state["validation_check_results"].items():
            for rule_result in result.get("per_rule", []):
                key = (table_name, rule_result.get("rule_id"))
                rule_sample_lookup[key] = rule_result.get("sample_violations", [])

        revised_rules = dict(state["validation_rules"])

        for item in to_revise:
            table = item["table"]
            rule_id = int(item["rule_id"])
            fix_hint = item.get("suggested_fix", "")
            print(f"  [revise_rules] Revising Rule #{rule_id} in {table}:")
            print(f"    Fix hint: {fix_hint[:100]}")

            table_rules = list(revised_rules.get(table, []))
            for i, rule in enumerate(table_rules):
                if int(rule.get("rule_id", -1)) != rule_id:
                    continue

                old_rule_text = rule.get("rule", "")
                sample_values = rule_sample_lookup.get((table, rule_id), [])

                prompt = REVISE_RULE_PROMPT_WITH_CONTEXT.replace(
                    "{fix_hint}", fix_hint
                ).replace(
                    "{original_rule_json}", json.dumps(rule, indent=2)
                ).replace(
                    "{sample_values_json}", json.dumps(sample_values, indent=2)
                )

                raw = llm_generator.call(prompt)
                try:
                    revised = clean_output(raw)
                    if isinstance(revised, list) and revised:
                        candidate = {**rule, **revised[0], "rule_id": rule_id}
                    elif isinstance(revised, dict):
                        candidate = {**rule, **revised, "rule_id": rule_id}
                    else:
                        print(f"    Unexpected revision output — keeping original")
                        continue

                    df = profile_results[table]["df"]
                    mask = _apply_rule_check(
                        df,
                        candidate.get("type"),
                        candidate.get("column") or (candidate.get("columns") or [""])[0],
                        candidate.get("check_params", {}),
                    )

                    if mask is not None:
                        new_rate = mask.fillna(False).sum() / len(df)
                        print(f"    Post-revision failure rate: {new_rate:.1%}")
                        if new_rate == 1.0:
                            print(f"    Still 100% — reverting to original")
                            new_history_entries.append({
                                "pass":               state["revision_count"] + 1,
                                "table":              table,
                                "rule_id":            rule_id,
                                "before":             old_rule_text,
                                "after":              candidate.get("rule", ""),
                                "post_revision_rate": round(new_rate, 3),
                                "reverted":           True,
                            })
                            continue
                    else:
                        new_rate = None
                        print(f"    Post-revision check: not evaluable deterministically "
                              f"— accepting")

                    table_rules[i] = candidate
                    new_rule_text = candidate.get("rule", "")
                    print(f"    Before type: {rule.get('type')} | "
                          f"params: {rule.get('check_params')}")
                    print(f"    After  type: {candidate.get('type')} | "
                          f"params: {candidate.get('check_params')}")

                    new_history_entries.append({
                        "pass":               state["revision_count"] + 1,
                        "table":              table,
                        "rule_id":            rule_id,
                        "before":             old_rule_text,
                        "after":              new_rule_text,
                        "post_revision_rate": round(new_rate, 3) if new_rate is not None else None,
                        "reverted":           False,
                    })

                except Exception as e:
                    print(f"    Revision failed: {e}")

            revised_rules[table] = table_rules

        revision_num = state["revision_count"] + 1
        rules_path = config.output_dir / f"rules_pass_{revision_num}.json"
        rules_path.write_text(
            json.dumps(revised_rules, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )
        print(f"  [revise_rules] Revised rules saved to rules_pass_{revision_num}.json")

        return {
            "validation_rules":         revised_rules,
            "revision_count":           revision_num,
            "validation_check_results": {},
            "inspection_notes":         [],
            "revision_history":         new_history_entries,
        }

    def route_after_assessment(state: PipelineState) -> list[Send] | str:
        tables = state.get("tables_to_regenerate", [])
        under_limit = state.get("regeneration_count", 0) < MAX_REGENERATIONS
        if tables and under_limit:
            return "regenerate_rules"
        return [
            Send("validate_table", {
                "table_name": table_name,
                "rules":      state["validation_rules"],
            })
            for table_name in state["validation_rules"]
        ]

    def route_after_inspection(state: PipelineState) -> str:
        has_notes = bool(state.get("inspection_notes"))
        under_revision_limit = state.get("revision_count", 0) < MAX_REVISIONS
        tables_to_regen = state.get("tables_to_regenerate", [])
        under_regen_limit = state.get("regeneration_count", 0) < MAX_REGENERATIONS

        if tables_to_regen and under_regen_limit:
            return "regenerate_rules"
        if has_notes and under_revision_limit:
            return "revise_rules"
        return END

    return (
        node_generate_rules,
        node_assess_rules,
        node_regenerate_rules,
        fan_out_validation,
        node_validate_table,
        node_inspect_rules,
        node_revise_rules,
        route_after_assessment,
        route_after_inspection,
    )