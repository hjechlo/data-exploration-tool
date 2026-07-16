"""
evaluate_models.py — Compare Kimi-K2.5 vs gpt-5.1 on structured output metrics.

Metrics (adapted from StructEval 2025 / Structured Output Benchmark 2026):
  1. Schema Compliance Rate     — % rules with all required JSON keys present
  2. Valid Rule Type Rate       — % rules whose 'type' is a known deterministic type
  3. Field-Level Accuracy       — % rules whose check_params match expected keys for that type
  4. Run-to-Run Consistency     — Jaccard overlap of rule types across 3 runs per column

Usage:
    python evaluate_models.py

Requires .env with:
    AZURE_OPENAI_KEY=...
    DEPLOYMENT_KIMI=...      ENDPOINT_KIMI=...
    DEPLOYMENT_GPT=...       ENDPOINT_GPT=...
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

# ── Add project root to path so 'profiling' package is importable ──────────
sys.path.insert(0, str(Path(__file__).parent))

from profiling import PipelineConfig, run_pipeline
from profiling.llm.llm_engine import AzureLLMEngine
from profiling.core.models import PipelineRunRequest

# ── Config ──────────────────────────────────────────────────────────────────

DATASETS = [
    Path("data/synthetic user accounts/Acc_Info_dirty.csv"),
    Path("data/synthetic user accounts/Cust_Bio_dirty.csv"),
]

# Columns to focus evaluation on — one of each interesting type
TARGET_COLUMNS = {
    "Acc_Info_dirty": ["Account_No", "JoinDate", "Subscription_Type", "SubscriptionFee", "FeedbackScore"],
    "Cust_Bio_dirty": ["ID_No", "Gender", "Contact_No", "Email", "Age"],
}

N_RUNS = 3

MODELS = {
    # "Kimi-K2.5": {
    #     "deployment": os.environ.get("DEPLOYMENT_KIMI", ""),
    #     "endpoint": os.environ.get("ENDPOINT_KIMI", ""),
    # },
    # "gpt-5.4": {
    #     "deployment": os.environ.get("DEPLOYMENT_GPT54", ""),
    #     "endpoint": os.environ.get("ENDPOINT_GPT54", ""),
    # },
    "gpt-oss-120b": {
        "deployment": os.environ.get("DEPLOYMENT_OSS120B", ""),
        "endpoint": os.environ.get("ENDPOINT_OSS120B", ""),
    },
}

# ── Valid rule types (from results.py _apply_rule_check) ────────────────────
VALID_RULE_TYPES = {
    "format",
    "enumeration",
    "numeric_sum",
    "date_ordering",
    "null_consistency",
    "valid_yyyymmdd_date",
    "numeric_parseable",
    "integer_parseable",
    "datetime_parseable",
    "date_not_future",
    "phone_validity",
    "referential_cross_table",
    "sentinel_check",
    "not_null",
    "range",
    "uniqueness",
    "custom",
    "nric_age_consistency",
    "nric_dob_consistency",
}

# Expected check_params keys per rule type
EXPECTED_PARAMS = {
    "format":                  {"regex"},
    "enumeration":             {"values"},
    "range":                   {"min", "max"},
    "date_ordering":           {"earlier_col", "later_col"},
    "numeric_sum":             {"columns", "expected_sum"},
    "null_consistency":        {"condition_col", "condition_value"},
    "valid_yyyymmdd_date":     set(),
    "numeric_parseable":       set(),
    "integer_parseable":       set(),
    "datetime_parseable":      set(),
    "date_not_future":         set(),
    "phone_validity":          set(),
    "referential_cross_table": {"source_table", "source_column"},
    "sentinel_check":          {"sentinel_values"},
    "not_null":                set(),
    "uniqueness":              set(),
    "nric_age_consistency":    {"nric_col", "age_col"},
    "nric_dob_consistency":    {"nric_col", "dob_col"},
    "custom":                  {"condition"},
}

REQUIRED_RULE_KEYS = {"type", "column", "check_params", "rule"}

# ── Scoring functions ────────────────────────────────────────────────────────

def score_schema_compliance(rules: list[dict]) -> float:
    """% of rules that have all required top-level keys."""
    if not rules:
        return 0.0
    compliant = sum(
        1 for r in rules
        if REQUIRED_RULE_KEYS.issubset(r.keys())
    )
    return compliant / len(rules)


def score_valid_rule_type(rules: list[dict]) -> float:
    """% of rules whose 'type' is a known deterministic rule type."""
    if not rules:
        return 0.0
    valid = sum(1 for r in rules if r.get("type") in VALID_RULE_TYPES)
    return valid / len(rules)


def score_field_level_accuracy(rules: list[dict]) -> float:
    """% of rules whose check_params contain the expected keys for that type."""
    if not rules:
        return 0.0
    correct = 0
    for r in rules:
        rule_type = r.get("type")
        if rule_type not in EXPECTED_PARAMS:
            continue
        expected = EXPECTED_PARAMS[rule_type]
        actual = set(r.get("check_params", {}).keys()) if r.get("check_params") else set()
        if expected.issubset(actual):
            correct += 1
    scoreable = sum(1 for r in rules if r.get("type") in EXPECTED_PARAMS)
    return correct / scoreable if scoreable else 0.0


def score_consistency(runs_rules: list[list[dict]]) -> float:
    """
    Jaccard overlap of rule type+column pairs across N runs.
    High score = same rules generated consistently across runs.
    """
    if len(runs_rules) < 2:
        return 0.0
    run_sets = [
        {(r.get("type", ""), r.get("column", "")) for r in run}
        for run in runs_rules
    ]
    intersection = run_sets[0]
    union = run_sets[0]
    for s in run_sets[1:]:
        intersection = intersection & s
        union = union | s
    return len(intersection) / len(union) if union else 0.0


def extract_target_rules(validation_rules: dict, target_columns: dict) -> list[dict]:
    """Filter rules to only those for target columns."""
    result = []
    for table_name, rules in validation_rules.items():
        target_cols = target_columns.get(table_name, [])
        for rule in rules:
            if rule.get("column") in target_cols:
                result.append(rule)
    return result


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation():
    api_key = os.environ.get("AZURE_OPENAI_KEY", "").strip()
    if not api_key:
        print("Error: AZURE_OPENAI_KEY not set in .env")
        sys.exit(1)

    results = {}  # model_name -> {metric -> scores across runs}

    for model_name, model_cfg in MODELS.items():
        if not model_cfg["deployment"] or not model_cfg["endpoint"]:
            print(f"Skipping {model_name} — deployment or endpoint not set in .env")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        all_runs_rules = []  # list of rule lists, one per run

        for run_idx in range(1, N_RUNS + 1):
            print(f"\n  Run {run_idx}/{N_RUNS}...")

            config = PipelineConfig(
                output_dir=f"eval_outputs/{model_name.replace('-','_')}_run{run_idx}",
                llm_model=model_cfg["deployment"],
                llm_endpoint=model_cfg["endpoint"],
                llm_resume=False,
                llm_is_native_azure=False,
            )

            llm_client = AzureLLMEngine(
                api_key=api_key,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            )

            request = PipelineRunRequest(
                dataset_paths=tuple(DATASETS),
                generate_word=False,
            )

            try:
                result = run_pipeline(
                    request=request,
                    config=config,
                    llm_client=llm_client,
                )
                target_rules = extract_target_rules(
                    result.validation_rules,
                    TARGET_COLUMNS,
                )
                all_runs_rules.append(target_rules)
                print(f"    Extracted {len(target_rules)} target-column rules")

            except Exception as e:
                import traceback
                print(f"    Run {run_idx} failed: {e}")
                traceback.print_exc()
                all_runs_rules.append([])

        # Score across all runs
        all_rules_flat = [r for run in all_runs_rules for r in run]

        schema_scores = [score_schema_compliance(run) for run in all_runs_rules]
        valid_type_scores = [score_valid_rule_type(run) for run in all_runs_rules]
        field_acc_scores = [score_field_level_accuracy(run) for run in all_runs_rules]
        consistency = score_consistency(all_runs_rules)

        results[model_name] = {
            "schema_compliance": sum(schema_scores) / len(schema_scores) if schema_scores else 0,
            "valid_rule_type":   sum(valid_type_scores) / len(valid_type_scores) if valid_type_scores else 0,
            "field_accuracy":    sum(field_acc_scores) / len(field_acc_scores) if field_acc_scores else 0,
            "consistency":       consistency,
            "total_rules":       len(all_rules_flat),
        }

    # ── Print comparison table ───────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Metric':<35} ", end="")
    for model_name in results:
        print(f"{model_name:<20}", end="")
    print()
    print("-" * 70)

    metrics = [
        ("Schema Compliance Rate",    "schema_compliance"),
        ("Valid Rule Type Rate",       "valid_rule_type"),
        ("Field-Level Accuracy",       "field_accuracy"),
        ("Run-to-Run Consistency",     "consistency"),
    ]

    for label, key in metrics:
        print(f"{label:<35} ", end="")
        for model_name, scores in results.items():
            val = scores.get(key, 0)
            print(f"{val:.1%}{'':14}", end="")
        print()

    print("-" * 70)
    print(f"{'Total rules evaluated':<35} ", end="")
    for model_name, scores in results.items():
        print(f"{scores.get('total_rules', 0):<20}", end="")
    print()

    # Save results to JSON
    out_path = Path("eval_outputs/comparison_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_evaluation()
