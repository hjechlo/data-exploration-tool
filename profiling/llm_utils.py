# profiling/llm_utils.py
"""
Stateless utility functions for LLM output parsing and validation.
"""

import json
import re


def strip_thinking(text: str) -> str:
    """Remove <thinking>...</thinking> blocks from LLM output."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<thinking>.*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def clean_output(raw: str) -> list[dict]:
    """Extract a JSON array from LLM response using multiple fallback strategies."""
    text = strip_thinking(raw)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass

    try:
        import json_repair  # type: ignore
        obj = json_repair.repair_json(text, return_objects=True)
        if isinstance(obj, list) and len(obj) > 0:
            return obj
        if isinstance(obj, dict):
            return [obj]
    except ImportError:
        pass

    raise ValueError(
        f"Could not extract a JSON array from the LLM response.\n"
        f"First 400 chars of raw response:\n{raw[:400]}"
    )


def validate_llm_rows(rows: list[dict], expected_names: list[str]) -> list[dict]:
    """Validate LLM output structure and reorder to match expected column order."""
    if not isinstance(rows, list):
        raise ValueError("LLM output is not a JSON array.")

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Item {i} is not an object.")
        missing = {"column_name", "description", "recommended_actions"} - set(row)
        if missing:
            raise ValueError(f"Item {i} missing keys: {missing}")
        if not isinstance(row["recommended_actions"], list):
            raise ValueError(f"Item {i} recommended_actions must be a list.")
        if not all(isinstance(x, str) for x in row["recommended_actions"]):
            raise ValueError(f"Item {i} recommended_actions must contain only strings.")

    returned = [r["column_name"] for r in rows]
    if set(returned) != set(expected_names) or len(returned) != len(expected_names):
        raise ValueError(f"Column name mismatch.\nExpected: {expected_names}\nReturned: {returned}")

    llm_map = {r["column_name"]: r for r in rows}
    return [llm_map[name] for name in expected_names]


def clean_actions(actions: list[str]) -> list[str]:
    """Deduplicate actions, remove no-ops, and ensure [No Immediate Action] sentinel."""
    cleaned = []
    seen = set()

    for action in actions or []:
        if not isinstance(action, str):
            continue
        a = action.strip()
        if not a:
            continue
        m = re.search(r"'([^']+)'\s*→\s*'([^']+)'", a)
        if m and m.group(1).strip() == m.group(2).strip():
            continue
        key = a.lower()
        if key not in seen:
            cleaned.append(a)
            seen.add(key)

    real_actions = [a for a in cleaned if a.strip() != "[No Immediate Action]"]
    return real_actions if real_actions else ["[No Immediate Action]"]


def chunks(items: list, size: int):
    """Yield successive fixed-size chunks from a list."""
    for i in range(0, len(items), size):
        yield items[i: i + size]