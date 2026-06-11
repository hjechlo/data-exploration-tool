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


def _semantic_chunks(
    reordered: list[dict],
    sim_matrix,
    original_order: list[int],
    max_chunk_size: int,
) -> list[list[dict]]:
    """
    Split a semantically reordered column list into variable-size chunks
    by finding natural breakpoints where similarity between adjacent columns
    drops sharply below the distribution of all adjacent similarities.

    Rules:
    - Compute similarity between each adjacent pair in the reordered sequence.
    - Split where similarity drops below (mean - 0.5 * std) of all adjacent sims.
    - Never let a chunk exceed max_chunk_size — force-split if needed.
    - Never produce an empty chunk.
    """
    import numpy as np

    n = len(reordered)
    if n <= 1:
        return [reordered]

    # Similarity between each adjacent pair in the reordered sequence
    adj_sims = []
    for i in range(n - 1):
        a = original_order[i]
        b = original_order[i + 1]
        adj_sims.append(float(sim_matrix[a][b]))

    mean_sim = float(np.mean(adj_sims))
    std_sim  = float(np.std(adj_sims))
    threshold = mean_sim - 1.0 * std_sim

    # Walk the sequence and split at breakpoints or when max size is hit
    result_chunks = []
    current_chunk = [reordered[0]]

    for i in range(1, n):
        similarity_to_prev = adj_sims[i - 1]
        is_breakpoint      = similarity_to_prev < threshold
        is_full            = len(current_chunk) >= max_chunk_size

        if is_breakpoint or is_full:
            result_chunks.append(current_chunk)
            current_chunk = [reordered[i]]
        else:
            current_chunk.append(reordered[i])

    if current_chunk:
        result_chunks.append(current_chunk)

    merged = []
    for chunk in result_chunks:
        if len(chunk) == 1 and merged and len(merged[-1]) < max_chunk_size:
            merged[-1].extend(chunk)
        else:
            merged.append(chunk)
    result_chunks = merged

    # Log the split for visibility
    sizes = [len(c) for c in result_chunks]
    names = [[col["column_name"] for col in c] for c in result_chunks]
    print(f"    [semantic chunks] {len(result_chunks)} chunks, sizes {sizes}: {names}")

    return result_chunks

def chunks(items: list, size: int):
    """Yield successive fixed-size chunks from a list."""
    for i in range(0, len(items), size):
        yield items[i: i + size]