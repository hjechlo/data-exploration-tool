"""Stateless utility functions for LLM output parsing and validation."""

import json
import re


def strip_thinking(text: str) -> str:
    """Remove <thinking>…</thinking> blocks produced by reasoning models."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<thinking>.*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def clean_output(raw: str) -> list[dict]:
    """Extract a JSON array from an LLM response using multiple fallback strategies."""
    text = strip_thinking(raw)
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```\s*$", "", text).strip()

    # Strategy 1: parse the whole text
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract the first [...] block
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass

    # Strategy 3: extract the first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return [obj]
        except json.JSONDecodeError:
            pass

    # Strategy 4: attempt JSON repair (optional dependency)
    try:
        import json_repair  # type: ignore
        obj = json_repair.repair_json(text, return_objects=True)
        if isinstance(obj, list) and obj:
            return obj
        if isinstance(obj, dict):
            return [obj]
    except ImportError:
        pass

    raise ValueError(
        f"Could not extract a JSON array from the LLM response.\n"
        f"First 400 chars:\n{raw[:400]}"
    )

def is_length_variation_only(pattern: str) -> bool:
    """Return True when a fingerprint pattern only varies by length, not structure.

    Suppresses abstract patterns such as ``aaaa``, ``aaa aaaa``, ``XXXXXX`` that
    describe text length or word count rather than a real format distinction.
    These are internal fingerprint codes that add no value in LLM evidence.
    """
    stripped = re.sub(r"[\s.'\-\/_(),:&]+", "", str(pattern))
    if not stripped:
        return False
    return all(c == "a" for c in stripped) or all(c == "X" for c in stripped)


def validate_llm_rows(rows: list[dict], expected_names: list[str]) -> list[dict]:
    """Validate LLM output structure and reorder to match expected column order."""
    if not isinstance(rows, list):
        raise ValueError("LLM output is not a JSON array.")

    required_keys = {"column_name", "description", "recommended_actions"}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Item {i} is not an object.")
        missing = required_keys - set(row)
        if missing:
            raise ValueError(f"Item {i} missing keys: {missing}")
        if not isinstance(row["recommended_actions"], list):
            raise ValueError(f"Item {i}: recommended_actions must be a list.")
        if not all(isinstance(x, str) for x in row["recommended_actions"]):
            raise ValueError(f"Item {i}: recommended_actions must contain only strings.")

    returned = [r["column_name"] for r in rows]
    if set(returned) != set(expected_names) or len(returned) != len(expected_names):
        raise ValueError(
            f"Column name mismatch.\nExpected: {expected_names}\nReturned: {returned}"
        )

    llm_map = {r["column_name"]: r for r in rows}
    return [llm_map[name] for name in expected_names]


def clean_actions(actions: list[str]) -> list[str]:
    """Deduplicate actions, remove no-ops, and ensure a [No Immediate Action] sentinel."""
    cleaned: list[str] = []
    seen: set[str] = set()

    for action in actions or []:
        if not isinstance(action, str):
            continue
        a = action.strip()
        if not a:
            continue
        # Drop trivial renames like 'X' → 'X'
        m = re.search(r"'([^']+)'\s*→\s*'([^']+)'", a)
        if m and m.group(1).strip() == m.group(2).strip():
            continue
        key = a.lower()
        if key not in seen:
            cleaned.append(a)
            seen.add(key)

    real_actions = [a for a in cleaned if a.strip() != "[No Immediate Action]"]
    return real_actions if real_actions else ["[No Immediate Action]"]


def semantic_chunks(
    reordered: list[dict],
    sim_matrix,
    original_order: list[int],
    max_chunk_size: int,
) -> list[list[dict]]:
    """Split a semantically reordered column list into variable-size chunks.

    Splits at natural breakpoints where similarity between adjacent columns
    drops sharply below the distribution of all adjacent similarities.

    Rules:
    - Split where similarity < (mean − 1×std) of all adjacent similarities.
    - Never exceed max_chunk_size (force-split if needed).
    - Never produce an empty chunk.
    - Merge orphan single-column chunks into the preceding chunk when possible.
    """
    import numpy as np

    n = len(reordered)
    if n <= 1:
        return [reordered]

    adj_sims = [
        float(sim_matrix[original_order[i]][original_order[i + 1]])
        for i in range(n - 1)
    ]
    threshold = float(np.mean(adj_sims)) - float(np.std(adj_sims))

    chunks: list[list[dict]] = []
    current: list[dict] = [reordered[0]]

    for i in range(1, n):
        if adj_sims[i - 1] < threshold or len(current) >= max_chunk_size:
            chunks.append(current)
            current = [reordered[i]]
        else:
            current.append(reordered[i])

    if current:
        chunks.append(current)

    # Merge orphan single-column chunks into the preceding chunk.
    merged: list[list[dict]] = []
    for chunk in chunks:
        if len(chunk) == 1 and merged and len(merged[-1]) < max_chunk_size:
            merged[-1].extend(chunk)
        else:
            merged.append(chunk)

    sizes = [len(c) for c in merged]
    names = [[col["column_name"] for col in c] for c in merged]
    print(f"    [semantic chunks] {len(merged)} chunks, sizes {sizes}: {names}")
    return merged


def chunks(items: list, size: int):
    """Yield successive fixed-size chunks from a list."""
    for i in range(0, len(items), size):
        yield items[i: i + size]