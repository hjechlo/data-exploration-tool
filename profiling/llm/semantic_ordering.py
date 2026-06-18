"""Semantic ordering of column evidence before LLM chunking."""

import os
import re

import numpy as np
from openai import AzureOpenAI


def group_by_logic(
    evidence: list[dict],
) -> tuple[list[dict], object, list[int]]:
    """
    Reorder columns by semantic similarity using sentence-transformers
    + greedy nearest-neighbour traversal.

    Instead of hard clustering (which produces variable-size groups),
    this builds a single ordered sequence where each column is followed
    by its most semantically similar unvisited neighbour. _chunks() then
    splits this sequence into equal-sized chunks, guaranteeing that
    similar columns end up in the same chunk without any group being
    too large or too small.
    """
    if len(evidence) < 2:
        n = len(evidence)
        return (evidence, [[1.0] * n for _ in range(n)], list(range(n)))
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        _ = SentenceTransformer  # dependency availability check
    except ImportError:
        print("    [semantic grouping] sentence-transformers not available — skipping.")
        n = len(evidence)
        return (evidence, [[1.0] * n for _ in range(n)], list(range(n)))

    def build_sentence(col: dict) -> str:
        """
        Describe the column in natural language so the embedding model
        can capture its semantic meaning.

        Example:
          "date of birth: datetime column with sample values 19801231, 19750101"
          "suburb: string column with sample values sydney, melbourne, brisbane"
        """
        col_name = col.get("column_name", "").replace("_", " ")
        dtype = col.get("data_type", "")
        intended = col.get("intended_data_type", "")
        samples = ", ".join((str(v) for v in col.get("sample_values", [])[:3]))
        type_desc = intended if intended and intended != dtype else dtype
        facts = col.get("column_facts", [])
        is_unique = any(("unique" in f.lower() and "all" in f.lower() for f in facts))
        perm = col.get("permissible_values")
        n_distinct = len(perm) if perm else None
        if is_unique:
            role = " unique identifier"
        elif n_distinct is not None and n_distinct <= 5:
            role = f" binary flag with {n_distinct} values"
        elif n_distinct is not None:
            role = f" categorical with {n_distinct} values"
        else:
            role = ""
        return f"{col_name}: {type_desc}{role} column with sample values {samples}"

    n = len(evidence)
    sentences = [build_sentence(col) for col in evidence]
    client = AzureOpenAI(
        azure_endpoint=os.environ.get("ENDPOINT_EMBED", ""),
        api_key=os.environ.get("AZURE_OPENAI_KEY", ""),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
    )
    response = client.embeddings.create(
        model=os.environ.get("DEPLOYMENT_EMBED", ""), input=sentences
    )
    embeddings = np.array([item.embedding for item in response.data])
    sim_matrix = cosine_similarity(embeddings)
    visited = [False] * n
    order = []
    current = 0
    for _ in range(n):
        visited[current] = True
        order.append(current)
        best_sim = -1
        best_next = -1
        for j in range(n):
            if not visited[j] and sim_matrix[current][j] > best_sim:
                best_sim = sim_matrix[current][j]
                best_next = j
        current = best_next if best_next != -1 else 0
    reordered = [evidence[i] for i in order]
    _unit_suffix = re.compile("^(.+?)(UnitMeasureCode|Unit|Units)$", re.IGNORECASE)
    _col_names = [c["column_name"] for c in reordered]
    for _col in list(_col_names):
        _m = _unit_suffix.match(_col)
        if not _m:
            continue
        _base = _m.group(1).rstrip("_")
        _val_col = next(
            (c for c in _col_names if c.lower() == _base.lower() and c != _col), None
        )
        if _val_col is None:
            continue
        _unit_idx = next(
            (i for i, c in enumerate(reordered) if c["column_name"] == _col)
        )
        _value_idx = next(
            (i for i, c in enumerate(reordered) if c["column_name"] == _val_col)
        )
        if _unit_idx != _value_idx + 1:
            _entry = reordered.pop(_unit_idx)
            _value_idx = next(
                (i for i, c in enumerate(reordered) if c["column_name"] == _val_col)
            )
            reordered.insert(_value_idx + 1, _entry)
            _col_names = [c["column_name"] for c in reordered]
    print(
        f"    [semantic ordering] {n} columns reordered: {[c['column_name'] for c in reordered]}"
    )
    return (reordered, sim_matrix, order)
