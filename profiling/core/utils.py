"""Shared utility functions for the profiling pipeline."""

import datetime as _dt
import math as _math

import numpy as _np
import pandas as pd
import pandas as _pd


def json_default(obj):
    """JSON serialiser for types not handled by the standard library."""
    if isinstance(obj, (_pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return None if _np.isnan(obj) else float(obj)
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    try:
        if _pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def clean_for_json(obj):
    """Recursively convert pandas/numpy/NaN values into strict JSON-safe types.

    Required because Python's json.dump can write NaN literals that
    Node.js JSON.parse rejects.
    """
    if obj is None:
        return None
    if obj is _pd.NaT:
        return None
    if isinstance(obj, (_pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return [clean_for_json(v) for v in obj.tolist()]
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return None if (_np.isnan(obj) or _np.isinf(obj)) else float(obj)
    if isinstance(obj, float):
        return None if (_math.isnan(obj) or _math.isinf(obj)) else obj
    if isinstance(obj, _np.bool_):
        return bool(obj)
    try:
        if _pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def is_sequential_ordinal(series: pd.Series) -> bool:
    """Return True if values form a contiguous 1..N integer sequence."""
    vals = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(vals) < 2 or not (vals % 1 == 0).all():
        return False
    vals_int = sorted(vals.astype(int).unique())
    n = len(vals_int)
    return vals_int[0] == 1 and vals_int[-1] == n


def email_local(v: str) -> str:
    """Return the local part of an email address (before the @)."""
    return v.split("@", 1)[0].lower().strip() if "@" in v else ""