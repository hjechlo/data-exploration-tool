import pandas as pd


def json_default(obj):
    import datetime as _dt
    import numpy as _np
    import pandas as _pd

    if isinstance(obj, (_pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()

    if isinstance(obj, _np.integer):
        return int(obj)

    if isinstance(obj, _np.floating):
        if _np.isnan(obj):
            return None
        return float(obj)

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
    """
    Recursively convert pandas/numpy/NaN values into strict JSON-safe values.
    This is needed because Python's json.dump can otherwise write NaN,
    which Node.js JSON.parse rejects.
    """
    import datetime as _dt
    import math as _math
    import numpy as _np
    import pandas as _pd

    if obj is None:
        return None

    if obj is _pd.NaT:
        return None

    if isinstance(obj, (_pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {
            str(k): clean_for_json(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [clean_for_json(v) for v in obj]

    if isinstance(obj, _np.ndarray):
        return [clean_for_json(v) for v in obj.tolist()]

    if isinstance(obj, _np.integer):
        return int(obj)

    if isinstance(obj, _np.floating):
        if _np.isnan(obj) or _np.isinf(obj):
            return None
        return float(obj)

    if isinstance(obj, float):
        if _math.isnan(obj) or _math.isinf(obj):
            return None
        return obj

    if isinstance(obj, _np.bool_):
        return bool(obj)

    try:
        if _pd.isna(obj):
            return None
    except Exception:
        pass

    return obj

def is_sequential_ordinal(series: "pd.Series") -> bool:
    """True if values are integers forming a contiguous 1..N sequence with no ID-like name."""
    import pandas as pd
    vals = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    if len(vals) < 2 or not (vals % 1 == 0).all():
        return False
    vals_int = sorted(vals.astype(int).unique())
    n = len(vals_int)
    return vals_int[0] == 1 and vals_int[-1] == n

def _email_local(v: str) -> str:
                return v.split("@", 1)[0].lower().strip() if "@" in v else ""