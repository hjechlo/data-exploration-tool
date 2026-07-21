"""DataLoader — responsible for reading files into DataFrames."""

import json
from pathlib import Path

import chardet
import pandas as pd

from ..core.config import SUPPORTED_EXTENSIONS


class DataLoader:
    """Load one or more datasets from disk into DataFrames."""

    # Tried in order when chardet confidence is low.
    _ENCODING_FALLBACKS: list[str] = [
        "utf-8",
        "utf-8-sig",
        "iso-8859-1",
        "windows-1252",
        "cp1252",
        "latin-1",
    ]

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        """Detect file encoding, falling back through common encodings."""
        with open(path, "rb") as fh:
            raw = fh.read(100_000)

        result = chardet.detect(raw)
        detected = result.get("encoding")
        confidence = result.get("confidence", 0)

        if detected and confidence > 0.8:
            try:
                raw.decode(detected)
                return detected
            except (UnicodeDecodeError, LookupError):
                pass

        for enc in DataLoader._ENCODING_FALLBACKS:
            try:
                raw.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue

        return "utf-8"

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        encoding = DataLoader._detect_encoding(path)
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8", errors="replace")

    @staticmethod
    def _load_json(path: Path) -> pd.DataFrame:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, dict):
            list_keys = [
                k for k, v in data.items()
                if isinstance(v, list) and v and isinstance(v[0], dict)
            ]
            if len(list_keys) == 1:
                rows = pd.json_normalize(data[list_keys[0]])
                # Inject top-level scalar fields into every row so downstream
                # profiling can flag constant/metadata columns.
                for k, v in data.items():
                    if k != list_keys[0] and not isinstance(v, (list, dict)):
                        rows[k] = v
                return rows

        return pd.read_json(path)

    @staticmethod
    def _load_geojson(path: Path) -> pd.DataFrame:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        rows = []
        for feature in data.get("features", []):
            row = dict(feature.get("properties") or {})
            geom = feature.get("geometry") or {}
            if geom:
                row["_geometry_type"] = geom.get("type")
            rows.append(row)
        return pd.DataFrame(rows)

    _READERS: dict = {
        ".csv":     _load_csv.__func__,           # type: ignore[attr-defined]
        ".xlsx":    staticmethod(lambda p: pd.read_excel(p)),
        ".xls":     staticmethod(lambda p: pd.read_excel(p)),
        ".json":    _load_json.__func__,           # type: ignore[attr-defined]
        ".geojson": _load_geojson.__func__,        # type: ignore[attr-defined]
        ".parquet": staticmethod(lambda p: pd.read_parquet(p)),
    }

    def load(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        reader = self._READERS.get(path.suffix.lower())
        if reader is None:
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported: {list(self._READERS)}"
            )
        return reader(path)

    def discover(self, data_dir: str | Path) -> list[Path]:
        """Return a sorted list of supported files found in *data_dir*."""
        data_dir = Path(data_dir)
        return sorted(
            f for f in data_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def load_all(self, paths: list[Path]) -> dict[str, pd.DataFrame]:
        """Load multiple datasets, keyed by file stem."""
        return {path.stem: self.load(path) for path in paths}