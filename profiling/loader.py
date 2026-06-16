"""
DataLoader — responsible for reading files into DataFrames.
"""

from pathlib import Path

import pandas as pd
import chardet
import json

from .config import SUPPORTED_EXTENSIONS


class DataLoader:
    """Loads one or more datasets from disk."""

    # Common encodings ordered by likelihood
    ENCODING_FALLBACKS = [
        'utf-8',
        'utf-8-sig',  
        'iso-8859-1',  
        'windows-1252', 
        'cp1252', 
        'latin-1',  
    ]
 
    @staticmethod
    def _detect_encoding(path: Path) -> str:
        """Detect file encoding with validation.
        
        Uses chardet for detection with high confidence threshold,
        otherwise tries common encodings in order.
        
        Returns:
            Detected encoding name
        """
        # Try chardet first
        with open(path, 'rb') as f:
            raw_data = f.read(100000)  
        
        result = chardet.detect(raw_data)
        detected_encoding = result.get('encoding')
        confidence = result.get('confidence', 0)
        
        # If chardet is very confident (>80%), try it first
        if detected_encoding and confidence > 0.8:
            try:
                raw_data.decode(detected_encoding)
                return detected_encoding
            except (UnicodeDecodeError, LookupError):
                pass  # Fall through to fallback chain
        
        # Try common encodings in order
        for encoding in DataLoader.ENCODING_FALLBACKS:
            try:
                raw_data.decode(encoding)
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Last resort: return UTF-8 with error handling
        return 'utf-8'
 
    @staticmethod
    def _load_csv_safe(path: Path) -> pd.DataFrame:
        """Load CSV with automatic encoding detection."""
        encoding = DataLoader._detect_encoding(path)
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='utf-8', errors='replace')
    
    @staticmethod
    def _load_geojson(path: Path) -> pd.DataFrame:
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        features = data.get("features", [])
        rows = []
        for feature in features:
            row = feature.get("properties", {}) or {}
            # Optionally include geometry type as a column
            geom = feature.get("geometry", {})
            if geom:
                row["_geometry_type"] = geom.get("type")
            rows.append(row)
        return pd.DataFrame(rows)
    
    @staticmethod
    def _load_json(path: Path) -> pd.DataFrame:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            list_keys = [k for k, v in data.items()
                         if isinstance(v, list) and v and isinstance(v[0], dict)]
            if len(list_keys) == 1:
                rows = pd.json_normalize(data[list_keys[0]])
                # Inject all top-level scalar fields into every row.
                # Constant/metadata fields will be flagged by downstream profiling.
                scalar_fields = {
                    k: v for k, v in data.items()
                    if k != list_keys[0]
                    and not isinstance(v, (list, dict))
                }
                for col, val in scalar_fields.items():
                    rows[col] = val
                return rows
        return pd.read_json(path)
 
    READERS = {
        ".csv": _load_csv_safe.__func__,
        ".xlsx": lambda p: pd.read_excel(p),
        ".xls": lambda p: pd.read_excel(p),
        ".json": _load_json.__func__,
        ".geojson": _load_geojson.__func__,
        ".parquet": lambda p: pd.read_parquet(p),
    }
 
    def load(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        reader = self.READERS.get(suffix)
        if reader is None:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {list(self.READERS)}"
            )
        return reader(path)
 
    def discover(self, data_dir: str | Path) -> list[Path]:
        """Return sorted list of supported files in data_dir."""
        data_dir = Path(data_dir)
        files = [
            f for f in data_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(files)
 
    def load_all(self, paths: list[Path]) -> dict[str, pd.DataFrame]:
        """Load multiple datasets, keyed by file stem."""
        return {path.stem: self.load(path) for path in paths}
 