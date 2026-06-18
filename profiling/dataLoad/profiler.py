"""
DataProfiler — generates ydata_profiling reports and saves HTML + JSON.
"""

from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport

from ..core.config import PipelineConfig
from .preprocessor import DataPreprocessor


class DataProfiler:
    """Wraps ydata_profiling to generate and persist profile reports."""

    def __init__(self, config: PipelineConfig, preprocessor: DataPreprocessor):
        self.config = config
        self.preprocessor = preprocessor

    def profile(
        self,
        raw_df: pd.DataFrame,
        dataset_name: str,
    ) -> tuple[pd.DataFrame, ProfileReport, dict]:
        """
        Preprocess, profile, and save reports for one dataset.

        Returns
        -------
        df          : preprocessed DataFrame
        report      : ProfileReport object
        summary     : dict with paths and shape info
        """
        df_profiling = self.preprocessor.fit_transform(raw_df)
         # Minimal preprocessing → for validation checks, format analysis, LLM
        df_clean = self.preprocessor.basic_preclean(raw_df, normalize_case=False)
        type_schema = self.preprocessor.infer_type_schema(df_profiling)

        report = ProfileReport(
            df_profiling,
            minimal=True,
            title=f"Profile Report — {dataset_name}",
            type_schema=type_schema,
            dataset={
                "description": f"Profiling report for {dataset_name}",
                "creator": "Data Dictionary Pipeline",
            },
            infer_dtypes=False,
        )

        html_path = self.config.output_dir / f"{dataset_name}_profile.html"
        json_path = self.config.output_dir / f"{dataset_name}_profile.json"
        report.to_file(html_path)
        report.to_file(json_path)

        summary = {
            "dataset_name": dataset_name,
            "rows": df_profiling.shape[0],
            "columns": df_profiling.shape[1],
            "html_report": str(html_path),
            "json_report": str(json_path),
        }
        return df_clean, report, summary


def profile_datasets(
    dataset_paths: list[Path],
    loader,
    profiler: DataProfiler,
) -> dict:
    """Load, preprocess, and profile all requested datasets."""
    profile_results = {}
    for path in dataset_paths:
        name = path.stem
        raw_df = loader.load(path)
        df, report, summary = profiler.profile(raw_df, name)
        profile_results[name] = {
            "raw_df": raw_df,
            "df": df,
            "report": report,
            "summary": summary,
            "path": path,
        }
        print(f"  Profiled {name}: {df.shape[0]} rows × {df.shape[1]} cols")
    return profile_results