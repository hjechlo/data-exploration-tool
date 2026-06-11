"""
profiling — Data Dictionary Pipeline package.

Public API
----------
>>> from profiling import DataDictionaryPipeline, PipelineConfig
>>> config = PipelineConfig(data_dir="data/raw", output_dir="profile_outputs")
>>> pipeline = DataDictionaryPipeline(config, llm_client=h2o_client)
>>> results = pipeline.run(dataset_paths)
"""

from .config import PipelineConfig
from .pipeline import DataDictionaryPipeline

# Individual components (for notebook-level exploration)
from .loader import DataLoader
from .preprocessor import DataPreprocessor
from .profiler import DataProfiler
from .column_analyzer import ColumnAnalyzer
from .minhash_analyzer import MinHashAnalyzer
from .llm_generator import LLMDictionaryGenerator
from .exporters import DataDictionaryExporter

__all__ = [
    "PipelineConfig",
    "DataDictionaryPipeline",
    "DataLoader",
    "DataPreprocessor",
    "DataProfiler",
    "ColumnAnalyzer",
    "MinHashAnalyzer",
    "LLMDictionaryGenerator",
    "DataDictionaryExporter",
]
