"""Structured request and result models for the deterministic pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineRunRequest:
    """All user-controlled options required for one pipeline execution."""

    dataset_paths: tuple[Path, ...]
    dataset_descriptions: dict[str, str] = field(default_factory=dict)
    generate_word: bool = True
    word_script: Path = Path("generate_word_report.js")
    report_title: str = ""
    join_hints: dict[str, list[str]] | None = None

    def __post_init__(self) -> None:
        normalized_paths = tuple(Path(path) for path in self.dataset_paths)
        object.__setattr__(self, "dataset_paths", normalized_paths)
        object.__setattr__(self, "word_script", Path(self.word_script))

        if not normalized_paths:
            raise ValueError("At least one dataset path must be provided.")


@dataclass
class PipelineRunResult:
    """Structured outputs from one complete deterministic pipeline execution."""

    run_directory: Path
    profile_results: dict[str, Any]
    column_summaries: dict[str, list[dict]]
    minhash_results: dict[str, Any]
    all_dictionaries: dict[str, list[dict]]
    dataset_summaries: dict[str, str]
    report_summary: str
    join_interpretation: str
    validation_rules: dict[str, list[dict]]
    validation_check_results: dict[str, Any]
    output_paths: dict[str, Any]
