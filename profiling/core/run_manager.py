"""Run-directory management for the deterministic pipeline."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import re

from .config import PipelineConfig


class RunManager:
    """Create one isolated configuration for each pipeline execution."""

    def __init__(self, base_config: PipelineConfig):
        self.base_config = base_config

    def create_run_config(self, dataset_paths: list[Path]) -> PipelineConfig:
        """Return a copy of the base config pointing to a timestamped run folder."""
        dataset_names = "_".join(
            re.sub(r"[^\w]+", "_", path.stem).strip("_")
            for path in dataset_paths
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.base_config.output_dir / f"{dataset_names}_{timestamp}"

        # PipelineConfig.__post_init__ creates the directory.
        return replace(self.base_config, output_dir=run_dir)