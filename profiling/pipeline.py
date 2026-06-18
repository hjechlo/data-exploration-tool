"""Top-level workflow for one complete data-dictionary pipeline run."""

from pathlib import Path

from .analysis.column_analyzer import ColumnAnalyzer, build_column_summaries
from .analysis.minhash_analyzer import MinHashAnalyzer, analyze_relationships
from .analysis.relationship_reporting import (
    annotate_relationship_roles,
    cleanup_fk_errors,
)
from .core.config import PipelineConfig
from .core.models import PipelineRunRequest, PipelineRunResult
from .core.run_manager import RunManager
from .dataLoad.loader import DataLoader
from .dataLoad.preprocessor import DataPreprocessor
from .dataLoad.profiler import DataProfiler, profile_datasets
from .llm.llm_generator import LLMDictionaryGenerator, generate_dictionaries
from .llm.summaries import (
    generate_join_interpretation,
    generate_report_summary,
)
from .reporting.exporters import DataDictionaryExporter, export_outputs
from .validation.failures import validate_tables
from .validation.rules import generate_rules_for_tables


def run(
    request: PipelineRunRequest,
    config: PipelineConfig,
    llm_client,
) -> PipelineRunResult:
    """Execute one complete LLM-powered data-dictionary workflow."""
    if llm_client is None:
        raise ValueError("An LLM client is required for the pipeline.")

    dataset_paths = [Path(path) for path in request.dataset_paths]
    missing_paths = [path for path in dataset_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Dataset files do not exist: {missing_paths}")

    run_config = RunManager(config).create_run_config(dataset_paths)
    print(f"Run folder: {run_config.output_dir}")

    loader = DataLoader()
    preprocessor = DataPreprocessor(run_config)
    profiler = DataProfiler(run_config, preprocessor)
    column_analyzer = ColumnAnalyzer(run_config)
    minhash_analyzer = MinHashAnalyzer(run_config)
    llm_generator = LLMDictionaryGenerator(llm_client, run_config)
    exporter = DataDictionaryExporter(run_config)

    print("\n── Step 1: Profiling ───────────────────────────────")
    profile_results = profile_datasets(
        dataset_paths,
        loader,
        profiler,
    )

    print("\n── Step 2: Column analysis ─────────────────────────")
    column_summaries = build_column_summaries(
        column_analyzer,
        profile_results,
    )

    print("\n── Step 3: MinHash analysis ────────────────────────")
    minhash_results = analyze_relationships(
        minhash_analyzer,
        column_summaries,
        profile_results,
    )

    print("\n── Step 3.5: Cleaning FK false positive errors ─────")
    column_summaries = cleanup_fk_errors(
        column_summaries,
        minhash_results,
    )
    column_summaries = annotate_relationship_roles(
        column_summaries,
        minhash_results,
    )

    print("\n── Step 4: LLM dictionary generation ───────────────")
    all_dictionaries, dataset_summaries = generate_dictionaries(
        generator=llm_generator,
        column_summaries=column_summaries,
        minhash_results=minhash_results,
        dataset_descriptions=request.dataset_descriptions,
        join_hints=request.join_hints,
    )

    print("\n  Generating report-level executive summary...")
    report_summary = generate_report_summary(
        call_llm=llm_generator.call,
        config=run_config,
        dataset_summaries=dataset_summaries,
        all_dictionaries=all_dictionaries,
        minhash_results=minhash_results,
        output_dir=run_config.output_dir,
    )

    print("\n  Generating join path interpretation...")
    join_paths_for_interpretation = minhash_results.get(
        "candidate_join_paths", []
    ) + minhash_results.get("join_paths", [])
    join_interpretation = (
        generate_join_interpretation(
            call_llm=llm_generator.call,
            config=run_config,
            join_paths=join_paths_for_interpretation,
            join_threshold=run_config.join_threshold,
            shingle_join_threshold=run_config.shingle_join_threshold,
            output_dir=run_config.output_dir,
        )
        if join_paths_for_interpretation
        else ""
    )

    print("\n── Step 5: Validation rules ────────────────────────")
    validation_rules = generate_rules_for_tables(
        config=run_config,
        llm_generator=llm_generator,
        column_summaries=column_summaries,
        minhash_results=minhash_results,
        profile_results=profile_results,
    )

    print("\n── Step 5b: Validation checks (record-wise) ────────")
    validation_check_results = validate_tables(
        config=run_config,
        llm_generator=llm_generator,
        validation_rules=validation_rules,
        profile_results=profile_results,
    )

    print("\n── Step 6: Export ──────────────────────────────────")
    output_paths = export_outputs(
        exporter=exporter,
        all_dictionaries=all_dictionaries,
        minhash_results=minhash_results,
        generate_word=request.generate_word,
        word_script=str(request.word_script),
        report_title=request.report_title,
        dataset_summaries=dataset_summaries,
        report_summary=report_summary,
        join_interpretation=join_interpretation,
        validation_rules=validation_rules,
        validation_check_results=validation_check_results,
    )

    print("\n✓ Pipeline complete.")
    return PipelineRunResult(
        run_directory=run_config.output_dir,
        profile_results=profile_results,
        column_summaries=column_summaries,
        minhash_results=minhash_results,
        all_dictionaries=all_dictionaries,
        dataset_summaries=dataset_summaries,
        report_summary=report_summary,
        join_interpretation=join_interpretation,
        validation_rules=validation_rules,
        validation_check_results=validation_check_results,
        output_paths=output_paths,
    )
