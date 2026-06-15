"""
main.py — CLI entrypoint for the Data Dictionary Pipeline.

Usage
-----
    # Interactive dataset selection (shows a numbered menu)
    python main.py

    # Specify datasets explicitly (skips the menu)
    python main.py --datasets data/raw/febrl4a.csv data/raw/febrl4b.csv

    # Override the directory that is scanned for the menu
    python main.py --data-dir path/to/my/datasets/

    # Skip Word document generation
    python main.py --no-word

    # Override output directory
    python main.py --output-dir results/

    # Adjust thresholds
    python main.py --join-threshold 0.4 --duplicate-threshold 0.85

    # Profile + MinHash only, skip LLM (useful for testing)
    python main.py --no-llm
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from profiling import DataDictionaryPipeline, PipelineConfig
from profiling.loader import DataLoader
from profiling.llm_engine import AzureLLMEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data Dictionary Pipeline — profiles datasets and generates "
                    "LLM-powered data dictionaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset selection
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory to scan for datasets when --datasets is not provided.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="FILE",
        default=None,
        help="Explicit list of dataset file paths. Overrides --data-dir.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="profile_outputs",
        help="Directory for all output files.",
    )

    # MinHash thresholds
    parser.add_argument(
        "--join-threshold",
        type=float,
        default=0.5,
        help="Minimum exact Jaccard resemblance to flag a join path.",
    )
    parser.add_argument(
        "--shingle-join-threshold",
        type=float,
        default=0.6,
        help="Minimum shingled Jaccard resemblance to flag a fuzzy join path.",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.9,
        help="Minimum exact Jaccard resemblance to flag a duplicate column.",
    )

    # LLM
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation (profile + MinHash only).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Number of columns per LLM prompt chunk. "
            "Defaults to the value in config.py (currently 5). "
            "Use 1 for reasoning/thinking models."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse previously cached LLM chunk results.",
    )
    parser.add_argument(
        "--llm-model",
        default="",
        metavar="MODEL_NAME",
        help="Azure deployment name for the primary model. Can also be set via DEPLOYMENT_GPT51 in .env.",
    )
    parser.add_argument(
        "--chunk-model",
        default="",
        metavar="MODEL_NAME",
        help=(
            "Instruct model for JSON generation (per-column descriptions). "
            "Falls back to --llm-model if not set."
        ),
    )

    parser.add_argument(
        "--dataset-descriptions",
        nargs="+",
        metavar="NAME:DESCRIPTION",
        default=None,
        help=(
            "Background descriptions per dataset, as key:value pairs. "
            "Example: --dataset-descriptions "
            "\"febrl4a:Synthetic Australian personal records\" "
            "\"febrl4b:Modified version with errors\""
        ),
    )


    # Word document
    parser.add_argument(
        "--no-word",
        action="store_true",
        help="Skip Word document generation.",
    )
    parser.add_argument(
        "--word-script",
        default="generate_word_report.js",
        help="Path to the Node.js Word generation script.",
    )

    return parser.parse_args()


def resolve_datasets(args: argparse.Namespace) -> list[Path]:
    """
    Return the list of dataset paths to process.

    If --datasets is provided, use those directly.
    Otherwise, scan --data-dir and show an interactive numbered menu
    so the user can pick which datasets to run.
    """
    if args.datasets:
        paths = [Path(p) for p in args.datasets]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"Error: the following dataset files do not exist: {missing}")
            sys.exit(1)
        return paths

    # Interactive selection
    loader = DataLoader()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory '{data_dir}' does not exist.")
        sys.exit(1)

    available = loader.discover(data_dir)
    if not available:
        print(f"Error: no supported datasets found in '{data_dir}'.")
        sys.exit(1)

    print(f"\nDatasets found in '{data_dir}':")
    for i, f in enumerate(available):
        print(f"  {i}: {f.name}")

    while True:
        choice_str = input(
            "\nChoose dataset number(s), comma-separated (e.g. 0,1,3): "
        ).strip()

        try:
            chosen_indices = [int(x.strip()) for x in choice_str.split(",") if x.strip()]
        except ValueError:
            print("  Invalid input — enter numbers only, separated by commas.")
            continue

        if not chosen_indices:
            print("  No datasets selected — please enter at least one number.")
            continue

        invalid = [i for i in chosen_indices if i < 0 or i >= len(available)]
        if invalid:
            print(f"  Invalid number(s): {invalid}. Choose from 0 to {len(available) - 1}.")
            continue

        break

    selected = [available[i] for i in chosen_indices]
    print("\nSelected datasets:")
    for p in selected:
        print(f"  - {p}")
    print(f"Output folder: {args.output_dir}\n")

    return selected


def build_config(args: argparse.Namespace) -> PipelineConfig:
    kwargs = dict(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        join_threshold=args.join_threshold,
        shingle_join_threshold=args.shingle_join_threshold,
        duplicate_threshold=args.duplicate_threshold,
        llm_resume=not args.no_resume,
        llm_model=args.llm_model or os.environ.get("DEPLOYMENT_GPT54", ""),
        llm_chunk_model=args.chunk_model or os.environ.get("DEPLOYMENT_GPT54", ""),
        llm_endpoint=os.environ.get("ENDPOINT_GPT54", ""),
        llm_chunk_endpoint=os.environ.get("ENDPOINT_GPT54", ""),
        llm_is_native_azure=True,  
        llm_chunk_is_native_azure=True,  
    )
    if args.chunk_size is not None:
        kwargs["llm_chunk_size"] = args.chunk_size
    return PipelineConfig(**kwargs)

def parse_dataset_descriptions(args: argparse.Namespace) -> dict[str, str]:
    """Parse --dataset-descriptions NAME:DESCRIPTION pairs into a dict."""
    if not args.dataset_descriptions:
        return {}
    descriptions = {}
    for item in args.dataset_descriptions:
        if ":" not in item:
            print(f"Warning: skipping malformed description '{item}' — expected NAME:DESCRIPTION")
            continue
        name, desc = item.split(":", 1)
        descriptions[name.strip()] = desc.strip()
    return descriptions



def main() -> None:
    load_dotenv()
    args = parse_args()
    dataset_paths = resolve_datasets(args)
    config = build_config(args)

    # LLM client (needed for --list-models too)
    llm_client = None
    if not args.no_llm:
        try:
            llm_client = AzureLLMEngine(
                api_key=os.environ["AZURE_OPENAI_KEY"].strip(),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            )
        except KeyError as e:
            print(f"Error: missing environment variable {e}.")
            print("Set AZURE_OPENAI_KEY in your .env file.")
            sys.exit(1)

    if args.llm_model:
        print(f"Using primary model: {args.llm_model}")
    if args.chunk_model:
        print(f"Using chunk model:   {args.chunk_model}")

    pipeline = DataDictionaryPipeline(config, llm_client=llm_client)

    if args.no_llm:
        # Profile + MinHash only
        print("\n── Profiling ───────────────────────────────────────")
        profile_results = pipeline.step_profile(dataset_paths)

        print("\n── Column analysis ─────────────────────────────────")
        column_summaries = pipeline.step_column_summaries(profile_results)

        print("\n── MinHash analysis ────────────────────────────────")
        pipeline.step_minhash(column_summaries, profile_results)

        print("\n✓ Done (LLM skipped). Profiles saved to:", config.output_dir)
    else:
        pipeline.run(
            dataset_paths=dataset_paths,
            generate_word=not args.no_word,
            word_script=args.word_script,
            dataset_descriptions=parse_dataset_descriptions(args),
        )


if __name__ == "__main__":
    main()