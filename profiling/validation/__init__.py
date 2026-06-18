"""Validation rule generation, failure detection, and result building."""

from .failures import identify_validation_failures, validate_tables
from .results import run_validation_checks
from .rules import generate_rules_for_tables, generate_validation_rules

__all__ = [
    "generate_validation_rules",
    "generate_rules_for_tables",
    "identify_validation_failures",
    "validate_tables",
    "run_validation_checks",
]
