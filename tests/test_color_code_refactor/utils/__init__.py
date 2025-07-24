"""
Test utilities for ColorCode refactoring equivalence testing.

This package provides comprehensive comparison utilities and helper functions
for testing equivalence between legacy and refactored implementations.
"""

from .comparison_utils import (
    ComparisonResult,
    create_test_instances,
    compare_circuits,
    compare_tanner_graphs,
    compare_qubit_groups,
    compare_full_instances,
    print_comparison_report,
    assert_equivalence
)

__all__ = [
    "ComparisonResult",
    "create_test_instances", 
    "compare_circuits",
    "compare_tanner_graphs",
    "compare_qubit_groups",
    "compare_full_instances",
    "print_comparison_report",
    "assert_equivalence"
]