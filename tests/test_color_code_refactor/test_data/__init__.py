"""
Test data and parameter sets for ColorCode refactoring tests.

This package provides comprehensive test case definitions and parameter
combinations for systematic equivalence testing.
"""

from .comprehensive_test_cases import (
    get_test_cases_by_category,
    get_extended_test_cases,
    get_comprehensive_test_suite,
    get_quick_test_suite,
    get_stress_test_cases,
    get_edge_case_test_cases,
    get_test_case_name
)

__all__ = [
    "get_test_cases_by_category",
    "get_extended_test_cases", 
    "get_comprehensive_test_suite",
    "get_quick_test_suite",
    "get_stress_test_cases",
    "get_edge_case_test_cases",
    "get_test_case_name"
]