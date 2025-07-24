"""
Shared test cases for circuit equivalence verification.

This module defines comprehensive test parameters for all circuit types
to ensure consistent testing across main and dev branches.
"""

from typing import Dict, List, Any

# Triangular circuit test cases
TRIANGULAR_TEST_CASES: List[Dict[str, Any]] = [
    # Basic cases with different distances
    {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.0},
    {"d": 3, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001},
    {"d": 5, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.0},
    {"d": 5, "rounds": 3, "circuit_type": "tri", "p_circuit": 0.001},
    
    # Different CNOT schedules
    {"d": 3, "rounds": 1, "circuit_type": "tri", "cnot_schedule": "tri_optimal", "p_circuit": 0.001},
    {"d": 3, "rounds": 1, "circuit_type": "tri", "cnot_schedule": "tri_optimal_reversed", "p_circuit": 0.001},
    {"d": 3, "rounds": 1, "circuit_type": "tri", "cnot_schedule": "LLB", "p_circuit": 0.001},
    
    # Perfect initialization and final measurement variations
    {"d": 3, "rounds": 1, "circuit_type": "tri", "perfect_init_final": True, "p_circuit": 0.001},
    {"d": 3, "rounds": 1, "circuit_type": "tri", "perfect_init_final": False, "p_circuit": 0.001},
    
    # Comparative decoding variations
    {"d": 3, "rounds": 1, "circuit_type": "tri", "comparative_decoding": True, "p_circuit": 0.001},
    {"d": 3, "rounds": 1, "circuit_type": "tri", "comparative_decoding": False, "p_circuit": 0.001},
    
    # Different temporal boundary types
    {"d": 3, "rounds": 1, "circuit_type": "tri", "temp_bdry_type": "X", "p_circuit": 0.001},
    {"d": 3, "rounds": 1, "circuit_type": "tri", "temp_bdry_type": "Z", "p_circuit": 0.001},
    {"d": 3, "rounds": 1, "circuit_type": "tri", "temp_bdry_type": "Y", "p_circuit": 0.001},
    
    # Exclude non-essential Pauli detectors
    {"d": 3, "rounds": 1, "circuit_type": "tri", "temp_bdry_type": "Z", "exclude_non_essential_pauli_detectors": True, "p_circuit": 0.001},
    {"d": 3, "rounds": 1, "circuit_type": "tri", "temp_bdry_type": "X", "exclude_non_essential_pauli_detectors": True, "p_circuit": 0.001},
]

# Rectangular circuit test cases
RECTANGULAR_TEST_CASES: List[Dict[str, Any]] = [
    # Basic rectangular cases
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.0},
    {"d": 4, "d2": 4, "rounds": 2, "circuit_type": "rec", "p_circuit": 0.001},
    {"d": 4, "d2": 6, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.0},
    {"d": 4, "d2": 6, "rounds": 3, "circuit_type": "rec", "p_circuit": 0.001},
    {"d": 6, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
    
    # Perfect initialization variations
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "perfect_init_final": True, "p_circuit": 0.001},
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "perfect_init_final": False, "p_circuit": 0.001},
    
    # Comparative decoding
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "comparative_decoding": True, "p_circuit": 0.001},
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "comparative_decoding": False, "p_circuit": 0.001},
    
    # Different temporal boundary types
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "temp_bdry_type": "X", "p_circuit": 0.001},
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "temp_bdry_type": "Z", "p_circuit": 0.001},
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "temp_bdry_type": "Y", "p_circuit": 0.001},
]

# Stability circuit test cases (with corrected rounds parameters)
STABILITY_TEST_CASES: List[Dict[str, Any]] = [
    # Updated parameters based on user's manual correction
    {"d": 4, "d2": 4, "rounds": 4, "circuit_type": "rec_stability", "p_circuit": 0.001},
    {"d": 4, "d2": 4, "rounds": 8, "circuit_type": "rec_stability", "p_circuit": 0.001},
    {"d": 6, "d2": 4, "rounds": 4, "circuit_type": "rec_stability", "p_circuit": 0.001},
    {"d": 6, "d2": 4, "rounds": 8, "circuit_type": "rec_stability", "p_circuit": 0.001},
    
    # Different d2 values
    {"d": 4, "d2": 6, "rounds": 4, "circuit_type": "rec_stability", "p_circuit": 0.001},
    {"d": 4, "d2": 6, "rounds": 8, "circuit_type": "rec_stability", "p_circuit": 0.001},
    
    # Perfect initialization variations
    {"d": 4, "d2": 4, "rounds": 4, "circuit_type": "rec_stability", "perfect_init_final": True, "p_circuit": 0.001},
    {"d": 4, "d2": 4, "rounds": 4, "circuit_type": "rec_stability", "perfect_init_final": False, "p_circuit": 0.001},
    
    # Comparative decoding (if supported)
    {"d": 4, "d2": 4, "rounds": 4, "circuit_type": "rec_stability", "comparative_decoding": False, "p_circuit": 0.001},
]

# Growing circuit test cases
GROWING_TEST_CASES: List[Dict[str, Any]] = [
    # Basic growing cases
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001},
    {"d": 3, "d2": 5, "rounds": 2, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001},
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "Z", "p_circuit": 0.001},
    {"d": 3, "d2": 5, "rounds": 2, "circuit_type": "growing", "temp_bdry_type": "Z", "p_circuit": 0.001},
    
    # Different distance combinations
    {"d": 5, "d2": 7, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001},
    {"d": 5, "d2": 7, "rounds": 2, "circuit_type": "growing", "temp_bdry_type": "Z", "p_circuit": 0.001},
    
    # Y boundary type
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "Y", "p_circuit": 0.001},
    
    # Perfect initialization variations
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "perfect_init_final": True, "p_circuit": 0.001},
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "perfect_init_final": False, "p_circuit": 0.001},
    
    # Comparative decoding
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "comparative_decoding": True, "p_circuit": 0.001},
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "comparative_decoding": False, "p_circuit": 0.001},
    
    # Exclude non-essential Pauli detectors
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "Z", "exclude_non_essential_pauli_detectors": True, "p_circuit": 0.001},
]

# Edge cases with individual error rates
EDGE_CASE_TEST_CASES: List[Dict[str, Any]] = [
    # Individual error rates (triangular)
    {
        "d": 3, "rounds": 2, "circuit_type": "tri",
        "p_bitflip": 0.0005, "p_reset": 0.001, "p_meas": 0.0015, 
        "p_cnot": 0.002, "p_idle": 0.0008
    },
    
    # Individual error rates (rectangular)
    {
        "d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec",
        "p_bitflip": 0.001, "p_reset": 0.0008, "p_meas": 0.0012, 
        "p_cnot": 0.0015, "p_idle": 0.0005
    },
    
    # Custom CNOT schedule
    {
        "d": 3, "rounds": 1, "circuit_type": "tri",
        "cnot_schedule": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "p_circuit": 0.001
    },
    
    # Complex parameter combinations
    {
        "d": 3, "rounds": 2, "circuit_type": "tri", 
        "temp_bdry_type": "Y", "perfect_init_final": True, 
        "comparative_decoding": True, "p_circuit": 0.001
    },
    
    # Growing with complex parameters
    {
        "d": 3, "d2": 5, "rounds": 2, "circuit_type": "growing",
        "temp_bdry_type": "Y", "perfect_init_final": False,
        "comparative_decoding": True, "exclude_non_essential_pauli_detectors": True,
        "p_circuit": 0.001
    },
]

# Combine all test cases for easy access
ALL_TEST_CASES = {
    "triangular": TRIANGULAR_TEST_CASES,
    "rectangular": RECTANGULAR_TEST_CASES,
    "stability": STABILITY_TEST_CASES,
    "growing": GROWING_TEST_CASES,
    "edge_cases": EDGE_CASE_TEST_CASES,
}

def get_test_case_name(params: Dict[str, Any]) -> str:
    """Generate a descriptive test case name from parameters."""
    circuit_type = params.get("circuit_type", "unknown")
    d = params.get("d", "?")
    d2 = params.get("d2", "")
    rounds = params.get("rounds", "?")
    
    name_parts = [f"{circuit_type}_d{d}"]
    if d2:
        name_parts.append(f"d2{d2}")
    name_parts.append(f"r{rounds}")
    
    # Add distinctive parameters
    if params.get("temp_bdry_type") and params["temp_bdry_type"] != "Z":
        name_parts.append(f"bdry{params['temp_bdry_type']}")
    if params.get("cnot_schedule") and isinstance(params["cnot_schedule"], str):
        name_parts.append(f"sched{params['cnot_schedule']}")
    if params.get("perfect_init_final"):
        name_parts.append("perfect")
    if params.get("comparative_decoding"):
        name_parts.append("comp")
    if params.get("exclude_non_essential_pauli_detectors"):
        name_parts.append("exclude")
    
    # Add error info
    if params.get("p_circuit"):
        name_parts.append(f"p{params['p_circuit']}")
    elif any(k.startswith("p_") for k in params.keys()):
        name_parts.append("individual_errors")
    
    return "_".join(name_parts)