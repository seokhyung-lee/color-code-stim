"""
Comprehensive test parameter sets for ColorCode equivalence testing.

This module defines systematic test cases covering all circuit types,
parameter combinations, and edge cases for thorough equivalence testing.
"""

from typing import Dict, List, Any, Tuple


# Base test cases for each circuit type
TRIANGULAR_BASE_CASES: List[Dict[str, Any]] = [
    # Basic triangular circuits
    {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.0},
    {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
    {"d": 3, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001},
    {"d": 5, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
    {"d": 5, "rounds": 3, "circuit_type": "tri", "p_circuit": 0.001},
]

RECTANGULAR_BASE_CASES: List[Dict[str, Any]] = [
    # Basic rectangular circuits
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.0},
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
    {"d": 4, "d2": 6, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
    {"d": 6, "d2": 4, "rounds": 2, "circuit_type": "rec", "p_circuit": 0.001},
    {"d": 6, "d2": 8, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
]

STABILITY_BASE_CASES: List[Dict[str, Any]] = [
    # Stability experiment circuits
    {"d": 4, "d2": 4, "rounds": 4, "circuit_type": "rec_stability", "p_circuit": 0.001},
    {"d": 4, "d2": 4, "rounds": 8, "circuit_type": "rec_stability", "p_circuit": 0.001},
    {"d": 4, "d2": 6, "rounds": 4, "circuit_type": "rec_stability", "p_circuit": 0.001},
    {"d": 6, "d2": 4, "rounds": 4, "circuit_type": "rec_stability", "p_circuit": 0.001},
    {"d": 6, "d2": 6, "rounds": 8, "circuit_type": "rec_stability", "p_circuit": 0.001},
]

GROWING_BASE_CASES: List[Dict[str, Any]] = [
    # Growing operation circuits
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001},
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "Z", "p_circuit": 0.001},
    {"d": 3, "d2": 5, "rounds": 2, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001},
    {"d": 5, "d2": 7, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "Z", "p_circuit": 0.001},
    {"d": 3, "d2": 7, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "Y", "p_circuit": 0.001},
]

CULTIVATION_BASE_CASES: List[Dict[str, Any]] = [
    # Cultivation + growing circuits
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "cult+growing", "temp_bdry_type": "X", "p_circuit": 0.001},
    {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "cult+growing", "temp_bdry_type": "Z", "p_circuit": 0.001},
    {"d": 3, "d2": 5, "rounds": 2, "circuit_type": "cult+growing", "temp_bdry_type": "X", "p_circuit": 0.001},
    {"d": 5, "d2": 7, "rounds": 1, "circuit_type": "cult+growing", "temp_bdry_type": "Z", "p_circuit": 0.001},
]


# Parameter variations for systematic testing
CNOT_SCHEDULE_VARIATIONS = [
    "tri_optimal",
    "tri_optimal_reversed", 
    "LLB",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Custom schedule
]

TEMPORAL_BOUNDARY_VARIATIONS = ["X", "Z", "Y"]

BOOLEAN_FLAG_VARIATIONS = [
    {"perfect_init_final": True},
    {"perfect_init_final": False},
    {"comparative_decoding": True},
    {"comparative_decoding": False},
    {"exclude_non_essential_pauli_detectors": True},
    {"exclude_non_essential_pauli_detectors": False},
]

NOISE_VARIATIONS = [
    # Circuit-level noise
    {"p_circuit": 0.0},
    {"p_circuit": 0.001},
    {"p_circuit": 0.005},
    
    # Individual error rates
    {"p_bitflip": 0.001, "p_reset": 0.0008, "p_meas": 0.0012, "p_cnot": 0.0015, "p_idle": 0.0005},
    {"p_bitflip": 0.0005, "p_reset": 0.001, "p_meas": 0.0015, "p_cnot": 0.002, "p_idle": 0.0008},
]


def generate_parameter_combinations(base_cases: List[Dict[str, Any]], 
                                   variations: List[Dict[str, Any]], 
                                   max_combinations: int = 50) -> List[Dict[str, Any]]:
    """
    Generate systematic parameter combinations from base cases and variations.
    
    Parameters
    ----------
    base_cases : list of dict
        Base test cases to extend
    variations : list of dict
        Parameter variations to apply
    max_combinations : int
        Maximum number of combinations to generate
        
    Returns
    -------
    list of dict
        Generated parameter combinations
    """
    combinations = []
    
    for base_case in base_cases:
        # Add base case as-is
        combinations.append(base_case.copy())
        
        # Add variations
        for variation in variations:
            if len(combinations) >= max_combinations:
                break
                
            combined = base_case.copy()
            combined.update(variation)
            combinations.append(combined)
        
        if len(combinations) >= max_combinations:
            break
    
    return combinations[:max_combinations]


def get_test_cases_by_category() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get organized test cases by circuit type category.
    
    Returns
    -------
    dict
        Test cases organized by category
    """
    return {
        "triangular": TRIANGULAR_BASE_CASES,
        "rectangular": RECTANGULAR_BASE_CASES, 
        "stability": STABILITY_BASE_CASES,
        "growing": GROWING_BASE_CASES,
        "cultivation": CULTIVATION_BASE_CASES,
    }


def get_extended_test_cases(category: str, max_per_category: int = 25) -> List[Dict[str, Any]]:
    """
    Get extended test cases with parameter variations for a specific category.
    
    Parameters
    ----------
    category : str
        Circuit type category ('triangular', 'rectangular', etc.)
    max_per_category : int
        Maximum test cases per category
        
    Returns
    -------
    list of dict
        Extended test cases with variations
    """
    base_cases = get_test_cases_by_category()[category]
    
    # Apply relevant variations based on category
    if category == "triangular":
        variations = BOOLEAN_FLAG_VARIATIONS + [{"cnot_schedule": s} for s in CNOT_SCHEDULE_VARIATIONS[:3]]
        variations += [{"temp_bdry_type": t} for t in TEMPORAL_BOUNDARY_VARIATIONS]
    elif category in ["rectangular", "stability"]:
        variations = BOOLEAN_FLAG_VARIATIONS[:4]  # Skip some for rectangular
    elif category in ["growing", "cultivation"]:
        variations = BOOLEAN_FLAG_VARIATIONS + [{"temp_bdry_type": t} for t in TEMPORAL_BOUNDARY_VARIATIONS]
    else:
        variations = BOOLEAN_FLAG_VARIATIONS
    
    return generate_parameter_combinations(base_cases, variations, max_per_category)


def get_comprehensive_test_suite() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get the complete comprehensive test suite for all categories.
    
    Returns
    -------
    dict
        Complete test suite organized by category
    """
    return {
        category: get_extended_test_cases(category, max_per_category=20)
        for category in get_test_cases_by_category().keys()
    }


def get_quick_test_suite() -> List[Dict[str, Any]]:
    """
    Get a quick test suite for rapid validation (subset of full suite).
    
    Returns
    -------
    list of dict
        Quick test cases covering all circuit types supported by both legacy and refactored code
    """
    # Manually selected quick cases that work with both legacy and refactored code
    # Note: Legacy code has restrictions on temp_bdry_type with certain circuit types
    quick_cases = [
        # Triangular cases
        {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.0},
        {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
        
        # Rectangular cases  
        {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.0},
        {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
        
        # Stability cases (no temp_bdry_type to avoid legacy assertion)
        {"d": 4, "d2": 4, "rounds": 4, "circuit_type": "rec_stability", "p_circuit": 0.001},
        
        # Growing cases (legacy supports these)
        {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "p_circuit": 0.001},
        
        # Additional triangular with variations
        {"d": 5, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
    ]
    
    return quick_cases


def get_stress_test_cases() -> List[Dict[str, Any]]:
    """
    Get stress test cases with larger parameters and complex configurations.
    
    Returns
    -------
    list of dict
        Stress test cases
    """
    return [
        # Large triangular
        {"d": 7, "rounds": 4, "circuit_type": "tri", "p_circuit": 0.001, "comparative_decoding": True},
        
        # Large rectangular  
        {"d": 8, "d2": 10, "rounds": 3, "circuit_type": "rec", "p_circuit": 0.001, "perfect_init_final": False},
        
        # Complex stability
        {"d": 8, "d2": 6, "rounds": 12, "circuit_type": "rec_stability", "p_circuit": 0.001},
        
        # Large growing
        {"d": 5, "d2": 9, "rounds": 2, "circuit_type": "growing", "temp_bdry_type": "Y", "p_circuit": 0.001},
        
        # Complex cultivation
        {"d": 3, "d2": 7, "rounds": 3, "circuit_type": "cult+growing", "temp_bdry_type": "Y", 
         "comparative_decoding": True, "exclude_non_essential_pauli_detectors": True, "p_circuit": 0.001},
    ]


def get_edge_case_test_cases() -> List[Dict[str, Any]]:
    """
    Get edge case test cases for boundary condition testing.
    
    Returns
    -------
    list of dict
        Edge case test parameters
    """
    return [
        # Minimum parameters
        {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.0},
        {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.0},
        
        # Zero noise
        {"d": 3, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.0, "perfect_init_final": True},
        
        # Custom individual error rates  
        {"d": 3, "rounds": 1, "circuit_type": "tri", 
         "p_bitflip": 0.001, "p_reset": 0.0, "p_meas": 0.001, "p_cnot": 0.002, "p_idle": 0.0},
        
        # Equal d and d2
        {"d": 5, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "Z", "p_circuit": 0.001},
    ]


def get_test_case_name(params: Dict[str, Any]) -> str:
    """
    Generate a descriptive test case name from parameters.
    
    Parameters
    ----------
    params : dict
        Test case parameters
        
    Returns
    -------
    str
        Descriptive test case name
    """
    circuit_type = params.get("circuit_type", "unknown")
    d = params.get("d", "?")
    d2 = params.get("d2", "")
    rounds = params.get("rounds", "?")
    
    name_parts = [f"{circuit_type}_d{d}"]
    if d2 and d2 != d:
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
    if params.get("p_circuit") is not None:
        p_circuit = params["p_circuit"]
        if p_circuit == 0.0:
            name_parts.append("noiseless")
        else:
            name_parts.append(f"p{p_circuit}")
    elif any(k.startswith("p_") for k in params.keys()):
        name_parts.append("individual_errors")
    
    return "_".join(name_parts)