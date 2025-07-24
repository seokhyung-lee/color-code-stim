"""
Comprehensive test parameter sets for ColorCode equivalence testing.

This module defines systematic test cases covering all circuit types,
parameter combinations, and edge cases for thorough equivalence testing.
"""

from typing import Dict, List, Any, Tuple


# Base test cases for each circuit type - noisy scenarios only
TRIANGULAR_BASE_CASES: List[Dict[str, Any]] = [
    # Basic triangular circuits - excluding noiseless cases
    {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
    {"d": 3, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001},
    {"d": 5, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
    {"d": 5, "rounds": 3, "circuit_type": "tri", "p_circuit": 0.001},
    {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.002},
]

RECTANGULAR_BASE_CASES: List[Dict[str, Any]] = [
    # Basic rectangular circuits - excluding noiseless cases
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
    {"d": 4, "d2": 6, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
    {"d": 6, "d2": 4, "rounds": 2, "circuit_type": "rec", "p_circuit": 0.001},
    {"d": 6, "d2": 8, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
    {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.002},
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
    # Circuit-level noise - noisy cases only
    {"p_circuit": 0.001},
    {"p_circuit": 0.002},
    {"p_circuit": 0.005},
    
    # Individual error rates
    {"p_bitflip": 0.001, "p_reset": 0.0008, "p_meas": 0.0012, "p_cnot": 0.0015, "p_idle": 0.0005},
    {"p_bitflip": 0.0005, "p_reset": 0.001, "p_meas": 0.0015, "p_cnot": 0.002, "p_idle": 0.0008},
]


def validate_parameter_combination(params: Dict[str, Any]) -> bool:
    """
    Validate that a parameter combination is supported by ColorCode.
    
    Parameters
    ---------- 
    params : dict
        Parameter combination to validate
        
    Returns
    -------
    bool
        True if parameter combination is valid, False otherwise
    """
    circuit_type = params.get("circuit_type", "tri")
    comparative_decoding = params.get("comparative_decoding", False)
    temp_bdry_type = params.get("temp_bdry_type")
    
    # rec_stability doesn't support comparative_decoding=True
    if circuit_type == "rec_stability" and comparative_decoding:
        return False
    
    # rec_stability and cult+growing have fixed temporal boundary types
    # and cannot accept custom temp_bdry_type values (legacy assertion)
    if circuit_type in ["rec_stability", "cult+growing"] and temp_bdry_type is not None:
        return False
    
    # Ensure d2 > d for growing operations  
    if circuit_type in ["growing", "cult+growing"]:  
        d = params.get("d", 3)
        d2 = params.get("d2", d)
        if d2 <= d:
            return False
    
    # cult+growing requires odd d and d2 values
    if circuit_type == "cult+growing":
        d = params.get("d", 3)
        d2 = params.get("d2", d)
        if d % 2 == 0 or d2 % 2 == 0:
            return False
    
    # Validate d and d2 are >= 3
    d = params.get("d", 3)
    d2 = params.get("d2", d)
    if d < 3 or d2 < 3:
        return False
    
    # Validate rounds >= 1
    rounds = params.get("rounds", 1)
    if rounds < 1:
        return False
    
    return True


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
        Generated parameter combinations (all validated)
    """
    combinations = []
    
    for base_case in base_cases:
        # Add base case if valid
        if validate_parameter_combination(base_case):
            combinations.append(base_case.copy())
        
        # Add variations
        for variation in variations:
            if len(combinations) >= max_combinations:
                break
                
            combined = base_case.copy()
            combined.update(variation)
            
            # Only add if the combination is valid
            if validate_parameter_combination(combined):
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
        Quick test cases covering all circuit types with noisy scenarios only
        (excludes noiseless cases which don't generate meaningful detector outcomes)
    """
    # Manually selected quick cases that work with both legacy and refactored code
    # NOTE: Only noisy cases (p_circuit > 0) since noiseless cases don't generate
    # meaningful detector outcomes for decoder testing
    quick_cases = [
        # Triangular cases - noisy only
        {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
        {"d": 3, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.002},
        
        # Rectangular cases - noisy only
        {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
        {"d": 4, "d2": 6, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
        
        # Stability cases - NOTE: rec_stability doesn't support comparative_decoding=True
        {"d": 4, "d2": 4, "rounds": 4, "circuit_type": "rec_stability", "p_circuit": 0.001},
        
        # Growing cases
        {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "p_circuit": 0.001},
        
        # Additional triangular with variations
        {"d": 5, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
        
        # Comparative decoding cases - only for supported circuit types
        {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001, "comparative_decoding": True},
        {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001, "comparative_decoding": True},
    ]
    
    # Filter out any invalid combinations
    validated_cases = [case for case in quick_cases if validate_parameter_combination(case)]
    
    return validated_cases


def get_stress_test_cases() -> List[Dict[str, Any]]:
    """
    Get stress test cases with larger parameters and complex configurations.
    
    Returns
    -------
    list of dict
        Stress test cases (all validated)
    """
    stress_cases = [
        # Large triangular
        {"d": 7, "rounds": 4, "circuit_type": "tri", "p_circuit": 0.001, "comparative_decoding": True},
        
        # Large rectangular  
        {"d": 8, "d2": 10, "rounds": 3, "circuit_type": "rec", "p_circuit": 0.001, "perfect_init_final": False},
        
        # Complex stability - NOTE: rec_stability doesn't support comparative_decoding=True
        {"d": 8, "d2": 6, "rounds": 12, "circuit_type": "rec_stability", "p_circuit": 0.001},
        
        # Large growing
        {"d": 5, "d2": 9, "rounds": 2, "circuit_type": "growing", "temp_bdry_type": "Y", "p_circuit": 0.001},
        
        # Complex cultivation
        {"d": 3, "d2": 7, "rounds": 3, "circuit_type": "cult+growing", "temp_bdry_type": "Y", 
         "comparative_decoding": True, "exclude_non_essential_pauli_detectors": True, "p_circuit": 0.001},
    ]
    
    # Filter out any invalid combinations
    validated_cases = [case for case in stress_cases if validate_parameter_combination(case)]
    
    return validated_cases


def get_edge_case_test_cases() -> List[Dict[str, Any]]:
    """
    Get edge case test cases for boundary condition testing.
    
    Returns
    -------
    list of dict
        Edge case test parameters - noisy scenarios only (all validated)
    """
    edge_cases = [
        # Minimum parameters - with noise
        {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
        {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
        
        # Low noise scenarios
        {"d": 3, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.0005, "perfect_init_final": True},
        
        # Custom individual error rates  
        {"d": 3, "rounds": 1, "circuit_type": "tri", 
         "p_bitflip": 0.001, "p_reset": 0.0, "p_meas": 0.001, "p_cnot": 0.002, "p_idle": 0.0},
        
        # Note: Equal d and d2 is invalid for growing operations (d2 must be > d)
        # Use valid growing case instead
        {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "Z", "p_circuit": 0.001},
        
        # Higher noise edge case
        {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.01},
    ]
    
    # Filter out any invalid combinations
    validated_cases = [case for case in edge_cases if validate_parameter_combination(case)]
    
    return validated_cases


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
    
    # Add error info - all cases should be noisy now
    if params.get("p_circuit") is not None:
        p_circuit = params["p_circuit"]
        name_parts.append(f"p{p_circuit}")
    elif any(k.startswith("p_") for k in params.keys()):
        name_parts.append("individual_errors")
    
    return "_".join(name_parts)