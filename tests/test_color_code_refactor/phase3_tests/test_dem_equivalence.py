"""
Phase 3: DEM Generation Equivalence Tests

This module tests the equivalence of DEM (Detector Error Model) generation
between the legacy ColorCode implementation and the refactored DEMManager module.

Focus Areas:
- Detector error model generation and structure
- DEM decomposition by color
- Detector ID mappings and metadata
- Observable matrix generation
- Error probability calculations
"""

import pytest
import sys
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(utils_path))

test_data_path = Path(__file__).parent.parent / "test_data"
sys.path.insert(0, str(test_data_path))

from comparison_utils import (
    create_test_instances,
    assert_equivalence,
    print_comparison_report,
    compare_dem_generation,
    compare_detector_info,
    compare_matrices,
    compare_dem_decomposition
)
from comprehensive_test_cases import (
    get_quick_test_suite,
    get_comprehensive_test_suite,
    get_test_case_name
)


class TestPhase3DEMEquivalence:
    """Test suite for Phase 3: DEM generation equivalence testing."""
    
    def test_quick_dem_equivalence(self):
        """Quick validation test for DEM generation equivalence."""
        for test_params in get_quick_test_suite():
            legacy, refactored = create_test_instances(test_params)
            
            # Test DEM generation
            dem_result = compare_dem_generation(legacy, refactored)
            
            # Test detector information
            detector_result = compare_detector_info(legacy, refactored)
            
            # Test matrices
            matrix_result = compare_matrices(legacy, refactored)
            
            # Test DEM decomposition
            decomp_result = compare_dem_decomposition(legacy, refactored)
            
            # Assert all tests pass
            assert_equivalence([dem_result, detector_result, matrix_result, decomp_result], 
                             get_test_case_name(test_params))
    
    def test_comprehensive_dem_equivalence(self):
        """Comprehensive DEM generation equivalence testing."""
        comprehensive_suite = get_comprehensive_test_suite()
        
        for category, test_cases in comprehensive_suite.items():
            for test_params in test_cases:
                legacy, refactored = create_test_instances(test_params)
                
                # Complete DEM validation
                dem_result = compare_dem_generation(legacy, refactored)
                detector_result = compare_detector_info(legacy, refactored)
                matrix_result = compare_matrices(legacy, refactored)
                decomp_result = compare_dem_decomposition(legacy, refactored)
                
                # Combined validation
                assert_equivalence([dem_result, detector_result, matrix_result, decomp_result],
                                 f"{category} - {get_test_case_name(test_params)}")
    
    def test_dem_generation_equivalence(self):
        """Test that DEM generation produces equivalent results."""
        for test_params in get_quick_test_suite():
            legacy, refactored = create_test_instances(test_params)
            
            dem_result = compare_dem_generation(legacy, refactored)
            assert_equivalence([dem_result], get_test_case_name(test_params))
    
    def test_dem_decomposition_equivalence(self):
        """Test that DEM decomposition by color is equivalent."""
        for test_params in get_quick_test_suite():
            legacy, refactored = create_test_instances(test_params)
            
            decomp_result = compare_dem_decomposition(legacy, refactored)
            assert_equivalence([decomp_result], get_test_case_name(test_params))
    
    def test_detector_info_equivalence(self):
        """Test that detector information mapping is equivalent."""
        for test_params in get_quick_test_suite():
            legacy, refactored = create_test_instances(test_params)
            
            detector_result = compare_detector_info(legacy, refactored)
            assert_equivalence([detector_result], get_test_case_name(test_params))
    
    def test_observable_matrix_equivalence(self):
        """Test that observable matrix generation is equivalent."""
        for test_params in get_quick_test_suite():
            legacy, refactored = create_test_instances(test_params)
            
            matrix_result = compare_matrices(legacy, refactored)
            assert_equivalence([matrix_result], get_test_case_name(test_params))


class TestPhase3DEMFeatures:
    """Individual feature testing for DEM generation components."""
    
    def test_error_probability_calculation(self):
        """Test that error probability calculations are equivalent."""
        for test_params in get_quick_test_suite():
            legacy, refactored = create_test_instances(test_params)
            
            # Compare error probabilities specifically
            matrix_result = compare_matrices(legacy, refactored)
            assert_equivalence([matrix_result], f"Error probabilities - {get_test_case_name(test_params)}")
    
    def test_detector_id_mapping(self):
        """Test that detector ID mappings are consistent."""
        for test_params in get_quick_test_suite():
            legacy, refactored = create_test_instances(test_params)
            
            # Focus on detector ID mappings
            detector_result = compare_detector_info(legacy, refactored)
            assert_equivalence([detector_result], f"Detector ID mapping - {get_test_case_name(test_params)}")
    
    def test_color_based_decomposition(self):
        """Test that color-based DEM decomposition is correct."""
        for test_params in get_quick_test_suite():
            legacy, refactored = create_test_instances(test_params)
            
            # Focus on DEM decomposition
            decomp_result = compare_dem_decomposition(legacy, refactored)
            assert_equivalence([decomp_result], f"Color decomposition - {get_test_case_name(test_params)}")
    
    def test_parity_check_matrices(self):
        """Test that parity check matrices are equivalent."""
        for test_params in get_quick_test_suite():
            legacy, refactored = create_test_instances(test_params)
            
            # Test parity check matrix specifically
            matrix_result = compare_matrices(legacy, refactored)
            assert_equivalence([matrix_result], f"Parity check matrices - {get_test_case_name(test_params)}")
    
    def test_observable_matrices(self):
        """Test that observable matrices are equivalent."""
        for test_params in get_quick_test_suite():
            legacy, refactored = create_test_instances(test_params)
            
            # Test observable matrix specifically
            matrix_result = compare_matrices(legacy, refactored)
            assert_equivalence([matrix_result], f"Observable matrices - {get_test_case_name(test_params)}")


def manual_dem_test():
    """Manual DEM testing for Phase 3 validation."""
    print("Phase 3 DEM Equivalence Tests - Running manual validation")
    print("Testing DEM generation, decomposition, and detector mapping equivalence")
    
    # Run a quick test with basic parameters
    test_params = {"d": 3, "rounds": 1, "circuit_type": "tri"}
    
    try:
        legacy, refactored = create_test_instances(test_params)
        
        # Test all DEM components
        dem_result = compare_dem_generation(legacy, refactored)
        detector_result = compare_detector_info(legacy, refactored)
        matrix_result = compare_matrices(legacy, refactored)
        decomp_result = compare_dem_decomposition(legacy, refactored)
        
        results = [dem_result, detector_result, matrix_result, decomp_result]
        print_comparison_report(results, "Manual Phase 3 DEM Test")
        
        if all(r.passed for r in results):
            print("✅ Manual test PASSED - DEM equivalence confirmed!")
        else:
            print("❌ Manual test FAILED - Check equivalence issues")
            
    except Exception as e:
        print(f"❌ Manual test ERROR: {e}")


if __name__ == "__main__":
    manual_dem_test()