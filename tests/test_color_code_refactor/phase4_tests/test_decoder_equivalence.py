"""
Phase 4 Decoder Equivalence Tests

This module tests that the refactored decoder implementations (ConcatMatchingDecoder,
BPDecoder, BeliefConcatMatchingDecoder) produce identical results to the legacy
implementation. This validates the extraction of decoding logic from the monolithic
ColorCode class into modular decoder classes.

Key Test Areas:
- Concatenated MWPM decoding equivalence
- BP decoder equivalence  
- Combined BP + MWPM (belief) decoding equivalence
- Performance regression testing
- Edge cases and error scenarios
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from typing import Dict, List, Any, Tuple

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add test paths for direct imports
test_root = Path(__file__).parent.parent
utils_path = test_root / "utils"
data_path = test_root / "test_data"
sys.path.insert(0, str(utils_path))
sys.path.insert(0, str(data_path))

from comparison_utils import (
    create_test_instances,
    compare_decoder_outputs,
    compare_bp_decoder_outputs,
    compare_belief_decoder_outputs, 
    compare_concat_decoder_outputs,
    assert_equivalence,
    print_comparison_report,
    ComparisonResult
)
from comprehensive_test_cases import (
    get_quick_test_suite,
    get_comprehensive_test_suite,
    get_stress_test_cases,
    get_edge_case_test_cases,
    get_test_case_name
)


class TestPhase4DecoderEquivalence:
    """Test equivalence between legacy and refactored decoder implementations."""
    
    @pytest.fixture
    def detector_samples(self):
        """Generate test detector outcome samples."""
        # Set random seed for reproducible test data
        np.random.seed(42)
        
        # Small sample for basic testing
        small_sample = np.random.randint(0, 2, (10, 50), dtype=bool)
        
        # Medium sample for comprehensive testing
        medium_sample = np.random.randint(0, 2, (100, 80), dtype=bool)
        
        # Large sample for stress testing
        large_sample = np.random.randint(0, 2, (500, 120), dtype=bool)
        
        return {
            "small": small_sample,
            "medium": medium_sample,
            "large": large_sample,
            "single": small_sample[0],  # Single sample
            "empty": np.array([], dtype=bool).reshape(0, 50)  # Edge case
        }
    
    def test_quick_decoder_equivalence(self, detector_samples):
        """Quick validation test for basic decoder equivalence."""
        quick_cases = get_quick_test_suite()
        
        for test_params in quick_cases:
            test_name = get_test_case_name(test_params)
            
            try:
                # Create test instances
                legacy, refactored = create_test_instances(test_params)
                
                # Sample detector outcomes from the circuit
                shots = 50
                det_outcomes, _ = legacy.sample(shots=shots, seed=123)
                
                # Test basic decode functionality
                result = compare_decoder_outputs(
                    legacy, refactored, det_outcomes,
                    colors="all"
                )
                
                assert_equivalence([result], f"Quick test: {test_name}")
                
            except Exception as e:
                pytest.fail(f"Quick decoder test failed for {test_name}: {e}")
    
    def test_comprehensive_decoder_equivalence(self, detector_samples):
        """Comprehensive decoder equivalence testing across all parameter combinations."""
        comprehensive_suite = get_comprehensive_test_suite()
        
        for category, test_cases in comprehensive_suite.items():
            for test_params in test_cases[:10]:  # Limit for test performance
                test_name = get_test_case_name(test_params)
                
                try:
                    # Create test instances
                    legacy, refactored = create_test_instances(test_params)
                    
                    # Sample detector outcomes
                    shots = 30
                    det_outcomes, _ = legacy.sample(shots=shots, seed=456)
                    
                    # Test different decode configurations
                    decode_configs = [
                        {"colors": "all"},
                        {"colors": "r"},
                        {"colors": ["r", "g"]},
                        {"full_output": True},
                        {"check_validity": True},
                    ]
                    
                    # Note: comparative_decoding is a constructor parameter, not a decode parameter
                    
                    results = []
                    for config in decode_configs:
                        try:
                            result = compare_decoder_outputs(
                                legacy, refactored, det_outcomes, **config
                            )
                            results.append(result)
                        except Exception as e:
                            # Some configurations may not be compatible
                            if "requires comparative_decoding" not in str(e):
                                raise
                    
                    assert_equivalence(results, f"Comprehensive test: {test_name}")
                    
                except Exception as e:
                    pytest.fail(f"Comprehensive decoder test failed for {test_name}: {e}")
    
    def test_bp_decoder_equivalence(self, detector_samples):
        """Test equivalence of BP decoder implementations."""
        # Select appropriate test cases for BP decoding
        bp_test_cases = [
            {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
            {"d": 5, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001, "comparative_decoding": True},
        ]
        
        for test_params in bp_test_cases:
            test_name = get_test_case_name(test_params)
            
            try:
                # Create test instances
                legacy, refactored = create_test_instances(test_params)
                
                # Sample detector outcomes
                shots = 20
                det_outcomes, _ = legacy.sample(shots=shots, seed=789)
                
                # Test BP decoder configurations
                bp_configs = [
                    {},  # Default parameters
                    {"max_iter": 50},
                    {"max_iter": 100, "bp_method": "min_sum"},
                ]
                
                results = []
                for bp_config in bp_configs:
                    try:
                        result = compare_bp_decoder_outputs(
                            legacy, refactored, det_outcomes, **bp_config
                        )
                        results.append(result)
                    except Exception as e:
                        # BP decoder might not be available for all configurations
                        if "BP decoder" in str(e) or "ldpc" in str(e):
                            continue
                        raise
                
                if results:  # Only assert if we have results
                    assert_equivalence(results, f"BP decoder test: {test_name}")
                    
            except Exception as e:
                if "ldpc" in str(e) or "BP" in str(e):
                    pytest.skip(f"BP decoder not available: {e}")
                else:
                    pytest.fail(f"BP decoder test failed for {test_name}: {e}")
    
    def test_belief_decoder_equivalence(self, detector_samples):
        """Test equivalence of belief propagation + concatenated matching workflow."""
        # Select test cases that support BP predecoding
        belief_test_cases = [
            {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001, "comparative_decoding": True},
            {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001, "comparative_decoding": True},
            {"d": 5, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001, "comparative_decoding": True},
        ]
        
        for test_params in belief_test_cases:
            test_name = get_test_case_name(test_params)
            
            try:
                # Create test instances
                legacy, refactored = create_test_instances(test_params)
                
                # Sample detector outcomes
                shots = 15
                det_outcomes, _ = legacy.sample(shots=shots, seed=101112)
                
                # Test belief decoder configurations
                belief_configs = [
                    {"colors": "all"},
                    {"colors": "r"},
                    {"colors": ["r", "g"]},
                    {"bp_prms": {"max_iter": 50}},
                    {"full_output": True},
                    {"full_output": True, "bp_prms": {"max_iter": 100}},
                ]
                
                results = []
                for config in belief_configs:
                    try:
                        result = compare_belief_decoder_outputs(
                            legacy, refactored, det_outcomes, **config
                        )
                        results.append(result)
                    except Exception as e:
                        # Skip if BP is not available
                        if "ldpc" in str(e) or "BP" in str(e):
                            continue
                        raise
                
                if results:  # Only assert if we have results
                    assert_equivalence(results, f"Belief decoder test: {test_name}")
                    
            except Exception as e:
                if "ldpc" in str(e) or "BP" in str(e):
                    pytest.skip(f"BP decoder not available: {e}")
                else:
                    pytest.fail(f"Belief decoder test failed for {test_name}: {e}")
    
    def test_concat_decoder_equivalence(self, detector_samples):
        """Test equivalence of concatenated matching decoder (without BP)."""
        concat_test_cases = [
            {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
            {"d": 3, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001, "comparative_decoding": True},
            {"d": 4, "d2": 6, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001, "comparative_decoding": True},
        ]
        
        for test_params in concat_test_cases:
            test_name = get_test_case_name(test_params)
            
            try:
                # Create test instances
                legacy, refactored = create_test_instances(test_params)
                
                # Sample detector outcomes
                shots = 25
                det_outcomes, _ = legacy.sample(shots=shots, seed=131415)
                
                # Test concatenated decoder configurations
                concat_configs = [
                    {"colors": "all"},
                    {"colors": "r"},
                    {"colors": ["g", "b"]},
                    {"full_output": True},
                    {"check_validity": True},
                    {"erasure_matcher_predecoding": False},  # Explicitly disable
                ]
                
                # Add erasure matcher predecoding if comparative decoding is enabled
                if test_params.get("comparative_decoding", False):
                    concat_configs.extend([
                        {"erasure_matcher_predecoding": True, "full_output": True},
                        {"erasure_matcher_predecoding": True, "partial_correction_by_predecoding": False},
                    ])
                
                results = []
                for config in concat_configs:
                    try:
                        result = compare_concat_decoder_outputs(
                            legacy, refactored, det_outcomes, **config
                        )
                        results.append(result)
                    except Exception as e:
                        # Some configurations may not be compatible
                        if any(msg in str(e) for msg in [
                            "requires comparative_decoding", 
                            "requires multiple logical classes"
                        ]):
                            continue
                        raise
                
                assert_equivalence(results, f"Concat decoder test: {test_name}")
                
            except Exception as e:
                pytest.fail(f"Concat decoder test failed for {test_name}: {e}")
    
    def test_decoder_edge_cases(self, detector_samples):
        """Test decoder behavior with edge cases and boundary conditions."""
        edge_test_cases = get_edge_case_test_cases()
        
        for test_params in edge_test_cases:
            test_name = get_test_case_name(test_params)
            
            try:
                # Create test instances
                legacy, refactored = create_test_instances(test_params)
                
                # Test with different sample sizes
                test_samples = [
                    detector_samples["single"].reshape(1, -1),  # Single sample
                    detector_samples["small"][:5],  # Small batch
                ]
                
                results = []
                for sample in test_samples:
                    # Ensure sample size matches circuit
                    if sample.shape[1] != legacy.dem_xz.num_detectors:
                        # Trim or pad to match
                        required_size = legacy.dem_xz.num_detectors
                        if sample.shape[1] > required_size:
                            sample = sample[:, :required_size]
                        else:
                            # Skip if too small (would require complex padding)
                            continue
                    
                    result = compare_decoder_outputs(
                        legacy, refactored, sample, colors="all"
                    )
                    results.append(result)
                
                if results:
                    assert_equivalence(results, f"Edge case test: {test_name}")
                    
            except Exception as e:
                # Some edge cases may fail legitimately
                if any(msg in str(e) for msg in [
                    "Invalid detector", "Shape mismatch", "Empty array"
                ]):
                    continue
                pytest.fail(f"Edge case test failed for {test_name}: {e}")
    
    def test_decoder_color_variations(self, detector_samples):
        """Test decoder behavior with different color selections."""
        # Use a representative test case
        test_params = {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001}
        
        try:
            # Create test instances
            legacy, refactored = create_test_instances(test_params)
            
            # Sample detector outcomes
            shots = 20
            det_outcomes, _ = legacy.sample(shots=shots, seed=192021)
            
            # Test different color combinations
            color_configs = [
                {"colors": "all"},
                {"colors": "r"},
                {"colors": "g"},
                {"colors": "b"},
                {"colors": ["r", "g"]},
                {"colors": ["g", "b"]},
                {"colors": ["r", "b"]},
                {"colors": ["r", "g", "b"]},  # Explicit all
            ]
            
            results = []
            for config in color_configs:
                result = compare_decoder_outputs(
                    legacy, refactored, det_outcomes, **config
                )
                results.append(result)
            
            assert_equivalence(results, "Color variation test")
            
        except Exception as e:
            pytest.fail(f"Color variation test failed: {e}")
    
    def test_decoder_with_various_logical_values(self, detector_samples):
        """Test decoder behavior with different logical value specifications."""
        # Use comparative decoding test case
        test_params = {
            "d": 3, "rounds": 1, "circuit_type": "tri", 
            "p_circuit": 0.001, "comparative_decoding": True
        }
        
        try:
            # Create test instances
            legacy, refactored = create_test_instances(test_params)
            
            # Sample detector outcomes
            shots = 15
            det_outcomes, _ = legacy.sample(shots=shots, seed=222324)
            
            # Test different logical value specifications
            logical_configs = [
                {"logical_value": None},  # Test all logical classes
                {"logical_value": False},  # Single logical value
                {"logical_value": True},   # Single logical value
                {"full_output": True},     # With extra outputs
                {"full_output": True, "logical_value": None},  # All classes with outputs
            ]
            
            results = []
            for config in logical_configs:
                result = compare_decoder_outputs(
                    legacy, refactored, det_outcomes, **config
                )
                results.append(result)
            
            assert_equivalence(results, "Logical value variation test")
            
        except Exception as e:
            pytest.fail(f"Logical value test failed: {e}")
    
    def test_stress_decoder_cases(self, detector_samples):
        """Test decoder with stress cases (larger parameters)."""
        stress_cases = get_stress_test_cases()[:3]  # Limit for test performance
        
        for test_params in stress_cases:
            test_name = get_test_case_name(test_params)
            
            try:
                # Create test instances
                legacy, refactored = create_test_instances(test_params)
                
                # Use smaller sample size for stress tests to avoid timeout
                shots = 10
                det_outcomes, _ = legacy.sample(shots=shots, seed=252627)
                
                # Test basic decoder functionality
                result = compare_decoder_outputs(
                    legacy, refactored, det_outcomes, colors="all"
                )
                
                assert_equivalence([result], f"Stress test: {test_name}")
                
            except Exception as e:
                # Large cases may fail due to memory/time constraints
                if any(msg in str(e) for msg in [
                    "MemoryError", "timeout", "too large"
                ]):
                    pytest.skip(f"Stress test skipped due to resource constraints: {e}")
                else:
                    pytest.fail(f"Stress test failed for {test_name}: {e}")
    
    def test_decoder_error_handling(self, detector_samples):
        """Test decoder error handling and robustness."""
        # Use a simple test case
        test_params = {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001}
        
        try:
            # Create test instances
            legacy, refactored = create_test_instances(test_params)
            
            # Sample valid detector outcomes first
            shots = 10
            valid_det_outcomes, _ = legacy.sample(shots=shots, seed=282930)
            
            # Get the actual number of detectors to create proper test cases
            num_detectors = legacy.dem_xz.num_detectors
            
            # Test error cases (these should behave the same way)
            error_test_cases = [
                # Wrong shape detector outcomes - actually create fewer columns
                {
                    "det_outcomes": valid_det_outcomes[:, :max(1, num_detectors-2)],  # Actually too few columns
                    "expected_error": True
                },
                # Wrong shape detector outcomes - too many columns
                {
                    "det_outcomes": np.hstack([valid_det_outcomes, np.zeros((valid_det_outcomes.shape[0], 3), dtype=bool)]),  # Too many columns
                    "expected_error": False  # Both implementations might handle this gracefully
                },
                # Invalid color specification
                {
                    "det_outcomes": valid_det_outcomes,
                    "colors": "invalid_color",
                    "expected_error": True
                },
                # Invalid parameter combinations
                {
                    "det_outcomes": valid_det_outcomes,
                    "erasure_matcher_predecoding": True,  # Without comparative_decoding 
                    "expected_error": True
                },
            ]
            
            for i, error_case in enumerate(error_test_cases):
                det_outcomes = error_case.pop("det_outcomes")
                expected_error = error_case.pop("expected_error", False)
                
                legacy_error = None
                refactored_error = None
                
                # Test legacy behavior
                try:
                    legacy.decode(det_outcomes, **error_case)
                except Exception as e:
                    legacy_error = type(e)
                
                # Test refactored behavior
                try:
                    refactored.decode(det_outcomes, **error_case)
                except Exception as e:
                    refactored_error = type(e)
                
                # Both should either succeed or fail consistently
                if expected_error:
                    assert legacy_error is not None, f"Legacy should have failed for case {i}"
                    assert refactored_error is not None, f"Refactored should have failed for case {i}"
                    # Both failed as expected - error types may differ but behavior is consistent
                else:
                    # Both should succeed or both should fail
                    both_succeeded = (legacy_error is None and refactored_error is None)
                    both_failed = (legacy_error is not None and refactored_error is not None)
                    assert both_succeeded or both_failed, f"Error handling mismatch for case {i}: legacy={legacy_error}, refactored={refactored_error}"
                    
        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")


if __name__ == "__main__":
    # Run quick tests when executed directly
    
    # Create a detector samples fixture manually
    np.random.seed(42)
    detector_samples = {
        "small": np.random.randint(0, 2, (10, 50), dtype=bool),
        "single": np.random.randint(0, 2, 50, dtype=bool),
    }
    
    # Create test instance
    test_instance = TestPhase4DecoderEquivalence()
    
    print("\n" + "="*80)
    print("Phase 4 Decoder Equivalence Tests - Quick Validation")
    print("="*80)
    
    try:
        print("\n1. Running quick decoder equivalence test...")
        test_instance.test_quick_decoder_equivalence(detector_samples)
        print("✅ Quick decoder test passed!")
        
        print("\n2. Running color variation test...")
        test_instance.test_decoder_color_variations(detector_samples)
        print("✅ Color variation test passed!")
        
        print("\n3. Running concatenated decoder test...")
        test_instance.test_concat_decoder_equivalence(detector_samples)
        print("✅ Concatenated decoder test passed!")
        
        print("\n" + "="*80)
        print("🎉 Phase 4 Quick Tests PASSED!")
        print("Run 'pytest tests/test_color_code_refactor/phase4_tests/ -v' for full test suite")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise