"""
Phase 1: Circuit Generation Equivalence Tests

This module tests the equivalence of circuit generation between the legacy
ColorCode implementation and the refactored CircuitBuilder module.

Focus Areas:
- Circuit structure and instruction sequences
- Detector error model generation
- Parameter handling and validation
- Edge cases and boundary conditions
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
    compare_circuits,
    compare_full_instances,
    assert_equivalence,
    print_comparison_report
)
from comprehensive_test_cases import (
    get_test_cases_by_category,
    get_extended_test_cases,
    get_quick_test_suite,
    get_stress_test_cases,
    get_edge_case_test_cases,
    get_test_case_name
)


class TestPhase1CircuitEquivalence:
    """Test suite for Phase 1: Circuit generation equivalence testing."""
    
    @pytest.mark.parametrize("test_params", get_quick_test_suite())
    def test_quick_circuit_equivalence(self, test_params):
        """Quick circuit equivalence test for all circuit types."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
        
        if not circuit_result.passed:
            print_comparison_report([circuit_result], f"Quick Test: {test_name}")
        
        assert_equivalence([circuit_result], test_name)
    
    @pytest.mark.parametrize("test_params", get_test_cases_by_category()["triangular"])
    def test_triangular_circuit_generation(self, test_params):
        """Test triangular circuit generation equivalence."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to circuit-specific results for Phase 1
        circuit_results = [r for r in results if r.component == "Circuit Generation"]
        
        if any(not r.passed for r in circuit_results):
            print_comparison_report(circuit_results, f"Triangular: {test_name}")
        
        assert_equivalence(circuit_results, test_name)
    
    @pytest.mark.parametrize("test_params", get_test_cases_by_category()["rectangular"])
    def test_rectangular_circuit_generation(self, test_params):
        """Test rectangular circuit generation equivalence."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to circuit-specific results for Phase 1
        circuit_results = [r for r in results if r.component == "Circuit Generation"]
        
        if any(not r.passed for r in circuit_results):
            print_comparison_report(circuit_results, f"Rectangular: {test_name}")
        
        assert_equivalence(circuit_results, test_name)
    
    @pytest.mark.parametrize("test_params", get_test_cases_by_category()["stability"])
    def test_stability_circuit_generation(self, test_params):
        """Test stability circuit generation equivalence."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to circuit-specific results for Phase 1
        circuit_results = [r for r in results if r.component == "Circuit Generation"]
        
        if any(not r.passed for r in circuit_results):
            print_comparison_report(circuit_results, f"Stability: {test_name}")
        
        assert_equivalence(circuit_results, test_name)
    
    @pytest.mark.parametrize("test_params", get_test_cases_by_category()["growing"])
    def test_growing_circuit_generation(self, test_params):
        """Test growing circuit generation equivalence."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to circuit-specific results for Phase 1
        circuit_results = [r for r in results if r.component == "Circuit Generation"]
        
        if any(not r.passed for r in circuit_results):
            print_comparison_report(circuit_results, f"Growing: {test_name}")
        
        assert_equivalence(circuit_results, test_name)
    
    @pytest.mark.parametrize("test_params", get_test_cases_by_category()["cultivation"])
    def test_cultivation_circuit_generation(self, test_params):
        """Test cultivation circuit generation equivalence."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to circuit-specific results for Phase 1
        circuit_results = [r for r in results if r.component == "Circuit Generation"]
        
        if any(not r.passed for r in circuit_results):
            print_comparison_report(circuit_results, f"Cultivation: {test_name}")
        
        assert_equivalence(circuit_results, test_name)


class TestPhase1ExtendedEquivalence:
    """Extended circuit equivalence tests with parameter variations."""
    
    @pytest.mark.parametrize("test_params", get_extended_test_cases("triangular", max_per_category=15))
    def test_triangular_extended_variations(self, test_params):
        """Test triangular circuits with extended parameter variations."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
        
        if not circuit_result.passed:
            print_comparison_report([circuit_result], f"Extended Triangular: {test_name}")
        
        assert_equivalence([circuit_result], test_name)
    
    @pytest.mark.parametrize("test_params", get_extended_test_cases("rectangular", max_per_category=10))
    def test_rectangular_extended_variations(self, test_params):
        """Test rectangular circuits with extended parameter variations."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
        
        if not circuit_result.passed:
            print_comparison_report([circuit_result], f"Extended Rectangular: {test_name}")
        
        assert_equivalence([circuit_result], test_name)
    
    @pytest.mark.parametrize("test_params", get_extended_test_cases("growing", max_per_category=10))
    def test_growing_extended_variations(self, test_params):
        """Test growing circuits with extended parameter variations."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
        
        if not circuit_result.passed:
            print_comparison_report([circuit_result], f"Extended Growing: {test_name}")
        
        assert_equivalence([circuit_result], test_name)


class TestPhase1StressAndEdgeCases:
    """Stress tests and edge cases for circuit generation."""
    
    @pytest.mark.parametrize("test_params", get_stress_test_cases())
    def test_circuit_stress_cases(self, test_params):
        """Test circuit generation with large/complex parameters."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
        
        if not circuit_result.passed:
            print_comparison_report([circuit_result], f"Stress Test: {test_name}")
        
        assert_equivalence([circuit_result], test_name)
    
    @pytest.mark.parametrize("test_params", get_edge_case_test_cases())
    def test_circuit_edge_cases(self, test_params):
        """Test circuit generation edge cases and boundary conditions."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
        
        if not circuit_result.passed:
            print_comparison_report([circuit_result], f"Edge Case: {test_name}")
        
        assert_equivalence([circuit_result], test_name)


class TestPhase1CircuitFeatures:
    """Individual feature testing for circuit generation components."""
    
    def test_detector_error_model_generation(self):
        """Test that detector error models are equivalent."""
        test_params = {"d": 3, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001}
        
        legacy, refactored = create_test_instances(test_params)
        
        # Extract and compare DEMs specifically
        legacy_dem = legacy.circuit.detector_error_model()
        refactored_dem = refactored.circuit.detector_error_model()
        
        assert str(legacy_dem) == str(refactored_dem), "Detector error models should be identical"
    
    def test_circuit_qubit_count_consistency(self):
        """Test that circuit qubit counts match tanner graph vertex counts."""
        test_params = {"d": 5, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001}
        
        legacy, refactored = create_test_instances(test_params)
        
        # Verify qubit count consistency
        legacy_circuit_qubits = legacy.circuit.num_qubits
        legacy_graph_qubits = legacy.tanner_graph.vcount()
        
        refactored_circuit_qubits = refactored.circuit.num_qubits
        refactored_graph_qubits = refactored.tanner_graph.vcount()
        
        assert legacy_circuit_qubits == legacy_graph_qubits, "Legacy: Circuit and graph qubit counts should match"
        assert refactored_circuit_qubits == refactored_graph_qubits, "Refactored: Circuit and graph qubit counts should match"
        assert legacy_circuit_qubits == refactored_circuit_qubits, "Circuit qubit counts should be equivalent"
    
    def test_noise_parameter_application(self):
        """Test that noise parameters are applied correctly."""
        # Test with individual error rates
        test_params = {
            "d": 3, "rounds": 1, "circuit_type": "tri",
            "p_bitflip": 0.001, "p_reset": 0.0008, "p_meas": 0.0012, 
            "p_cnot": 0.0015, "p_idle": 0.0005
        }
        
        legacy, refactored = create_test_instances(test_params)
        circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
        
        assert_equivalence([circuit_result], "Individual error rates")
        
        # Test with circuit-level noise
        test_params_circuit = {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.002}
        
        legacy_circuit, refactored_circuit = create_test_instances(test_params_circuit)
        circuit_result_circuit = compare_circuits(legacy_circuit.circuit, refactored_circuit.circuit)
        
        assert_equivalence([circuit_result_circuit], "Circuit-level noise")
    
    def test_cnot_schedule_variations(self):
        """Test different CNOT schedule configurations."""
        cnot_schedules = ["tri_optimal", "tri_optimal_reversed", "LLB"]
        
        for schedule in cnot_schedules:
            test_params = {
                "d": 3, "rounds": 1, "circuit_type": "tri", 
                "cnot_schedule": schedule, "p_circuit": 0.001
            }
            
            legacy, refactored = create_test_instances(test_params)
            circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
            
            assert_equivalence([circuit_result], f"CNOT schedule: {schedule}")
    
    def test_temporal_boundary_types(self):
        """Test different temporal boundary type configurations."""
        boundary_types = ["X", "Z", "Y"]
        
        for bdry_type in boundary_types:
            test_params = {
                "d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing",
                "temp_bdry_type": bdry_type, "p_circuit": 0.001
            }
            
            legacy, refactored = create_test_instances(test_params)
            circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
            
            assert_equivalence([circuit_result], f"Temporal boundary: {bdry_type}")


if __name__ == "__main__":
    # Run quick test for manual verification
    print("Running Phase 1 Circuit Equivalence Quick Test...")
    
    quick_cases = get_quick_test_suite()[:3]  # Just first 3 for manual testing
    
    for test_params in quick_cases:
        test_name = get_test_case_name(test_params)
        print(f"\nTesting: {test_name}")
        
        try:
            legacy, refactored = create_test_instances(test_params)
            circuit_result = compare_circuits(legacy.circuit, refactored.circuit)
            
            if circuit_result.passed:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                print(circuit_result.get_report())
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\nManual test completed.")