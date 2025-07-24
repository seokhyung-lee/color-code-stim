"""
Integration Tests: End-to-End Equivalence Testing

This module provides comprehensive integration testing that validates the complete
equivalence between legacy and refactored ColorCode implementations across all
currently implemented phases.

Focus Areas:
- Complete workflow equivalence (graph → circuit → ready for simulation)
- Cross-phase data consistency
- Performance characteristics
- Memory usage patterns
- Error handling and edge cases
"""

import pytest
import sys
import time
import tracemalloc
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(utils_path))

test_data_path = Path(__file__).parent.parent / "test_data"
sys.path.insert(0, str(test_data_path))

from comparison_utils import (
    create_test_instances,
    compare_full_instances,
    assert_equivalence,
    print_comparison_report,
    ComparisonResult
)
from comprehensive_test_cases import (
    get_comprehensive_test_suite,
    get_quick_test_suite,
    get_stress_test_cases,
    get_edge_case_test_cases,
    get_test_case_name
)


class TestEndToEndEquivalence:
    """Comprehensive end-to-end equivalence testing."""
    
    @pytest.mark.parametrize("test_params", get_quick_test_suite())
    def test_complete_workflow_equivalence(self, test_params):
        """Test complete workflow equivalence for all circuit types."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # All currently implemented phases should pass
        implemented_phases = ["Tanner Graph Structure", "Qubit Groups", "Circuit Generation"]
        implemented_results = [r for r in results if r.component in implemented_phases]
        
        if any(not r.passed for r in implemented_results):
            print_comparison_report(implemented_results, f"End-to-End: {test_name}")
        
        assert_equivalence(implemented_results, f"Complete workflow for {test_name}")
    
    def test_cross_phase_data_consistency(self):
        """Test that data flows correctly between phases."""
        test_params = {"d": 5, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001}
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        
        # Verify data consistency across phases
        consistency_results = []
        
        # Phase 1-2 consistency: Graph vertices should match circuit qubits
        legacy_graph_qubits = legacy.tanner_graph.vcount()
        legacy_circuit_qubits = legacy.circuit.num_qubits
        refactored_graph_qubits = refactored.tanner_graph.vcount()
        refactored_circuit_qubits = refactored.circuit.num_qubits
        
        consistency_result = ComparisonResult("Cross-Phase Data Consistency")
        
        if legacy_graph_qubits != legacy_circuit_qubits:
            consistency_result.add_difference(f"Legacy graph-circuit mismatch: {legacy_graph_qubits} vs {legacy_circuit_qubits}")
        
        if refactored_graph_qubits != refactored_circuit_qubits:
            consistency_result.add_difference(f"Refactored graph-circuit mismatch: {refactored_graph_qubits} vs {refactored_circuit_qubits}")
        
        if legacy_graph_qubits != refactored_graph_qubits:
            consistency_result.add_difference(f"Graph qubit count mismatch: {legacy_graph_qubits} vs {refactored_graph_qubits}")
        
        if legacy_circuit_qubits != refactored_circuit_qubits:
            consistency_result.add_difference(f"Circuit qubit count mismatch: {legacy_circuit_qubits} vs {refactored_circuit_qubits}")
        
        if consistency_result.passed:
            consistency_result.set_summary("All cross-phase data is consistent")
            consistency_result.add_detail(f"Consistent qubit count: {legacy_graph_qubits}")
        else:
            consistency_result.set_summary("Cross-phase data inconsistencies detected")
        
        consistency_results.append(consistency_result)
        
        if any(not r.passed for r in consistency_results):
            print_comparison_report(consistency_results, f"Cross-Phase Consistency: {test_name}")
        
        assert_equivalence(consistency_results, f"Cross-phase consistency for {test_name}")


class TestIntegrationStressAndPerformance:
    """Integration stress testing and performance validation."""
    
    @pytest.mark.parametrize("test_params", get_stress_test_cases())
    def test_stress_integration(self, test_params):
        """Test integration with large/complex parameters."""
        test_name = get_test_case_name(test_params)
        
        # Time the creation and comparison
        start_time = time.time()
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Filter to implemented phases
        implemented_phases = ["Tanner Graph Structure", "Qubit Groups", "Circuit Generation"]
        implemented_results = [r for r in results if r.component in implemented_phases]
        
        # Add performance information
        perf_result = ComparisonResult("Performance Characteristics")
        perf_result.add_detail(f"Total test time: {total_time:.3f}s")
        perf_result.add_detail(f"Legacy graph vertices: {legacy.tanner_graph.vcount()}")
        perf_result.add_detail(f"Legacy circuit operations: {len(legacy.circuit)}")
        
        if total_time > 30:  # Flag if test takes more than 30 seconds
            perf_result.add_difference(f"Test took {total_time:.3f}s (>30s threshold)")
        else:
            perf_result.set_summary(f"Performance acceptable: {total_time:.3f}s")
        
        all_results = implemented_results + [perf_result]
        
        if any(not r.passed for r in all_results):
            print_comparison_report(all_results, f"Stress Integration: {test_name}")
        
        assert_equivalence(implemented_results, f"Stress integration for {test_name}")
    
    def test_memory_usage_comparison(self):
        """Compare memory usage between legacy and refactored implementations."""
        test_params = {"d": 7, "rounds": 3, "circuit_type": "tri", "p_circuit": 0.001}
        test_name = get_test_case_name(test_params)
        
        # Measure legacy memory usage
        tracemalloc.start()
        legacy, _ = create_test_instances(test_params)
        # Access all properties to trigger lazy loading
        _ = legacy.tanner_graph.vcount()
        _ = legacy.circuit.num_qubits
        _ = len(legacy.qubit_groups["data"])
        legacy_memory = tracemalloc.get_traced_memory()[1]  # Peak memory
        tracemalloc.stop()
        
        # Measure refactored memory usage
        tracemalloc.start()
        _, refactored = create_test_instances(test_params)
        # Access all properties to trigger lazy loading
        _ = refactored.tanner_graph.vcount()
        _ = refactored.circuit.num_qubits
        _ = len(refactored.qubit_groups["data"])
        refactored_memory = tracemalloc.get_traced_memory()[1]  # Peak memory
        tracemalloc.stop()
        
        memory_result = ComparisonResult("Memory Usage Comparison")
        memory_result.add_detail(f"Legacy peak memory: {legacy_memory / 1024 / 1024:.2f} MB")
        memory_result.add_detail(f"Refactored peak memory: {refactored_memory / 1024 / 1024:.2f} MB")
        
        memory_ratio = refactored_memory / legacy_memory if legacy_memory > 0 else 1.0
        memory_result.add_detail(f"Memory ratio (refactored/legacy): {memory_ratio:.3f}")
        
        # Flag if memory usage increases significantly (>20%)
        if memory_ratio > 1.2:
            memory_result.add_difference(f"Memory usage increased by {(memory_ratio-1)*100:.1f}%")
        else:
            memory_result.set_summary(f"Memory usage acceptable: {memory_ratio:.3f}x")
        
        if not memory_result.passed:
            print_comparison_report([memory_result], f"Memory Usage: {test_name}")
        
        # Don't fail test for memory differences, just report
        print(f"\nMemory Usage Report for {test_name}:")
        print(memory_result.get_report())


class TestIntegrationEdgeCases:
    """Integration testing for edge cases and error conditions."""
    
    @pytest.mark.parametrize("test_params", get_edge_case_test_cases())
    def test_edge_case_integration(self, test_params):
        """Test integration with edge cases and boundary conditions."""
        test_name = get_test_case_name(test_params)
        
        legacy, refactored = create_test_instances(test_params)
        results = compare_full_instances(legacy, refactored)
        
        # Filter to implemented phases
        implemented_phases = ["Tanner Graph Structure", "Qubit Groups", "Circuit Generation"]
        implemented_results = [r for r in results if r.component in implemented_phases]
        
        if any(not r.passed for r in implemented_results):
            print_comparison_report(implemented_results, f"Edge Case Integration: {test_name}")
        
        assert_equivalence(implemented_results, f"Edge case integration for {test_name}")
    
    def test_parameter_validation_consistency(self):
        """Test that parameter validation behaves consistently."""
        # Test cases that should fail in both implementations
        invalid_cases = [
            {"d": 2, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},  # d too small for tri
            {"d": 3, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},  # d odd for rec
        ]
        
        for invalid_params in invalid_cases:
            test_name = get_test_case_name(invalid_params)
            
            legacy_failed = False
            refactored_failed = False
            
            try:
                legacy, _ = create_test_instances(invalid_params)
                # If we get here, legacy didn't fail as expected
            except Exception:
                legacy_failed = True
            
            try:
                _, refactored = create_test_instances(invalid_params)
                # If we get here, refactored didn't fail as expected
            except Exception:
                refactored_failed = True
            
            # Both should fail or both should succeed
            assert legacy_failed == refactored_failed, f"Parameter validation inconsistency for {test_name}: legacy_failed={legacy_failed}, refactored_failed={refactored_failed}"


class TestComprehensiveEquivalence:
    """Comprehensive testing across all parameter combinations."""
    
    @pytest.mark.slow
    def test_comprehensive_triangular_equivalence(self):
        """Comprehensive test of all triangular circuit variations."""
        test_suite = get_comprehensive_test_suite()
        triangular_cases = test_suite["triangular"]
        
        failed_cases = []
        
        for test_params in triangular_cases:
            test_name = get_test_case_name(test_params)
            
            try:
                legacy, refactored = create_test_instances(test_params)
                results = compare_full_instances(legacy, refactored)
                
                # Filter to implemented phases
                implemented_phases = ["Tanner Graph Structure", "Qubit Groups", "Circuit Generation"]
                implemented_results = [r for r in results if r.component in implemented_phases]
                
                if any(not r.passed for r in implemented_results):
                    failed_cases.append((test_name, implemented_results))
            except Exception as e:
                failed_cases.append((test_name, [f"Exception: {e}"]))
        
        if failed_cases:
            print(f"\n{len(failed_cases)} out of {len(triangular_cases)} triangular test cases failed:")
            for test_name, results in failed_cases[:5]:  # Show first 5 failures
                print(f"  • {test_name}")
                if isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'get_report'):
                    for result in results:
                        if not result.passed:
                            print(f"    - {result.component}: {result.summary}")
        
        # Assert that we have a high success rate (>95%)
        success_rate = (len(triangular_cases) - len(failed_cases)) / len(triangular_cases)
        assert success_rate >= 0.95, f"Comprehensive triangular test success rate too low: {success_rate:.1%} ({len(failed_cases)} failures)"
    
    @pytest.mark.slow
    def test_comprehensive_rectangular_equivalence(self):
        """Comprehensive test of all rectangular circuit variations."""
        test_suite = get_comprehensive_test_suite()
        rectangular_cases = test_suite["rectangular"]
        
        failed_cases = []
        
        for test_params in rectangular_cases:
            test_name = get_test_case_name(test_params)
            
            try:
                legacy, refactored = create_test_instances(test_params)
                results = compare_full_instances(legacy, refactored)
                
                # Filter to implemented phases
                implemented_phases = ["Tanner Graph Structure", "Qubit Groups", "Circuit Generation"]
                implemented_results = [r for r in results if r.component in implemented_phases]
                
                if any(not r.passed for r in implemented_results):
                    failed_cases.append((test_name, implemented_results))
            except Exception as e:
                failed_cases.append((test_name, [f"Exception: {e}"]))
        
        if failed_cases:
            print(f"\n{len(failed_cases)} out of {len(rectangular_cases)} rectangular test cases failed:")
            for test_name, results in failed_cases[:5]:  # Show first 5 failures
                print(f"  • {test_name}")
        
        # Assert that we have a high success rate (>95%)
        success_rate = (len(rectangular_cases) - len(failed_cases)) / len(rectangular_cases)
        assert success_rate >= 0.95, f"Comprehensive rectangular test success rate too low: {success_rate:.1%} ({len(failed_cases)} failures)"


if __name__ == "__main__":
    # Run quick integration test for manual verification
    print("Running Integration Tests Quick Verification...")
    
    quick_cases = get_quick_test_suite()[:2]  # Just first 2 for manual testing
    
    for test_params in quick_cases:
        test_name = get_test_case_name(test_params)
        print(f"\nTesting Integration: {test_name}")
        
        try:
            start_time = time.time()
            legacy, refactored = create_test_instances(test_params)
            results = compare_full_instances(legacy, refactored)
            end_time = time.time()
            
            # Filter to implemented phases
            implemented_phases = ["Tanner Graph Structure", "Qubit Groups", "Circuit Generation"]
            implemented_results = [r for r in results if r.component in implemented_phases]
            
            passed = all(r.passed for r in implemented_results)
            
            if passed:
                print(f"✅ {test_name}: PASSED ({end_time - start_time:.3f}s)")
            else:
                print(f"❌ {test_name}: FAILED")
                for result in implemented_results:
                    if not result.passed:
                        print(f"  - {result.component}: {result.summary}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\nIntegration test manual verification completed.")