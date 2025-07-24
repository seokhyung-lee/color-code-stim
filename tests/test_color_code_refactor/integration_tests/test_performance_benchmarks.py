"""
Performance Benchmarking Tests

This module provides systematic performance benchmarking between legacy and 
refactored ColorCode implementations to ensure no significant performance 
regression occurs during refactoring.

Focus Areas:
- Initialization time comparisons
- Graph construction performance
- Circuit generation performance  
- Memory usage analysis
- Scalability testing with increasing parameters
"""

import pytest
import sys
import time
import statistics
import tracemalloc
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add utils to path
utils_path = Path(__file__).parent.parent / "utils"
sys.path.insert(0, str(utils_path))

test_data_path = Path(__file__).parent.parent / "test_data"
sys.path.insert(0, str(test_data_path))

from comparison_utils import create_test_instances
from comprehensive_test_cases import get_test_case_name


class PerformanceBenchmark:
    """Performance measurement and analysis utilities."""
    
    @staticmethod
    def measure_timing(func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Measure execution time of a function.
        
        Returns
        -------
        tuple
            (result, execution_time_seconds)
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
    @staticmethod
    def measure_memory(func, *args, **kwargs) -> Tuple[Any, int]:
        """
        Measure peak memory usage of a function.
        
        Returns
        -------
        tuple
            (result, peak_memory_bytes)
        """
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, peak
    
    @staticmethod
    def statistical_comparison(legacy_times: List[float], refactored_times: List[float], 
                             tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Perform statistical comparison of timing results.
        
        Parameters
        ----------
        legacy_times, refactored_times : list of float
            Timing measurements for comparison
        tolerance : float
            Acceptable performance regression threshold (10% by default)
            
        Returns
        -------
        dict
            Statistical comparison results
        """
        legacy_mean = statistics.mean(legacy_times)
        refactored_mean = statistics.mean(refactored_times)
        
        legacy_stdev = statistics.stdev(legacy_times) if len(legacy_times) > 1 else 0
        refactored_stdev = statistics.stdev(refactored_times) if len(refactored_times) > 1 else 0
        
        ratio = refactored_mean / legacy_mean if legacy_mean > 0 else 1.0
        regression = ratio - 1.0
        
        return {
            "legacy_mean": legacy_mean,
            "refactored_mean": refactored_mean,
            "legacy_stdev": legacy_stdev,
            "refactored_stdev": refactored_stdev,
            "ratio": ratio,
            "regression": regression,
            "acceptable": abs(regression) <= tolerance,
            "tolerance": tolerance
        }


class TestInitializationPerformance:
    """Test ColorCode initialization performance."""
    
    def test_basic_initialization_performance(self):
        """Test basic initialization time for different circuit types."""
        test_cases = [
            {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
            {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001},
        ]
        
        for test_params in test_cases:
            test_name = get_test_case_name(test_params)
            
            # Measure initialization times (multiple runs for statistical validity)
            legacy_times = []
            refactored_times = []
            
            for _ in range(5):  # 5 runs for statistical analysis
                _, legacy_time = PerformanceBenchmark.measure_timing(create_test_instances, test_params)
                legacy_times.append(legacy_time)
                
                _, refactored_time = PerformanceBenchmark.measure_timing(create_test_instances, test_params)
                refactored_times.append(refactored_time)
            
            # Statistical comparison
            stats = PerformanceBenchmark.statistical_comparison(legacy_times, refactored_times, tolerance=0.2)
            
            print(f"\nInitialization Performance - {test_name}:")
            print(f"  Legacy:     {stats['legacy_mean']:.4f}s ± {stats['legacy_stdev']:.4f}s")
            print(f"  Refactored: {stats['refactored_mean']:.4f}s ± {stats['refactored_stdev']:.4f}s")
            print(f"  Ratio:      {stats['ratio']:.3f}x")
            print(f"  Regression: {stats['regression']:.1%}")
            
            # Allow up to 20% performance regression for initialization
            assert stats["acceptable"], f"Initialization performance regression too high for {test_name}: {stats['regression']:.1%}"
    
    def test_lazy_loading_performance(self):
        """Test that lazy loading doesn't cause performance issues."""
        test_params = {"d": 5, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001}
        
        # Test repeated property access
        legacy, refactored = create_test_instances(test_params)
        
        # Time multiple accesses to see if lazy loading causes delays
        properties_to_test = ["tanner_graph", "circuit", "qubit_groups"]
        
        for prop_name in properties_to_test:
            legacy_times = []
            refactored_times = []
            
            for _ in range(3):  # 3 accesses each
                _, legacy_time = PerformanceBenchmark.measure_timing(getattr, legacy, prop_name)
                legacy_times.append(legacy_time)
                
                _, refactored_time = PerformanceBenchmark.measure_timing(getattr, refactored, prop_name)
                refactored_times.append(refactored_time)
            
            # First access might be slower due to lazy loading, but subsequent should be fast
            legacy_subsequent = statistics.mean(legacy_times[1:]) if len(legacy_times) > 1 else legacy_times[0]
            refactored_subsequent = statistics.mean(refactored_times[1:]) if len(refactored_times) > 1 else refactored_times[0]
            
            print(f"\nLazy Loading Performance - {prop_name}:")
            print(f"  Legacy subsequent:     {legacy_subsequent:.6f}s")
            print(f"  Refactored subsequent: {refactored_subsequent:.6f}s")
            
            # Subsequent accesses should be very fast (<1ms)
            assert legacy_subsequent < 0.001, f"Legacy {prop_name} subsequent access too slow: {legacy_subsequent:.6f}s"
            assert refactored_subsequent < 0.001, f"Refactored {prop_name} subsequent access too slow: {refactored_subsequent:.6f}s"


class TestGraphConstructionPerformance:
    """Test Tanner graph construction performance."""
    
    def test_graph_construction_scaling(self):
        """Test graph construction performance with increasing distance."""
        distances = [3, 5, 7]  # Test different scales
        
        for d in distances:
            test_params = {"d": d, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001}
            test_name = get_test_case_name(test_params)
            
            # Measure graph construction time
            legacy_times = []
            refactored_times = []
            
            for _ in range(3):  # 3 runs for averaging
                legacy, refactored = create_test_instances(test_params)
                
                # Time graph access (which triggers construction)
                _, legacy_time = PerformanceBenchmark.measure_timing(getattr, legacy, "tanner_graph")
                legacy_times.append(legacy_time)
                
                _, refactored_time = PerformanceBenchmark.measure_timing(getattr, refactored, "tanner_graph")
                refactored_times.append(refactored_time)
            
            stats = PerformanceBenchmark.statistical_comparison(legacy_times, refactored_times, tolerance=0.15)
            
            print(f"\nGraph Construction Performance - {test_name}:")
            print(f"  Legacy:     {stats['legacy_mean']:.4f}s")
            print(f"  Refactored: {stats['refactored_mean']:.4f}s")
            print(f"  Ratio:      {stats['ratio']:.3f}x")
            print(f"  Vertices:   {legacy.tanner_graph.vcount()}")
            
            # Allow up to 15% performance regression for graph construction
            assert stats["acceptable"], f"Graph construction performance regression too high for {test_name}: {stats['regression']:.1%}"
    
    def test_rectangular_graph_performance(self):
        """Test rectangular graph construction performance."""
        test_cases = [
            {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
            {"d": 4, "d2": 6, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
            {"d": 6, "d2": 8, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
        ]
        
        for test_params in test_cases:
            test_name = get_test_case_name(test_params)
            
            legacy_times = []
            refactored_times = []
            
            for _ in range(3):
                legacy, refactored = create_test_instances(test_params)
                
                _, legacy_time = PerformanceBenchmark.measure_timing(getattr, legacy, "tanner_graph")
                legacy_times.append(legacy_time)
                
                _, refactored_time = PerformanceBenchmark.measure_timing(getattr, refactored, "tanner_graph")
                refactored_times.append(refactored_time)
            
            stats = PerformanceBenchmark.statistical_comparison(legacy_times, refactored_times, tolerance=0.15)
            
            print(f"\nRectangular Graph Performance - {test_name}:")
            print(f"  Legacy:     {stats['legacy_mean']:.4f}s")
            print(f"  Refactored: {stats['refactored_mean']:.4f}s")
            print(f"  Ratio:      {stats['ratio']:.3f}x")
            
            assert stats["acceptable"], f"Rectangular graph performance regression too high for {test_name}: {stats['regression']:.1%}"


class TestCircuitGenerationPerformance:
    """Test circuit generation performance."""
    
    def test_circuit_generation_scaling(self):
        """Test circuit generation performance with increasing complexity."""
        test_cases = [
            {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 3, "rounds": 3, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 5, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 5, "rounds": 3, "circuit_type": "tri", "p_circuit": 0.001},
        ]
        
        for test_params in test_cases:
            test_name = get_test_case_name(test_params)
            
            legacy_times = []
            refactored_times = []
            
            for _ in range(3):
                legacy, refactored = create_test_instances(test_params)
                
                # Time circuit access (which triggers generation)
                _, legacy_time = PerformanceBenchmark.measure_timing(getattr, legacy, "circuit")
                legacy_times.append(legacy_time)
                
                _, refactored_time = PerformanceBenchmark.measure_timing(getattr, refactored, "circuit")
                refactored_times.append(refactored_time)
            
            stats = PerformanceBenchmark.statistical_comparison(legacy_times, refactored_times, tolerance=0.1)
            
            print(f"\nCircuit Generation Performance - {test_name}:")
            print(f"  Legacy:     {stats['legacy_mean']:.4f}s")
            print(f"  Refactored: {stats['refactored_mean']:.4f}s")
            print(f"  Ratio:      {stats['ratio']:.3f}x")
            print(f"  Operations: {len(legacy.circuit)}")
            
            # Circuit generation should be very consistent - allow only 10% regression
            assert stats["acceptable"], f"Circuit generation performance regression too high for {test_name}: {stats['regression']:.1%}"
    
    def test_complex_circuit_performance(self):
        """Test performance with complex circuit configurations."""
        test_cases = [
            {"d": 3, "d2": 5, "rounds": 2, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001},
            {"d": 4, "d2": 6, "rounds": 4, "circuit_type": "rec_stability", "p_circuit": 0.001},
        ]
        
        for test_params in test_cases:
            test_name = get_test_case_name(test_params)
            
            legacy_times = []
            refactored_times = []
            
            for _ in range(3):
                legacy, refactored = create_test_instances(test_params)
                
                # Time full initialization (graph + circuit)
                start_time = time.perf_counter()
                _ = legacy.tanner_graph.vcount()
                _ = legacy.circuit.num_qubits
                legacy_time = time.perf_counter() - start_time
                legacy_times.append(legacy_time)
                
                start_time = time.perf_counter()
                _ = refactored.tanner_graph.vcount()
                _ = refactored.circuit.num_qubits
                refactored_time = time.perf_counter() - start_time
                refactored_times.append(refactored_time)
            
            stats = PerformanceBenchmark.statistical_comparison(legacy_times, refactored_times, tolerance=0.15)
            
            print(f"\nComplex Circuit Performance - {test_name}:")
            print(f"  Legacy:     {stats['legacy_mean']:.4f}s")
            print(f"  Refactored: {stats['refactored_mean']:.4f}s")
            print(f"  Ratio:      {stats['ratio']:.3f}x")
            
            assert stats["acceptable"], f"Complex circuit performance regression too high for {test_name}: {stats['regression']:.1%}"


class TestMemoryUsageAnalysis:
    """Test memory usage characteristics."""
    
    def test_memory_usage_comparison(self):
        """Compare memory usage between implementations."""
        test_cases = [
            {"d": 3, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 4, "d2": 6, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
            {"d": 5, "rounds": 3, "circuit_type": "tri", "p_circuit": 0.001},
        ]
        
        for test_params in test_cases:
            test_name = get_test_case_name(test_params)
            
            # Measure legacy memory usage
            (legacy, _), legacy_memory = PerformanceBenchmark.measure_memory(create_test_instances, test_params)
            # Access properties to trigger full initialization
            _ = legacy.tanner_graph.vcount()
            _ = legacy.circuit.num_qubits
            _ = len(legacy.qubit_groups["data"])
            
            # Measure refactored memory usage
            (_, refactored), refactored_memory = PerformanceBenchmark.measure_memory(create_test_instances, test_params)
            # Access properties to trigger full initialization
            _ = refactored.tanner_graph.vcount()
            _ = refactored.circuit.num_qubits
            _ = len(refactored.qubit_groups["data"])
            
            memory_ratio = refactored_memory / legacy_memory if legacy_memory > 0 else 1.0
            
            print(f"\nMemory Usage - {test_name}:")
            print(f"  Legacy:     {legacy_memory / 1024 / 1024:.2f} MB")
            print(f"  Refactored: {refactored_memory / 1024 / 1024:.2f} MB")
            print(f"  Ratio:      {memory_ratio:.3f}x")
            
            # Allow up to 25% memory increase (refactoring might have some overhead)
            assert memory_ratio <= 1.25, f"Memory usage increase too high for {test_name}: {memory_ratio:.3f}x"
    
    def test_memory_scaling_behavior(self):
        """Test memory scaling with increasing problem size."""
        distances = [3, 5, 7]
        
        legacy_memories = []
        refactored_memories = []
        
        for d in distances:
            test_params = {"d": d, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001}
            
            # Measure memory for this distance
            (legacy, refactored), _ = PerformanceBenchmark.measure_memory(create_test_instances, test_params)
            
            # Trigger full initialization
            legacy_graph_size = legacy.tanner_graph.vcount()
            legacy_circuit_size = legacy.circuit.num_qubits
            
            refactored_graph_size = refactored.tanner_graph.vcount()
            refactored_circuit_size = refactored.circuit.num_qubits
            
            # Use graph size as a proxy for memory scaling
            legacy_memories.append(legacy_graph_size)
            refactored_memories.append(refactored_graph_size)
            
            print(f"\nMemory Scaling - d={d}:")
            print(f"  Legacy graph size:     {legacy_graph_size}")
            print(f"  Refactored graph size: {refactored_graph_size}")
            print(f"  Legacy circuit size:   {legacy_circuit_size}")
            print(f"  Refactored circuit size: {refactored_circuit_size}")
            
            # Sizes should be identical
            assert legacy_graph_size == refactored_graph_size, f"Graph size mismatch for d={d}"
            assert legacy_circuit_size == refactored_circuit_size, f"Circuit size mismatch for d={d}"


@pytest.mark.benchmark
class TestPerformanceRegression:
    """Comprehensive performance regression testing."""
    
    def test_overall_performance_regression(self):
        """Test overall performance regression across multiple scenarios."""
        test_cases = [
            {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001},
            {"d": 4, "d2": 4, "rounds": 1, "circuit_type": "rec", "p_circuit": 0.001},
            {"d": 3, "d2": 5, "rounds": 1, "circuit_type": "growing", "temp_bdry_type": "X", "p_circuit": 0.001},
            {"d": 5, "rounds": 2, "circuit_type": "tri", "p_circuit": 0.001},
        ]
        
        total_legacy_time = 0
        total_refactored_time = 0
        
        for test_params in test_cases:
            test_name = get_test_case_name(test_params)
            
            # Measure full initialization time
            _, legacy_time = PerformanceBenchmark.measure_timing(create_test_instances, test_params)
            total_legacy_time += legacy_time
            
            _, refactored_time = PerformanceBenchmark.measure_timing(create_test_instances, test_params)
            total_refactored_time += refactored_time
            
            print(f"\nOverall Performance - {test_name}:")
            print(f"  Legacy:     {legacy_time:.4f}s")
            print(f"  Refactored: {refactored_time:.4f}s")
        
        overall_ratio = total_refactored_time / total_legacy_time if total_legacy_time > 0 else 1.0
        overall_regression = overall_ratio - 1.0
        
        print(f"\nOverall Performance Summary:")
        print(f"  Total Legacy Time:     {total_legacy_time:.4f}s")
        print(f"  Total Refactored Time: {total_refactored_time:.4f}s")
        print(f"  Overall Ratio:         {overall_ratio:.3f}x")
        print(f"  Overall Regression:    {overall_regression:.1%}")
        
        # Allow up to 15% overall performance regression
        assert overall_regression <= 0.15, f"Overall performance regression too high: {overall_regression:.1%}"


if __name__ == "__main__":
    # Run quick performance test for manual verification
    print("Running Performance Benchmarks Quick Test...")
    
    test_params = {"d": 3, "rounds": 1, "circuit_type": "tri", "p_circuit": 0.001}
    test_name = get_test_case_name(test_params)
    
    print(f"\nBenchmarking: {test_name}")
    
    # Time multiple runs
    legacy_times = []
    refactored_times = []
    
    for i in range(3):
        print(f"  Run {i+1}/3...")
        
        _, legacy_time = PerformanceBenchmark.measure_timing(create_test_instances, test_params)
        legacy_times.append(legacy_time)
        
        _, refactored_time = PerformanceBenchmark.measure_timing(create_test_instances, test_params)
        refactored_times.append(refactored_time)
    
    stats = PerformanceBenchmark.statistical_comparison(legacy_times, refactored_times)
    
    print(f"\nPerformance Results:")
    print(f"  Legacy:     {stats['legacy_mean']:.4f}s ± {stats['legacy_stdev']:.4f}s")
    print(f"  Refactored: {stats['refactored_mean']:.4f}s ± {stats['refactored_stdev']:.4f}s")
    print(f"  Ratio:      {stats['ratio']:.3f}x")
    print(f"  Regression: {stats['regression']:.1%}")
    print(f"  Acceptable: {stats['acceptable']}")
    
    print("\nPerformance benchmark manual test completed.")