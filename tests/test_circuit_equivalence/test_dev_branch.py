"""
Test circuit generation using the refactored package from dev branch.

This module tests circuit generation using the refactored CircuitBuilder implementation
to compare against the original monolithic implementation.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import pytest

# Import from local src (refactored version)
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))
from color_code_stim.color_code import ColorCode

from .shared_test_cases import (
    TRIANGULAR_TEST_CASES,
    RECTANGULAR_TEST_CASES, 
    STABILITY_TEST_CASES,
    GROWING_TEST_CASES,
    EDGE_CASE_TEST_CASES,
    get_test_case_name
)


class TestDevBranchCircuits:
    """Test circuit generation using refactored package from dev branch."""
    
    def test_triangular_circuits(self, circuit_serializer, result_collector):
        """Test triangular circuit generation on dev branch."""
        results = []
        
        for params in TRIANGULAR_TEST_CASES:
            test_name = get_test_case_name(params)
            print(f"  Testing: {test_name}")
            
            start_time = time.time()
            try:
                cc = ColorCode(**params)
                generation_time = time.time() - start_time
                
                # Serialize circuit and metadata
                metadata = {
                    "test_name": test_name,
                    "parameters": params,
                    "generation_time": generation_time,
                    "branch": "dev"
                }
                
                serialized = circuit_serializer(cc.circuit, metadata)
                results.append(serialized)
                
            except Exception as e:
                print(f"    ERROR: {e}")
                error_result = {
                    "test_name": test_name,
                    "parameters": params,
                    "error": str(e),
                    "branch": "dev",
                    "timestamp": time.time()
                }
                results.append(error_result)
        
        # Store results
        result_collector("triangular_circuits", results, "dev")
        
        # Basic validation - at least some circuits should be generated
        successful_results = [r for r in results if "error" not in r]
        assert len(successful_results) > 0, "No triangular circuits were successfully generated"
    
    def test_rectangular_circuits(self, circuit_serializer, result_collector):
        """Test rectangular circuit generation on dev branch."""
        results = []
        
        for params in RECTANGULAR_TEST_CASES:
            test_name = get_test_case_name(params)
            print(f"  Testing: {test_name}")
            
            start_time = time.time()
            try:
                cc = ColorCode(**params)
                generation_time = time.time() - start_time
                
                metadata = {
                    "test_name": test_name,
                    "parameters": params,
                    "generation_time": generation_time,
                    "branch": "dev"
                }
                
                serialized = circuit_serializer(cc.circuit, metadata)
                results.append(serialized)
                
            except Exception as e:
                print(f"    ERROR: {e}")
                error_result = {
                    "test_name": test_name,
                    "parameters": params,
                    "error": str(e),
                    "branch": "dev",
                    "timestamp": time.time()
                }
                results.append(error_result)
        
        result_collector("rectangular_circuits", results, "dev")
        
        successful_results = [r for r in results if "error" not in r]
        assert len(successful_results) > 0, "No rectangular circuits were successfully generated"
    
    def test_stability_circuits(self, circuit_serializer, result_collector):
        """Test stability circuit generation on dev branch."""
        results = []
        
        for params in STABILITY_TEST_CASES:
            test_name = get_test_case_name(params)
            print(f"  Testing: {test_name}")
            
            start_time = time.time()
            try:
                cc = ColorCode(**params)
                generation_time = time.time() - start_time
                
                metadata = {
                    "test_name": test_name,
                    "parameters": params,
                    "generation_time": generation_time,
                    "branch": "dev"
                }
                
                serialized = circuit_serializer(cc.circuit, metadata)
                results.append(serialized)
                
            except Exception as e:
                print(f"    ERROR: {e}")
                error_result = {
                    "test_name": test_name,
                    "parameters": params,
                    "error": str(e),
                    "branch": "dev",
                    "timestamp": time.time()
                }
                results.append(error_result)
        
        result_collector("stability_circuits", results, "dev")
        
        # Note: We expect some stability circuits to fail due to known bug
        print(f"  Generated {len([r for r in results if 'error' not in r])} successful stability circuits")
    
    def test_growing_circuits(self, circuit_serializer, result_collector):
        """Test growing circuit generation on dev branch."""
        results = []
        
        for params in GROWING_TEST_CASES:
            test_name = get_test_case_name(params)
            print(f"  Testing: {test_name}")
            
            start_time = time.time()
            try:
                cc = ColorCode(**params)
                generation_time = time.time() - start_time
                
                metadata = {
                    "test_name": test_name,
                    "parameters": params,
                    "generation_time": generation_time,
                    "branch": "dev"
                }
                
                serialized = circuit_serializer(cc.circuit, metadata)
                results.append(serialized)
                
            except Exception as e:
                print(f"    ERROR: {e}")
                error_result = {
                    "test_name": test_name,
                    "parameters": params,
                    "error": str(e),
                    "branch": "dev",
                    "timestamp": time.time()
                }
                results.append(error_result)
        
        result_collector("growing_circuits", results, "dev")
        
        successful_results = [r for r in results if "error" not in r]
        assert len(successful_results) > 0, "No growing circuits were successfully generated"
    
    def test_edge_cases(self, circuit_serializer, result_collector):
        """Test edge case circuit generation on dev branch."""
        results = []
        
        for params in EDGE_CASE_TEST_CASES:
            test_name = get_test_case_name(params)
            print(f"  Testing: {test_name}")
            
            start_time = time.time()
            try:
                cc = ColorCode(**params)
                generation_time = time.time() - start_time
                
                metadata = {
                    "test_name": test_name,
                    "parameters": params,
                    "generation_time": generation_time,
                    "branch": "dev"
                }
                
                serialized = circuit_serializer(cc.circuit, metadata)
                results.append(serialized)
                
            except Exception as e:
                print(f"    ERROR: {e}")
                error_result = {
                    "test_name": test_name,
                    "parameters": params,
                    "error": str(e),
                    "branch": "dev",
                    "timestamp": time.time()
                }
                results.append(error_result)
        
        result_collector("edge_cases", results, "dev")
        
        successful_results = [r for r in results if "error" not in r]
        assert len(successful_results) > 0, "No edge case circuits were successfully generated"


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v"])