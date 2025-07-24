"""
Pytest configuration and fixtures for circuit equivalence tests.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List

import pytest
import stim


@pytest.fixture
def test_results_dir():
    """Fixture providing the test results directory."""
    results_dir = Path(__file__).parent / "fixtures"
    results_dir.mkdir(exist_ok=True)
    return results_dir


@pytest.fixture
def circuit_serializer():
    """Fixture providing circuit serialization utilities."""
    
    def serialize_circuit(circuit: stim.Circuit, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize a circuit and its metadata for comparison."""
        return {
            "circuit_string": str(circuit),
            "num_qubits": circuit.num_qubits,
            "num_instructions": len(circuit),
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
            "metadata": metadata,
            "timestamp": time.time()
        }
    
    return serialize_circuit


@pytest.fixture  
def result_collector(test_results_dir):
    """Fixture for collecting and storing test results."""
    
    def collect_results(test_name: str, results: List[Dict[str, Any]], branch: str):
        """Collect and store test results to JSON file."""
        filename = f"results_{branch}.json"
        filepath = test_results_dir / filename
        
        # Load existing results if file exists
        if filepath.exists():
            with open(filepath, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = {}
        
        # Add new results
        all_results[test_name] = results
        
        # Save back to file
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return filepath
    
    return collect_results


@pytest.fixture
def circuit_comparator():
    """Fixture providing circuit comparison utilities."""
    
    def compare_circuits(circuit1: stim.Circuit, circuit2: stim.Circuit) -> Dict[str, Any]:
        """Compare two circuits and return detailed comparison results."""
        comparison = {
            "identical": True,
            "differences": [],
            "properties": {}
        }
        
        # Compare basic properties
        props = ["num_qubits", "num_detectors", "num_observables"]
        for prop in props:
            val1, val2 = getattr(circuit1, prop), getattr(circuit2, prop)
            comparison["properties"][prop] = {"circuit1": val1, "circuit2": val2, "match": val1 == val2}
            if val1 != val2:
                comparison["identical"] = False
                comparison["differences"].append(f"{prop}: {val1} vs {val2}")
        
        # Compare instruction count
        len1, len2 = len(circuit1), len(circuit2)
        comparison["properties"]["num_instructions"] = {"circuit1": len1, "circuit2": len2, "match": len1 == len2}
        if len1 != len2:
            comparison["identical"] = False
            comparison["differences"].append(f"instruction_count: {len1} vs {len2}")
        
        # Compare circuit strings (most comprehensive)
        str1, str2 = str(circuit1), str(circuit2)
        comparison["properties"]["circuit_string_match"] = str1 == str2
        if str1 != str2:
            comparison["identical"] = False
            comparison["differences"].append("circuit_strings_differ")
            
            # Find first differing line for debugging
            lines1, lines2 = str1.split('\n'), str2.split('\n')
            for i, (line1, line2) in enumerate(zip(lines1, lines2)):
                if line1 != line2:
                    comparison["first_difference_line"] = i
                    comparison["first_difference"] = {"line1": line1, "line2": line2}
                    break
        
        return comparison
    
    return compare_circuits