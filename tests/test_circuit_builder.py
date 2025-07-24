"""
Unit tests for the CircuitBuilder class.

Tests the circuit builder functionality extracted during Phase 1 of the refactoring.
Note: These tests focus on structure and initialization rather than full circuit generation,
which requires complex dependencies like igraph and stim.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add src to path to import modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from color_code_stim.circuit_builder import CircuitBuilder
    from color_code_stim.config import CNOT_SCHEDULES
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestCircuitBuilderStructure:
    """Test CircuitBuilder class structure and initialization."""
    
    @pytest.fixture
    def mock_tanner_graph(self):
        """Create a mock Tanner graph."""
        graph = Mock()
        graph.vcount.return_value = 10
        graph.vs = []
        return graph
    
    @pytest.fixture
    def mock_qubit_groups(self):
        """Create mock qubit groups."""
        mock_seq = Mock()
        mock_seq.__len__ = Mock(return_value=5)
        mock_seq.__getitem__ = Mock(return_value=[0, 1, 2, 3, 4])
        
        return {
            "data": mock_seq,
            "anc": mock_seq,
            "anc_Z": mock_seq,
            "anc_X": mock_seq,
        }
    
    @pytest.fixture
    def basic_params(self, mock_tanner_graph, mock_qubit_groups):
        """Basic parameters for CircuitBuilder initialization."""
        return {
            "d": 3,
            "d2": None,
            "rounds": 1,
            "circuit_type": "tri",
            "cnot_schedule": CNOT_SCHEDULES["tri_optimal"],
            "temp_bdry_type": "Z",
            "physical_probs": {
                "bitflip": 0.001,
                "reset": 0.001,
                "meas": 0.001,
                "cnot": 0.001,
                "idle": 0.001,
            },
            "perfect_init_final": False,
            "tanner_graph": mock_tanner_graph,
            "qubit_groups": mock_qubit_groups,
        }
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_circuit_builder_initialization(self, basic_params):
        """Test CircuitBuilder can be initialized with basic parameters."""
        builder = CircuitBuilder(**basic_params)
        
        # Test parameter assignment
        assert builder.d == 3
        assert builder.d2 is None
        assert builder.rounds == 1
        assert builder.circuit_type == "tri"
        assert builder.temp_bdry_type == "Z"
        assert not builder.perfect_init_final
        
        # Test derived attributes
        assert builder.p_bitflip == 0.001
        assert builder.p_reset == 0.001
        assert builder.p_meas == 0.001
        assert builder.p_cnot == 0.001
        assert builder.p_idle == 0.001
        
        assert builder.num_qubits == 10
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_circuit_builder_with_optional_params(self, basic_params):
        """Test CircuitBuilder with optional parameters."""
        basic_params.update({
            "exclude_non_essential_pauli_detectors": True,
            "comparative_decoding": True,
        })
        
        builder = CircuitBuilder(**basic_params)
        assert builder.exclude_non_essential_pauli_detectors
        assert builder.comparative_decoding
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_circuit_builder_rectangular_params(self, basic_params):
        """Test CircuitBuilder initialization for rectangular circuit."""
        basic_params.update({
            "circuit_type": "rec",
            "d2": 4,
        })
        
        builder = CircuitBuilder(**basic_params)
        assert builder.circuit_type == "rec"
        assert builder.d2 == 4
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_circuit_type_parameters_method_exists(self, basic_params):
        """Test that circuit type parameter setup method exists."""
        builder = CircuitBuilder(**basic_params)
        assert hasattr(builder, '_setup_circuit_type_parameters')
        assert callable(getattr(builder, '_setup_circuit_type_parameters'))
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_build_method_exists(self, basic_params):
        """Test that main build method exists."""
        builder = CircuitBuilder(**basic_params)
        assert hasattr(builder, 'build')
        assert callable(getattr(builder, 'build'))
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_syndrome_extraction_methods_exist(self, basic_params):
        """Test that syndrome extraction methods exist."""
        builder = CircuitBuilder(**basic_params)
        
        expected_methods = [
            '_build_syndrome_extraction_circuits',
            '_build_base_syndrome_extraction',
            '_add_detectors',
        ]
        
        for method_name in expected_methods:
            assert hasattr(builder, method_name), f"Method {method_name} should exist"
            assert callable(getattr(builder, method_name)), f"Method {method_name} should be callable"
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_initialization_methods_exist(self, basic_params):
        """Test that initialization methods exist."""
        builder = CircuitBuilder(**basic_params)
        
        expected_methods = [
            '_add_data_qubit_initialization',
            '_add_ancilla_initialization',
        ]
        
        for method_name in expected_methods:
            assert hasattr(builder, method_name), f"Method {method_name} should exist"
            assert callable(getattr(builder, method_name)), f"Method {method_name} should be callable"


class TestCircuitBuilderValidation:
    """Test validation and error handling in CircuitBuilder."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Imports not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_circuit_builder_cnot_schedule_validation(self):
        """Test CNOT schedule validation."""
        # Test valid schedules
        for schedule_name, schedule in CNOT_SCHEDULES.items():
            assert len(schedule) == 12, f"Schedule {schedule_name} should have 12 elements"
            assert all(1 <= x <= 12 for x in schedule), f"Schedule {schedule_name} should have values 1-12"


if __name__ == "__main__":
    pytest.main([__file__])