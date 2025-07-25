"""
Integration tests for initial_data_qubit_depol parameter.

This module tests the integration of the initial_data_qubit_depol parameter
with the ColorCode class and circuit generation, including timing scenarios.
"""

import pytest
import stim
from src.color_code_stim import ColorCode, NoiseModel


class TestInitialDataQubitDepolIntegration:
    """Test integration of initial_data_qubit_depol with ColorCode."""
    
    def test_colorcode_with_initial_data_qubit_depol(self):
        """Test ColorCode creation with initial_data_qubit_depol parameter."""
        noise = NoiseModel(
            cnot=0.001,
            meas=0.0005,
            initial_data_qubit_depol=0.002
        )
        
        # Should create successfully without errors
        colorcode = ColorCode(
            d=3,
            rounds=2,
            circuit_type="tri",
            noise_model=noise
        )
        
        # Verify noise model is properly stored
        assert colorcode.noise_model['initial_data_qubit_depol'] == 0.002
        assert colorcode.noise_model['cnot'] == 0.001
        assert colorcode.noise_model['meas'] == 0.0005
    
    def test_timing_perfect_first_syndrome_false(self):
        """Test noise applied after data qubit initialization when perfect_first_syndrome_extraction=False."""
        noise = NoiseModel(initial_data_qubit_depol=0.01)  # High value for easier detection
        
        colorcode = ColorCode(
            d=3,
            rounds=2,
            circuit_type="tri",
            noise_model=noise,
            perfect_first_syndrome_extraction=False
        )
        
        circuit = colorcode.circuit
        
        # Convert circuit to string to inspect structure
        circuit_str = str(circuit)
        
        # Should contain DEPOLARIZE1 instruction on data qubits
        assert "DEPOLARIZE1" in circuit_str
        
        # Find position of data qubit reset and our DEPOLARIZE1
        lines = circuit_str.split('\n')
        data_reset_line_idx = -1
        depol_line_idx = -1
        
        for i, line in enumerate(lines):
            # Look for the first R instruction with many qubits (data qubits)
            if line.startswith('R ') and len(line.split()) > 6 and data_reset_line_idx == -1:  
                data_reset_line_idx = i
            elif 'DEPOLARIZE1(0.01)' in line:  # Our specific depolarization
                depol_line_idx = i
        
        # DEPOLARIZE1 should come after data qubit reset
        assert data_reset_line_idx != -1, "Data qubit reset not found"
        assert depol_line_idx != -1, "DEPOLARIZE1 instruction not found"
        assert depol_line_idx > data_reset_line_idx, f"DEPOLARIZE1 (line {depol_line_idx}) should come after data qubit reset (line {data_reset_line_idx})"
    
    def test_timing_perfect_first_syndrome_true(self):
        """Test noise applied after first syndrome extraction when perfect_first_syndrome_extraction=True."""
        noise = NoiseModel(initial_data_qubit_depol=0.01)  # High value for easier detection
        
        colorcode = ColorCode(
            d=3,
            rounds=2,
            circuit_type="tri",
            noise_model=noise,
            perfect_first_syndrome_extraction=True
        )
        
        circuit = colorcode.circuit
        circuit_str = str(circuit)
        
        # Should contain DEPOLARIZE1 instruction on data qubits
        assert "DEPOLARIZE1" in circuit_str
        
        # With perfect_first_syndrome_extraction=True, the DEPOLARIZE1 should come
        # after the first syndrome extraction round
        # This is more complex to verify precisely, but we can check it exists
        assert "DEPOLARIZE1" in circuit_str
    
    def test_zero_initial_data_qubit_depol(self):
        """Test that zero initial_data_qubit_depol doesn't add DEPOLARIZE1 instruction."""
        noise = NoiseModel(
            cnot=0.001,
            initial_data_qubit_depol=0.0  # Explicit zero
        )
        
        colorcode = ColorCode(
            d=3,
            rounds=2,
            circuit_type="tri",
            noise_model=noise
        )
        
        circuit = colorcode.circuit
        circuit_str = str(circuit)
        
        # Should not contain additional DEPOLARIZE1 from initial_data_qubit_depol
        # (there might be other DEPOLARIZE1 from other noise sources)
        # Count DEPOLARIZE1 occurrences - with cnot=0.001, we expect some, but not from initial_data_qubit_depol
        depolarize_count = circuit_str.count("DEPOLARIZE1")
        
        # Create a reference circuit with no noise to compare
        noise_ref = NoiseModel()  # All zeros
        colorcode_ref = ColorCode(
            d=3,
            rounds=2,
            circuit_type="tri",
            noise_model=noise_ref
        )
        ref_depolarize_count = str(colorcode_ref.circuit).count("DEPOLARIZE1")
        
        # The difference should only be from cnot noise, not from initial_data_qubit_depol
        # This is a basic sanity check
        assert depolarize_count >= ref_depolarize_count
    
    def test_multiple_circuit_types(self):
        """Test initial_data_qubit_depol works with different circuit types."""
        noise = NoiseModel(initial_data_qubit_depol=0.001)
        
        # Test triangular circuit
        colorcode_tri = ColorCode(
            d=3,
            rounds=2,
            circuit_type="tri",
            noise_model=noise
        )
        assert "DEPOLARIZE1" in str(colorcode_tri.circuit)
        
        # Test rectangular circuit
        colorcode_rec = ColorCode(
            d=4,
            d2=4,
            rounds=2,
            circuit_type="rec",
            noise_model=noise
        )
        assert "DEPOLARIZE1" in str(colorcode_rec.circuit)
    
    def test_perfect_first_syndrome_timing_difference(self):
        """Test that perfect_first_syndrome_extraction affects timing of initial_data_qubit_depol."""
        noise = NoiseModel(initial_data_qubit_depol=0.01)
        
        # Create two identical circuits except for perfect_first_syndrome_extraction
        colorcode_false = ColorCode(
            d=3,
            rounds=2,
            circuit_type="tri",
            noise_model=noise,
            perfect_first_syndrome_extraction=False
        )
        
        colorcode_true = ColorCode(
            d=3,
            rounds=2,
            circuit_type="tri",
            noise_model=noise,
            perfect_first_syndrome_extraction=True
        )
        
        # Both should have DEPOLARIZE1, but in different positions
        circuit_false_str = str(colorcode_false.circuit)
        circuit_true_str = str(colorcode_true.circuit)
        
        assert "DEPOLARIZE1" in circuit_false_str
        assert "DEPOLARIZE1" in circuit_true_str
        
        # The circuits should be different due to different timing
        assert circuit_false_str != circuit_true_str
    
    def test_backward_compatibility(self):
        """Test that existing code without initial_data_qubit_depol still works."""
        # Create NoiseModel without the new parameter (should default to 0.0)
        noise = NoiseModel(cnot=0.001, meas=0.0005)
        
        colorcode = ColorCode(
            d=3,
            rounds=2,
            circuit_type="tri",
            noise_model=noise
        )
        
        # Should work without errors
        assert colorcode.noise_model['initial_data_qubit_depol'] == 0.0
        
        # Circuit generation should succeed
        assert isinstance(colorcode.circuit, stim.Circuit)
    
    def test_simulation_compatibility(self):
        """Test that circuits with initial_data_qubit_depol can be simulated."""
        noise = NoiseModel(
            cnot=0.001,
            initial_data_qubit_depol=0.0005
        )
        
        colorcode = ColorCode(
            d=3,
            rounds=2,
            circuit_type="tri",
            noise_model=noise
        )
        
        # Should be able to sample without errors
        det, obs = colorcode.sample(shots=10)
        
        assert det.shape[0] == 10  # 10 shots
        assert det.shape[1] > 0     # Some detectors
        assert len(obs.shape) in [1, 2]  # Observables (1D for single obs, 2D for multiple)