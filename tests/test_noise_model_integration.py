"""
Integration tests for NoiseModel with ColorCode.

This module tests that NoiseModel integrates properly with ColorCode
and maintains backward compatibility.
"""

import pytest
from src.color_code_stim import ColorCode
from src.color_code_stim.noise_model import NoiseModel


class TestNoiseModelColorCodeIntegration:
    """Test NoiseModel integration with ColorCode."""
    
    def test_colorcode_with_noise_model(self):
        """Test ColorCode constructor with NoiseModel."""
        noise = NoiseModel(depol=0.001, cnot=0.002, meas=0.001)
        
        cc = ColorCode(
            d=3,
            rounds=1,
            circuit_type="tri",
            noise_model=noise
        )
        
        # Verify noise model is stored correctly
        assert cc.noise_model['depol'] == 0.001
        assert cc.noise_model['cnot'] == 0.002
        assert cc.noise_model['meas'] == 0.001
        assert cc.noise_model['bitflip'] == 0.0
    
    def test_colorcode_with_uniform_circuit_noise(self):
        """Test ColorCode with uniform circuit noise model."""
        noise = NoiseModel.uniform_circuit_noise(0.001)
        
        cc = ColorCode(
            d=3,
            rounds=1,
            circuit_type="tri",
            noise_model=noise
        )
        
        # Verify uniform noise is applied correctly
        assert cc.noise_model['reset'] == 0.001
        assert cc.noise_model['meas'] == 0.001
        assert cc.noise_model['cnot'] == 0.001
        assert cc.noise_model['idle'] == 0.001
        assert cc.noise_model['bitflip'] == 0.0
        assert cc.noise_model['depol'] == 0.0
    
    def test_backward_compatibility_individual_params(self):
        """Test backward compatibility with individual noise parameters."""
        # Old way - should still work
        cc1 = ColorCode(
            d=3,
            rounds=1,
            circuit_type="tri",
            p_depol=0.001,
            p_cnot=0.002,
            p_meas=0.001
        )
        
        # Verify NoiseModel was created internally
        assert isinstance(cc1.noise_model, NoiseModel)
        assert cc1.noise_model['depol'] == 0.001
        assert cc1.noise_model['cnot'] == 0.002
        assert cc1.noise_model['meas'] == 0.001
    
    def test_backward_compatibility_p_circuit(self):
        """Test backward compatibility with p_circuit parameter."""
        # Old way with p_circuit
        cc = ColorCode(
            d=3,
            rounds=1,
            circuit_type="tri",
            p_circuit=0.001
        )
        
        # Verify p_circuit logic was applied correctly
        assert cc.noise_model['reset'] == 0.001
        assert cc.noise_model['meas'] == 0.001
        assert cc.noise_model['cnot'] == 0.001
        assert cc.noise_model['idle'] == 0.001
    
    def test_noise_model_takes_precedence(self):
        """Test that noise_model parameter takes precedence over individual params."""
        noise = NoiseModel(depol=0.005, cnot=0.006)
        
        cc = ColorCode(
            d=3,
            rounds=1,
            circuit_type="tri",
            noise_model=noise,
            # These should be ignored
            p_depol=0.001,
            p_cnot=0.002
        )
        
        # Should use noise_model values, not individual params
        assert cc.noise_model['depol'] == 0.005
        assert cc.noise_model['cnot'] == 0.006
    
    def test_cult_growing_with_noise_model(self):
        """Test cult+growing circuit type with NoiseModel."""
        noise = NoiseModel.uniform_circuit_noise(0.001)
        
        cc = ColorCode(
            d=3,
            d2=5,
            rounds=1,
            circuit_type="cult+growing",
            noise_model=noise,
            p_circuit=0.001  # Still required for validation
        )
        
        # Should work without error
        assert cc.circuit_type == "cult+growing"
        assert cc.noise_model['cnot'] == 0.001
        assert cc.noise_model['cult'] == 0.001  # Defaults to cnot
    
    def test_cult_growing_validation_with_noise_model(self):
        """Test cult+growing validation still works with NoiseModel."""
        noise = NoiseModel(bitflip=0.001)  # Should fail validation
        
        with pytest.raises(ValueError, match="p_bitflip must be 0"):
            ColorCode(
                d=3,
                d2=5,
                rounds=1,
                circuit_type="cult+growing",
                noise_model=noise,
                p_circuit=0.001,
                p_bitflip=0.001  # This should trigger validation
            )
    
    def test_circuit_generation_with_noise_model(self):
        """Test that circuits are generated correctly with NoiseModel."""
        noise = NoiseModel(depol=0.001, cnot=0.002)
        
        cc = ColorCode(
            d=3,
            rounds=1,
            circuit_type="tri",
            noise_model=noise
        )
        
        # Should generate circuit successfully
        assert cc.circuit is not None
        assert len(cc.circuit) > 0
        
        # Basic smoke test - should be able to sample
        det, obs = cc.sample(shots=10)
        assert det.shape[0] == 10
        assert obs.shape[0] == 10


class TestNoiseModelEquivalence:
    """Test equivalence between old and new approaches."""
    
    def test_equivalent_results_individual_vs_noise_model(self):
        """Test that individual params and NoiseModel give equivalent results."""
        # Create with individual parameters
        cc1 = ColorCode(
            d=3,
            rounds=1,
            circuit_type="tri",
            p_depol=0.001,
            p_cnot=0.002,
            p_meas=0.001,
            p_reset=0.0005
        )
        
        # Create with equivalent NoiseModel
        noise = NoiseModel(
            depol=0.001,
            cnot=0.002,
            meas=0.001,
            reset=0.0005
        )
        cc2 = ColorCode(
            d=3,
            rounds=1,
            circuit_type="tri",
            noise_model=noise
        )
        
        # Should have equivalent noise models
        for key in ['depol', 'cnot', 'meas', 'reset', 'bitflip', 'idle']:
            assert cc1.noise_model[key] == cc2.noise_model[key]
    
    def test_equivalent_results_p_circuit_vs_uniform(self):
        """Test equivalence between p_circuit and uniform_circuit_noise."""
        # Old way with p_circuit
        cc1 = ColorCode(
            d=3,
            rounds=1,
            circuit_type="tri",
            p_circuit=0.001
        )
        
        # New way with uniform_circuit_noise
        noise = NoiseModel.uniform_circuit_noise(0.001)
        cc2 = ColorCode(
            d=3,
            rounds=1,
            circuit_type="tri",
            noise_model=noise
        )
        
        # Should have equivalent noise models
        for key in ['reset', 'meas', 'cnot', 'idle', 'bitflip', 'depol']:
            assert cc1.noise_model[key] == cc2.noise_model[key]