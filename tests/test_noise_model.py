"""
Tests for NoiseModel class.

This module tests the NoiseModel class functionality including
dictionary-like access, validation, and convenience methods.
"""

import pytest
from src.color_code_stim.noise_model import NoiseModel


class TestNoiseModelBasics:
    """Test basic NoiseModel functionality."""
    
    def test_init_default(self):
        """Test default initialization."""
        noise = NoiseModel()
        assert noise['bitflip'] == 0.0
        assert noise['depol'] == 0.0
        assert noise['reset'] == 0.0
        assert noise['meas'] == 0.0
        assert noise['cnot'] == 0.0
        assert noise['idle'] == 0.0
        assert noise['cult'] == 0.0  # Defaults to cnot when cnot is 0
        assert noise['idle_during_cnot'] == 0.0  # Defaults to idle when idle is 0
        assert noise['idle_during_meas'] == 0.0  # Defaults to idle when idle is 0
    
    def test_init_with_parameters(self):
        """Test initialization with specific parameters."""
        noise = NoiseModel(
            bitflip=0.001,
            depol=0.002,
            reset=0.003,
            meas=0.004,
            cnot=0.005,
            idle=0.006,
            cult=0.007
        )
        assert noise['bitflip'] == 0.001
        assert noise['depol'] == 0.002
        assert noise['reset'] == 0.003
        assert noise['meas'] == 0.004
        assert noise['cnot'] == 0.005
        assert noise['idle'] == 0.006
        assert noise['cult'] == 0.007
    
    def test_cult_defaults_to_cnot(self):
        """Test that cult parameter defaults to cnot value when not provided."""
        noise = NoiseModel(cnot=0.005)
        assert noise['cult'] == 0.005
        
        # When cult is explicitly None
        noise2 = NoiseModel(cnot=0.003, cult=None)
        assert noise2['cult'] == 0.003


class TestNoiseModelDictionaryAccess:
    """Test dictionary-like access functionality."""
    
    def test_getitem(self):
        """Test __getitem__ access."""
        noise = NoiseModel(depol=0.001, cnot=0.002)
        assert noise['depol'] == 0.001
        assert noise['cnot'] == 0.002
        assert noise['bitflip'] == 0.0
    
    def test_setitem(self):
        """Test __setitem__ assignment."""
        noise = NoiseModel()
        noise['depol'] = 0.005
        noise['cnot'] = 0.003
        assert noise['depol'] == 0.005
        assert noise['cnot'] == 0.003
    
    def test_contains(self):
        """Test __contains__ operator."""
        noise = NoiseModel()
        assert 'depol' in noise
        assert 'cnot' in noise
        assert 'invalid_key' not in noise
    
    def test_keys(self):
        """Test keys() method."""
        noise = NoiseModel()
        keys = list(noise.keys())
        expected_keys = ['bitflip', 'depol', 'reset', 'meas', 'cnot', 'idle', 'initial_data_qubit_depol', 'depol1_after_cnot', 'idle_during_cnot', 'idle_during_meas', 'cult']
        assert keys == expected_keys
    
    def test_values(self):
        """Test values() method."""
        noise = NoiseModel(bitflip=0.001, depol=0.002, cnot=0.003)
        values = list(noise.values())
        # cult should default to cnot value, idle_during_* should default to idle value
        expected_values = [0.001, 0.002, 0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003]
        assert values == expected_values
    
    def test_items(self):
        """Test items() method."""
        noise = NoiseModel(depol=0.001, cnot=0.002)
        items = dict(noise.items())
        assert items['depol'] == 0.001
        assert items['cnot'] == 0.002
        assert items['cult'] == 0.002  # Defaults to cnot


class TestNoiseModelErrors:
    """Test error handling."""
    
    def test_invalid_key_getitem(self):
        """Test error on invalid key access."""
        noise = NoiseModel()
        with pytest.raises(KeyError, match="Unknown noise parameter 'invalid'"):
            _ = noise['invalid']
    
    def test_invalid_key_setitem(self):
        """Test error on invalid key assignment."""
        noise = NoiseModel()
        with pytest.raises(KeyError, match="Unknown noise parameter 'invalid'"):
            noise['invalid'] = 0.001
    
    def test_negative_value_init(self):
        """Test error on negative values in init."""
        with pytest.raises(ValueError, match="must be non-negative"):
            NoiseModel(depol=-0.001)
    
    def test_negative_value_setitem(self):
        """Test error on negative values in setitem."""
        noise = NoiseModel()
        with pytest.raises(ValueError, match="must be non-negative"):
            noise['depol'] = -0.001


class TestNoiseModelUniformCircuitNoise:
    """Test uniform_circuit_noise class method."""
    
    def test_uniform_circuit_noise(self):
        """Test uniform circuit noise creation."""
        p = 0.001
        noise = NoiseModel.uniform_circuit_noise(p)
        
        # Should set circuit-level noise parameters
        assert noise['reset'] == p
        assert noise['meas'] == p
        assert noise['cnot'] == p
        assert noise['idle'] == p
        
        # Should not set bitflip or depol
        assert noise['bitflip'] == 0.0
        assert noise['depol'] == 0.0
        
        # Cult should default to cnot
        assert noise['cult'] == p
    
    def test_uniform_circuit_noise_zero(self):
        """Test uniform circuit noise with zero."""
        noise = NoiseModel.uniform_circuit_noise(0.0)
        assert noise.is_noiseless()


class TestNoiseModelUtilities:
    """Test utility methods."""
    
    def test_is_noiseless_true(self):
        """Test is_noiseless when all parameters are zero."""
        noise = NoiseModel()
        assert noise.is_noiseless()
    
    def test_is_noiseless_false(self):
        """Test is_noiseless when some parameters are non-zero."""
        noise = NoiseModel(depol=0.001)
        assert not noise.is_noiseless()
    
    def test_validate_success(self):
        """Test successful validation."""
        noise = NoiseModel(depol=0.001, cnot=0.002)
        noise.validate()  # Should not raise
    
    def test_str_noiseless(self):
        """Test string representation for noiseless model."""
        noise = NoiseModel()
        assert str(noise) == "NoiseModel(noiseless)"
    
    def test_str_with_noise(self):
        """Test string representation with noise."""
        noise = NoiseModel(depol=0.001, cnot=0.002)
        s = str(noise)
        assert "depol=0.001" in s
        assert "cnot=0.002" in s
        assert "NoiseModel(" in s
    
    def test_repr(self):
        """Test repr representation."""
        noise = NoiseModel(depol=0.001)
        repr_str = repr(noise)
        assert "NoiseModel(" in repr_str
        assert "depol=0.001" in repr_str


class TestNoiseModelIntegration:
    """Test integration scenarios."""
    
    def test_backward_compatibility_dict_like(self):
        """Test that NoiseModel can be used like a dict."""
        noise = NoiseModel(depol=0.001, cnot=0.002, meas=0.003)
        
        # Should work like the old noise_model dict
        p_depol = noise["depol"]
        p_cnot = noise["cnot"]
        p_meas = noise["meas"]
        
        assert p_depol == 0.001
        assert p_cnot == 0.002
        assert p_meas == 0.003
    
    def test_cult_parameter_handling(self):
        """Test special handling of cult parameter."""
        # Test explicit cult
        noise1 = NoiseModel(cnot=0.002, cult=0.005)
        assert noise1['cult'] == 0.005
        
        # Test cult defaulting to cnot
        noise2 = NoiseModel(cnot=0.003)
        assert noise2['cult'] == 0.003
        
        # Test cult=None defaulting to cnot
        noise3 = NoiseModel(cnot=0.004, cult=None)
        assert noise3['cult'] == 0.004


class TestInitialDataQubitDepol:
    """Test initial_data_qubit_depol parameter functionality."""
    
    def test_default_value(self):
        """Test that initial_data_qubit_depol defaults to 0.0."""
        noise = NoiseModel()
        assert noise['initial_data_qubit_depol'] == 0.0
    
    def test_explicit_value(self):
        """Test setting explicit value for initial_data_qubit_depol."""
        noise = NoiseModel(initial_data_qubit_depol=0.005)
        assert noise['initial_data_qubit_depol'] == 0.005
    
    def test_dictionary_access(self):
        """Test dictionary-like access and modification."""
        noise = NoiseModel()
        noise['initial_data_qubit_depol'] = 0.003
        assert noise['initial_data_qubit_depol'] == 0.003
    
    def test_in_uniform_circuit_noise(self):
        """Test that uniform_circuit_noise excludes initial_data_qubit_depol."""
        noise = NoiseModel.uniform_circuit_noise(0.001)
        assert noise['initial_data_qubit_depol'] == 0.0
        
        # Other circuit-level parameters should be set
        assert noise['cnot'] == 0.001
        assert noise['reset'] == 0.001
        assert noise['meas'] == 0.001
        assert noise['idle'] == 0.001
    
    def test_negative_value_validation(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            NoiseModel(initial_data_qubit_depol=-0.001)
    
    def test_negative_value_setitem(self):
        """Test that setting negative values raises ValueError."""
        noise = NoiseModel()
        with pytest.raises(ValueError, match="must be non-negative"):
            noise['initial_data_qubit_depol'] = -0.005
    
    def test_combination_with_other_parameters(self):
        """Test initial_data_qubit_depol in combination with other parameters."""
        noise = NoiseModel(
            cnot=0.002,
            meas=0.001,
            initial_data_qubit_depol=0.0005
        )
        assert noise['cnot'] == 0.002
        assert noise['meas'] == 0.001
        assert noise['initial_data_qubit_depol'] == 0.0005
        assert noise['depol'] == 0.0  # Should remain default


class TestIdleContextParameters:
    """Test idle_during_cnot and idle_during_meas parameter functionality."""
    
    def test_default_fallback_behavior(self):
        """Test that idle context parameters default to idle when None."""
        noise = NoiseModel(idle=0.003)
        assert noise['idle_during_cnot'] == 0.003  # Falls back to idle
        assert noise['idle_during_meas'] == 0.003  # Falls back to idle
    
    def test_explicit_override_behavior(self):
        """Test that explicit values override idle parameter."""
        noise = NoiseModel(
            idle=0.002,
            idle_during_cnot=0.005,
            idle_during_meas=0.001
        )
        assert noise['idle'] == 0.002
        assert noise['idle_during_cnot'] == 0.005  # Overrides idle
        assert noise['idle_during_meas'] == 0.001  # Overrides idle
    
    def test_zero_override_behavior(self):
        """Test that setting to 0 overrides idle parameter."""
        noise = NoiseModel(idle=0.002)
        noise['idle_during_cnot'] = 0.0  # Explicit 0 overrides idle
        noise['idle_during_meas'] = 0.0  # Explicit 0 overrides idle
        
        assert noise['idle'] == 0.002
        assert noise['idle_during_cnot'] == 0.0  # Overridden to 0
        assert noise['idle_during_meas'] == 0.0  # Overridden to 0
    
    def test_none_assignment_fallback(self):
        """Test that setting to None restores fallback behavior."""
        noise = NoiseModel(idle=0.003, idle_during_cnot=0.005)
        assert noise['idle_during_cnot'] == 0.005  # Explicit value
        
        noise['idle_during_cnot'] = None  # Reset to fallback
        assert noise['idle_during_cnot'] == 0.003  # Now falls back to idle
    
    def test_negative_value_validation(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            NoiseModel(idle_during_cnot=-0.001)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            NoiseModel(idle_during_meas=-0.001)
    
    def test_negative_setitem_validation(self):
        """Test that setting negative values raises ValueError."""
        noise = NoiseModel()
        
        with pytest.raises(ValueError, match="must be non-negative"):
            noise['idle_during_cnot'] = -0.005
            
        with pytest.raises(ValueError, match="must be non-negative"):
            noise['idle_during_meas'] = -0.005
    
    def test_in_uniform_circuit_noise(self):
        """Test that uniform_circuit_noise excludes idle context parameters."""
        noise = NoiseModel.uniform_circuit_noise(0.001)
        assert noise['idle_during_cnot'] == 0.001  # Falls back to idle
        assert noise['idle_during_meas'] == 0.001  # Falls back to idle
        
        # Internal storage should still be None
        assert noise._params['idle_during_cnot'] is None
        assert noise._params['idle_during_meas'] is None
    
    def test_combination_with_other_parameters(self):
        """Test idle context parameters in combination with other parameters."""
        noise = NoiseModel(
            cnot=0.002,
            idle=0.001,
            idle_during_cnot=0.0005,
            idle_during_meas=0.0015
        )
        assert noise['cnot'] == 0.002
        assert noise['idle'] == 0.001
        assert noise['idle_during_cnot'] == 0.0005
        assert noise['idle_during_meas'] == 0.0015