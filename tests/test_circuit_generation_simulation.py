"""
Test module for circuit generation and simulation across different parameters.

Tests ColorCode object creation and simulation for all circuit types with
various parameter combinations to ensure no errors are raised.
"""

import pytest
from color_code_stim import ColorCode
from color_code_stim.noise_model import NoiseModel


class TestCircuitGenerationSimulation:
    """Test circuit generation and simulation for all parameter combinations."""

    @pytest.fixture
    def noise_model(self):
        """Standard noise model for testing."""
        return NoiseModel.uniform_circuit_noise(1e-3)

    @pytest.mark.parametrize("temp_bdry_type", ["X", "Y", "Z"])
    @pytest.mark.parametrize("superdense_circuit", [True, False])
    @pytest.mark.parametrize("comparative_decoding", [True, False])
    def test_tri_circuit(self, noise_model, temp_bdry_type, superdense_circuit, comparative_decoding):
        """Test triangular circuit generation and simulation."""
        for d in [3, 5]:
            rounds = 3
            cc = ColorCode(
                d=d,
                rounds=rounds,
                circuit_type="tri",
                superdense_circuit=superdense_circuit,
                temp_bdry_type=temp_bdry_type,
                noise_model=noise_model,
                comparative_decoding=comparative_decoding,
            )
            # Should not raise any exceptions
            num_fails = cc.simulate(10)

    @pytest.mark.parametrize("temp_bdry_type", ["X", "Y", "Z"])
    @pytest.mark.parametrize("superdense_circuit", [True, False])
    @pytest.mark.parametrize("comparative_decoding", [True, False])
    def test_rec_circuit(self, noise_model, temp_bdry_type, superdense_circuit, comparative_decoding):
        """Test rectangular circuit generation and simulation."""
        configs = [(2, 4), (4, 6)]
        for d, d2 in configs:
            rounds = 3
            cc = ColorCode(
                d=d,
                d2=d2,
                rounds=rounds,
                circuit_type="rec",
                superdense_circuit=superdense_circuit,
                temp_bdry_type=temp_bdry_type,
                noise_model=noise_model,
                comparative_decoding=comparative_decoding,
            )
            # Should not raise any exceptions
            num_fails = cc.simulate(10)

    @pytest.mark.parametrize("superdense_circuit", [True, False])
    @pytest.mark.parametrize("comparative_decoding", [True, False])
    def test_rec_stability_circuit(self, noise_model, superdense_circuit, comparative_decoding):
        """Test rectangular stability circuit generation and simulation."""
        # Skip comparative_decoding=True for rec_stability as it's not implemented yet
        if comparative_decoding and True:  # circuit_type is "rec_stability" 
            pytest.skip("comparative_decoding=True not implemented for rec_stability circuit type")
        
        configs = [(4, 4), (6, 6)]
        for d, d2 in configs:
            rounds = 3
            cc = ColorCode(
                d=d,
                d2=d2,
                rounds=rounds,
                circuit_type="rec_stability",
                superdense_circuit=superdense_circuit,
                # temp_bdry_type defaults to 'r' for rec_stability
                noise_model=noise_model,
                comparative_decoding=comparative_decoding,
            )
            # Should not raise any exceptions
            num_fails = cc.simulate(10)

    @pytest.mark.parametrize("temp_bdry_type", ["X", "Y", "Z"])
    @pytest.mark.parametrize("superdense_circuit", [True, False])
    @pytest.mark.parametrize("comparative_decoding", [True, False])
    def test_growing_circuit(self, noise_model, temp_bdry_type, superdense_circuit, comparative_decoding):
        """Test growing circuit generation and simulation."""
        configs = [(3, 5), (5, 7)]
        for d, d2 in configs:
            rounds = 3
            cc = ColorCode(
                d=d,
                d2=d2,
                rounds=rounds,
                circuit_type="growing",
                superdense_circuit=superdense_circuit,
                temp_bdry_type=temp_bdry_type,
                noise_model=noise_model,
                comparative_decoding=comparative_decoding,
            )
            # Should not raise any exceptions
            num_fails = cc.simulate(10)

    @pytest.mark.parametrize("superdense_circuit", [True, False])
    @pytest.mark.parametrize("comparative_decoding", [True, False])
    def test_cult_growing_circuit(self, noise_model, superdense_circuit, comparative_decoding):
        """Test cultivation+growing circuit generation and simulation."""
        configs = [(3, 5), (5, 7)]
        for d, d2 in configs:
            rounds = 3
            cc = ColorCode(
                d=d,
                d2=d2,
                rounds=rounds,
                circuit_type="cult+growing",
                superdense_circuit=superdense_circuit,
                # temp_bdry_type defaults to 'Y' for cult+growing
                noise_model=noise_model,
                comparative_decoding=comparative_decoding,
            )
            # `simulate` is currently not supported for cult+growing circuit type
            # so just check whether DEM can be generated without error
            cc.circuit.detector_error_model()
