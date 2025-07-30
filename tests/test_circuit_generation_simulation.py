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

    @pytest.mark.parametrize("temp_bdry_type", ["X", "Y", "Z"])
    @pytest.mark.parametrize("comparative_decoding", [True, False])
    @pytest.mark.parametrize("set_all_faces_segmented", [True, False])
    def test_sdqc_memory_circuit(self, noise_model, temp_bdry_type, comparative_decoding, set_all_faces_segmented):
        """Test SDQC memory circuit generation and simulation."""
        # SDQC circuits require superdense_circuit=True
        for d in [3, 5, 7]:
            rounds = 3
            # Create noise model with shuttling error rates for SDQC
            sdqc_noise = NoiseModel(
                cnot=1e-3,
                meas=1e-3,
                reset=1e-3,
                idle=1e-3,
                shuttling_seg_init=1e-4,
                shuttling_non_seg_init=1e-4,
                shuttling_seg_final=1e-4,
                shuttling_non_seg_final=1e-4,
                depol1_on_anc_before_cnot=1e-5,
            )
            cc = ColorCode(
                d=d,
                rounds=rounds,
                circuit_type="sdqc_memory",
                superdense_circuit=True,  # Required for SDQC
                temp_bdry_type=temp_bdry_type,
                noise_model=sdqc_noise,
                comparative_decoding=comparative_decoding,
                set_all_faces_segmented=set_all_faces_segmented,
            )
            # Should not raise any exceptions
            num_fails = cc.simulate(10)

    def test_sdqc_memory_requires_superdense(self, noise_model):
        """Test that SDQC memory circuit requires superdense_circuit=True."""
        with pytest.raises(ValueError, match="sdqc_memory circuit type requires superdense_circuit=True"):
            ColorCode(
                d=3,
                rounds=3,
                circuit_type="sdqc_memory",
                superdense_circuit=False,  # Should raise error
                noise_model=noise_model,
            )

    def test_sdqc_segmentation_rules(self, noise_model):
        """Test SDQC face segmentation rules for different distances."""
        from color_code_stim.config import SDQC_SEGMENTATION_RULES
        
        for d in [3, 5, 7, 9, 11, 13]:
            cc = ColorCode(
                d=d,
                rounds=3,
                circuit_type="sdqc_memory",
                superdense_circuit=True,
                noise_model=noise_model,
            )
            # Check that segmented faces match configuration
            expected_faces = SDQC_SEGMENTATION_RULES[d]
            assert cc.segmented_faces == expected_faces

    def test_sdqc_unimplemented_distance(self, noise_model):
        """Test that SDQC circuits raise NotImplementedError for d>=15."""
        with pytest.raises(NotImplementedError, match="SDQC circuit with distance d=15 is not implemented"):
            ColorCode(
                d=15,
                rounds=3,
                circuit_type="sdqc_memory",
                superdense_circuit=True,
                noise_model=noise_model,
                set_all_faces_segmented=False,  # Don't override the distance check
            )

    def test_sdqc_set_all_faces_segmented_override(self, noise_model):
        """Test that set_all_faces_segmented=True allows d>=15."""
        # This should not raise an error
        cc = ColorCode(
            d=15,
            rounds=3,
            circuit_type="sdqc_memory",
            superdense_circuit=True,
            noise_model=noise_model,
            set_all_faces_segmented=True,  # Override distance check
        )
        # Should succeed without error
        assert cc.circuit_type == "sdqc_memory"
