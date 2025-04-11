import unittest  # For more structured testing (optional)
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, spmatrix

# ==========================================================
# 1. Mock Dependencies (Minimal required for the test)
# ==========================================================

COLOR_LABEL = Literal["r", "g", "b"]


class stim:
    # Dummy class to satisfy type hints and potential attribute access
    class DetectorErrorModel:
        def __init__(self):
            self.num_errors = 0
            self.num_detectors = 0

        def get_detector_coordinates(self):
            # Needs to return a dict if decompose_org_dem is called fully
            return {}

        def __len__(self):
            return 0  # Placeholder

        def __add__(self, other):
            return self  # Placeholder

        def __iadd__(self, other):
            return self  # Placeholder


# Dummy function assumed by the original __init__ structure
# We won't rely on its output for mapping, but need it for __init__ to run
def dem_to_parity_check(dem):
    # Return dummy values consistent with expected types
    print(f"Warning: Using dummy dem_to_parity_check for {type(dem)}")
    # Return shapes that might be expected, though values aren't used in this specific test focus
    num_dets = 10
    num_errs = 5
    return csc_matrix((num_dets, num_errs), dtype=bool), None, np.random.rand(num_errs)


class _DemSymbolic:
    # Dummy class to satisfy type hints and __init__ calls
    def __init__(self, probs_list, dets_list, obss_list, org_dem_dets, num_org_errors):
        # Store minimal info needed if __init__ uses it
        self.num_decomp_errors = len(probs_list)
        # In a real scenario, this matrix would be built more complexly
        # For testing map_errors_to_org_dem, we override this in TestableDemDecomp
        self.error_map_matrix = csr_matrix(
            (self.num_decomp_errors, num_org_errors), dtype=bool
        )

    def to_dem(self, org_prob, sort=False):
        # Return dummy stim.DEM and indices needed by __init__
        indices = np.arange(self.num_decomp_errors)
        if sort:
            # Simulate potential sorting if needed by __init__ logic
            pass  # No actual sorting needed for this test's focus
        return stim.DetectorErrorModel(), indices  # Return dummy DEM and indices


# ==========================================================
# 2. Paste the DemDecomp Class Definition Here
#    (Ensure it's the version with precomputation in __init__)
# ==========================================================


class DemDecomp:
    """
    Decomposition of a detector error model (DEM) into two stages for concatenated color
    code decoding.
    [... Full class docstring ...]

    Attributes
    ----------
    # ... (other attributes) ...
    org_prob : 1D numpy array (float)
        Error probabilities of the original DEM.
    error_map_matrices : 2-tuple of csr_matrix (bool)
        Matrices mapping errors in decomposed DEMs to errors in original DEM.
    _best_org_error_map : Tuple[np.ndarray, np.ndarray]
        Precomputed mapping from decomposed error index to best original error index
        for stage 1 and stage 2. Index is original error index, value is -1 if no mapping.
    """

    color: COLOR_LABEL
    dems: Tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]
    dems_symbolic: Tuple[_DemSymbolic, _DemSymbolic]
    Hs: Tuple[csc_matrix, csc_matrix]
    probs: Tuple[np.ndarray, np.ndarray]
    org_dem: stim.DetectorErrorModel
    org_prob: np.ndarray
    error_map_matrices: Tuple[csr_matrix, csr_matrix]
    _best_org_error_map: Tuple[np.ndarray, np.ndarray]

    # NOTE: This is a simplified __init__ for testing purposes.
    # It directly accepts org_prob and error_map_matrices,
    # bypassing the complex decomposition logic which is not the focus here.
    def __init__(
        self,
        *,
        org_prob: np.ndarray,
        error_map_matrices: Tuple[csr_matrix, csr_matrix],
        # Add dummy params to match original signature if needed elsewhere,
        # but we won't use them here.
        org_dem: stim.DetectorErrorModel = stim.DetectorErrorModel(),
        color: COLOR_LABEL = "r",
    ):
        """Simplified __init__ for testing map_errors_to_org_dem."""
        print("Using simplified __init__ for testing.")
        self.org_prob = org_prob
        self.error_map_matrices = error_map_matrices
        # Add dummy attributes needed by the class if any other methods were called
        self.org_dem = org_dem
        self.color = color
        self.dems = (stim.DetectorErrorModel(), stim.DetectorErrorModel())
        self.dems_symbolic = (None, None)  # Not needed for this test
        self.Hs = (csc_matrix((1, 1)), csc_matrix((1, 1)))
        self.probs = (np.array([]), np.array([]))

        # --- Precompute best original error index mapping ---
        best_maps = []
        org_prob_flat = np.asarray(self.org_prob).flatten()
        for stage_idx in range(2):  # For stage 1 (index 0) and stage 2 (index 1)
            error_map_matrix = self.error_map_matrices[stage_idx]
            num_decomp_errors, num_org_errors = error_map_matrix.shape

            if org_prob_flat.shape[0] != num_org_errors:
                raise ValueError(
                    f"Shape mismatch during init: org_prob length {org_prob_flat.shape[0]} != "
                    f"error_map_matrix cols {num_org_errors} for stage {stage_idx + 1}"
                )

            best_org_error_indices = np.full(num_decomp_errors, -1, dtype=int)
            for i in range(num_decomp_errors):
                start = error_map_matrix.indptr[i]
                end = error_map_matrix.indptr[i + 1]
                if start == end:
                    continue
                candidate_org_indices_j = error_map_matrix.indices[start:end]
                if candidate_org_indices_j.size > 0:
                    probs_for_candidates = org_prob_flat[candidate_org_indices_j]
                    max_prob_local_idx = np.argmax(probs_for_candidates)
                    best_j = candidate_org_indices_j[max_prob_local_idx]
                    best_org_error_indices[i] = best_j
            best_maps.append(best_org_error_indices)
        self._best_org_error_map = tuple(best_maps)
        # --- End of precomputation ---

    # We need the original decompose_org_dem method signature, but it won't
    # be called by the simplified init. Add a dummy version if needed.
    def decompose_org_dem(self, *args, **kwargs):
        print("Warning: Dummy decompose_org_dem called")
        # Needs to return two _DemSymbolic objects based on its logic
        # Return dummy objects consistent with expected types if absolutely necessary
        dummy_sym = _DemSymbolic([], [], [], [], 0)
        return dummy_sym, dummy_sym

    def map_errors_to_org_dem(
        self, errors: List[bool | int] | np.ndarray | spmatrix, *, stage: int
    ) -> np.ndarray:
        """
        Map errors from the decomposed DEM back to the original DEM format using
        precomputed mapping.
        [... Full method docstring ...]
        """
        if stage not in [1, 2]:
            raise ValueError("stage must be 1 or 2")

        # --- 1. Retrieve precomputed mapping ---
        best_org_error_indices = self._best_org_error_map[stage - 1]
        num_decomp_errors = best_org_error_indices.shape[0]
        num_org_errors = self.org_prob.shape[0]

        # --- 2. Process the input 'errors' array ---
        is_sparse_input = isinstance(errors, spmatrix)
        input_is_1d_semantic = False  # Flag to track if input represents a 1D vector

        if is_sparse_input:
            # Check sparse shape before converting to dense
            if errors.shape[0] == 1 or errors.shape[1] == 1:
                input_is_1d_semantic = True
            errors_np = errors.toarray().astype(np.uint8)
        elif not isinstance(errors, np.ndarray):
            # Includes list case
            errors_np = np.asarray(errors, dtype=np.uint8)
            if errors_np.ndim == 1:
                input_is_1d_semantic = True
        else:
            errors_np = errors.astype(np.uint8, copy=False)
            if errors_np.ndim == 1:
                input_is_1d_semantic = True

        # Validate shape consistency
        if errors_np.shape[-1] != num_decomp_errors:
            raise ValueError(
                f"Last dimension of input errors ({errors_np.shape[-1]}) does not match "
                f"number of decomposed errors ({num_decomp_errors}) for stage {stage}"
            )

        # --- 3. Map active decomposed errors to original errors using advanced indexing ---
        # Calculate output shape based on the (potentially 2D) errors_np shape
        output_shape = errors_np.shape[:-1] + (num_org_errors,)
        errors_org = np.zeros(output_shape, dtype=bool)

        active_indices = np.nonzero(errors_np)
        if active_indices[0].size == 0:
            # Return zeros with appropriate shape
            if input_is_1d_semantic and errors_org.ndim > 1:
                return np.zeros(num_org_errors, dtype=bool)  # Return 1D zeros
            else:
                return errors_org  # Return potentially multi-dim zeros

        decomp_error_idx_i = active_indices[-1]
        target_org_idx_j = best_org_error_indices[decomp_error_idx_i]
        valid_mapping_mask = target_org_idx_j != -1

        # Prepare final index tuple based on errors_np dimensions
        if np.all(valid_mapping_mask):
            if target_org_idx_j.size > 0:
                final_index_tuple = active_indices[:-1] + (target_org_idx_j,)
                errors_org[final_index_tuple] = True
        elif np.any(valid_mapping_mask):
            filtered_target_org_idx_j = target_org_idx_j[valid_mapping_mask]
            filtered_active_indices = tuple(
                idx_arr[valid_mapping_mask] for idx_arr in active_indices
            )
            final_index_tuple = filtered_active_indices[:-1] + (
                filtered_target_org_idx_j,
            )
            errors_org[final_index_tuple] = True

        # --- 4. Adjust final shape ---
        # If the input was semantically 1D and output is 2D (e.g., (1, N)), reshape to 1D
        if input_is_1d_semantic and errors_org.ndim > 1:
            # Specifically handle the case where output is (1, N)
            if errors_org.shape[0] == 1:
                errors_org = errors_org.reshape(-1)  # Reshape (1, N) to (N,)
            # Can add handling for (N, 1) output if that's possible and needs reshaping

        return errors_org


# ==========================================================
# 3. Test Setup and Execution
# ==========================================================


class TestDemDecompMapping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up mock data and instantiate the class once for all tests."""
        print("\n--- Setting up test data ---")
        # Original error probabilities
        cls.org_probabilities = np.array(
            [0.1, 0.5, 0.2, 0.8, 0.3]
        )  # num_org_errors = 5

        # Error map matrix for Stage 1 (shape: 3x5)
        # Decomp Err 0 -> Org Err 0 (0.1), 1 (0.5) => Best: 1
        # Decomp Err 1 -> Org Err 2 (0.2), 4 (0.3) => Best: 4
        # Decomp Err 2 -> Org Err 3 (0.8)          => Best: 3
        map1_data = np.array([1, 1, 1, 1, 1], dtype=bool)
        map1_indices = np.array([0, 1, 2, 4, 3])
        map1_indptr = np.array([0, 2, 4, 5])
        cls.error_map1 = csr_matrix(
            (map1_data, map1_indices, map1_indptr), shape=(3, 5)
        )
        cls.expected_map1 = np.array([1, 4, 3])

        # Error map matrix for Stage 2 (shape: 4x5)
        # Decomp Err 0 -> Org Err 1 (0.5)               => Best: 1
        # Decomp Err 1 -> Org Err 0 (0.1), 3 (0.8)      => Best: 3
        # Decomp Err 2 -> Org Err 1 (0.5), 2 (0.2), 4 (0.3) => Best: 1
        # Decomp Err 3 -> Org Err 3 (0.8)               => Best: 3
        map2_data = np.array([1, 1, 1, 1, 1, 1, 1], dtype=bool)
        map2_indices = np.array([1, 0, 3, 1, 2, 4, 3])
        map2_indptr = np.array([0, 1, 3, 6, 7])
        cls.error_map2 = csr_matrix(
            (map2_data, map2_indices, map2_indptr), shape=(4, 5)
        )
        cls.expected_map2 = np.array([1, 3, 1, 3])

        cls.error_maps = (cls.error_map1, cls.error_map2)

        print("Instantiating DemDecomp with mock data...")
        # Use the simplified __init__ for testing
        cls.decomp_instance = DemDecomp(
            org_prob=cls.org_probabilities, error_map_matrices=cls.error_maps
        )
        print("Instantiation complete.")

    def test_01_precomputed_maps(self):
        """Verify the precomputed best original error maps."""
        print("\n--- Test: Precomputed Maps ---")
        np.testing.assert_array_equal(
            self.decomp_instance._best_org_error_map[0],
            self.expected_map1,
            "Precomputed map stage 1 mismatch",
        )
        print("Stage 1 map OK")
        np.testing.assert_array_equal(
            self.decomp_instance._best_org_error_map[1],
            self.expected_map2,
            "Precomputed map stage 2 mismatch",
        )
        print("Stage 2 map OK")

    def test_02_map_stage1_numpy(self):
        """Test mapping for stage 1 with NumPy array input."""
        print("\n--- Test: Stage 1, NumPy Input ---")
        errors_stage1 = np.array([True, False, True])  # Decomp errors 0, 2 active
        # Expected: Maps to best org errs for 0 and 2 => 1 and 3
        expected_org1 = np.array([False, True, False, True, False])
        actual_org1 = self.decomp_instance.map_errors_to_org_dem(errors_stage1, stage=1)
        print(f"Input: {errors_stage1}, Output: {actual_org1}")
        np.testing.assert_array_equal(actual_org1, expected_org1)

    def test_03_map_stage2_list(self):
        """Test mapping for stage 2 with list input."""
        print("\n--- Test: Stage 2, List Input ---")
        errors_stage2 = [False, True, True, False]  # Decomp errors 1, 2 active
        # Expected: Maps to best org errs for 1 and 2 => 3 and 1
        expected_org2 = np.array([False, True, False, True, False])
        actual_org2 = self.decomp_instance.map_errors_to_org_dem(errors_stage2, stage=2)
        print(f"Input: {errors_stage2}, Output: {actual_org2}")
        np.testing.assert_array_equal(actual_org2, expected_org2)

    def test_04_map_stage2_multi_dim(self):
        """Test mapping for stage 2 with multi-dimensional NumPy input."""
        print("\n--- Test: Stage 2, Multi-Dim Input ---")
        errors_stage2_multi = np.array(
            [
                [False, True, True, False],  # Decomp 1, 2 active -> Org 3, 1
                [True, False, False, True],  # Decomp 0, 3 active -> Org 1, 3
            ]
        )
        # Expected:
        # Row 0 -> Org 1, 3 => [F, T, F, T, F]
        # Row 1 -> Org 1, 3 => [F, T, F, T, F]
        expected_org2_multi = np.array(
            [[False, True, False, True, False], [False, True, False, True, False]]
        )
        actual_org2_multi = self.decomp_instance.map_errors_to_org_dem(
            errors_stage2_multi, stage=2
        )
        print(f"Input:\n{errors_stage2_multi}\nOutput:\n{actual_org2_multi}")
        np.testing.assert_array_equal(actual_org2_multi, expected_org2_multi)

    def test_05_map_stage1_sparse(self):
        """Test mapping for stage 1 with sparse matrix input."""
        print("\n--- Test: Stage 1, Sparse Input ---")
        errors_stage1 = np.array([True, False, True])  # Decomp errors 0, 2 active
        errors_stage1_sparse = csr_matrix(errors_stage1)
        # Expected: Maps to best org errs for 0 and 2 => 1 and 3
        expected_org1 = np.array([False, True, False, True, False])
        actual_org1_sparse = self.decomp_instance.map_errors_to_org_dem(
            errors_stage1_sparse, stage=1
        )
        print(
            f"Input (dense): {errors_stage1_sparse.toarray()}, Output: {actual_org1_sparse}"
        )
        np.testing.assert_array_equal(actual_org1_sparse, expected_org1)

    def test_06_map_empty_errors(self):
        """Test mapping when input errors array is all False/zeros."""
        print("\n--- Test: Empty Errors Input ---")
        errors_empty = np.zeros(3, dtype=bool)  # For stage 1 (size 3)
        expected_empty = np.zeros(5, dtype=bool)  # Expect all False org errors
        actual_empty = self.decomp_instance.map_errors_to_org_dem(errors_empty, stage=1)
        print(f"Input: {errors_empty}, Output: {actual_empty}")
        np.testing.assert_array_equal(actual_empty, expected_empty)

    def test_07_map_no_valid_mapping(self):
        """Test case where an active error has no valid map (best_org_idx = -1)."""
        print("\n--- Test: No Valid Mapping ---")
        # Create a temporary map where decomp error 1 maps to nothing
        temp_map1_data = np.array([1, 1, 1], dtype=bool)
        temp_map1_indices = np.array([0, 1, 3])  # Error 1 maps to nothing now
        temp_map1_indptr = np.array([0, 2, 2, 3])  # Indptr reflects empty row 1
        temp_error_map1 = csr_matrix(
            (temp_map1_data, temp_map1_indices, temp_map1_indptr), shape=(3, 5)
        )
        temp_error_maps = (
            temp_error_map1,
            self.error_map2,
        )  # Use original map for stage 2

        # Instantiate with this temporary map
        temp_decomp_instance = DemDecomp(
            org_prob=self.org_probabilities, error_map_matrices=temp_error_maps
        )
        # Expected precomputed map for stage 1: [1, -1, 3]
        np.testing.assert_array_equal(
            temp_decomp_instance._best_org_error_map[0], [1, -1, 3]
        )

        # Input where only error 1 is active (which has no mapping)
        errors_no_map = np.array([False, True, False])
        expected_org_no_map = np.array(
            [False, False, False, False, False]
        )  # Should map to nothing
        actual_org_no_map = temp_decomp_instance.map_errors_to_org_dem(
            errors_no_map, stage=1
        )
        print(f"Input: {errors_no_map}, Output: {actual_org_no_map}")
        np.testing.assert_array_equal(actual_org_no_map, expected_org_no_map)

        # Input where error 0 (maps to 1) and 1 (maps to nothing) are active
        errors_mixed_map = np.array([True, True, False])
        expected_org_mixed = np.array(
            [False, True, False, False, False]
        )  # Only error 0's mapping should appear
        actual_org_mixed = temp_decomp_instance.map_errors_to_org_dem(
            errors_mixed_map, stage=1
        )
        print(f"Input: {errors_mixed_map}, Output: {actual_org_mixed}")
        np.testing.assert_array_equal(actual_org_mixed, expected_org_mixed)


# ==========================================================
# 4. Run Tests
# ==========================================================

if __name__ == "__main__":
    print("Starting DemDecomp mapping tests...")
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
    print("\nTesting finished.")
