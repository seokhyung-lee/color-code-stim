import stim
from src.color_code_stim.stim_utils import get_observable_matrix_from_dem


# --- Example Usage ---
if __name__ == "__main__":
    dem_str = """
        error(0.1) D0 L0      # Error 0: Affects L0
        error(0.2) D1 L1      # Error 1: Affects L1
        detector D0           # Ignored
        error(0.3) D0 D1 L0 L1 # Error 2: Affects L0, L1
        logical_observable L0 # Ignored
        error(0.4) D2         # Error 3: Affects no observables
        REPEAT 2 {
            error(0.05) D3 L1 # Error 4 (loop 1), Error 5 (loop 2): Affect L1
        }
    """
    # Flattened errors and affected observables:
    # Err 0: L0
    # Err 1: L1
    # Err 2: L0, L1
    # Err 3: (None)
    # Err 4: L1
    # Err 5: L1

    dem_obj = stim.DetectorErrorModel(dem_str)
    print("--- Input DEM ---")
    print(dem_obj)
    print(f"Number of observables: {dem_obj.num_observables}")  # Should be 2 (L0, L1)

    obs_mat = get_observable_matrix_from_dem(dem_obj)

    print("\n--- Observable Matrix (CSC format) ---")
    print(repr(obs_mat))
    print("\n--- Observable Matrix (Dense Array) ---")
    # Convert to dense for easier visualization (use for small matrices only)
    try:
        print(obs_mat.toarray())
    except MemoryError:
        print("Matrix too large to display as dense array.")

    # Expected Dense Array (Rows=Observables, Cols=Errors):
    #        E0     E1     E2     E3     E4     E5
    # L0 [[ True, False,  True, False, False, False],
    # L1  [False,  True,  True, False,  True,  True]]

    # Verify shape
    num_flat_errors = 0
    for inst in dem_obj.flattened():
        if isinstance(inst, stim.DemInstruction) and inst.type == "error":
            num_flat_errors += 1

    assert obs_mat.shape == (
        dem_obj.num_observables,
        num_flat_errors,
    ), "Matrix shape mismatch"
    print(f"\nMatrix shape: {obs_mat.shape} (Correct)")

    # Verify some specific entries
    assert obs_mat[0, 0] == True, "Entry (0, 0) should be True"
    assert obs_mat[1, 0] == False, "Entry (1, 0) should be False"
    assert obs_mat[0, 2] == True, "Entry (0, 2) should be True"
    assert obs_mat[1, 2] == True, "Entry (1, 2) should be True"
    assert obs_mat[0, 3] == False, "Entry (0, 3) should be False"
    assert obs_mat[1, 3] == False, "Entry (1, 3) should be False"
    assert obs_mat[1, 4] == True, "Entry (1, 4) should be True"
    assert obs_mat[1, 5] == True, "Entry (1, 5) should be True"
    print("Specific entry checks passed!")

    # Test with no observables
    dem_no_obs = stim.DetectorErrorModel(
        """
        error(0.1) D0
        error(0.2) D1
    """
    )
    obs_mat_no_obs = get_observable_matrix_from_dem(dem_no_obs)
    print("\n--- Matrix with No Observables ---")
    print(repr(obs_mat_no_obs))
    print(f"Shape: {obs_mat_no_obs.shape}")
    assert obs_mat_no_obs.shape == (0, 2)

    # Test with no errors
    dem_no_err = stim.DetectorErrorModel(
        """
        detector D0
        logical_observable L0
    """
    )
    obs_mat_no_err = get_observable_matrix_from_dem(dem_no_err)
    print("\n--- Matrix with No Errors ---")
    print(repr(obs_mat_no_err))
    print(f"Shape: {obs_mat_no_err.shape}")
    assert obs_mat_no_err.shape == (1, 0)
