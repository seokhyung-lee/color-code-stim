import numpy as np
import scipy.sparse
import stim

from src.color_code_stim.stim_symbolic import _DemSymbolic, _ErrorMechanismSymbolic

# --- Example Usage ---
if __name__ == "__main__":
    # Assume _DemSymbolic and _ErrorMechanismSymbolic are fully defined above

    # Create EMs for testing compression
    em_a1 = _ErrorMechanismSymbolic(
        prob_vars=[0], prob_muls=[0.1], dets=[stim.target_relative_detector_id(0)]
    )
    em_a2 = _ErrorMechanismSymbolic(
        prob_vars=[1], prob_muls=[0.2], dets=[stim.target_relative_detector_id(0)]
    )  # Same dets as a1
    em_b1 = _ErrorMechanismSymbolic(
        prob_vars=[0],
        prob_muls=[0.3],
        dets=[stim.target_relative_detector_id(1)],
        obss={stim.target_logical_observable_id(0)},
    )
    em_c1 = _ErrorMechanismSymbolic(
        prob_vars=[1], prob_muls=[0.4], dets=[stim.target_relative_detector_id(2)]
    )  # Unique
    em_a3 = _ErrorMechanismSymbolic(
        prob_vars=[0], prob_muls=[0.5], dets=[stim.target_relative_detector_id(0)]
    )  # Same dets as a1, a2

    # Dummy dets_org and initial map
    dummy_dets_org = stim.DetectorErrorModel(
        "detector D0\ndetector D1\ndetector D2\nlogical_observable L0"
    )
    # Initial map needs correct dimensions (#ems, #vars)
    # Variables used: 0, 1. Max index is 1, so num_vars = 2.
    # Number of EMs = 5
    initial_map = scipy.sparse.csr_matrix((5, 2), dtype=bool)  # Placeholder

    symbolic_dem_compress_test = _DemSymbolic.FromEms(
        [em_a1, em_a2, em_b1, em_c1, em_a3], dummy_dets_org
    )
    # Calculate the correct initial map before compression
    symbolic_dem_compress_test.update_error_map_matrix()

    print("--- Before Compression ---")
    print(f"Number of EMs: {len(symbolic_dem_compress_test._ems)}")
    for i, em in enumerate(symbolic_dem_compress_test._ems):
        print(f"  EM {i}: {em}")
    print("Error Map Matrix (Dense):")
    print(symbolic_dem_compress_test.error_map_matrix.toarray())
    # Expected initial map (5 EMs, 2 Vars p0, p1):
    #        p0     p1
    # E0 [[ True, False], # em_a1 (p0)
    # E1  [False,  True], # em_a2 (p1)
    # E2  [ True, False], # em_b1 (p0)
    # E3  [False,  True], # em_c1 (p1)
    # E4  [ True, False]] # em_a3 (p0)

    # --- Perform Compression ---
    symbolic_dem_compress_test.compress()

    print("\n--- After Compression ---")
    print(f"Number of EMs: {len(symbolic_dem_compress_test._ems)}")
    # Expected: em_a1, em_a2, em_a3 merged into one. em_b1, em_c1 remain. Total 3 EMs.
    # The merged EM (a1+a2+a3) should have:
    #   dets = {D0}
    #   obss = {}
    #   prob_vars = [0, 1] (sorted)
    #   prob_muls = [0.1 + 0.5, 0.2] = [0.6, 0.2]
    for i, em in enumerate(symbolic_dem_compress_test._ems):
        print(f"  EM {i}: {em}")
    print("Error Map Matrix (Dense):")
    print(symbolic_dem_compress_test.error_map_matrix.toarray())
    # Expected compressed map (3 EMs, 2 Vars p0, p1):
    #        p0     p1
    # E0 [[ True,  True], # Merged a1, a2, a3 (depends on p0, p1)
    # E1  [ True, False], # em_b1 (depends on p0)
    # E2  [False,  True]] # em_c1 (depends on p1)
    # Note: The order of rows might change depending on dictionary iteration order.

    # --- Verification ---
    assert len(symbolic_dem_compress_test._ems) == 3

    # Find the merged EM (should have D0)
    merged_em_found = None
    other_ems = []
    for em in symbolic_dem_compress_test._ems:
        if em.dets == {stim.target_relative_detector_id(0)}:
            merged_em_found = em
        else:
            other_ems.append(em)

    assert merged_em_found is not None, "Merged EM with D0 not found"
    assert np.array_equal(merged_em_found.prob_vars, np.array([0, 1]))
    assert np.allclose(merged_em_found.prob_muls, np.array([0.6, 0.2]))
    assert merged_em_found.obss == set()

    # Check the other EMs are present
    assert em_b1 in other_ems
    assert em_c1 in other_ems

    # Check the recomputed map matrix shape
    assert symbolic_dem_compress_test.error_map_matrix.shape == (3, 2)  # 3 EMs, 2 Vars

    # Check map content (assuming the order after compression is merged_a, b1, c1)
    # This order depends on hash/dict iteration, so check based on EM content
    expected_map_dense_compressed = np.zeros((3, 2), dtype=bool)
    for i, em in enumerate(symbolic_dem_compress_test._ems):
        if em.dets == {stim.target_relative_detector_id(0)}:  # Merged A
            expected_map_dense_compressed[i, 0] = True
            expected_map_dense_compressed[i, 1] = True
        elif em.dets == {stim.target_relative_detector_id(1)}:  # B1
            expected_map_dense_compressed[i, 0] = True
        elif em.dets == {stim.target_relative_detector_id(2)}:  # C1
            expected_map_dense_compressed[i, 1] = True

    assert np.array_equal(
        symbolic_dem_compress_test.error_map_matrix.toarray(),
        expected_map_dense_compressed,
    )

    print("\nCompression assertions passed!")
