import stim

from src.color_code_stim.cultivation import _isolate_final_mpps

# --- Example Usage ---
if __name__ == "__main__":
    # MPP using GateTarget and combiners, producing 2 measurements
    circuit_in_gatetarget = stim.Circuit(
        """
        # Part 1: Keep this
        H 0 1
        CX 0 2 1 3
        M 0 1 # Measurements at index 0, 1 (abs) before the TICK
        TICK
        # Part 2: Remove this block
        MPP Z1 * Z3 X0 * X2 # Meas index 2 (Z basis {1, 3}), Meas index 3 (X basis {0, 2})
        OBSERVABLE_INCLUDE(0) rec[-1] # Depends on X0*X2
        DETECTOR rec[-1] rec[-3] # Depends on X0*X2 (MPP) and M 1 (before TICK) -> ('X', {0, 2}), -3
        DETECTOR rec[-2] rec[-4] # Depends on Z1*Z3 (MPP) and M 0 (before TICK) -> ('Z', {1, 3}), -4
        DETECTOR rec[-1] rec[-2] # Depends on X0*X2 (MPP) and Z1*Z3 (MPP) -> ('X', {0, 2}), ('Z', {1, 3})
    """
    )  # Total 4 measurements

    print("--- Original Circuit (GateTarget MPP) ---")
    print(circuit_in_gatetarget)
    print(f"Original num_measurements: {circuit_in_gatetarget.num_measurements}")

    mpp_instr_check = circuit_in_gatetarget[-5]  # Get the MPP instruction
    print(f"MPP Instruction: {mpp_instr_check}")
    print(f"MPP num_measurements property: {mpp_instr_check.num_measurements}")
    # print(f"MPP targets_copy(): {mpp_instr_check.targets_copy()}") # Debug: See the flat list

    trimmed_c_gt, d_map_gt = _isolate_final_mpps(circuit_in_gatetarget)

    print("\n--- Trimmed Circuit (GateTarget MPP) ---")
    print(trimmed_c_gt)
    print(f"Trimmed num_measurements: {trimmed_c_gt.num_measurements}")

    print("\n--- Detector Qubit Map (GateTarget MPP) ---")
    # Expected output based on example:
    # Detector 1: rec[-1] (X0*X2 -> ('X', {0, 2})), rec[-3] (M 1 -> -3)
    # Detector 2: rec[-2] (Z1*Z3 -> ('Z', {1, 3})), rec[-4] (M 0 -> -4)
    # Detector 3: rec[-1] (X0*X2 -> ('X', {0, 2})), rec[-2] (Z1*Z3 -> ('Z', {1, 3}))
    expected_map_gt = [
        [("X", frozenset({0, 2})), -3],
        [("Z", frozenset({1, 3})), -4],
        [("X", frozenset({0, 2})), ("Z", frozenset({1, 3}))],
    ]

    print("Expected:")
    for i, item in enumerate(expected_map_gt):
        print(f"  Detector {i}: {item}")

    print("\nActual:")
    for i, item in enumerate(d_map_gt):
        print(f"  Detector {i}: {item}")

    assert trimmed_c_gt == stim.Circuit(
        """
        H 0 1
        CX 0 2 1 3
        M 0 1
    """
    ), "Trimmed circuit mismatch (GateTarget MPP)"
    assert d_map_gt == expected_map_gt, "Detector map mismatch (GateTarget MPP)"
    print("\nAssertions Passed!")
