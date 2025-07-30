#!/usr/bin/env python3
"""
Debug script to investigate parity issues in SDQC comparative decoding.
"""

import numpy as np
from color_code_stim import ColorCode
from color_code_stim.noise_model import NoiseModel


def investigate_parity_issue():
    """Investigate the parity issue in SDQC comparative decoding."""
    print("=== Investigating Parity Issue in SDQC Comparative Decoding ===")

    d = 5
    rounds = 3

    # Create noise models
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

    tri_noise = NoiseModel.uniform_circuit_noise(1e-3)

    # Create circuits
    print("Creating circuits...")

    # SDQC without comparative decoding (works)
    cc_sdqc_normal = ColorCode(
        d=d,
        rounds=rounds,
        circuit_type="sdqc_memory",
        superdense_circuit=True,
        temp_bdry_type="Z",
        noise_model=sdqc_noise,
        comparative_decoding=False,
    )

    # SDQC with comparative decoding (fails)
    cc_sdqc_comp = ColorCode(
        d=d,
        rounds=rounds,
        circuit_type="sdqc_memory",
        superdense_circuit=True,
        temp_bdry_type="Z",
        noise_model=sdqc_noise,
        comparative_decoding=True,
    )

    # Regular tri with comparative decoding (works)
    cc_tri_comp = ColorCode(
        d=d,
        rounds=rounds,
        circuit_type="tri",
        superdense_circuit=True,
        temp_bdry_type="Z",
        noise_model=tri_noise,
        comparative_decoding=True,
    )

    print("All circuits created successfully")

    # Compare observables and detector structures
    print(
        f"\nSDQC normal - detectors: {cc_sdqc_normal.dem_xz.num_detectors}, observables: {cc_sdqc_normal.dem_xz.num_observables}"
    )
    print(
        f"SDQC comp   - detectors: {cc_sdqc_comp.dem_xz.num_detectors}, observables: {cc_sdqc_comp.dem_xz.num_observables}"
    )
    print(
        f"Tri comp    - detectors: {cc_tri_comp.dem_xz.num_detectors}, observables: {cc_tri_comp.dem_xz.num_observables}"
    )

    # Let's check the observable matrices
    print(f"\nSDQC normal - obs matrix shape: {cc_sdqc_normal.obs_matrix.shape}")
    print(f"SDQC comp   - obs matrix shape: {cc_sdqc_comp.obs_matrix.shape}")
    print(f"Tri comp    - obs matrix shape: {cc_tri_comp.obs_matrix.shape}")

    # Check if the observable matrices are different
    if cc_sdqc_normal.obs_matrix.shape == cc_sdqc_comp.obs_matrix.shape:
        obs_diff = (cc_sdqc_normal.obs_matrix != cc_sdqc_comp.obs_matrix).nnz
        print(f"SDQC normal vs comp observable matrix differences: {obs_diff} entries")

    if cc_tri_comp.obs_matrix.shape == cc_sdqc_comp.obs_matrix.shape:
        obs_diff = (cc_tri_comp.obs_matrix != cc_sdqc_comp.obs_matrix).nnz
        print(
            f"Tri comp vs SDQC comp observable matrix differences: {obs_diff} entries"
        )

    # Try sampling with no noise to see if there are fundamental issues
    print("\nTesting with zero noise...")

    zero_noise = NoiseModel(
        cnot=0.0,
        meas=0.0,
        reset=0.0,
        idle=0.0,
        shuttling_seg_init=0.0,
        shuttling_non_seg_init=0.0,
        shuttling_seg_final=0.0,
        shuttling_non_seg_final=0.0,
        depol1_on_anc_before_cnot=0.0,
    )

    cc_sdqc_zero = ColorCode(
        d=d,
        rounds=rounds,
        circuit_type="sdqc_memory",
        superdense_circuit=True,
        temp_bdry_type="Z",
        noise_model=zero_noise,
        comparative_decoding=True,
    )

    try:
        result = cc_sdqc_zero.simulate(shots=5)
        print(f"SDQC with zero noise succeeded: {result} failures")
    except Exception as e:
        print(f"SDQC with zero noise failed: {type(e).__name__}: {e}")

    # Try different boundary types
    print("\nTesting different boundary types for SDQC...")
    for bdry_type in ["X", "Y", "Z"]:
        try:
            cc_test = ColorCode(
                d=d,
                rounds=rounds,
                circuit_type="sdqc_memory",
                superdense_circuit=True,
                temp_bdry_type=bdry_type,
                noise_model=zero_noise,
                comparative_decoding=True,
            )
            result = cc_test.simulate(shots=2)
            print(f"SDQC with boundary {bdry_type}: SUCCESS ({result} failures)")
        except Exception as e:
            print(f"SDQC with boundary {bdry_type}: FAILED - {type(e).__name__}: {e}")

    # Test smaller distance
    print(f"\nTesting smaller distance (d=3)...")
    try:
        cc_small = ColorCode(
            d=3,
            rounds=3,
            circuit_type="sdqc_memory",
            superdense_circuit=True,
            temp_bdry_type="Z",
            noise_model=zero_noise,
            comparative_decoding=True,
        )
        result = cc_small.simulate(shots=2)
        print(f"SDQC d=3 with zero noise: SUCCESS ({result} failures)")
    except Exception as e:
        print(f"SDQC d=3 with zero noise: FAILED - {type(e).__name__}: {e}")


def sample_and_check_parity():
    """Sample detector outcomes and check parity manually."""
    print("\n=== Manual Parity Check ===")

    d = 5
    zero_noise = NoiseModel(
        cnot=0.0,
        meas=0.0,
        reset=0.0,
        idle=0.0,
        shuttling_seg_init=0.0,
        shuttling_non_seg_init=0.0,
        shuttling_seg_final=0.0,
        shuttling_non_seg_final=0.0,
        depol1_on_anc_before_cnot=0.0,
    )

    # Sample from SDQC circuit
    cc_sdqc = ColorCode(
        d=d,
        rounds=3,
        circuit_type="sdqc_memory",
        superdense_circuit=True,
        temp_bdry_type="Z",
        noise_model=zero_noise,
        comparative_decoding=True,
    )

    # Sample a few shots and check syndrome parity
    print("Sampling detector outcomes...")
    try:
        # Sample without decoding
        det_outcomes, obs_outcomes = cc_sdqc.simulator.sample(shots=5)
        print(
            f"Sample successful: detector shape {det_outcomes.shape}, obs shape {obs_outcomes.shape}"
        )

        # Check parity of detector outcomes
        for shot in range(det_outcomes.shape[0]):
            det_parity = np.sum(det_outcomes[shot]) % 2
            obs_parity = obs_outcomes[shot, 0] if obs_outcomes.shape[1] > 0 else 0
            print(f"Shot {shot}: detector parity={det_parity}, observable={obs_parity}")

        # Try decoding one shot manually
        print("\nTrying manual decode...")
        single_det = det_outcomes[0:1]  # Just first shot
        pred = cc_sdqc.decode(single_det)
        print(f"Manual decode successful: {pred}")

    except Exception as e:
        print(f"Sampling failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    investigate_parity_issue()
    sample_and_check_parity()
