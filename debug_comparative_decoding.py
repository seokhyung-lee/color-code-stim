#!/usr/bin/env python3
"""
Debug script to investigate comparative decoding issues with SDQC circuits.
"""

import traceback
from color_code_stim import ColorCode
from color_code_stim.noise_model import NoiseModel


def test_comparative_decoding_sdqc():
    """Test comparative decoding for SDQC circuits with detailed error reporting."""
    print("=== Testing SDQC Memory Circuit with Comparative Decoding ===")

    # Create noise model with shuttling error rates
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

    for d in [3, 5]:
        print(f"\n--- Testing distance d={d} ---")

        try:
            # Create SDQC circuit with comparative decoding
            cc_sdqc = ColorCode(
                d=d,
                rounds=3,
                circuit_type="sdqc_memory",
                superdense_circuit=True,
                temp_bdry_type="Z",
                noise_model=sdqc_noise,
                comparative_decoding=True,  # This is what fails
                set_all_faces_segmented=False,
            )

            print(f"SDQC ColorCode created successfully for d={d}")
            print(f"Circuit has {len(cc_sdqc.circuit)} instructions")
            print(f"DEM has {cc_sdqc.dem_xz.num_detectors} detectors")
            print(f"DEM has {cc_sdqc.dem_xz.num_observables} observables")
            print(f"Segmented faces: {cc_sdqc.segmented_faces}")

            # Try simulation
            print("Attempting simulation...")
            num_fails = cc_sdqc.simulate(shots=10)
            print(f"Simulation successful: {num_fails} failures")

        except Exception as e:
            print(f"ERROR for d={d}: {type(e).__name__}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            print()


def test_comparative_decoding_regular_tri():
    """Test comparative decoding for regular triangular circuits."""
    print("\n=== Testing Regular Triangular Circuit with Comparative Decoding ===")

    noise = NoiseModel.uniform_circuit_noise(1e-3)

    for d in [3, 5]:
        print(f"\n--- Testing distance d={d} ---")

        try:
            # Create regular tri circuit with comparative decoding
            cc_tri = ColorCode(
                d=d,
                rounds=3,
                circuit_type="tri",
                superdense_circuit=True,  # Use same settings as SDQC
                temp_bdry_type="Z",
                noise_model=noise,
                comparative_decoding=True,
            )

            print(f"Regular tri ColorCode created successfully for d={d}")
            print(f"Circuit has {len(cc_tri.circuit)} instructions")
            print(f"DEM has {cc_tri.dem_xz.num_detectors} detectors")
            print(f"DEM has {cc_tri.dem_xz.num_observables} observables")

            # Try simulation
            print("Attempting simulation...")
            num_fails = cc_tri.simulate(shots=10)
            print(f"Simulation successful: {num_fails} failures")

        except Exception as e:
            print(f"ERROR for d={d}: {type(e).__name__}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            print()


def compare_circuits_structure():
    """Compare circuit structure between SDQC and regular tri."""
    print("\n=== Comparing Circuit Structures ===")

    # Create both circuits with same parameters (except circuit type)
    d = 5
    rounds = 3

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

    try:
        # SDQC circuit without comparative decoding (should work)
        cc_sdqc = ColorCode(
            d=d,
            rounds=rounds,
            circuit_type="sdqc_memory",
            superdense_circuit=True,
            temp_bdry_type="Z",
            noise_model=sdqc_noise,
            comparative_decoding=False,
        )

        # Regular tri circuit without comparative decoding
        cc_tri = ColorCode(
            d=d,
            rounds=rounds,
            circuit_type="tri",
            superdense_circuit=True,
            temp_bdry_type="Z",
            noise_model=tri_noise,
            comparative_decoding=False,
        )

        print(f"SDQC circuit length: {len(cc_sdqc.circuit)}")
        print(f"Regular tri circuit length: {len(cc_tri.circuit)}")
        print(f"SDQC detectors: {cc_sdqc.dem_xz.num_detectors}")
        print(f"Regular tri detectors: {cc_tri.dem_xz.num_detectors}")
        print(f"SDQC observables: {cc_sdqc.dem_xz.num_observables}")
        print(f"Regular tri observables: {cc_tri.dem_xz.num_observables}")

        # Check if DEMs have same structure
        print(f"SDQC segmented faces: {cc_sdqc.segmented_faces}")

        # Now try adding comparative decoding to see where it fails
        print("\n--- Trying to add comparative decoding ---")

        try:
            cc_sdqc_comp = ColorCode(
                d=d,
                rounds=rounds,
                circuit_type="sdqc_memory",
                superdense_circuit=True,
                temp_bdry_type="Z",
                noise_model=sdqc_noise,
                comparative_decoding=True,  # This should fail
            )
            print("SDQC with comparative decoding: SUCCESS")
        except Exception as e:
            print(f"SDQC with comparative decoding failed: {type(e).__name__}: {e}")
            traceback.print_exc()

        try:
            cc_tri_comp = ColorCode(
                d=d,
                rounds=rounds,
                circuit_type="tri",
                superdense_circuit=True,
                temp_bdry_type="Z",
                noise_model=tri_noise,
                comparative_decoding=True,
            )
            print("Regular tri with comparative decoding: SUCCESS")
        except Exception as e:
            print(
                f"Regular tri with comparative decoding failed: {type(e).__name__}: {e}"
            )
            traceback.print_exc()

    except Exception as e:
        print(f"Comparison setup failed: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting comparative decoding investigation...")

    test_comparative_decoding_sdqc()
    test_comparative_decoding_regular_tri()
    compare_circuits_structure()

    print("\nInvestigation complete.")
