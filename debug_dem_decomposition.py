#!/usr/bin/env python3
"""
Debug script to investigate DEM decomposition issues for SDQC circuits.
"""

from color_code_stim import ColorCode
from color_code_stim.noise_model import NoiseModel


def investigate_dem_decomposition():
    """Investigate DEM decomposition for SDQC vs tri circuits."""
    print("=== Investigating DEM Decomposition ===")

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

    print("Creating circuits...")

    # Create all combinations
    cc_sdqc_normal = ColorCode(
        d=d,
        rounds=rounds,
        circuit_type="sdqc_memory",
        superdense_circuit=True,
        temp_bdry_type="Z",
        noise_model=sdqc_noise,
        comparative_decoding=False,
    )

    cc_sdqc_comp = ColorCode(
        d=d,
        rounds=rounds,
        circuit_type="sdqc_memory",
        superdense_circuit=True,
        temp_bdry_type="Z",
        noise_model=sdqc_noise,
        comparative_decoding=True,
    )

    cc_tri_comp = ColorCode(
        d=d,
        rounds=rounds,
        circuit_type="tri",
        superdense_circuit=True,
        temp_bdry_type="Z",
        noise_model=tri_noise,
        comparative_decoding=True,
    )

    print("Checking DEM decomposition...")

    # Check decomposed DEMs
    print(f"\nSDQC normal:")
    print(f"  Detectors: {cc_sdqc_normal.dem_xz.num_detectors}")
    print(f"  Observables: {cc_sdqc_normal.dem_xz.num_observables}")

    try:
        dems_decomp = cc_sdqc_normal.dems_decomposed
        print(f"  Decomposed DEMs available: {list(dems_decomp.keys())}")
        for color, dem_decomp in dems_decomp.items():
            print(
                f"    {color}: stage1 detectors={dem_decomp.dem1.num_detectors}, stage2 detectors={dem_decomp.dem2.num_detectors}"
            )
    except Exception as e:
        print(f"  Decomposition failed: {e}")

    print(f"\nSDQC comp:")
    print(f"  Detectors: {cc_sdqc_comp.dem_xz.num_detectors}")
    print(f"  Observables: {cc_sdqc_comp.dem_xz.num_observables}")

    try:
        dems_decomp = cc_sdqc_comp.dems_decomposed
        print(f"  Decomposed DEMs available: {list(dems_decomp.keys())}")
        for color, dem_decomp in dems_decomp.items():
            print(
                f"    {color}: stage1 detectors={dem_decomp.dem1.num_detectors}, stage2 detectors={dem_decomp.dem2.num_detectors}"
            )
    except Exception as e:
        print(f"  Decomposition failed: {e}")
        import traceback

        traceback.print_exc()

    print(f"\nTri comp:")
    print(f"  Detectors: {cc_tri_comp.dem_xz.num_detectors}")
    print(f"  Observables: {cc_tri_comp.dem_xz.num_observables}")

    try:
        dems_decomp = cc_tri_comp.dems_decomposed
        print(f"  Decomposed DEMs available: {list(dems_decomp.keys())}")
        for color, dem_decomp in dems_decomp.items():
            print(
                f"    {color}: stage1 detectors={dem_decomp.dem1.num_detectors}, stage2 detectors={dem_decomp.dem2.num_detectors}"
            )
    except Exception as e:
        print(f"  Decomposition failed: {e}")
        import traceback

        traceback.print_exc()

    # Check decoder details
    print(f"\nChecking decoder internals...")

    try:
        decoder_sdqc_comp = cc_sdqc_comp.concat_matching_decoder
        decoder_tri_comp = cc_tri_comp.concat_matching_decoder

        print(
            f"SDQC comp stage2 matchings available: {list(decoder_sdqc_comp.stage2_matchings.keys())}"
        )
        print(
            f"Tri comp stage2 matchings available: {list(decoder_tri_comp.stage2_matchings.keys())}"
        )

        # Check individual stage2 matching graphs
        for color in ["r", "g", "b"]:
            if color in decoder_sdqc_comp.stage2_matchings:
                matching = decoder_sdqc_comp.stage2_matchings[color]
                print(
                    f"SDQC comp {color} matching: num_nodes={matching.num_nodes}, num_edges={matching.num_edges}"
                )

            if color in decoder_tri_comp.stage2_matchings:
                matching = decoder_tri_comp.stage2_matchings[color]
                print(
                    f"Tri comp {color} matching: num_nodes={matching.num_nodes}, num_edges={matching.num_edges}"
                )

    except Exception as e:
        print(f"Decoder inspection failed: {e}")
        import traceback

        traceback.print_exc()


def check_circuit_differences():
    """Check what's different in the actual circuits."""
    print("\n=== Checking Circuit Differences ===")

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

    # Create SDQC circuits with and without comparative decoding
    cc_sdqc_normal = ColorCode(
        d=d,
        rounds=rounds,
        circuit_type="sdqc_memory",
        superdense_circuit=True,
        temp_bdry_type="Z",
        noise_model=sdqc_noise,
        comparative_decoding=False,
    )

    cc_sdqc_comp = ColorCode(
        d=d,
        rounds=rounds,
        circuit_type="sdqc_memory",
        superdense_circuit=True,
        temp_bdry_type="Z",
        noise_model=sdqc_noise,
        comparative_decoding=True,
    )

    print("Circuit lengths:")
    print(f"  SDQC normal: {len(cc_sdqc_normal.circuit)} instructions")
    print(f"  SDQC comp:   {len(cc_sdqc_comp.circuit)} instructions")

    # Check if circuits are actually different
    if str(cc_sdqc_normal.circuit) == str(cc_sdqc_comp.circuit):
        print("  Circuits are identical!")
    else:
        print("  Circuits are different!")
        # Find first difference
        normal_lines = str(cc_sdqc_normal.circuit).split("\n")
        comp_lines = str(cc_sdqc_comp.circuit).split("\n")

        for i, (line1, line2) in enumerate(zip(normal_lines, comp_lines)):
            if line1 != line2:
                print(f"  First difference at line {i}:")
                print(f"    Normal: {line1}")
                print(f"    Comp:   {line2}")
                break

    # Check detector error models
    print(f"\nDEM string lengths:")
    print(f"  SDQC normal: {len(str(cc_sdqc_normal.dem_xz))} chars")
    print(f"  SDQC comp:   {len(str(cc_sdqc_comp.dem_xz))} chars")

    if str(cc_sdqc_normal.dem_xz) == str(cc_sdqc_comp.dem_xz):
        print("  DEMs are identical!")
    else:
        print("  DEMs are different!")

        # Find differences in DEM
        normal_dem_lines = str(cc_sdqc_normal.dem_xz).split("\n")
        comp_dem_lines = str(cc_sdqc_comp.dem_xz).split("\n")

        print(f"  Normal DEM has {len(normal_dem_lines)} lines")
        print(f"  Comp DEM has {len(comp_dem_lines)} lines")

        # Show first few differences
        differences = 0
        for i, (line1, line2) in enumerate(zip(normal_dem_lines, comp_dem_lines)):
            if line1 != line2 and differences < 3:  # Show first 3 differences
                print(f"    Difference at line {i}:")
                print(f"      Normal: {line1}")
                print(f"      Comp:   {line2}")
                differences += 1


if __name__ == "__main__":
    investigate_dem_decomposition()
    check_circuit_differences()
