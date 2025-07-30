#!/usr/bin/env python3
"""
Debug script to trace why NotImplementedError isn't raised for SDQC comparative decoding.
"""

import traceback
from color_code_stim import ColorCode
from color_code_stim.noise_model import NoiseModel


def trace_sdqc_creation():
    """Trace SDQC circuit creation to see why NotImplementedError isn't raised."""
    print("=== Tracing SDQC Circuit Creation ===")

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

    print("Creating SDQC circuit with comparative_decoding=True...")

    try:
        cc = ColorCode(
            d=5,
            rounds=3,
            circuit_type="sdqc_memory",
            superdense_circuit=True,
            temp_bdry_type="Z",
            noise_model=sdqc_noise,
            comparative_decoding=True,
        )
        print("SUCCESS - SDQC circuit created without NotImplementedError")
        print(f"Circuit length: {len(cc.circuit)}")
        print(f"Detectors: {cc.dem_xz.num_detectors}")

        # Try to manually check if _add_final_measurements_and_detectors would raise
        print("\nTesting if _add_final_measurements_and_detectors should raise...")

        # Access the circuit builder directly
        builder = cc._circuit_builder
        print(f"Builder comparative_decoding: {builder.comparative_decoding}")

        # The NotImplementedError should be raised during the building process
        # If we got here, something prevented it from being raised

    except NotImplementedError as e:
        print(f"EXPECTED EXCEPTION: NotImplementedError: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"UNEXPECTED EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()


def check_final_measurements_method():
    """Check the _add_final_measurements_and_detectors method more carefully."""
    print("\n=== Checking Final Measurements Method ===")

    from color_code_stim.circuit_builder import CircuitBuilder
    from color_code_stim.graph_builder import TannerGraphBuilder
    from color_code_stim.noise_model import NoiseModel
    import stim

    # Create a minimal setup to test the method
    d = 5
    noise = NoiseModel(
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

    # Build tanner graph
    graph_builder = TannerGraphBuilder(d, circuit_type="sdqc_memory")
    tanner_graph = graph_builder.build()
    qubit_groups = graph_builder.get_qubit_groups()

    # Create circuit builder with comparative_decoding=True
    print("Creating CircuitBuilder with comparative_decoding=True...")
    try:
        builder = CircuitBuilder(
            d=d,
            d2=None,
            rounds=3,
            circuit_type="sdqc_memory",
            superdense_circuit=True,
            temp_bdry_type="Z",
            cnot_schedule="superdense_default",
            noise_model=noise,
            perfect_first_syndrome_extraction=False,
            perfect_logical_measurement=False,
            tanner_graph=tanner_graph,
            qubit_groups=qubit_groups,
            comparative_decoding=True,  # This should cause NotImplementedError
        )
        print("CircuitBuilder created successfully")

        # Try to build the circuit
        print("Building circuit...")
        circuit = builder.build()
        print(f"Circuit built successfully with {len(circuit)} instructions")

    except NotImplementedError as e:
        print(f"EXPECTED: NotImplementedError: {e}")
    except Exception as e:
        print(f"UNEXPECTED: {type(e).__name__}: {e}")
        traceback.print_exc()


def check_method_source():
    """Check if the method implementation has changed."""
    print("\n=== Checking Method Source ===")

    from color_code_stim.circuit_builder import CircuitBuilder
    import inspect

    # Get the source of _add_final_measurements_and_detectors
    try:
        method = getattr(CircuitBuilder, "_add_final_measurements_and_detectors")
        source = inspect.getsource(method)

        # Check if NotImplementedError is still there
        if "raise NotImplementedError" in source:
            print("NotImplementedError is still in the method source")
            # Find the lines
            lines = source.split("\n")
            for i, line in enumerate(lines):
                if (
                    "comparative_decoding" in line
                    and "raise NotImplementedError" in lines[i + 1]
                ):
                    print(f"Line {i}: {line.strip()}")
                    print(f"Line {i+1}: {lines[i+1].strip()}")
        else:
            print("NotImplementedError was removed from the method")

    except Exception as e:
        print(f"Failed to get method source: {e}")


if __name__ == "__main__":
    trace_sdqc_creation()
    check_final_measurements_method()
    check_method_source()
