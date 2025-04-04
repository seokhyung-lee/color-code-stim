import stim

from src.color_code_stim.stim_utils import _annotate_detectors_before_tick

# --- Example Usage ---
if __name__ == "__main__":
    circuit_in = stim.Circuit(
        """
        DETECTOR(1, 0) rec[-1]       # Before any TICK (tick_counter=0) -> Annotate if tick > 1
        TICK                          # tick_counter becomes 1
        DETECTOR(2, 0) rec[-1]       # Before 2nd TICK (tick_counter=1) -> Annotate if tick > 2
        REPEAT 2 {
            SHIFT_COORDS(0, 10)
            DETECTOR(3, 0) rec[-1]    # iter 0: coords (3, 10), tick_counter=1 -> Annotate if tick > 2
                                      # iter 1: coords (3, 20), tick_counter=1 -> Annotate if tick > 2
        }
        DETECTOR rec[-1]              # No coords - never annotated
        TICK                          # tick_counter becomes 2
        DETECTOR(4, 0) rec[-1]       # Before 3rd TICK (tick_counter=2) -> Annotate if tick > 3
    """
    )

    print("--- Original Circuit ---")
    print(circuit_in)

    print("\n--- Annotate before tick=1 (Should do nothing) ---")
    annotated_1 = _annotate_detectors_before_tick(circuit_in, tick=1, annotation=99)
    # Since flattening happens, direct comparison isn't easy. Print flattened.
    print(annotated_1.flattened())

    print("\n--- Annotate before tick=2 (Annotate first detector) ---")
    annotated_2 = _annotate_detectors_before_tick(circuit_in, tick=2, annotation=-1)
    print(annotated_2.flattened())
    # Expected: DETECTOR(1, 0, -1) rec[-1] ... others unchanged coords

    print("\n--- Annotate before tick=3 (Annotate dets before 2nd TICK) ---")
    annotated_3 = _annotate_detectors_before_tick(circuit_in, tick=3, annotation=5.5)
    print(annotated_3.flattened())
    # Expected: DETECTOR(1, 0, 5.5) rec[-1]
    #           TICK
    #           DETECTOR(2, 0, 5.5) rec[-1]
    #           DETECTOR(3, 10, 5.5) rec[-1] # From first loop iteration
    #           DETECTOR(3, 20, 5.5) rec[-1] # From second loop iteration
    #           DETECTOR rec[-1]
    #           TICK
    #           DETECTOR(4, 0) rec[-1] # This one is NOT annotated

    print("\n--- Annotate before tick=10 (Annotate all with coords) ---")
    annotated_10 = _annotate_detectors_before_tick(circuit_in, tick=10, annotation=-9)
    print(annotated_10.flattened())
    # Expected: All detectors with coords should have -9 appended.
