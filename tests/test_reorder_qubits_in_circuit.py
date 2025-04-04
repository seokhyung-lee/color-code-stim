from typing import Dict, Tuple

import stim

from src.color_code_stim.cultivation import _reorder_qubits_in_circuit

if __name__ == "__main__":
    # Example Circuit
    circuit_in = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(0, 1) 2
        QUBIT_COORDS(1, 1) 3
        H 0
        CX 0 1 2 3
        X_ERROR(0.1) 1
        DEPOLARIZE2(0.01) 0 2
        MPP X0*Z1 Y2*Y3 Z0*X2
        M 0 1 2 3
    """
    )

    # Define the desired new order based on original coordinates
    # Map (0,0) -> 3, (1,0) -> 2, (0,1) -> 1, (1,1) -> 0
    new_order: Dict[Tuple[int, int], int] = {
        (0, 0): 3,
        (1, 0): 2,
        (0, 1): 1,
        (1, 1): 0,
    }

    # Perform the reordering
    try:
        circuit_out = _reorder_qubits_in_circuit(circuit_in, new_order)

        print("--- Original Circuit ---")
        print(circuit_in)
        print("\n--- Reordered Circuit ---")
        print(circuit_out)

        # Verification (check specific instructions)
        # Original CX 0 1 2 3 -> New CX 3 2 1 0
        # Original MPP X0*Z1 Y2*Y3 Z0*X2 -> New MPP X3*Z2 Y1*Y0 Z3*X1
        # Original M 0 1 2 3 -> New M 3 2 1 0
        # QUBIT_COORDS targets should also be remapped: QUBIT_COORDS(0,0) 3 etc.

    except ValueError as e:
        print(f"Error during qubit reordering: {e}")

    print("\n--- Example with Missing Mapping ---")
    circuit_missing = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(5, 5) 5 # This one won't be in the map
        H 0
        CX 0 5
        M 0 5
    """
    )
    order_missing: Dict[Tuple[int, int], int] = {(0, 0): 99}
    try:
        _reorder_qubits_in_circuit(circuit_missing, order_missing)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n--- Example with Uncoordinated Qubit ---")
    circuit_nocoords = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        # Qubit 1 has no coordinates defined
        H 0
        CX 0 1
        M 0 1
    """
    )
    order_nocoords: Dict[Tuple[int, int], int] = {(0, 0): 10}
    try:
        _reorder_qubits_in_circuit(circuit_nocoords, order_nocoords)
    except ValueError as e:
        print(f"Caught expected error: {e}")
