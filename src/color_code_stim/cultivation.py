import math
from typing import List

import stim

from .utils import get_project_folder


def _is_data_qubit(ox: int, oy: int) -> bool:
    """
    Determines if a qubit at original coordinates (ox, oy) is a data qubit
    based on the provided rules.

    Args:
        ox: Original X coordinate.
        oy: Original Y coordinate.

    Returns:
        True if it's a data qubit, False otherwise (meaning it's an ancillary qubit).
    """
    if oy % 2 == 0:  # y is even
        # x=0, 3, 4, 7, 8, 11, 12, ...
        # Pattern: x % 4 is 0 or 3
        return ox % 4 == 0 or ox % 4 == 3
    else:  # y is odd
        # x=1, 2, 5, 6, 9, 10, 13, 14, ...
        # Pattern: x % 4 is 1 or 2
        return ox % 4 == 1 or ox % 4 == 2


def _get_ancilla_type(ox: int, oy: int) -> str:
    """
    Determines the type (X or Z) of an ancillary qubit.
    Assumes the input coordinates belong to an ancillary qubit.

    Args:
        ox: Original X coordinate.
        oy: Original Y coordinate.

    Returns:
        'X' or 'Z'.
    """
    if (ox + oy) % 2 != 0:  # Odd sum
        return "X"
    else:  # Even sum
        return "Z"


def _map_qubit_coords(ox: int, oy: int, d: int) -> tuple[float, int]:
    """
    Maps original qubit coordinates (ox, oy) to new coordinates (nx, ny).

    Args:
        ox: Original X coordinate.
        oy: Original Y coordinate.
        d: The code distance.

    Returns:
        A tuple (nx, ny) representing the new coordinates.
        nx can be a float for ancillary qubits. ny is always an integer.
    """
    # Rule 1: New Y coordinate is the same as the original
    ny = oy

    k = d - 1

    # Rule 2: Determine qubit type and calculate New X accordingly
    if _is_data_qubit(ox, oy):
        nx = math.ceil(k * 3 - 1.5 * ox)
        nx = float(nx)
    else:
        anc_type = _get_ancilla_type(ox, oy)

        group_index = (ox - 1) // 2
        nx_base = 3 * k - 2 - 3 * group_index

        # Apply offset based on ancilla type
        if anc_type == "X":
            nx = nx_base + 0.5
        else:  # anc_type == 'Z'
            nx = nx_base - 0.5

    return (nx, ny)


def _transform_coords(coords: List[float], d: int) -> List[float]:
    """
    Transforms the first coordinate (x) using _map_qubit_coords logic,
    keeping other coordinates (y, z, ...) the same.

    Args:
        coords: The list of original coordinates [x, y, z, ...].
        d: The code distance parameter for _map_qubit_coords.

    Returns:
        A list of transformed coordinates [x', y, z, ...].
    """
    if not coords:
        return []

    x = coords[0]
    rest = coords[1:]  # y, z, ...

    # Determine the effective 'oy' to use for mapping.
    # Use round(y) if y exists, otherwise default to 0.
    oy_for_map = round(coords[1]) if len(coords) >= 2 else 0

    new_x: float
    # Use a small tolerance for checking if x is effectively an integer
    if abs(x - round(x)) < 1e-9:
        # --- Integer Case ---
        ox_int = round(x)
        # Get the transformed x-coordinate using the mapping function
        new_x = _map_qubit_coords(ox_int, oy_for_map, d)[0]
    else:
        # --- Float Case ---
        x1 = math.floor(x)
        x2 = math.ceil(x)

        # Handle edge case where floor and ceil are the same (should be rare with tolerance check)
        if x1 == x2:
            new_x = _map_qubit_coords(int(x1), oy_for_map, d)[0]
        else:
            # Map the integer coordinates bounding the float coordinate
            nx1 = _map_qubit_coords(int(x1), oy_for_map, d)[0]
            nx2 = _map_qubit_coords(int(x2), oy_for_map, d)[0]

            # Linearly interpolate between the transformed coordinates
            # based on the original float's position between the integers.
            ratio = (x - x1) / (x2 - x1)
            new_x = nx1 + ratio * (nx2 - nx1)

    # Combine the new x with the original rest of the coordinates
    return [new_x] + rest


def _adjust_cultivation_circuit_coords(circuit: stim.Circuit, d: int) -> stim.Circuit:
    """
    Adjusts the coordinates of qubits and detectors in a cultivation circuit.

    Iterates through the circuit, modifying QUBIT_COORDS and DETECTOR instructions
    that specify coordinates. It applies a transformation to the first coordinate (x)
    based on the `_map_qubit_coords` logic, leaving other coordinate dimensions
    unchanged.

    Note: This function implicitly flattens REPEAT blocks and SHIFT_COORDS
    by processing instruction by instruction and creating a new circuit.
    If precise handling of nested structures without flattening is needed,
    a more complex recursive approach would be required. For coordinate
    adjustment, operating on a flattened representation is often simpler.

    Args:
        circuit: The input stim.Circuit, potentially containing coordinate data.
        d: The code distance, used by the coordinate transformation logic.

    Returns:
        A new stim.Circuit with adjusted coordinates in QUBIT_COORDS and
        DETECTOR instructions. Other instructions are preserved.
    """
    # It's often easier to work with a flattened circuit when modifying
    # elements affected by loops or coordinate shifts.
    flattened_circuit = circuit.flattened()
    new_circuit = stim.Circuit()

    for instruction in flattened_circuit:
        # Since we flattened, we only expect stim.CircuitInstruction here
        if not isinstance(instruction, stim.CircuitInstruction):
            # This case should not happen after flattening, but handle defensively
            print(
                f"Warning: Encountered non-instruction in flattened circuit: {type(instruction)}"
            )
            # If it were a repeat block, we might recursively process its body
            # but flattening avoids this. We'll just skip non-instructions here.
            continue

        name = instruction.name
        targets = instruction.targets_copy()
        args = (
            instruction.gate_args_copy()
        )  # Parens arguments (like probabilities or coordinates)

        # Instructions that define coordinates that need transformation
        if name == "QUBIT_COORDS":
            if args:
                new_args = _transform_coords(args, d)
                # Append the instruction with modified coordinates (args)
                new_circuit.append(name, targets, new_args)
            else:
                # QUBIT_COORDS should always have args, but handle defensively
                new_circuit.append(instruction)
        elif name == "DETECTOR":
            if args:  # Check if the detector instruction *has* coordinates specified
                new_args = _transform_coords(args, d)
                # Append the instruction with modified coordinates (args)
                new_circuit.append(name, targets, new_args)
            else:
                # Detector instruction without coordinates, append as is
                new_circuit.append(instruction)
        else:
            # For all other instructions, append them unmodified
            new_circuit.append(instruction)

    return new_circuit


# Combine circuits
