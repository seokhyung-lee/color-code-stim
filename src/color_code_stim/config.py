"""
Configuration constants and utilities for color code quantum error correction.

This module centralizes configuration data, type definitions, and utility functions
used throughout the color_code_stim package.
"""

from typing import Dict, List, Literal, Tuple

# Type definitions
PAULI_LABEL = Literal["X", "Y", "Z"]
COLOR_LABEL = Literal["r", "g", "b"]
CIRCUIT_TYPE = Literal["tri", "rec", "rec_stability", "growing", "cult+growing", "sdqc_memory"]
PATCH_TYPE = Literal["tri", "rec", "rec_stability"]

# CNOT Schedules for triangular color codes
# Moved from ColorCode.__init__ method for centralized configuration
CNOT_SCHEDULES: Dict[str, List[int]] = {
    "tri_optimal": [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2],
    "tri_optimal_reversed": [3, 4, 7, 6, 5, 2, 2, 3, 6, 5, 4, 1],
    "LLB": [2, 3, 6, 5, 4, 1, 3, 4, 7, 6, 5, 2],  # Alias for tri_optimal
    "superdense_default": [3, 1, 2, 3, 1, 2, 6, 4, 5, 6, 4, 5],
}

# SDQC segmentation rules for different code distances
# Maps distance to list of face_x coordinates that are segmented
SDQC_SEGMENTATION_RULES: Dict[int, List[int]] = {
    3: [],  # No segmented faces
    5: [],  # No segmented faces
    7: [20],
    9: [20, 32],
    11: [20, 32, 44],
    13: [20, 32, 38, 44, 56],
}


def color_to_color_val(color: COLOR_LABEL) -> int:
    """
    Convert color label to numeric value.

    Parameters
    ----------
    color : COLOR_LABEL
        Color label ("r", "g", or "b").

    Returns
    -------
    int
        Numeric value (0=r, 1=g, 2=b).
    """
    return {"r": 0, "g": 1, "b": 2}[color]


def color_val_to_color(color_val: Literal[0, 1, 2]) -> COLOR_LABEL:
    """
    Convert numeric value to color label.

    Parameters
    ----------
    color_val : Literal[0, 1, 2]
        Numeric color value (0, 1, or 2).

    Returns
    -------
    COLOR_LABEL
        Color label ("r", "g", or "b").
    """
    return {0: "r", 1: "g", 2: "b"}[color_val]


def get_qubit_coords(qubit) -> Tuple[int, int]:
    """
    Extract coordinates from qubit vertex.

    Parameters
    ----------
    qubit : ig.Vertex
        Qubit vertex with 'x' and 'y' attributes.

    Returns
    -------
    Tuple[int, int]
        Qubit coordinates (x, y).
    """
    return (qubit["x"], qubit["y"])