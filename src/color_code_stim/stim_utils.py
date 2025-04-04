import io
import re
from typing import List, Tuple

import numpy as np
import stim
from cairosvg import svg2png
from scipy.sparse import csc_matrix


def dem_to_str(dem: stim.DetectorErrorModel) -> str:
    """
    Convert a detector error model to its string representation.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to convert.

    Returns
    -------
    str
        String representation of the detector error model.
    """
    buffer = io.StringIO()
    dem.to_file(buffer)
    s = buffer.getvalue()
    return s


def str_to_dem(s: str) -> stim.DetectorErrorModel:
    """
    Convert a string representation back to a detector error model.

    Parameters
    ----------
    s : str
        String representation of a detector error model.

    Returns
    -------
    stim.DetectorErrorModel
        The reconstructed detector error model.
    """
    buffer = io.StringIO(s)
    return stim.DetectorErrorModel.from_file(buffer)


def remove_obs_from_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """
    Remove detectors acting as observables from a detector error model.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to process.

    Returns
    -------
    stim.DetectorErrorModel
        A new detector error model with all detectors acting as observables removed.
    """
    num_dets = dem.num_detectors
    num_obss = dem.num_observables
    s = dem_to_str(dem)

    # Remove last lines corresponding to observables
    s = "\n".join(s.splitlines()[:-num_obss]) + "\n"

    # Build a regex pattern to match the exact text "D{num_dets - 1}"
    patterns = [r"\bD" + str(num_dets - num_obss + i) + r"\b" for i in range(num_obss)]
    # Remove all occurrences of that pattern
    for pattern in patterns:
        s = re.sub(pattern, "", s)

    # Clean up extra spaces in each line (e.g. converting "D0  D1  L0" to "D0 D1 L0")
    s = "\n".join(" ".join(line.split()) for line in s.splitlines())

    return str_to_dem(s)


def dem_to_parity_check(dem: stim.DetectorErrorModel) -> Tuple[csc_matrix, np.ndarray]:
    """
    Convert a detector error model (DEM) into a parity check matrix and probability vector.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to convert.

    Returns
    -------
    H : csc_matrix
        A boolean matrix of shape (number of detectors, number of errors)
        where H[i, j] = True if detector i is involved in error j.
    p : np.ndarray
        A 1D numpy array of probabilities corresponding to each error.
    """
    lines = dem_to_str(dem).splitlines()
    probabilities = []
    error_detector_indices = []
    max_detector = -1

    for line in lines:
        if line.startswith("error("):
            # Extract probability using regex.
            prob_match = re.search(r"error\(([\d.eE+-]+)\)", line)
            if not prob_match:
                continue
            prob = float(prob_match.group(1))
            probabilities.append(prob)

            # Tokenize the line and collect detectors (ignore observables starting with 'L')
            tokens = line.split()
            detectors = []
            for token in tokens:
                if token.startswith("D"):
                    try:
                        det = int(token[1:])
                        detectors.append(det)
                        max_detector = max(max_detector, det)
                    except ValueError:
                        pass
            error_detector_indices.append(detectors)

    num_errors = len(probabilities)
    num_detectors = max_detector + 1

    # Build the sparse parity check matrix.
    rows = []
    cols = []
    data = []
    for col, dets in enumerate(error_detector_indices):
        for d in dets:
            rows.append(d)
            cols.append(col)
            data.append(True)

    H = csc_matrix((data, (rows, cols)), shape=(num_detectors, num_errors), dtype=bool)
    p = np.array(probabilities)

    return H, p


def get_observable_matrix_from_dem(
    dem: stim.DetectorErrorModel,
) -> csc_matrix:
    """
    Extracts the observable matrix from a Detector Error Model.

    The resulting matrix indicates which error mechanisms (columns) affect
    which logical observables (rows).

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The input Detector Error Model.

    Returns
    -------
    obs_matrix : scipy.sparse.csc_matrix
        A boolean sparse matrix in CSC format. obs_matrix[i, j] is True
        if and only if the j-th error mechanism (in the flattened DEM order)
        affects the i-th logical observable (L{i}).
        The number of rows equals dem.num_observables.
        The number of columns equals the total count of 'error' instructions
        in the flattened DEM.
    """
    # Flatten the DEM to handle loops and get a linear sequence of errors
    try:
        flattened_dem = dem.flattened()
    except Exception as e:
        print(f"Error during DEM flattening: {e}")
        # Re-raise or handle as appropriate
        raise e

    num_observables = dem.num_observables
    if num_observables == 0:
        # If there are no observables, the matrix has 0 rows.
        # We still need to count errors to determine the number of columns.
        num_errors = 0
        for instruction in flattened_dem:
            if (
                isinstance(instruction, stim.DemInstruction)
                and instruction.type == "error"
            ):
                num_errors += 1
        return csc_matrix((0, num_errors), dtype=bool)

    # Data for constructing the CSC matrix directly
    row_indices: List[int] = (
        []
    )  # Stores the observable index (row) for each non-zero entry
    col_pointers: List[int] = [0]  # indptr array for CSC format
    data: List[bool] = []  # Stores the non-zero values (always True for us)

    error_index_counter = 0  # Tracks the current error mechanism (column index)

    for instruction in flattened_dem:
        # We only care about 'error' instructions for the columns
        if isinstance(instruction, stim.DemInstruction) and instruction.type == "error":
            instruction_targets = instruction.targets_copy()

            # Check which observables this error affects
            for target in instruction_targets:
                if target.is_logical_observable_id():
                    observable_id = int(str(target)[1:])
                    if observable_id >= num_observables:
                        # This shouldn't happen if num_observables is correct
                        raise ValueError(
                            f"Error instruction {error_index_counter} targets "
                            f"observable L{observable_id}, but DEM only reports "
                            f"{num_observables} observables."
                        )
                    row_indices.append(observable_id)
                    data.append(True)

            # Update the column pointer: points to the start of the *next* column's data
            col_pointers.append(len(row_indices))
            error_index_counter += 1

        # Ignore other instruction types like 'detector', 'logical_observable'

    num_columns = error_index_counter

    # Create the CSC matrix
    obs_matrix = csc_matrix(
        (data, row_indices, col_pointers),
        shape=(num_observables, num_columns),
        dtype=bool,
    )

    return obs_matrix


def save_circuit_diagram(circuit: stim.Circuit, path: str, type: str = "timeline-svg"):
    """
    Save a circuit diagram to a file.
    """
    print(circuit.diagram(type=type), file=open(path, "w"))
