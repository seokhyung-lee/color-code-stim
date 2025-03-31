import io
import re

import numpy as np
import stim
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


def dem_to_parity_check(dem: stim.DetectorErrorModel):
    """
    Convert a detector error model (DEM) into a parity check matrix and probability vector.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model to convert.

    Returns
    -------
    H : csc_matrix
        A binary matrix of shape (number of detectors, number of errors)
        where H[i, j] = 1 if detector i is involved in error j.
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
            data.append(1)

    H = csc_matrix(
        (data, (rows, cols)), shape=(num_detectors, num_errors), dtype=np.int8
    )
    p = np.array(probabilities)

    return H, p
