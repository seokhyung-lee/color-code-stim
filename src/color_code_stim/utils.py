import time
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from statsmodels.stats.proportion import proportion_confint


def timeit(func: Callable) -> Callable:
    @wraps(func)
    def wrap(*args, **kwargs):
        if args[0].benchmarking:
            start = time.time()
            res = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"Elapsed time for function '{func.__name__}': {elapsed:.2e} s")
        else:
            res = func(*args, **kwargs)
        return res

    return wrap


def get_pfail(
    shots: int | np.ndarray,
    fails: int | np.ndarray,
    alpha: float = 0.01,
    confint_method: str = "wilson",
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """
    Calculate the failure probability and confidence interval.

    This function computes the estimated failure probability and the half-width
    of its confidence interval based on the number of shots and failures.

    Parameters
    ----------
    shots : int or array-like
        Total number of experimental shots.
    fails : int or array-like
        Number of failures observed.
    alpha : float, default 0.01
        Significance level for the confidence interval (e.g., 0.01 for 99%
        confidence).
    confint_method : str, default "wilson"
        Method to calculate confidence intervals. See
        statsmodels.stats.proportion.proportion_confint for available options.

    Returns
    -------
    pfail : float or array-like
        Estimated failure probability (midpoint of confidence interval).
    delta_pfail : float or array-like
        Half-width of the confidence interval.
    """
    pfail_low, pfail_high = proportion_confint(
        fails, shots, alpha=alpha, method=confint_method
    )
    pfail = (pfail_low + pfail_high) / 2
    delta_pfail = pfail_high - pfail

    return pfail, delta_pfail


def _get_final_predictions(
    preds_obs: List[Dict[str, np.ndarray]], weights: List[Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Get final predictions and weights from multiple logical values and colors

    Args:
        preds_obs: A list where each element is a dictionary.
                   Keys are color strings, values are 2D numpy arrays (int)
                   of shape (N, M) representing predictions.
                   preds_obs[i][c] is the prediction array for set i, color c.
        weights: A list with the same structure as preds_obs.
                 Keys are color strings, values are 1D numpy arrays (float)
                 of shape (N,) representing weights.
                 weights[i][c][k] is the weight for the k-th element's
                 prediction in set i, color c.

    Returns:
        A tuple containing:
        - preds_obs_final: 2D numpy array (bool) of shape (N, M).
                           preds_obs_final[k, l] is the boolean converted value from
                           preds_obs[i][c][k, l] corresponding to the (i, c)
                           pair that yields the minimum weight[i][c][k].
        - best_colors: 1D numpy array (str) of shape (N,).
                       best_colors[k] is the color 'c' corresponding to the
                       minimum weight for element k.
        - weights_final: 1D numpy array (float) of shape (N,).
                         weights_final[k] is the minimum weight found across
                         all sets 'i' and colors 'c' for element k.
        - logical_gap: 1D numpy array (float) of shape (N,) or None.
                       The difference between the second smallest and the smallest
                       minimum weight found per set 'i' for each element 'k'.
                       Calculated only if len(preds_obs) > 1, otherwise None.

    Raises:
        ValueError: If input lists are empty, shapes are inconsistent, or
                    dictionaries are empty.
    """
    num_sets = len(preds_obs)
    if num_sets == 0 or len(weights) != num_sets:
        raise ValueError(
            "Input lists 'preds_obs' and 'weights' must be non-empty and have the same length."
        )

    # --- Data Preparation and Validation ---
    all_weights_flat = []
    all_preds_flat = []
    all_indices_flat = []  # Stores tuples of (set_index, color_str)
    num_k = -1  # Number of primary elements (N)
    num_l = -1  # Secondary dimension (M)

    for i, (preds_dict, weights_dict) in enumerate(zip(preds_obs, weights)):
        if not weights_dict:  # Check for empty dictionary in weights
            raise ValueError(f"Weight dictionary at index {i} is empty.")
        if not preds_dict:  # Check for empty dictionary in preds
            raise ValueError(f"Prediction dictionary at index {i} is empty.")
        if weights_dict.keys() != preds_dict.keys():
            raise ValueError(f"Keys mismatch between weights and preds at index {i}.")

        for color, w_arr in weights_dict.items():
            if color not in preds_dict:
                # This case is covered by the keys mismatch check above,
                # but included for clarity if that check were removed.
                raise ValueError(
                    f"Color '{color}' found in weights but not in preds at index {i}."
                )

            p_arr = preds_dict[color]

            # Validate shapes and determine N, M on first valid entry
            if num_k == -1:
                if w_arr.ndim != 1 or p_arr.ndim != 2:
                    raise ValueError(
                        f"Weight must be 1D and Preds must be 2D. Found shapes {w_arr.shape} and {p_arr.shape} for i={i}, c='{color}'."
                    )
                num_k = w_arr.shape[0]
                num_l = p_arr.shape[1]
                if num_k == 0 or num_l == 0:
                    raise ValueError("Dimensions N and M must be greater than 0.")
            elif w_arr.shape != (num_k,) or p_arr.shape != (num_k, num_l):
                raise ValueError(
                    f"Inconsistent shapes found for i={i}, c='{color}'. "
                    f"Expected weight shape ({num_k},) but got {w_arr.shape}. "
                    f"Expected prediction shape ({num_k}, {num_l}) but got {p_arr.shape}."
                )

            all_weights_flat.append(w_arr)
            all_preds_flat.append(p_arr)
            all_indices_flat.append((i, color))  # Store set index and color

    if num_k == -1:
        # This happens if all dictionaries were empty, though caught earlier
        raise ValueError("No valid prediction/weight data found.")

    # --- Combine data for efficient processing ---
    # Stack weights: rows are combinations of (set, color), columns are k
    # Shape: (num_combinations, N)
    weights_stack = np.stack(all_weights_flat, axis=0)

    # Stack predictions: first dim is combinations, then N, then M
    # Shape: (num_combinations, N, M)
    preds_stack = np.stack(all_preds_flat, axis=0)

    # Convert indices to numpy array for potential advanced indexing (though used differently here)
    # Using object dtype because it contains strings (colors)
    indices_arr = np.array(
        all_indices_flat, dtype=object
    )  # Shape: (num_combinations, 2)

    # --- Find minimum weight and corresponding items for each k ---
    # Find the index of the minimum weight along the combinations dimension (axis=0) for each k
    # Shape: (N,)
    best_combo_indices = np.argmin(weights_stack, axis=0)

    # Get the minimum weights themselves (this is weights_final)
    # Shape: (N,)
    weights_final = np.min(weights_stack, axis=0)
    # Alternatively: weights_final = weights_stack[best_combo_indices, np.arange(num_k)]

    # Retrieve the corresponding (set_index, color) pairs for the minimums
    # Shape: (N, 2)
    winning_indices = indices_arr[best_combo_indices]

    # Extract the best colors
    # Shape: (N,)
    best_colors = winning_indices[:, 1].astype(str)  # Ensure correct dtype

    # Select the corresponding prediction rows using advanced indexing
    # We select from preds_stack using the best combination index for each k,
    # and we want all elements along the M dimension.
    # Shape: (N, M)
    preds_obs_final_int = preds_stack[best_combo_indices, np.arange(num_k), :]

    # Convert the final predictions to boolean as requested
    preds_obs_final = preds_obs_final_int.astype(bool)

    # --- Calculate Logical Gap (if applicable) ---
    logical_gap = None
    if num_sets > 1:
        min_weights_per_set = []
        # Group weights by original set index i
        weights_by_set = [[] for _ in range(num_sets)]
        for idx, w_arr in enumerate(all_weights_flat):
            set_idx, _ = all_indices_flat[idx]
            weights_by_set[set_idx].append(w_arr)

        # Calculate min weight within each set for each k
        for i in range(num_sets):
            if not weights_by_set[i]:
                # Handle case where a set might have no valid entries (though prevented by earlier checks)
                min_w_for_set_i = np.full(
                    (num_k,), np.inf
                )  # Assign infinity if set was empty
            else:
                # Stack weights for the current set: Shape (num_colors_in_set_i, N)
                current_set_weights = np.stack(weights_by_set[i], axis=0)
                # Find min across colors for each k: Shape (N,)
                min_w_for_set_i = np.min(current_set_weights, axis=0)
            min_weights_per_set.append(min_w_for_set_i)

        # Stack the minimum weights across sets: Shape (num_sets, N)
        min_weights_per_set_stack = np.stack(min_weights_per_set, axis=0)

        # Partition to find the two smallest minimums for each k
        # We need the 0th (smallest) and 1st (second smallest) elements along axis 0 (sets)
        # np.partition is efficient for finding k-th smallest elements
        partitioned_weights = np.partition(min_weights_per_set_stack, kth=1, axis=0)

        # Logical gap is the difference between the second smallest and smallest
        # Shape: (N,)
        logical_gap = partitioned_weights[1, :] - partitioned_weights[0, :]

    return preds_obs_final, best_colors, weights_final, logical_gap


def get_project_folder() -> Path:
    project_folder = Path(__file__).resolve().parents[2]
    return project_folder
