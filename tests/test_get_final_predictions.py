import numpy as np
from typing import Tuple, Optional


def _get_final_predictions(
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Finds the best logical class and color index that minimizes the weight
    for each sample from a 3D weights array, and calculates the gap between
    the best and second-best logical class minimum weights.

    Parameters
    ----------
    weights : np.ndarray
        A 3D float array with shape (num_logical_classes, num_colors, num_samples).
        `weights[i, c, j]` is the weight for the j-th sample, i-th logical
        class, and c-th color index.

    Returns
    -------
    best_logical_classes : np.ndarray
        1D int array. The index of the best logical class for each sample.
    best_color_indices : np.ndarray
        1D int array. The index of the best color for each sample.
    weights_final : np.ndarray
        1D float array. The minimum weight found for each sample.
    logical_gap : Optional[np.ndarray]
        1D float array or None. The difference between the second smallest and
        smallest minimum weight across logical classes for each sample.
        Calculated only if `num_logical_classes > 1`.
    """
    if weights.size == 0:
        # Handle empty input array
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=float)
        return (
            empty_int,
            empty_int.copy(),
            empty_float,
            None,
        )  # Return two empty int arrays

    num_logical_classes, num_colors, num_samples = weights.shape

    if num_samples == 0:
        # Handle case with zero samples
        empty_int = np.array([], dtype=int)
        empty_float = np.array([], dtype=float)
        return (
            empty_int,
            empty_int.copy(),
            empty_float,
            None,
        )  # Return two empty int arrays

    # 1. Find the overall minimum weight and its index for each sample
    # Reshape to (logical_class * color, sample) to find the flat index of the min
    reshaped_weights = weights.reshape(num_logical_classes * num_colors, num_samples)

    flat_min_indices = np.argmin(reshaped_weights, axis=0)
    weights_final = np.min(
        reshaped_weights, axis=0
    )  # Or: reshaped_weights[flat_min_indices, np.arange(num_samples)]

    # 2. Decode the flat index back into logical class and color index
    best_logical_classes = flat_min_indices // num_colors
    best_color_indices = (
        flat_min_indices % num_colors
    )  # This is already the desired int array

    # 3. Calculate logical_gap if needed
    logical_gap: Optional[np.ndarray] = None
    if num_logical_classes > 1:
        # Find the minimum weight per logical class for each sample
        # Shape: (num_logical_classes, num_samples)
        min_weights_per_class = np.min(weights, axis=1)  # Min across color axis

        # Sort along the logical class axis to find the smallest two
        # Shape: (num_logical_classes, num_samples)
        sorted_min_weights = np.sort(min_weights_per_class, axis=0)

        # Calculate the gap between the second smallest and the smallest
        # Shape: (num_samples,)
        logical_gap = sorted_min_weights[1] - sorted_min_weights[0]

    # Return best_color_indices directly instead of mapped string colors
    return best_logical_classes, best_color_indices, weights_final, logical_gap


# Example Usage:
# Recreate the 3D numpy array
weights_data_list = [
    {
        "r": np.array([0.5, 0.1, 0.8]),
        "g": np.array([0.2, 0.9, 0.3]),
        "b": np.array([0.7, 0.4, 0.6]),
    },  # LC 0
    {
        "r": np.array([0.3, 0.6, 0.4]),
        "g": np.array([0.8, 0.2, 0.1]),
        "b": np.array([0.4, 0.7, 0.9]),
    },  # LC 1
]

num_lc = len(weights_data_list)
num_samp = len(weights_data_list[0]["r"])
weights_array = np.zeros((num_lc, 3, num_samp))

color_map_inv = {"r": 0, "g": 1, "b": 2}
for i, class_data in enumerate(weights_data_list):
    for color, values in class_data.items():
        weights_array[i, color_map_inv[color], :] = values

# Call the updated function
best_lc, best_c_idx, final_w, gap = _get_final_predictions(weights_array)

# Print results
print("--- Example 1 (Multiple Logical Classes - Array Input, Int Color Output) ---")
print(f"Input Array Shape: {weights_array.shape}")
print(f"Best Logical Classes: {best_lc}")
# Expected: [0 0 1]
print(f"Best Color Indices: {best_c_idx}")  # Now returns indices
# Expected: [1 0 1] (Indices for 'g', 'r', 'g')
print(f"Final Weights: {final_w}")
# Expected: [0.2 0.1 0.1]
print(f"Logical Gap: {gap}")
# Expected: [0.1 0.1 0.2]


# Example with a single logical class
weights_single_list = [
    {
        "r": np.array([0.5, 0.1, 0.8]),
        "g": np.array([0.2, 0.9, 0.3]),
        "b": np.array([0.7, 0.4, 0.6]),
    }
]
weights_single_array = np.zeros((1, 3, num_samp))
for color, values in weights_single_list[0].items():
    weights_single_array[0, color_map_inv[color], :] = values

best_lc_single, best_c_idx_single, final_w_single, gap_single = (
    _get_final_predictions(weights_single_array)
)

print("\n--- Example 2 (Single Logical Class - Array Input, Int Color Output) ---")
print(f"Input Array Shape: {weights_single_array.shape}")
print(f"Best Logical Classes: {best_lc_single}")  # Expected: [0 0 0]
print(f"Best Color Indices: {best_c_idx_single}")  # Now returns indices
# Expected: [1 0 1] (Indices for 'g', 'r', 'g')
print(f"Final Weights: {final_w_single}")  # Expected: [0.2 0.1 0.3]
print(f"Logical Gap: {gap_single}")  # Expected: None