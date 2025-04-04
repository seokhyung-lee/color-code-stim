import numpy as np

np.random.seed(42)  # For reproducibility


def original_approach(error_preds_all, weights_all):
    """Original implementation with for loop"""
    num_samples = error_preds_all.shape[0]
    num_logical_value_combs = weights_all.shape[1]
    predecoding_preds_all = []

    # Make a copy to avoid modifying the original
    weights_all_copy = weights_all.copy()

    for i in range(num_logical_value_combs):
        min_weight_idx = np.argmin(weights_all_copy, axis=1)
        preds = error_preds_all[np.arange(num_samples), min_weight_idx]
        if i < num_logical_value_combs - 1:
            weights_all_copy[np.arange(num_samples), min_weight_idx] = np.inf
        predecoding_preds_all.append(preds)

    return predecoding_preds_all


def concise_approach(error_preds_all, weights_all):
    """Concise implementation without for loop"""
    num_samples = error_preds_all.shape[0]

    # Get the sorted indices for weights (ascending order)
    idx_sorted = np.argsort(weights_all, axis=1)

    # Use advanced indexing to get predictions in sorted order
    sorted_preds = error_preds_all[np.arange(num_samples)[:, np.newaxis], idx_sorted]

    # Transpose to match original output shape
    predecoding_preds_all = np.transpose(sorted_preds, (1, 0, 2))

    return list(predecoding_preds_all)  # Convert to list for direct comparison


def test_equivalence():
    # Generate test data
    num_samples = 5
    num_logical_value_combs = 4
    num_org_errors = 3

    # Random boolean array for error predictions
    error_preds_all = np.random.choice(
        [True, False], size=(num_samples, num_logical_value_combs, num_org_errors)
    )

    # Random weights (using floats between 0 and 10)
    weights_all = np.random.random((num_samples, num_logical_value_combs)) * 10

    # Run both approaches
    original_result = original_approach(error_preds_all, weights_all)
    concise_result = concise_approach(error_preds_all, weights_all)

    # Compare results
    are_equal = True
    for i in range(len(original_result)):
        if not np.array_equal(original_result[i], concise_result[i]):
            are_equal = False
            print(f"Mismatch at index {i}")
            print(f"Original: {original_result[i]}")
            print(f"Concise:  {concise_result[i]}")

    if are_equal:
        print("✅ Both implementations produce identical results!")
    else:
        print("❌ The implementations produce different results.")

    # Also verify entire array structure matches
    print(
        f"\nOriginal result shape: {len(original_result)} items, each with shape {original_result[0].shape}"
    )
    print(
        f"Concise result shape: {len(concise_result)} items, each with shape {concise_result[0].shape}"
    )

    # Show a couple examples for visual verification
    print(f"\nExample output (first item):")
    print(f"Original: {original_result[0]}")
    print(f"Concise:  {concise_result[0]}")

    # Detailed comparison with weights to verify sorting
    print("\nDetailed comparison for first sample:")
    sample_idx = 0
    print(f"Weights: {weights_all[sample_idx]}")

    # Show indices in order selected by original algorithm
    orig_order = []
    weights_copy = weights_all[sample_idx].copy()
    for _ in range(num_logical_value_combs):
        idx = np.argmin(weights_copy)
        orig_order.append(idx)
        weights_copy[idx] = np.inf

    # Show indices from sorted order
    sorted_order = np.argsort(weights_all[sample_idx])

    print(f"Original selection order: {orig_order}")
    print(f"Sorted indices order:     {sorted_order}")

    # Extra: check full array equality
    orig_array = np.array(original_result)
    concise_array = np.array(concise_result)
    print(f"\nFull array equality: {np.array_equal(orig_array, concise_array)}")


if __name__ == "__main__":
    test_equivalence()
