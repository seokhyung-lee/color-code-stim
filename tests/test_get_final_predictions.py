import numpy as np

from color_code_stim import _get_final_predictions

# --- Create Sample Data ---
# N = 3 (number of primary elements)
# M = 4 (secondary dimension)
# num_sets = 2

# Set 0
preds_obs_0 = {
    "red": np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]]),  # Shape (3, 4)
    "blue": np.array([[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]]),
}
weights_0 = {
    "red": np.array([0.5, 0.8, 0.1]),  # Shape (3,) - Min for k=2
    "blue": np.array([0.6, 0.2, 0.9]),  # Min for k=1
}

# Set 1
preds_obs_1 = {
    "red": np.array([[0, 0, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]]),
    "green": np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]]),
}
weights_1 = {
    "red": np.array([0.4, 0.7, 0.3]),  # Min for k=0
    "green": np.array([0.9, 0.3, 0.6]),  # Potential 2nd min for k=1
}

# Combine into lists
preds_obs_list = [preds_obs_0, preds_obs_1]
weights_list = [weights_0, weights_1]

# --- Expected Results ---
# k=0: min weight is 0.4 (set 1, red). Preds: [0, 0, 0, 0] -> [F, F, F, F]
# k=1: min weight is 0.2 (set 0, blue). Preds: [1, 1, 0, 0] -> [T, T, F, F]
# k=2: min weight is 0.1 (set 0, red). Preds: [1, 1, 1, 1] -> [T, T, T, T]

expected_preds_final = np.array(
    [[False, False, False, False], [True, True, False, False], [True, True, True, True]]
)
expected_best_colors = np.array(["red", "blue", "red"])
expected_weights_final = np.array([0.4, 0.2, 0.1])

# Logical Gap Calculation:
# Min weights per set:
# Set 0: min( [0.5, 0.8, 0.1], [0.6, 0.2, 0.9] ) -> [0.5, 0.2, 0.1]
# Set 1: min( [0.4, 0.7, 0.3], [0.9, 0.3, 0.6] ) -> [0.4, 0.3, 0.3]
# Stacked min weights: [[0.5, 0.2, 0.1], [0.4, 0.3, 0.3]] (shape 2, 3)
#
# k=0: values are [0.5, 0.4]. Sorted: [0.4, 0.5]. Gap: 0.5 - 0.4 = 0.1
# k=1: values are [0.2, 0.3]. Sorted: [0.2, 0.3]. Gap: 0.3 - 0.2 = 0.1
# k=2: values are [0.1, 0.3]. Sorted: [0.1, 0.3]. Gap: 0.3 - 0.1 = 0.2
expected_logical_gap = np.array([0.1, 0.1, 0.2])


# --- Run the function ---
print("--- Inputs ---")
print("preds_obs:")
for i, d in enumerate(preds_obs_list):
    print(f" Set {i}:")
    for k, v in d.items():
        print(f"  '{k}':\n{v}")
print("\nweights:")
for i, d in enumerate(weights_list):
    print(f" Set {i}:")
    for k, v in d.items():
        print(f"  '{k}': {v}")
print("-" * 20)

preds_final, best_colors, weights_final, logical_gap = _get_final_predictions(
    preds_obs_list, weights_list
)

# --- Print and Verify Outputs ---
print("\n--- Outputs ---")
print(f"preds_obs_final:\n{preds_final}")
print(f"best_colors: {best_colors}")
print(f"weights_final: {weights_final}")
print(f"logical_gap: {logical_gap}")
print("-" * 20)

print("\n--- Verification ---")
try:
    np.testing.assert_array_equal(preds_final, expected_preds_final)
    print("✅ preds_obs_final: OK")
except AssertionError as e:
    print(f"❌ preds_obs_final: FAIL\n{e}")

try:
    np.testing.assert_array_equal(best_colors, expected_best_colors)
    print("✅ best_colors: OK")
except AssertionError as e:
    print(f"❌ best_colors: FAIL\n{e}")

try:
    np.testing.assert_allclose(weights_final, expected_weights_final)
    print("✅ weights_final: OK")
except AssertionError as e:
    print(f"❌ weights_final: FAIL\n{e}")

try:
    assert logical_gap is not None, "Logical gap should not be None"
    np.testing.assert_allclose(logical_gap, expected_logical_gap)
    print("✅ logical_gap: OK")
except AssertionError as e:
    print(f"❌ logical_gap: FAIL\n{e}")

print("-" * 20)

# --- Test Case: Single Set ---
print("\n--- Test Case: Single Set ---")
preds_obs_single = [preds_obs_0]
weights_single = [weights_0]

# Expected for single set
expected_preds_single = np.array(
    [
        [True, False, True, False],  # k=0, min=0.5 ('red')
        [True, True, False, False],  # k=1, min=0.2 ('blue')
        [True, True, True, True],  # k=2, min=0.1 ('red')
    ]
)
expected_colors_single = np.array(["red", "blue", "red"])
expected_weights_single = np.array([0.5, 0.2, 0.1])
expected_gap_single = None  # Should be None for single set

preds_final_s, colors_s, weights_s, gap_s = _get_final_predictions(
    preds_obs_single, weights_single
)

print(f"preds_obs_final:\n{preds_final_s}")
print(f"best_colors: {colors_s}")
print(f"weights_final: {weights_s}")
print(f"logical_gap: {gap_s}")

try:
    np.testing.assert_array_equal(preds_final_s, expected_preds_single)
    print("✅ preds_obs_final (single): OK")
except AssertionError as e:
    print(f"❌ preds_obs_final (single): FAIL\n{e}")

try:
    np.testing.assert_array_equal(colors_s, expected_colors_single)
    print("✅ best_colors (single): OK")
except AssertionError as e:
    print(f"❌ best_colors (single): FAIL\n{e}")

try:
    np.testing.assert_allclose(weights_s, expected_weights_single)
    print("✅ weights_final (single): OK")
except AssertionError as e:
    print(f"❌ weights_final (single): FAIL\n{e}")

try:
    assert (
        gap_s is expected_gap_single
    ), f"Expected logical_gap to be None, but got {gap_s}"
    print("✅ logical_gap (single): OK")
except AssertionError as e:
    print(f"❌ logical_gap (single): FAIL\n{e}")

print("-" * 20)
