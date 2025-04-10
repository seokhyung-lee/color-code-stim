import stim
import math
from src.color_code_stim.stim_utils import separate_depolarizing_errors


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create an example circuit with depolarizing noise
    original_circuit = stim.Circuit(
        """
        H 0 1
        CNOT 0 1
        DEPOLARIZE1(0.01) 0
        DEPOLARIZE2(0.02) 0 1
        M 0 1
    """
    )

    print("Original Circuit:")
    print(original_circuit)
    print("-" * 20)

    # 2. Separate the errors using the new formulas
    separated_circuit = separate_depolarizing_errors(original_circuit)

    print("Circuit with Separated Errors (v2):")
    print(separated_circuit)
    print("-" * 20)

    # 3. Verify the probabilities (optional manual check with v2 formulas)
    p1 = 0.01
    if p1 == 0:
        q1 = 0.0
        pxz = 0.0
    else:
        term_under_sqrt = 1.0 - 4.0 * p1 / 3.0
        if term_under_sqrt < 0:
            term_under_sqrt = 0
        q1 = (1.0 - math.sqrt(term_under_sqrt)) / 2.0
        pxz = 2.0 * q1 * (1.0 - q1)
    print(f"DEPOLARIZE1(p={p1}) -> q1={q1:.6f}, pxz = {pxz:.6f}")

    p2 = 0.02
    if p2 == 0:
        q2 = 0.0
        p_component = 0.0
    else:
        term_in_power = 1.0 - 16.0 * p2 / 15.0
        if term_in_power < 0:
            term_in_power = 0
        q2 = (1.0 - term_in_power ** (1.0 / 8.0)) / 2.0
        p_component = 4.0 * q2 * (1.0 - q2) ** 3 + 4.0 * q2**3 * (1.0 - q2)
    print(f"DEPOLARIZE2(p={p2}) -> q2={q2:.6f}, p_component = {p_component:.6f}")

    # Example with REPEAT block
    circuit_with_repeat = stim.Circuit(
        """
        REPEAT 10 {
            H 0
            DEPOLARIZE1(0.03) 0
            CNOT 0 1
            DEPOLARIZE2(0.04) 0 1
        }
        M 0 1
    """
    )
    print("\nOriginal Circuit with REPEAT:")
    print(circuit_with_repeat)
    print("-" * 20)

    separated_repeat_circuit = separate_depolarizing_errors(circuit_with_repeat)
    print("Circuit with Separated Errors (v2, with REPEAT):")
    print(separated_repeat_circuit)