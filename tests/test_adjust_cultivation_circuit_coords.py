from src.color_code_stim.cultivation import _adjust_cultivation_circuit_coords
from pathlib import Path

import stim

if __name__ == "__main__":
    current_folder = Path(__file__).parent
    circuit = stim.Circuit.from_file(
        current_folder / "assets" / "cultivation_circuits" / "d3_p0.001.stim"
    )
    adjusted_circuit = _adjust_cultivation_circuit_coords(circuit, 3)
    print(adjusted_circuit)