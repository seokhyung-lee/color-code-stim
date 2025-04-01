from src.color_code_stim.cultivation import _adjust_cultivation_circuit_coords
from src.color_code_stim.utils import get_project_folder

import stim

if __name__ == "__main__":
    project_folder = get_project_folder()
    circuit = stim.Circuit.from_file(
        project_folder / "assets" / "cultivation_circuits" / "d3_p0.001.stim"
    )
    adjusted_circuit = _adjust_cultivation_circuit_coords(circuit, 3)
    print(adjusted_circuit)