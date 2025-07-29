# color-code-stim

Python package for simulating & decoding 2D color code circuits via the [concatenated MWPM decoder](https://quantum-journal.org/papers/q-2025-01-27-1609).

## Installation

Requires Python >= 3.11

```bash
pip install color-code-stim
```

## Quick Start

```python
from color_code_stim import ColorCode, NoiseModel

# Create noise model
noise = NoiseModel.uniform_circuit_noise(1e-3)

# Create color code instance
colorcode = ColorCode(
    d=5,                    # Code distance
    rounds=5,              # Syndrome extraction rounds
    circuit_type="tri",    # Circuit type
    noise_model=noise      # Noise configuration
)

# Run simulation
num_fails, info = colorcode.simulate(shots=10000, full_output=True)
```

See the [Getting Started Notebook](https://github.com/seokhyung-lee/color-code-stim/blob/main/getting_started.ipynb) for more details.

## Features

- Simulation of 2D color code circuits using [Stim](https://github.com/quantumlib/Stim)
- Concatenated Minimum-Weight Perfect Matching (MWPM) decoder implementation
- Support for multiple circuit types: memory experiments with triangular/rectangular patches, stability experiments, growing operation, and cultivation+growing circuit
- Monte Carlo simulation for decoder performance evaluation
- Comparative decoding with logical gap calculation

## Citation

```bibtex
@article{lee2025color,
  doi = {10.22331/q-2025-01-27-1609},
  title = {Color code decoder with improved scaling for correcting circuit-level noise},
  author = {Lee, Seok-Hyung and Li, Andrew and Bartlett, Stephen D.},
  journal = {{Quantum}},
  volume = {9},
  pages = {1609},
  year = {2025}
}
```